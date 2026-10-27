/* GX module -> PICA200 (citro3d).
 *
 * GX state is kept host-side, set by the hand-written GX calls below. Vertex
 * data reaches us as a byte stream: GXBegin pushes the draw command (opcode,
 * vertex count) and the game then writes the vertices to the write-gather
 * pipe 0xCC008000 itself (inline GXPosition* etc.), see gx_fifo_write. A
 * primitive is decoded and drawn as soon as its last byte arrives.
 *
 * Vertices are decoded on the CPU (Dolphin's vertex loader semantics) into
 * untransformed PICA vertices; the GX transform, COLOR0/ALPHA0 lighting and
 * texgen run in the PICA vertex shader (vshader.v.pica) from per-draw
 * uniforms, which also map the EFB to the 400x240 top screen. TEV stages are translated to PICA texenv stages; what the
 * PICA cannot express is logged once (docs/recomp_unknowns.md, "GX"). */
#include <3ds.h>
#include <citro3d.h>
#include <stdio.h>
#include <stdlib.h>
#include "ppc.h"
#include "vshader_shbin.h"

#define LOG() RECOMP_LOG()
/* unemulated GX features: the first hit is logged with its frame, and every
 * site's hit count goes into the "# gx gaps" line with the frame stats */
#define GAP_SITES 64
static struct { const char *fmt; u32 n; } gap[GAP_SITES];
static int gap_n;
static u32 frames;        /* GXCopyDisp handled (GX thread) */
static u32 game_frames;   /* GXCopyDisp called (game thread); equal to frames without the GX thread */
u32 recomp_gx_dbg[3];
/* sites are hit from both threads (ONCE: GX thread / handlers, ONCE_G: game thread):
 * the slot is taken atomically, the line is one recomp_log call */
#define ONCE_AT(fr, ...) do { static int id_ = -1; \
    if (id_ < 0) { int k_ = __atomic_fetch_add(&gap_n, 1, __ATOMIC_RELAXED); id_ = k_ < GAP_SITES ? k_ : GAP_SITES - 1; \
        gap[id_].fmt = #__VA_ARGS__; char m_[256]; snprintf(m_, sizeof m_, __VA_ARGS__); \
        recomp_log("# gx: frame %lu: %s", (unsigned long)(fr), m_); } \
    gap[id_].n++; } while (0)
#define ONCE(...) ONCE_AT(frames, __VA_ARGS__)
#define ONCE_G(...) ONCE_AT(game_frames, __VA_ARGS__)
static void gap_dump(void) {
    int n = gap_n < GAP_SITES ? gap_n : GAP_SITES;
    if (!n) return;
    recomp_log("# gx gaps (hits since start):");
    for (int i = 0; i < n; i++) if (gap[i].fmt) recomp_log(" [%lu] %.40s;", (unsigned long)gap[i].n, gap[i].fmt + 1);   /* fmt is set just after the slot is taken */
    recomp_log("\n");
}
#define GUEST(a) (((a) & RAM_MASK) | 0x80000000u)

/* ---------------------------------------------------------------- state */
enum { A_PNMTX = 0, A_POS = 9, A_NRM = 10, A_CLR0 = 11, A_CLR1 = 12, A_TEX0 = 13, A_MAX = 26 };

typedef struct { u8 cnt, type, frac; } VtxFmt;
typedef struct { u8 a, b, c, d, op, bias, scale, clamp, out; } TevSide;
typedef struct { TevSide c, a; u8 coord, map, chan, ksel_c, ksel_a, ras_swap, tex_swap; } TevStage;
typedef struct { u8 enable, amb_src, mat_src, attn_fn, diff_fn; u8 light_mask; } Chan;
typedef struct { float pos[3], dir[3], a[3], k[3]; u8 color[4]; } Light;
typedef struct { u8 func, src, normalize; u32 mtx; } TexGen;
/* min_lod/max_lod in 1/16 (TX_SETMODE1), lod_bias in 1/32 (TX_SETMODE0) */
typedef struct { u32 img; u16 w, h; u8 fmt, wrap_s, wrap_t, min_f, mag_f, ci; u32 tlut; u8 min_lod, max_lod; s8 lod_bias; } TexMap;
typedef struct { u32 addr; u8 fmt; u16 n; } Tlut;

static struct {
    u8 desc[A_MAX];
    VtxFmt vat[8][A_MAX];
    u32 array_base[12], array_stride[12];
    float xf[0x600];           /* XF matrix memory: pos/tex 0x000, normal 0x400 */
    u32 cur_mtx, tex_mtx_idx[8];
    float proj[6]; u32 proj_ortho;
    float vp[6];
    s32 sc[4], sc_off[2];
    u32 n_texgens, n_chans, n_stages;
    TexGen texgen[8];
    TevStage tev[16];
    u8 tev_reg[4][4], kcolor[4][4];  /* PREV, REG0..2 / K0..3 (RGBA, 0..255) */
    s16 tev_s10[4][4];         /* the same registers as set, signed (GXSetTevColorS10) */
    Chan chan[4];              /* COLOR0, COLOR1, ALPHA0, ALPHA1 */
    u8 mat[2][4], amb[2][4];
    Light light[8];
    TexMap map[8];
    Tlut tlut[20];
    u8 blend_type, blend_src, blend_dst, logic_op;
    u8 z_enable, z_func, z_update, color_update, alpha_update;
    u8 cull, alpha_cmp[5];
    u8 clear_color[4]; u32 clear_z;
    u16 efb_w, efb_h;
} gx;   /* GX thread (see "GX thread" below) */
/* the game thread's own copies of what it reads itself: the vertex format (where a
 * FIFO primitive ends) and projection/viewport (GXGetProjectionv, GXGetViewportv) */
static struct { u8 desc[A_MAX]; VtxFmt vat[8][A_MAX]; float proj[6]; u32 proj_ortho; float vp[6]; } gs;
static u32 cur_r1;   /* guest r1 recorded with the draw being handled (range_clean) */
/* write-tracking epoch for a cache entry about to hash guest memory: taken before the
 * hash (the game thread advances it at GXCopyDisp, after a barrier), so every write
 * stamped below it happened before the hash read */
static u32 epoch_now(void) { u32 e = *(volatile u32 *)&recomp_wr_epoch; __dmb(); return e; }
/* time buckets of a draw: one tick read per boundary, each closing the interval since the
 * previous one (svcGetSystemTick is an SVC; Citra adds 150 ticks to every call, its cost
 * measured on an o3DS). t_start opens a draw's chain; time taken out of it (the texture
 * cache's slow path, timed on its own) moves t_last forward. */
static u64 t_last;
static void t_start(void) { t_last = recomp_ticks(); }
static void t_mark(int b) { u64 t = recomp_ticks(); recomp_prof[b] += t - t_last; t_last = t; }
static void t_mark2(int b, int part) { u64 t = recomp_ticks(); recomp_prof[b] += t - t_last; recomp_prof[part] += t - t_last; t_last = t; }

/* ---------------------------------------------------------------- PICA side */
#define VB_VERTS 0x6000      /* per frame half: immediate draws and display-list draws the vertex cache does not take; u16 indices are relative to vb_base (rebased) */
#define IB_INDICES 0x30000   /* per frame half (immediate draws; cached lists are drawn from vc_ixa) */
#define VC_VERTS 0x14000     /* vertex cache arena (5.4 MB) */
typedef struct { float pos[4]; u8 mat[4]; float nrm[3], tc[3][3]; } PVtx;   /* vshader.v.pica inputs */
/* Vertex packing (sdmc:/cr_vtxpack_off.txt = every vertex a PVtx; bottom-screen button "Vertex
 * pack"). Display-list draws (the vertex / display-list caches) store only the shader inputs the
 * draw reads, in the GX source format where the PICA has one (s16 -> GPU_SHORT, s8 -> GPU_BYTE,
 * u8 -> GPU_UNSIGNED_BYTE, f32 -> GPU_FLOAT; u16 is converted to float): the position scale 2^-frac
 * goes to the vscale uniform (exact: the PICA converts the integer to float24 exactly and the
 * power-of-2 scale commutes with rounding), a texcoord's into its texm rows. The normal is left out
 * when the light loop is off and it feeds no texgen; texcoord units nothing feeds read a fixed
 * attribute (0, 0, 1, 1), the value PVtx has there. A texcoord from TEXn is 2 components (the
 * loader's z = 0) when its texm column 2 is zero, else z = 1 is stored. Every attribute is a
 * multiple of 4 bytes, so offsets stay aligned. One shader program; per-draw C3D_AttrInfo. */
typedef struct { u8 stride, nl, pk, c2z; u8 reg[6], fmt[6], cnt[6]; float vs[2], ts[3]; u8 tn[3]; u32 id, ush; } VLay;   /* loaders in memory order; ts/tn: texm scale and columns per unit; id: lay_id; ush: vs/ts/tn encoded (plan_finish), 0 = no scale */
/* attribute configuration as one number: per register, loaded (1) | format (2) | count - 1 (2) */
static u32 lay_id(const VLay *l) {
    u32 id = 0;
    for (int i = 0; i < l->nl; i++) id |= (1u | (u32)l->fmt[i] << 1 | (u32)(l->cnt[i] - 1) << 3) << (5 * l->reg[i]);
    return id;
}
static VLay lay_full = {sizeof(PVtx), 6, 0, 0, {0, 1, 2, 3, 4, 5}, {GPU_FLOAT, GPU_UNSIGNED_BYTE, GPU_FLOAT, GPU_FLOAT, GPU_FLOAT, GPU_FLOAT},
                              {4, 4, 3, 3, 3, 3}, {1, 1}, {1, 1, 1}, {0, 0, 0}, 0, 0};   /* id set in recomp_gx_init */
static VLay cur_lay;      /* layout of the PICA attribute buffer (use_buf; stride 0: none yet) */
int recomp_vtxpack_off;   /* read in recomp_gx_start; flipped by the menu between frames */
static int va_on;   /* this frame is a stats frame (va_count; set in frame_begin: ARM11 has no divide) */
static u32 st_dec_v, st_dec_b, st_lay_sw, st_vu;   /* vertices decoded and their bytes; C3D_AttrInfo changes; set_vtx_unif uploads (300-frame line) */

static C3D_RenderTarget *target;
static DVLB_s *vsh_dvlb;
static shaderProgram_s vsh;
static PVtx *vbuf[2];
static u16 *ibuf[2];
static u32 vb_used, ib_used, vb_half;
/* display-list vertex cache, see vc_draw: decoded vertices in vc_arena, index lists in vc_ixa,
 * both linear memory the GPU reads in place, each allocated through a ring (ring_alloc) */
#define VC_IXN 0x30000       /* index ring (u16) */
static u8 *vc_arena;   /* allocations in 4-byte units (vring), aligned to the vertex size */
static u16 *vc_ixa;
static u32 vc_hits, vc_misses;
#define VC_REC 32768   /* allocation records per ring */
typedef struct { u32 off, len, last, owner; } VcRec;   /* owner: OWN_DL | dlcache index, a vcache index, or OWN_NONE */
typedef struct {
    u8 *base; u32 unit, size;   /* units of unit bytes */
    u32 top, live, evicted, fails, dlo, dhi;   /* dlo..dhi: written since the last flush */
    u32 nfree, qh, qn;          /* free records; queue of live ones in arena order from top */
    u16 freel[VC_REC], q[VC_REC];
    VcRec rec[VC_REC];
} VcRing;
static VcRing vring, iring;   /* vertices (4-byte units), indices (u16) */
static void ring_flush(VcRing *g);
static u8 mat_vtx[4], mat_reg[4];   /* material colour per component: vertex CLR0 or register */
static float ushadow[96][4];        /* last value written per float uniform */
static int u_clip, u_pos, u_nrm, u_tex, u_amb, u_lcfg, u_lights, u_blit, u_iloop, u_eye = -1, u_vsc;
static int frame_open, clear_pending = 1;
static int scissor_lost = 1;   /* C3D_FrameDrawOn -> C3D_SetViewport disables the scissor: resend it */

/* Draw batching: a draw's indices are appended to the pending C3D_DrawElements
 * while no PICA state changed since the previous draw (same vertex buffer,
 * indices contiguous in ibuf). Every citro3d state call is preceded by
 * pend_flush(), since citro3d applies state when the draw is issued.
 * sdmc:/cr_batch_off.txt = one C3D_DrawElements per draw (draws are still
 * counted as mergeable). */
static u16 *pend_ix;
static u16 *last_end;   /* end of the previous draw's indices; NULL after a state change */
static u32 pend_n, st_merged, batch_off;   /* batch_off: read in recomp_gx_start */
static void pend_issue(void) {
    if (!pend_n) return;
    C3D_DrawElements(GPU_TRIANGLES, pend_n, C3D_UNSIGNED_SHORT, pend_ix);
    recomp_prof[PROF_GPU_DRAWS]++; recomp_prof[PROF_GPU_IDX] += pend_n; recomp_prof[PROF_GPU_VBYTES] += pend_n * cur_lay.stride;
    pend_n = 0;
}
static void pend_flush(void) { last_end = NULL; pend_issue(); }

static C3D_RenderTarget *target_r;   /* right eye; allocated here, before VRAM textures can fill VRAM */
static int st_out;                     /* target_r's output is set (the slider was raised once) */
void recomp_gx_init(void) {
    C3D_Init(C3D_DEFAULT_CMDBUF_SIZE * 8);   /* 2 MB: battle frames need over 1 MB (~112 words per draw) */
    target = C3D_RenderTargetCreate(240, 400, GPU_RB_RGBA8, GPU_RB_DEPTH24_STENCIL8);
    target_r = C3D_RenderTargetCreate(240, 400, GPU_RB_RGBA8, GPU_RB_DEPTH24_STENCIL8);
    C3D_RenderTargetSetOutput(target, GFX_TOP, GFX_LEFT,
        GX_TRANSFER_FLIP_VERT(0) | GX_TRANSFER_OUT_TILED(0) | GX_TRANSFER_RAW_COPY(0) |
        GX_TRANSFER_IN_FORMAT(GX_TRANSFER_FMT_RGBA8) | GX_TRANSFER_OUT_FORMAT(GX_TRANSFER_FMT_RGB8) |
        GX_TRANSFER_SCALING(GX_TRANSFER_SCALE_NO));
    vsh_dvlb = DVLB_ParseFile((u32 *)vshader_shbin, vshader_shbin_size);
    shaderProgramInit(&vsh);
    shaderProgramSetVsh(&vsh, &vsh_dvlb->DVLE[0]);
    C3D_BindProgram(&vsh);
    /* C3D_AttrInfo: use_buf, per vertex layout. Fixed attribute values per attribute slot: a
     * packed layout's fixed slots (after its loaders) read (0, 0, 1, 1), constant forever */
    for (int i = 0; i < 12; i++) C3D_FixedAttribSet(i, 0, 0, 1, 1);
    lay_full.id = lay_id(&lay_full);
    vbuf[0] = linearAlloc(sizeof(PVtx) * VB_VERTS * 2);
    vc_arena = linearAlloc(sizeof(PVtx) * VC_VERTS);   /* NULL: no vertex cache */
    if (!vc_arena) recomp_log("# gx: no linear memory for the vertex cache\n");
    vbuf[1] = vbuf[0] + VB_VERTS;
    ibuf[0] = linearAlloc(2 * IB_INDICES * 2);
    if (!vbuf[0] || !ibuf[0]) { recomp_log("# gx: linearAlloc of vertex/index buffers failed (linear free %lu)\n",(unsigned long)linearSpaceFree()); abort(); }
    ibuf[1] = ibuf[0] + IB_INDICES;
    vc_ixa = vc_arena ? linearAlloc(VC_IXN * 2) : NULL;
    if (vc_arena && !vc_ixa) { recomp_log("# gx: no linear memory for the index ring: no vertex cache\n"); linearFree(vc_arena); vc_arena = NULL; }
    VcRing *rg[2] = {&vring, &iring};
    for (int k = 0; k < 2; k++) {
        VcRing *g = rg[k];
        g->base = k ? (u8 *)vc_ixa : vc_arena; g->unit = k ? 2 : 4; g->size = k ? VC_IXN : VC_VERTS * sizeof(PVtx) / 4;
        g->dlo = 0xffffffffu; g->nfree = VC_REC;
        for (u32 i = 0; i < VC_REC; i++) g->freel[i] = (u16)(VC_REC - 1 - i);
    }
    shaderInstance_s *si = vsh.vertexShader;
    u_clip = shaderInstanceGetUniformLocation(si, "clipm"); u_pos = shaderInstanceGetUniformLocation(si, "posm");
    u_nrm = shaderInstanceGetUniformLocation(si, "nrmm"); u_tex = shaderInstanceGetUniformLocation(si, "texm");
    u_amb = shaderInstanceGetUniformLocation(si, "amb"); u_lcfg = shaderInstanceGetUniformLocation(si, "lcfg");
    u_lights = shaderInstanceGetUniformLocation(si, "lights"); u_blit = shaderInstanceGetUniformLocation(si, "blit");
    u_iloop = shaderInstanceGetUniformLocation(si, "iloop"); u_eye = shaderInstanceGetUniformLocation(si, "eye");
    u_vsc = shaderInstanceGetUniformLocation(si, "vscale");
    memset(ushadow, 0xff, sizeof ushadow);
    gx.efb_w = 640; gx.efb_h = 480;
    gx.color_update = gx.alpha_update = 1;
    gx.z_enable = 1; gx.z_func = 3; gx.z_update = 1;
    gx.clear_z = 0xffffff;
    gx.n_chans = 1; gx.n_stages = 1;
    for (int i = 0; i < 4; i++) gx.mat[i & 1][i >> 1] = 255;
    for (int i = 0; i < 16; i++) { gx.tev[i].coord = gx.tev[i].map = 0xff; gx.tev[i].chan = 4; }
    gx.alpha_cmp[0] = gx.alpha_cmp[3] = 7;
    for (int i = 0; i < 3; i++) gx.xf[i * 5] = 1;   /* PNMTX0 = identity */
}

/* EFB -> screen: the EFB copy source scaled to fit 400x240, centered */
static float efb_scale(void) {
    float sx = 400.0f / gx.efb_w, sy = 240.0f / gx.efb_h;
    return sx < sy ? sx : sy;
}
static float efb_x0(void) { return (400 - gx.efb_w * efb_scale()) * 0.5f; }
static float efb_y0(void) { return (240 - gx.efb_h * efb_scale()) * 0.5f; }

/* u16 indices reach 0x10000 vertices: past that, later draws index from a new base */
static u32 vb_base;
static const void *cur_buf;   /* PICA attribute buffer: vbuf + vb_base or a vertex cache window */
static void use_buf(const void *p, const VLay *l) {
    int same = cur_lay.stride && cur_lay.id == l->id;   /* the id fixes the stride too */
    if (p == cur_buf && same) return;
    pend_flush();
    if (!same) {
        C3D_AttrInfo *ai = C3D_GetAttrInfo();
        AttrInfo_Init(ai);
        u32 have = 0;
        for (int i = 0; i < l->nl; i++) { AttrInfo_AddLoader(ai, l->reg[i], l->fmt[i], l->cnt[i]); have |= 1u << l->reg[i]; }
        for (int r = 0; r < 6; r++) if (!(have >> r & 1)) AttrInfo_AddFixed(ai, r);
        cur_lay = *l;
        st_lay_sw++;
    }
    cur_buf = p;
    C3D_BufInfo *bi = C3D_GetBufInfo();
    BufInfo_Init(bi);
    BufInfo_Add(bi, p, l->stride, l->nl, 0x543210 & ((1u << (4 * l->nl)) - 1));
}

/* Stereo 3D (the mw3ds technique): the left eye is drawn as usual while its
 * command words are recorded; at frame end the right target is bound, the
 * shader's eye uniform gets the other sign and the same words are appended
 * again (GPUCMD_AddRawCommands), so the right eye costs GPU time but almost no
 * CPU. The recorded words must not depend on state from earlier frames: at
 * frame begin the left framebuffer is emitted first (outside the recording),
 * then all other citro3d state is marked dirty, so the first draw re-emits it
 * inside the recording. eye is written raw and never through citro3d, so no
 * recorded word sets it. Mid-frame submits split the words into ranges (the
 * FINALIZE padding between them is not recorded). On with the 3D slider;
 * sdmc:/cr_stereo_force.txt = "slider [sep] [conv]" forces it (Citra). Slider 0
 * = the mono path, no extra words. */
typedef struct { gxCmdQueue_s q; u32 *cmdBuf; size_t cmdBufSize; float cmdBufUsage; u32 flags; } C3DCtxHead;   /* citro3d 1.7.1 source/internal.h C3D_Context */
#define C3DiF_DrawUsed_ BIT(1)
/* AttrInfo BufInfo Effect Viewport Scissor Program TexEnvBuf LightEnv, TexAll, TexEnvAll (no FrameBuf, no shader code) */
#define C3DiF_RESEND (BIT(2) | BIT(3) | BIT(4) | BIT(6) | BIT(7) | BIT(8) | BIT(9) | BIT(10) | (7u << 23) | (0x3Fu << 26))
void C3Di_UpdateContext(void);
void C3Di_DirtyUniforms(GPU_SHADER_TYPE type);
extern C3DCtxHead __C3D_Context;   /* declared with only its head; gxQueue is the first member */
#define SR_MAX 16
static int eye_init;
static float st_force = -1, st_sep = 0.025f, st_conv = 3.0f;   /* sep: parallax per eye at w = inf, in NDC (200 px); conv: w at zero parallax (battle w is ~3-60) */
static int stereo, stereo_prev;
static u32 *sr_a[SR_MAX], sr_n[SR_MAX], *sr_open;
static int sr_k, sr_bad, cb_full, clear_r, st_r_done;
static u32 st_stereo, st_st_fall, st_st_words;
static u32 *cb_pos(void) { u32 *a, s, o; GPUCMD_GetBuffer(&a, &s, &o); return a + o; }
static void sr_close(void) {
    if (!sr_open) return;
    if (sr_k < SR_MAX) { sr_a[sr_k] = sr_open; sr_n[sr_k++] = cb_pos() - sr_open; } else sr_bad = 1;
    sr_open = NULL;
}
static void eye_write(float s) {
    C3D_FVec v; v.x = 0; v.y = s * st_sep; v.z = st_conv; v.w = 0;
    GPUCMD_AddWrite(GPUREG_VSH_FLOATUNIFORM_CONFIG, 0x80000000 | u_eye);
    GPUCMD_AddWrites(GPUREG_VSH_FLOATUNIFORM_DATA, (u32 *)&v, 4);
}
static float stereo_slider(void) { return st_force >= 0 ? st_force : osGet3DSliderState(); }
/* after C3D_FrameDrawOn(target): pick mono / stereo, start the recording */
static void stereo_begin(void) {
    stereo = stereo_slider() > 0 && u_eye >= 0;
    if (stereo && !st_out) {
        if (!target_r) { recomp_log("# gx: no VRAM for the right-eye target: stereo off\n"); u_eye = -1; stereo = 0; }
        else {
            C3D_RenderTargetSetOutput(target_r, GFX_TOP, GFX_RIGHT,
                GX_TRANSFER_FLIP_VERT(0) | GX_TRANSFER_OUT_TILED(0) | GX_TRANSFER_RAW_COPY(0) |
                GX_TRANSFER_IN_FORMAT(GX_TRANSFER_FMT_RGBA8) | GX_TRANSFER_OUT_FORMAT(GX_TRANSFER_FMT_RGB8) |
                GX_TRANSFER_SCALING(GX_TRANSFER_SCALE_NO));
            st_out = 1;
            gfxSet3D(true);   /* frames without the right target still show the left one to both eyes */
            recomp_log("# gx: stereo on at frame %lu (slider %.2f, sep %.3f, conv %.1f)\n", (unsigned long)frames, stereo_slider(), st_sep, st_conv);
        }
    }
    sr_k = 0; sr_bad = 0; sr_open = NULL; cb_full = 0; st_r_done = 0;
    if (!stereo) {
        if (stereo_prev || !eye_init) { eye_write(0); eye_init = 1; }
        stereo_prev = 0;
        return;
    }
    if (!stereo_prev) clear_r = 1;   /* stale since the last stereo frame: frame_begin clears it */
    stereo_prev = 1;
    C3Di_UpdateContext();   /* left framebuffer + any pending state, before the recording */
    eye_write(stereo_slider());   /* left eye: +, right eye: - (far objects shift apart, near ones cross) */
    C3DCtxHead *ctx = &__C3D_Context;
    ctx->flags |= C3DiF_RESEND;
    C3Di_DirtyUniforms(GPU_VERTEX_SHADER);   /* float uniforms citro3d ever wrote, bools, shader constants */
    for (int i = 0; i < 4; i++) C3D_IVUnifDirty[GPU_VERTEX_SHADER][i] = true;   /* DirtyUniforms misses ints */
    sr_open = cb_pos();
}
/* before C3D_FrameEnd: append the left eye's words for the right eye */
static void stereo_end(void) {
    if (!stereo) return;
    sr_close();
    u32 total = 0;
    for (int i = 0; i < sr_k; i++) total += sr_n[i];
    u32 *a, size, off;
    GPUCMD_GetBuffer(&a, &size, &off);
    if (sr_bad || cb_full || off + total + 256 > size) {   /* the right target is not used: this frame is shown in 2D */
        if (!st_st_fall++) recomp_log("# gx: stereo fallback at frame %lu (ranges %d%s%s, %lu words, %lu free)\n", (unsigned long)frames,
                                      sr_k, sr_bad ? " overflow" : "", cb_full ? ", cmdbuf full" : "", (unsigned long)total, (unsigned long)(size - off));
        return;
    }
    C3D_FrameDrawOn(target_r);
    C3Di_UpdateContext();   /* framebuffer flush of the left eye, right framebuffer, viewport */
    eye_write(-stereo_slider());
    for (int i = 0; i < sr_k; i++) GPUCMD_AddRawCommands(sr_a[i], sr_n[i]);
    __C3D_Context.flags |= C3DiF_DrawUsed_;   /* the replayed draws: C3D_FrameEnd flushes the right framebuffer */
    st_stereo++; st_st_words += total; st_r_done = 1;
}

/* Mid-frame GPU submit. C3D_FrameBegin stops citro3d's GX queue, and a
 * C3D_FrameSplit chunk only reaches the GPU once the queue runs, which citro3d
 * does in C3D_FrameEnd: so the GPU idled while the CPU built the frame. Every
 * KICK_WORDS command words the pending draw is issued, the vertex/index data
 * written since the last submit is flushed from the CPU cache, the chunk is
 * split off and the queue is started (gxCmdQueueRun), so the GPU draws it while
 * the CPU builds the rest. libctru has no getter for the queue: it is the first
 * member (gxQueue) of citro3d's non-static global C3D_Context __C3D_Context
 * (source/base.c, source/internal.h). Each submit takes one of the queue's 32
 * entries (C3D_Init), so at most KICK_MAX per frame. sdmc:/cr_gpukick_off.txt =
 * start with the old one submit at C3D_FrameEnd; pressing ZL toggles it. */
#define KICK_WORDS 2048
#define KICK_MAX 12
static int kick_off;   /* sdmc:/cr_gpukick_off.txt, read in recomp_gx_start */
int recomp_gx_kick_on;
u32 recomp_gx_kick_flips;
static u32 vb_fl, ib_fl, kicks, st_kick_cap;   /* vb_fl/ib_fl: vbuf/ibuf flushed up to here this frame */
static u64 kick_t0;   /* ticks of the frame's first submit (0: none yet) */
static void flush_new(void) {
    if (vb_used > vb_fl) GSPGPU_FlushDataCache(vbuf[vb_half] + vb_fl, (vb_used - vb_fl) * sizeof(PVtx));
    if (ib_used > ib_fl) GSPGPU_FlushDataCache(ibuf[vb_half] + ib_fl, (ib_used - ib_fl) * 2);
    ring_flush(&vring); ring_flush(&iring);
    vb_fl = vb_used; ib_fl = ib_used;
}
static void gpu_kick(void) {
    if (kicks >= KICK_MAX) {
        if (!st_kick_cap++) recomp_log("# gx: mid-frame submit cap (%d) hit at frame %lu\n", KICK_MAX, (unsigned long)frames);
        return;
    }
    pend_issue();   /* not pend_flush: the next draw may still merge (same stats as with submits off) */
    flush_new();
    sr_close();   /* the FINALIZE words C3D_FrameSplit adds are not recorded */
    C3D_FrameSplit(GX_CMDLIST_FLUSH);   /* void in citro3d 1.7.1; cb_off >= KICK_WORDS, so there is a chunk */
    if (stereo) sr_open = cb_pos();
    gxCmdQueueRun(&__C3D_Context.q);   /* the queue may have drained and stopped since the last submit */
    if (!kicks++) kick_t0 = recomp_ticks();
}
static u64 fe_t;   /* ticks at the previous C3D_FrameEnd */

/* Frame pacing. C3D_FRAME_SYNCDRAW runs C3D_FrameSync, which always waits for the
 * next vblank: every frame idled up to 16.7 ms at its first draw, counted as game
 * time (hw run 13 battle: ~14 of 53.6 ms). Now a frame waits for a vblank only when
 * none has passed since the previous frame began (a 60 Hz cap), then only for the
 * GPU to finish the previous frame. sdmc:/cr_vsync_on.txt = the old wait. With the GX
 * thread this wait is on that thread; the game thread meets it at GXDrawDone. */
static int vsync_on;   /* read in recomp_gx_start */
static u32 last_vbl;

static u8 tex_gen;
static void vt_frame(void);
static void frame_begin(void) {
    if (frame_open) return;
    recomp_gx_kick_on = !kick_off;
    u64 ts = recomp_ticks();
    if (!vsync_on && C3D_FrameCounter(0) == last_vbl) C3D_FrameSync();
    if (!C3D_FrameBegin(vsync_on ? C3D_FRAME_SYNCDRAW : 0)) recomp_log("# gx: C3D_FrameBegin failed at frame %lu\n", (unsigned long)frames);
    last_vbl = C3D_FrameCounter(0);
    recomp_prof[PROF_GX_SYNC] += recomp_ticks() - ts;
    vt_frame();
    /* the previous frame's queue (draws + display transfer) is finished here: its GPU time, in ms */
    if (frames) {
        u64 g = (u64)(C3D_GetDrawingTime() * (SYSCLOCK_ARM11 / 1000.0));
        recomp_prof[PROF_GPU] += g;
        recomp_prof[PROF_GPU_SPAN] += (kick_t0 ? fe_t - kick_t0 : 0) + g;
    }
    kicks = 0; kick_t0 = 0; vb_fl = ib_fl = 0;
    if (!C3D_FrameDrawOn(target)) recomp_log("# gx: C3D_FrameDrawOn failed at frame %lu\n", (unsigned long)frames);
    stereo_begin();
    frame_open = 1; scissor_lost = 1;
    vb_half = frames & 1; vb_used = ib_used = 0;
    vb_base = 0; cur_buf = NULL; va_on = frames % 300 == 0;
    use_buf(vbuf[vb_half], &lay_full);
    if (clear_pending || (stereo && clear_r)) {
        u32 c = gx.clear_color[0] << 24 | gx.clear_color[1] << 16 | gx.clear_color[2] << 8 | gx.clear_color[3];
        pend_flush();
        if (clear_pending) C3D_RenderTargetClear(target, C3D_CLEAR_ALL, c, gx.clear_z);
        if (stereo) C3D_RenderTargetClear(target_r, C3D_CLEAR_ALL, c, gx.clear_z);
        clear_pending = clear_r = 0;
    }
}

static u32 st_draws, st_tris, st_tex, st_evict, st_upload, st_upbytes, st_split;
volatile u32 recomp_gx_stat[4];
static RecursiveLock lin_lock;   /* below, at the linearAlloc wrappers */
static u32 st_dl_calls, st_dl_prims, st_dl_hits, st_dl_misses, st_dl_slow, pm_hits, pm_misses;
static void fb_dump(void);
static void va_log(void);
static void tex_resident(u32 *bytes, u32 *etc1);
static void vt_count(void);
static int vramtex_off;
static u32 st_vt_moves, st_vt_kb, st_vt_cap, st_vt_full, st_vt_evict;   /* VRAM textures, per 300-frame stats line */

static int in_draw;
static void frame_end(void) {
    if (in_draw) recomp_log("# gx: frame ended inside a draw at frame %lu (guest lr %08lx)\n", (unsigned long)frames, (unsigned long)cpu.lr);
    frame_begin();
    if (frames % 300 == 0) {   /* draw stats every 300 copies */
        recomp_log("# gx frame %lu: draws %lu tris %lu textured %lu evicted %lu uploads %lu (%lu KB) efb %dx%d clear %02x%02x%02x%02x cmdbuf-full skips %lu, cmdbuf %d%%, mid-frame submits %lu (cap skips %lu)\n", (unsigned long)frames,
            (unsigned long)st_draws, (unsigned long)st_tris, (unsigned long)st_tex, (unsigned long)st_evict, (unsigned long)st_upload, (unsigned long)(st_upbytes >> 10), gx.efb_w, gx.efb_h,
            gx.clear_color[0], gx.clear_color[1], gx.clear_color[2], gx.clear_color[3],
            (unsigned long)st_split, (int)(C3D_GetCmdBufUsage() * 100), (unsigned long)kicks, (unsigned long)st_kick_cap);
        recomp_log("# gx batch frame %lu: draws %lu mergeable %lu (%lu%%) -> %lu DrawElements%s\n", (unsigned long)frames, (unsigned long)st_draws,
            (unsigned long)st_merged, st_draws ? (unsigned long)(st_merged * 100 / st_draws) : 0ul,
            (unsigned long)(batch_off ? st_draws : st_draws - st_merged), batch_off ? " (batching off)" : "");
    }
    if (frames % 300 == 0) {
        u32 tb, te;
        tex_resident(&tb, &te);
        recomp_log("# gx tex frame %lu: resident %lu KB (as ETC1/ETC1A4 %lu KB)\n", (unsigned long)frames,
                   (unsigned long)(tb >> 10), (unsigned long)(te >> 10));
        vt_count();
        recomp_log("# gx vram tex frame %lu: %s; in VRAM %lu (%lu KB), linear %lu (%lu KB), VRAM free %lu KB; moves %lu (%lu KB), over cap %lu, VRAM full %lu, moved back to linear %lu\n",
                   (unsigned long)frames, vramtex_off ? "off" : "on", (unsigned long)recomp_vt_stat[0], (unsigned long)recomp_vt_stat[1],
                   (unsigned long)recomp_vt_stat[2], (unsigned long)recomp_vt_stat[3], (unsigned long)(vramSpaceFree() >> 10),
                   (unsigned long)st_vt_moves, (unsigned long)st_vt_kb, (unsigned long)st_vt_cap, (unsigned long)st_vt_full, (unsigned long)st_vt_evict);
        st_vt_moves = st_vt_kb = st_vt_cap = st_vt_full = st_vt_evict = 0;
        gap_dump();
        recomp_log("# gx vcache frame %lu: hits %lu misses %lu (draws), arena %lu KB, index ring %lu KB, evicted %lu + %lu, full %lu + %lu (since start), linear free %lu KB\n", (unsigned long)frames,
                   (unsigned long)vc_hits, (unsigned long)vc_misses, (unsigned long)(vring.live * 4 >> 10), (unsigned long)(iring.live * 2 >> 10),
                   (unsigned long)vring.evicted, (unsigned long)iring.evicted, (unsigned long)vring.fails, (unsigned long)iring.fails, (unsigned long)(linearSpaceFree() >> 10));
        recomp_log("# gx vpack frame %lu: %s; decoded this frame %lu vertices, %lu bytes (%lu per vertex); layout switches %lu, vscale/texm uploads %lu\n", (unsigned long)frames, recomp_vtxpack_off ? "off" : "on",
                   (unsigned long)st_dec_v, (unsigned long)st_dec_b, st_dec_v ? (unsigned long)(st_dec_b / st_dec_v) : 0ul, (unsigned long)st_lay_sw, (unsigned long)st_vu);
    }
    if (frames % 300 == 0) va_log();
    if (frames % 300 == 0)
        recomp_log("# gx dl frame %lu: calls %lu prims %lu (%lu per call) cache hits %lu misses %lu slow %lu, plan memo hits %lu misses %lu\n", (unsigned long)frames,
                   (unsigned long)st_dl_calls, (unsigned long)st_dl_prims, st_dl_calls ? (unsigned long)(st_dl_prims / st_dl_calls) : 0ul,
                   (unsigned long)st_dl_hits, (unsigned long)st_dl_misses, (unsigned long)st_dl_slow, (unsigned long)pm_hits, (unsigned long)pm_misses);
    if (frames % 300 == 0 && st_out) {
        recomp_log("# gx stereo frame %lu: slider %.2f, stereo frames %lu, fallbacks %lu, replayed %lu words per stereo frame\n", (unsigned long)frames,
                   stereo_slider(), (unsigned long)st_stereo, (unsigned long)st_st_fall, st_stereo ? (unsigned long)(st_st_words / st_stereo) : 0ul);
        st_stereo = st_st_fall = st_st_words = 0;
    }
    /* running totals for the bottom-screen stats (log.c reads them on the game thread); linear free under lin_lock */
    recomp_gx_stat[1] += st_draws; recomp_gx_stat[2] += st_tris; __dmb(); recomp_gx_stat[0]++;
    if (recomp_gx_stat[0] % 30 == 0) vt_count();
    if (recomp_gx_stat[0] % 60 == 0) { RecursiveLock_Lock(&lin_lock); recomp_gx_stat[3] = linearSpaceFree(); RecursiveLock_Unlock(&lin_lock); }
    vc_hits = vc_misses = st_dec_v = st_dec_b = st_lay_sw = st_vu = 0;
    st_dl_calls = st_dl_prims = st_dl_hits = st_dl_misses = st_dl_slow = pm_hits = pm_misses = 0;
    st_draws = st_tris = st_tex = st_evict = st_upload = st_upbytes = st_split = st_merged = st_kick_cap = 0;
    /* a frame with no draws still shows the clear color (frame_begin above) */
    pend_flush();
    last_end = NULL;
    u64 t0 = recomp_ticks();
    /* the buffers written since the last mid-frame submit (all of them with submits off) */
    flush_new();
    stereo_end();
    fe_t = recomp_ticks();
    C3D_FrameEnd(GX_CMDLIST_FLUSH);   /* buffers + textures flushed above; skips the whole-linear-heap flush */
    fb_dump();
    recomp_prof[PROF_GX_END] += recomp_ticks() - t0;
    frame_open = 0;
    frames++;   /* recomp_wr_epoch: advanced by sdk_GXCopyDisp on the game thread */
}

/* ---------------------------------------------------------------- textures */
static const u8 tile_w[16] = {8, 8, 8, 4, 4, 4, 4, 0, 8, 8, 4, 0, 0, 0, 8, 0};
static const u8 tile_h[16] = {8, 4, 4, 4, 4, 4, 4, 0, 8, 4, 4, 0, 0, 0, 8, 0};
static const u8 tex_bpp[16] = {4, 8, 8, 16, 16, 16, 32, 0, 4, 8, 16, 0, 0, 0, 4, 0};

static u32 rgb565(u16 v) {
    u32 r = (v >> 11) & 31, g = (v >> 5) & 63, b = v & 31;
    return (r << 3 | r >> 2) | (g << 2 | g >> 4) << 8 | (b << 3 | b >> 2) << 16 | 0xffu << 24;
}
static u32 rgb5a3(u16 v) {
    if (v & 0x8000) {   /* bit replication as Dolphin (Convert5To8 / Convert3To8) */
        u32 r = (v >> 10) & 31, g = (v >> 5) & 31, b = v & 31;
        return (r << 3 | r >> 2) | (g << 3 | g >> 2) << 8 | (b << 3 | b >> 2) << 16 | 0xffu << 24;
    }
    u32 a = (v >> 12) & 7;
    return ((v >> 8) & 15) * 17 | ((v >> 4) & 15) * 17 << 8 | (v & 15) * 17 << 16 | (a << 5 | a << 2 | a >> 1) << 24;
}
static u32 ia8(u16 v) { u32 i = v & 0xff; return i | i << 8 | i << 16 | (u32)(v >> 8) << 24; }
static u32 pal_color(const Tlut *t, u32 i) {
    if (i >= t->n) i = t->n ? t->n - 1 : 0;
    u16 v = rd16(t->addr + i * 2);
    return t->fmt == 0 ? ia8(v) : t->fmt == 1 ? rgb565(v) : rgb5a3(v);
}

/* decode a GX texture into w*h RGBA (byte order R,G,B,A in a little-endian u32) */
static void tex_decode(u32 *out, u32 src, int w, int h, int fmt, const Tlut *tl) {
    int tw = tile_w[fmt], th = tile_h[fmt];
    int bw = (w + tw - 1) / tw, bh = (h + th - 1) / th;
    const u8 *p = ram + (src & RAM_MASK);
    for (int by = 0; by < bh; by++) for (int bx = 0; bx < bw; bx++) {
        if (fmt == 14) {   /* CMPR: 2x2 DXT1 sub-blocks per 8x8 tile */
            for (int s = 0; s < 4; s++, p += 8) {
                u16 c0 = p[0] << 8 | p[1], c1 = p[2] << 8 | p[3];
                u32 pal[4]; pal[0] = rgb565(c0); pal[1] = rgb565(c1);
                for (int k = 0; k < 3; k++) {
                    u32 a = (pal[0] >> (8 * k)) & 255, b = (pal[1] >> (8 * k)) & 255;
                    u32 m2 = c0 > c1 ? (2 * a + b) / 3 : (a + b) / 2, m3 = c0 > c1 ? (a + 2 * b) / 3 : 0;
                    if (!k) { pal[2] = 0; pal[3] = 0; }
                    pal[2] |= m2 << (8 * k); pal[3] |= m3 << (8 * k);
                }
                pal[2] |= 0xffu << 24; if (c0 > c1) pal[3] |= 0xffu << 24;
                for (int y = 0; y < 4; y++) for (int x = 0; x < 4; x++) {
                    int px = bx * 8 + (s & 1) * 4 + x, py = by * 8 + (s >> 1) * 4 + y;
                    if (px < w && py < h) out[py * w + px] = pal[(p[4 + y] >> (6 - 2 * x)) & 3];
                }
            }
            continue;
        }
        for (int y = 0; y < th; y++) for (int x = 0; x < tw; x++) {
            int px = bx * tw + x, py = by * th + y, i = y * tw + x;
            u32 c;
            switch (fmt) {
            case 0: { u32 v = (p[i >> 1] >> ((i & 1) ? 0 : 4)) & 15; v *= 17; c = v | v << 8 | v << 16 | v << 24; break; }
            case 1: { u32 v = p[i]; c = v | v << 8 | v << 16 | v << 24; break; }
            case 2: { u32 v = (p[i] & 15) * 17, a = (p[i] >> 4) * 17; c = v | v << 8 | v << 16 | a << 24; break; }
            case 3: c = ia8(p[2 * i] << 8 | p[2 * i + 1]); break;
            case 4: c = rgb565(p[2 * i] << 8 | p[2 * i + 1]); break;
            case 5: c = rgb5a3(p[2 * i] << 8 | p[2 * i + 1]); break;
            case 6: c = p[2 * i + 1] | p[32 + 2 * i] << 8 | p[32 + 2 * i + 1] << 16 | (u32)p[2 * i] << 24; break;
            case 8: c = pal_color(tl, (p[i >> 1] >> ((i & 1) ? 0 : 4)) & 15); break;
            case 9: c = pal_color(tl, p[i]); break;
            case 10: c = pal_color(tl, (p[2 * i] << 8 | p[2 * i + 1]) & 0x3fff); break;
            default: c = 0xffff00ff; break;
            }
            if (px < w && py < h) out[py * w + px] = c;
        }
        p += tw * th * tex_bpp[fmt] / 8;   /* RGBA8: 64-byte tile, AR block then GB block */
    }
}

static u32 pow2(u32 v) { u32 p = 8; while (p < v && p < 1024) p <<= 1; return p; }

/* upf: frame of the last upload; dmf: frame + 1 of its last move VRAM -> linear (tex_to_vram); lv: PICA levels allocated, lk: tex_lk of the texmap it was made for; etc1: bytes the texture would take as ETC1/ETC1A4 (estimate only) */
typedef struct { u32 img, tlut_addr, hash, last, pkey, vep, tlut_n, etc1, upf, dmf; u16 w, h; u8 fmt, tlut_fmt, used, lv, lk, gen; C3D_Tex tex; } TexEntry;
#define TEX_CACHE 192
static TexEntry tcache[TEX_CACHE];
static u32 *decode_buf;

static u32 hash_bytes(u32 addr, u32 n) {
    const u8 *p = ram + (addr & RAM_MASK);
    u32 h = 2166136261u;
    u32 i = 0;
    if (n >= 32) {   /* four independent lanes: the single chain is bound by multiply latency */
        u32 h1 = h ^ 0x85ebca6bu, h2 = h ^ 0xc2b2ae35u, h3 = h ^ 0x27d4eb2fu, v0, v1, v2, v3;
        for (; i + 16 <= n; i += 16) {
            memcpy(&v0, p + i, 4); memcpy(&v1, p + i + 4, 4); memcpy(&v2, p + i + 8, 4); memcpy(&v3, p + i + 12, 4);
            h = (h ^ v0) * 16777619u; h1 = (h1 ^ v1) * 16777619u; h2 = (h2 ^ v2) * 16777619u; h3 = (h3 ^ v3) * 16777619u;
        }
        h = ((h * 0x85ebca6bu ^ h1) * 0xc2b2ae35u ^ h2) * 0x27d4eb2fu ^ h3;
    }
    for (; i + 4 <= n; i += 4) { u32 v; memcpy(&v, p + i, 4); h = (h ^ v) * 16777619u; }
    return h;
}

static u32 tex_bytes(int w, int h, int fmt) {
    int tw = tile_w[fmt], th = tile_h[fmt];
    if (!tw) return 0;
    return ((w + tw - 1) / tw * tw) * ((h + th - 1) / th * th) * tex_bpp[fmt] / 8;
}
static int lvdim(int v, int l) { v >>= l; return v ? v : 1; }
/* the texmap fields tex_levels depends on besides the size: entries are per (img, w, h, fmt, lk) */
static u8 tex_lk(const TexMap *m) { return m->min_f < 2 ? 0 : (u8)(m->max_lod + 1); }

/* PICA levels for a texmap: the GX chain (mip min filter: max_lod rounded up + 1, as Dolphin,
 * at most down to 1x1) cut at the PICA minimum of 8x8 for the W x H allocation */
static int tex_levels(const TexMap *m, u32 W, u32 H) {
    if (m->min_f < 2) return 1;
    int n = (m->max_lod + 15) / 16 + 1, lg = 0, pl = 1;
    for (u32 v = m->w > m->h ? m->w : m->h; v > 1; v >>= 1) lg++;
    if (n > lg + 1) n = lg + 1;
    for (u32 v = W < H ? W : H; v > 8; v >>= 1) pl++;
    return n < pl ? n : pl;
}
/* guest bytes of levels 0..lv-1 (the levels follow each other in RAM) */
static u32 tex_chain_bytes(const TexMap *m, int lv) {
    u32 n = 0;
    for (int l = 0; l < lv; l++) n += tex_bytes(lvdim(m->w, l), lvdim(m->h, l), m->fmt);
    return n;
}

static C3D_Tex *tex_bind_params(TexEntry *e, const TexMap *m, float *su, float *sv);

/* PICA format holding a GX format exactly: I4/IA4 -> LA4, I8/IA8 -> LA8 (GX
 * intensity also feeds alpha), RGB565 -> RGB565; the rest RGBA8 (tex_classify
 * may then pick a 16-bit one) */
static GPU_TEXCOLOR pica_fmt(int fmt) {
    switch (fmt) {
    case 0: case 2: return GPU_LA4;
    case 1: case 3: return GPU_LA8;
    case 4: return GPU_RGB565;
    default: return GPU_RGBA8;
    }
}

/* decode levels 0..lv-1 into decode_buf (level after level, w_l * h_l texels each) and pick
 * the smallest PICA format that holds every texel exactly: LA4 (grey, 4-bit l and a), then
 * RGB565 (opaque, bit-replicated 5/6/5), RGBA5551 (alpha 0 or 255, 5/5/5), LA8 (grey),
 * RGBA4 (every channel a 4-bit value * 17), else RGBA8. *alpha: some texel is not opaque */
static u32 lv_off[11];
static GPU_TEXCOLOR tex_classify(const TexMap *m, const Tlut *tl, int lv, int *alpha) {
    int w = m->w > 1024 ? 1024 : m->w, h = m->h > 1024 ? 1024 : m->h;
    if (!decode_buf) decode_buf = malloc(1024 * 1024 * 4 * 4 / 3 + 64);
    u32 src = m->img, o = 0;
    int f565 = 1, f5551 = f565, fla8 = f565, fla4 = f565, f4 = f565, a = 0;
    for (int l = 0; l < lv; l++) {
        int wl = lvdim(w, l), hl = lvdim(h, l);
        lv_off[l] = o;
        tex_decode(decode_buf + o, src, wl, hl, m->fmt, tl);
        for (u32 i = o, n = o + wl * hl; i < n; i++) {
            u32 c = decode_buf[i], r = c & 255, g = (c >> 8) & 255, b = (c >> 16) & 255, al = c >> 24;
            int rb5 = r == ((r >> 3) << 3 | r >> 5) && b == ((b >> 3) << 3 | b >> 5);
            if (al != 255) a = 1;
            if (f565 && !(al == 255 && rb5 && g == ((g >> 2) << 2 | g >> 6))) f565 = 0;
            if (f5551 && !((al == 255 || !al) && rb5 && g == ((g >> 3) << 3 | g >> 5))) f5551 = 0;
            int grey = r == g && g == b, c4 = r % 17 == 0 && g % 17 == 0 && b % 17 == 0 && al % 17 == 0;
            if (fla8 && !grey) fla8 = 0;
            if (fla4 && !(grey && c4)) fla4 = 0;
            if (f4 && !c4) f4 = 0;
        }
        o += wl * hl;
        src += tex_bytes(wl, hl, m->fmt);
    }
    *alpha = a;
    return fla4 ? GPU_LA4 : f565 ? GPU_RGB565 : f5551 ? GPU_RGBA5551 : fla8 ? GPU_LA8 : f4 ? GPU_RGBA4 : GPU_RGBA8;
}

static const u8 morton_x[8] = {0, 1, 4, 5, 16, 17, 20, 21}, morton_y[8] = {0, 2, 8, 10, 32, 34, 40, 42};
static u32 col_src[1024], col_dst[1024];

/* one GX level (w x h at srca) -> one W x H PICA level at dv, straight from the GX tiles
 * (the formats pica_fmt maps to LA4/LA8/RGB565) */
static void tex_level_native(void *dv, u32 W, u32 H, u32 srca, int w, int h, int fmt) {
    int tw = tile_w[fmt], th = tile_h[fmt], tn = tw * th, bw = (w + tw - 1) / tw;
    const u8 *src = ram + (srca & RAM_MASK);
    for (u32 x = 0; x < W; x++) {   /* texel index of column x inside its GX tile row */
        u32 sx = x < (u32)w ? x : w - 1;
        col_src[x] = sx / tw * tn + sx % tw;
        col_dst[x] = (x >> 3) * 64 + morton_x[x & 7];
    }
    for (u32 y = 0; y < H; y++) {
        u32 sy = y < (u32)h ? y : h - 1;
        u32 rs = sy / th * bw * tn + sy % th * tw, rd = (y >> 3) * (W >> 3) * 64 + morton_y[y & 7];
        switch (fmt) {
        case 0: { u8 *d = dv; for (u32 x = 0; x < W; x++) { u32 i = rs + col_src[x], v = (src[i >> 1] >> ((i & 1) ? 0 : 4)) & 15; d[rd + col_dst[x]] = (u8)(v << 4 | v); } break; }
        case 2: { u8 *d = dv; for (u32 x = 0; x < W; x++) { u8 v = src[rs + col_src[x]]; d[rd + col_dst[x]] = (u8)(v << 4 | v >> 4); } break; }
        case 1: { u16 *d = dv; for (u32 x = 0; x < W; x++) { u16 v = src[rs + col_src[x]]; d[rd + col_dst[x]] = v << 8 | v; } break; }
        case 3: { u16 *d = dv; for (u32 x = 0; x < W; x++) { const u8 *p = src + 2 * (rs + col_src[x]); d[rd + col_dst[x]] = (u16)(p[1] << 8 | p[0]); } break; }
        default: { u16 *d = dv; for (u32 x = 0; x < W; x++) { const u8 *p = src + 2 * (rs + col_src[x]); d[rd + col_dst[x]] = (u16)(p[0] << 8 | p[1]); } break; }
        }
    }
}

/* GX texture -> W x H PICA tiled texture with e->lv levels; GX row y is PICA memory row y,
 * sampled at v = 1 - y/H (see set_texgen); padding repeats the edge. The native compact
 * formats are converted straight from the GX tiles, the rest from decode_buf (tex_classify) */
static void tex_upload(TexEntry *e, const TexMap *m, u32 W, u32 H) {
    int w = m->w > 1024 ? 1024 : m->w, h = m->h > 1024 ? 1024 : m->h, fmt = m->fmt;
    GPU_TEXCOLOR pf = e->tex.fmt;
    u32 src = m->img;
    for (int l = 0; l < e->lv; l++) {
        int wl = lvdim(w, l), hl = lvdim(h, l);
        u32 Wl = W >> l, Hl = H >> l;
        void *dl = C3D_TexGetImagePtr(&e->tex, e->tex.data, l, NULL);
        if (pica_fmt(fmt) != GPU_RGBA8) tex_level_native(dl, Wl, Hl, src, wl, hl, fmt);
        else {
            const u32 *lvb = decode_buf + lv_off[l];
            for (u32 x = 0; x < Wl; x++) col_dst[x] = (x >> 3) * 64 + morton_x[x & 7];
            for (u32 y = 0; y < Hl; y++) {
                const u32 *row = lvb + (y < (u32)hl ? y : hl - 1) * wl;
                u32 rd = (y >> 3) * (Wl >> 3) * 64 + morton_y[y & 7];
                if (pf == GPU_RGBA8) {
                    u32 *d = dl;
                    for (u32 x = 0; x < Wl; x++) d[rd + col_dst[x]] = __builtin_bswap32(row[x < (u32)wl ? x : wl - 1]);   /* R,G,B,A bytes -> PICA A,B,G,R */
                } else if (pf == GPU_LA4) {
                    u8 *d = dl;
                    for (u32 x = 0; x < Wl; x++) { u32 c = row[x < (u32)wl ? x : wl - 1]; d[rd + col_dst[x]] = (u8)((c & 0xf0) | c >> 28); }
                } else if (pf == GPU_LA8 || pf == GPU_RGBA4) {
                    u16 *d = dl;
                    for (u32 x = 0; x < Wl; x++) {
                        u32 c = row[x < (u32)wl ? x : wl - 1];
                        d[rd + col_dst[x]] = pf == GPU_LA8 ? (u16)((c & 0xff) << 8 | c >> 24)
                                                           : (u16)((c & 0xf0) << 8 | (c & 0xf000) >> 4 | (c & 0xf00000) >> 16 | c >> 28);
                    }
                } else {
                    u16 *d = dl;
                    for (u32 x = 0; x < Wl; x++) {
                        u32 c = row[x < (u32)wl ? x : wl - 1];
                        d[rd + col_dst[x]] = pf == GPU_RGB565 ? (u16)((c & 0xf8) << 8 | (c & 0xfc00) >> 5 | (c & 0xf80000) >> 19)
                                                              : (u16)((c & 0xf8) << 8 | (c & 0xf800) >> 5 | (c & 0xf80000) >> 18 | c >> 31);
                    }
                }
            }
        }
        src += tex_bytes(wl, hl, fmt);
    }
    u32 total = C3D_TexCalcTotalSize(e->tex.size, e->lv - 1);
    GSPGPU_FlushDataCache(e->tex.data, total);
    st_upload++; st_upbytes += total;
    e->upf = frames;
}

/* first upload of each texture (up to 400 lines): the texture inventory */
static void tex_note(const TexMap *m, const TexEntry *e) {
    static u32 seen[400];
    static int n;
    u32 key = GUEST(m->img) ^ (u32)m->fmt << 28 ^ (u32)m->w << 16 ^ m->h;
    for (int i = 0; i < n; i++) if (seen[i] == key) return;
    if (n == 400) return;
    seen[n++] = key;
    static const char *const pn[] = {"RGBA8", "RGB8", "RGBA5551", "RGB565", "RGBA4", "LA8", "HILO8", "L8", "A8", "LA4", "L4", "A4", "ETC1", "ETC1A4"};
    recomp_log("# gx tex new: img %08lx fmt %d %dx%d min %d mag %d lod %d..%d bias %d -> %s %ux%u levels %d, %lu bytes (etc1 %lu) frame %lu\n",
               (unsigned long)GUEST(m->img), m->fmt, m->w, m->h, m->min_f, m->mag_f, m->min_lod, m->max_lod, m->lod_bias,
               pn[e->tex.fmt], e->tex.width, e->tex.height, e->lv,
               (unsigned long)C3D_TexCalcTotalSize(e->tex.size, e->lv - 1), (unsigned long)e->etc1, (unsigned long)frames);
}

/* resident texture bytes and their ETC1 estimate, for the 300-frame stats line */
static void tex_resident(u32 *bytes, u32 *etc1) {
    u32 b = 0, t = 0;
    for (int i = 0; i < TEX_CACHE; i++) if (tcache[i].used) { b += C3D_TexCalcTotalSize(tcache[i].tex.size, tcache[i].lv - 1); t += tcache[i].etc1; }
    *bytes = b; *etc1 = t;
}

/* GXCopyTex destinations: EFB -> texture copies are not emulated, so RAM there
 * holds stale data; such textures are not drawn (were 600x440 per-frame uploads
 * of garbage in the campaign) */
static u32 copy_dst[8];

/* a texmap that draws without its texture (TEXC then reads white): each
 * distinct image is logged once, up to 48 */
static void tex_miss(const char *why, const TexMap *m) {
    static u32 seen[48];
    static int n;
    u32 key = GUEST(m->img) ^ (u32)m->fmt << 28 ^ (u32)m->w << 16 ^ m->h;
    for (int i = 0; i < n; i++) if (seen[i] == key) return;
    if (n == 48) return;
    seen[n++] = key;
    recomp_log("# gx tex miss: %s img %08lx fmt %d %dx%d ci %d frame %lu\n", why, (unsigned long)GUEST(m->img),
               m->fmt, m->w, m->h, m->ci, (unsigned long)frames);
}

static int range_clean(const u8 *p, u32 n, u32 vep);
static C3D_Tex *tex_find(const TexMap *m, const Tlut *tl, u32 tla, u32 fk, float *su, float *sv);
static TexEntry *fidx[256];
/* used entries chained by (img, w, h, fmt), each chain in tcache index order: the lookups below
 * visit the same candidates in the same order as a scan of the whole cache. Links are index + 1. */
#define TEX_HASH 256
static u8 tex_head[TEX_HASH], tex_next[TEX_CACHE];
static u32 tex_key(u32 img, u32 w, u32 h, u32 fmt) { return (img * 0x9e3779b1u ^ w << 20 ^ h << 8 ^ fmt) >> 24; }
static void tex_link(TexEntry *e) {
    u8 i = (u8)(e - tcache + 1), *p = &tex_head[tex_key(e->img, e->w, e->h, e->fmt)];
    while (*p && *p < i) p = &tex_next[*p - 1];
    tex_next[i - 1] = *p; *p = i;
}
static void tex_unlink(TexEntry *e) {
    u8 i = (u8)(e - tcache + 1), *p = &tex_head[tex_key(e->img, e->w, e->h, e->fmt)];
    while (*p != i) p = &tex_next[*p - 1];
    *p = tex_next[i - 1];
}
/* VRAM textures. A texture whose contents stayed the same for 2 frames moves from linear FCRAM to
 * VRAM (campaign is partly fill-bound, hw run 17b). tex_upload still decodes into linear memory;
 * the move is a GX_TextureCopy added to citro3d's GPU queue (bound by C3D_Init), which runs in
 * order: after the work already queued, before this frame's command list (queued at a mid-frame
 * submit or C3D_FrameEnd). So the copy is done before any draw reads the VRAM copy, and VRAM freed
 * by an entry not drawn this frame is not read after a copy overwrites it. The linear block is
 * freed at the next frame_begin (C3D_FrameBegin waited for the queue). Not C3D_TexInitVRAM +
 * C3D_TexLoadImage: its C3D_SyncTextureCopy waits for and clears the queue mid-frame.
 * At most VT_COPIES moves per frame, and only while the queue (32 entries) keeps room for the
 * frame's submits: a texture over the cap stays linear and moves on a later frame. When VRAM
 * (less VT_RESERVE) is full, VRAM entries not drawn for VT_STALE frames move back to linear memory
 * (a queued copy, ordered before the move that takes their VRAM; deleting them made battle r3000
 * decode them all again in one 337 ms frame in Citra); their new linear block is the copy's target
 * until the queue runs, so that frame (dmf) it is not reused, rewritten or freed. With no stale
 * VRAM entry the texture stays linear (a fallback). VRAM entries are never rewritten in
 * place: changed contents go to a new linear allocation. sdmc:/cr_vramtex_off.txt = linear only;
 * the bottom-screen button toggles it between frames (texture cache invalidated). */
#define VT_COPIES 8
#define VT_RESERVE (256u << 10)
#define VT_STALE 60
static volatile int vt_req;   /* recomp_vt_toggle (bottom-screen button), applied by frame_begin */
void recomp_vt_toggle(void) { vt_req = 1; }
int recomp_vt_on;
u32 recomp_vt_flips;
volatile u32 recomp_vt_stat[6];   /* VRAM textures, KB; linear textures, KB; moves, fallbacks (running totals) */
static void *vt_free[VT_COPIES];
static int vt_nfree;
static int tex_in_vram(const TexEntry *e) { return (u32)e->tex.data - OS_VRAM_VADDR < OS_VRAM_SIZE; }
static void tex_to_vram(TexEntry *e) {
    if (vramtex_off || !frame_open || tex_in_vram(e) || frames - e->upf < 2) return;
    const gxCmdQueue_s *q = &__C3D_Context.q;
    if (vt_nfree == VT_COPIES || q->numEntries + 8 + KICK_MAX - kicks >= q->maxEntries) { st_vt_cap++; return; }
    u32 size = C3D_TexCalcTotalSize(e->tex.size, e->lv - 1);
    void *v = NULL;
    while (vramSpaceFree() < size + VT_RESERVE || !(v = vramAlloc(size))) {
        TexEntry *old = NULL;
        for (int i = 0; i < TEX_CACHE; i++) {
            TexEntry *t = &tcache[i];
            if (t->used && tex_in_vram(t) && t->last + VT_STALE < frames && (!old || t->last < old->last)) old = t;
        }
        if (!old) { st_vt_full++; recomp_vt_stat[5]++; return; }
        if (vt_nfree + 1 >= VT_COPIES || q->numEntries + 9 + KICK_MAX - kicks >= q->maxEntries) { st_vt_cap++; return; }   /* room for both copies */
        u32 osz = C3D_TexCalcTotalSize(old->tex.size, old->lv - 1);
        void *l = linearAlloc(osz);
        if (l) {
            GX_TextureCopy((u32 *)old->tex.data, 0, (u32 *)l, 0, osz, 8);
            vramFree(old->tex.data);
            old->tex.data = l; old->dmf = frames + 1;
            vt_nfree++; vt_free[vt_nfree - 1] = NULL;   /* counts toward VT_COPIES (linearFree(NULL) is a no-op) */
        } else { C3D_TexDelete(&old->tex); tex_unlink(old); old->used = 0; }
        st_vt_evict++;
    }
    GX_TextureCopy((u32 *)e->tex.data, 0, (u32 *)v, 0, size, 8);   /* flags as citro3d's C3D_TexLoadImage */
    vt_free[vt_nfree++] = e->tex.data;
    e->tex.data = v;
    st_vt_moves++; st_vt_kb += size >> 10; recomp_vt_stat[4]++;
}
/* frame_begin, after C3D_FrameBegin: the copies queued last frame are done; a button toggle */
static void vt_frame(void) {
    for (int i = 0; i < vt_nfree; i++) if (vt_free[i]) linearFree(vt_free[i]);
    vt_nfree = 0;
    if (vt_req) {
        vt_req = 0; vramtex_off ^= 1; recomp_vt_on = !vramtex_off; recomp_vt_flips++;
        tex_gen++; memset(fidx, 0, sizeof fidx);   /* every texture decodes again (into linear memory first) */
        recomp_log("# gx r%lu: VRAM textures %s (bottom-screen tap)\n", (unsigned long)recomp_retrace_count(), vramtex_off ? "off" : "on");
    }
}
/* resident textures per pool, for the # time lines (log.c) and the 300-frame line */
static void vt_count(void) {
    u32 n[2] = {0, 0}, b[2] = {0, 0};
    for (int i = 0; i < TEX_CACHE; i++) if (tcache[i].used) {
        int v = tex_in_vram(&tcache[i]);
        n[v]++; b[v] += C3D_TexCalcTotalSize(tcache[i].tex.size, tcache[i].lv - 1);
    }
    recomp_vt_stat[0] = n[1]; recomp_vt_stat[1] = b[1] >> 10; recomp_vt_stat[2] = n[0]; recomp_vt_stat[3] = b[0] >> 10;
}

#define TEX_SAME_KEY(t, m)((t)->img == (m)->img && (t)->w == (m)->w && (t)->h == (m)->h && (t)->fmt == (m)->fmt && (t)->lk == tex_lk(m) && (t)->gen == tex_gen)
static C3D_Tex *tex_get(const TexMap *m, float *su, float *sv) {
    if (!m->img) { tex_miss("no image", m); return NULL; }
    if (!tile_w[m->fmt & 15]) { tex_miss("format not decoded", m); return NULL; }
    if (!m->w || !m->h) { tex_miss("zero size", m); return NULL; }
    for (int i = 0; i < 8; i++) if (copy_dst[i] && copy_dst[i] == GUEST(m->img)) { ONCE("EFB copy texture not drawn\n"); tex_miss("EFB copy destination", m); return NULL; }
    const Tlut *tl = m->ci ? &gx.tlut[m->tlut < 20 ? m->tlut : 0] : NULL;
    /* hashed once per frame: the title draws one big texture as many strips.
     * Entries used this frame are found through a direct-mapped index (they are
     * not evicted or reused within the frame) */
    u32 tla = tl ? tl->addr : 0;
    u32 fk = (m->img * 0x9e3779b1u ^ (u32)m->w << 20 ^ (u32)m->h << 8 ^ m->fmt ^ tla * 0x85ebca6bu) >> 24;
    TexEntry *t = fidx[fk];
    if (t && t->used && t->last == frames && TEX_SAME_KEY(t, m) && t->tlut_addr == tla) return tex_bind_params(t, m, su, sv);
    u64 t0 = recomp_ticks();   /* the slow path is the gx tex bucket, taken out of the draw's chain */
    C3D_Tex *r = tex_find(m, tl, tla, fk, su, sv);
    u64 d = recomp_ticks() - t0;
    recomp_prof[PROF_GX_TEX] += d; t_last += d;
    return r;
}
static C3D_Tex *tex_find(const TexMap *m, const Tlut *tl, u32 tla, u32 fk, float *su, float *sv) {
    TexEntry *t;
    u8 chain = tex_head[tex_key(m->img, m->w, m->h, m->fmt)];
    for (u8 k = chain; k; k = tex_next[k - 1]) {
        t = &tcache[k - 1];
        if (t->last == frames && TEX_SAME_KEY(t, m) && t->tlut_addr == tla) { fidx[fk] = t; return tex_bind_params(t, m, su, sv); }
    }
    u32 W = pow2(m->w), H = pow2(m->h);
    int lv = tex_levels(m, W, H);
    u32 n = tex_chain_bytes(m, lv), hash;
    /* contents not written since the entry was last hashed (see range_clean): reuse it without hashing */
    u32 tln = tl ? tl->n : 0;
    for (u8 k = chain; k; k = tex_next[k - 1]) {
        t = &tcache[k - 1];
        if (t->vep && TEX_SAME_KEY(t, m) && t->tlut_addr == tla &&
            t->tlut_n == tln && range_clean(ram + (m->img & RAM_MASK), n, t->vep) &&
            (!tl || range_clean(ram + (tl->addr & RAM_MASK), tl->n * 2, t->vep))) {
            tex_to_vram(t);
            t->last = frames; fidx[fk] = t;
            return tex_bind_params(t, m, su, sv);
        }
    }
    u32 ep = epoch_now();
    hash = hash_bytes(m->img, n);
    if (tl) hash ^= hash_bytes(tl->addr, tl->n * 2) * 31 + tl->fmt;
    TexEntry *e = NULL, *victim = NULL, *same = NULL;
    for (u8 k = chain; k; k = tex_next[k - 1]) {
        t = &tcache[k - 1];
        if (!TEX_SAME_KEY(t, m)) continue;
        if (t->hash == hash) { e = t; e->vep = ep; e->tlut_n = tln; break; }
        if (t->last < frames && t->dmf != frames + 1) same = t;   /* changed contents: reuse its memory (the GPU is done with it) */
    }
    if (!e) for (int i = 0; i < TEX_CACHE; i++) {   /* a new texture: the least recently used entry */
        t = &tcache[i];
        if (!t->used) { if (!victim || victim->used) victim = t; }
        else if ((!victim || (victim->used && t->last < victim->last)) && t->last + 2 < frames && t->dmf != frames + 1) victim = t;
    }
    if (!e) pend_flush();   /* the entry may be bound to a pending draw */
    GPU_TEXCOLOR pf = pica_fmt(m->fmt);
    int alpha = m->fmt == 2 || m->fmt == 3;
    if (!e && pf == GPU_RGBA8) pf = tex_classify(m, tl, lv, &alpha);   /* decodes into decode_buf for tex_upload */
    if (!e && same && same->tex.fmt == pf && !tex_in_vram(same)) {   /* (lv is the same: same key); a VRAM entry is replaced below */
        e = same;
        tex_upload(e, m, W, H);
        e->hash = hash; e->vep = ep; e->tlut_n = tln;
    } else if (!e) {
        if (same) victim = same;   /* contents now need another PICA format */
        if (!victim) { ONCE("texture cache full\n"); tex_miss("cache full", m); return NULL; }
        e = victim;
        if (e->used) { C3D_TexDelete(&e->tex); tex_unlink(e); }
        e->used = 0;
        while (!C3D_TexInitWithParams(&e->tex, NULL, (C3D_TexInitParams){W, H, lv - 1, pf, GPU_TEX_2D, false})) {   /* out of linear memory: free the least recently used */
            TexEntry *old = NULL;
            for (int i = 0; i < TEX_CACHE; i++)   /* (a VRAM entry frees no linear memory) */
                if (tcache[i].used && tcache[i].last < frames && !tex_in_vram(&tcache[i]) && tcache[i].dmf != frames + 1 && (!old || tcache[i].last < old->last)) old = &tcache[i];
            if (!old) { ONCE("C3D_TexInit failed with nothing left to evict\n"); tex_miss("C3D_TexInit failed", m); return NULL; }
            C3D_TexDelete(&old->tex); tex_unlink(old); old->used = 0;
            st_evict++;
        }
        e->lv = (u8)lv; e->lk = tex_lk(m); e->gen = tex_gen;
        tex_upload(e, m, W, H);
        e->img = m->img; e->w = m->w; e->h = m->h; e->fmt = m->fmt; e->hash = hash; e->used = 1; e->vep = ep; e->tlut_n = tln;
        e->pkey = 0;
        u32 px = 0;   /* ETC1 is 4 bits per texel, ETC1A4 8: what it would take (lossy, so not used) */
        for (int l = 0; l < lv; l++) px += (W >> l) * (H >> l);
        e->etc1 = alpha ? px : px / 2;
        tex_link(e);
        tex_note(m, e);
    }
    if (e->last < frames) tex_to_vram(e);   /* not drawn this frame yet; fresh uploads wait (upf) */
    e->last = frames;
    e->tlut_addr = tla;
    fidx[fk] = e;
    return tex_bind_params(e, m, su, sv);
}

static C3D_Tex *tex_bind_params(TexEntry *e, const TexMap *m, float *su, float *sv) {
    static const GPU_TEXTURE_WRAP_PARAM wrap[4] = {GPU_CLAMP_TO_EDGE, GPU_REPEAT, GPU_MIRRORED_REPEAT, GPU_REPEAT};
    *su = (float)m->w / e->tex.width; *sv = (float)m->h / e->tex.height;
    u32 pk = (u32)(m->wrap_s & 3) | (u32)(m->wrap_t & 3) << 2 | (u32)m->mag_f << 4 | (u32)m->min_f << 5 |
             (u32)m->max_lod << 8 | (u32)m->min_lod << 16 | (u32)(u8)m->lod_bias << 24;
    if (e->pkey == pk + 1) return &e->tex;   /* wrap/filter/LOD already set */
    e->pkey = pk + 1;
    u32 param = e->tex.param, lodp = e->tex.lodParam;
    C3D_TexSetWrap(&e->tex, wrap[m->wrap_s & 3], wrap[m->wrap_t & 3]);
    /* GXTexFilter: odd = linear texel filter; 2/3 nearest mip, 4/5 linear mip */
    C3D_TexSetFilter(&e->tex, m->mag_f ? GPU_LINEAR : GPU_NEAREST, (m->min_f & 1) ? GPU_LINEAR : GPU_NEAREST);
    C3D_TexSetFilterMipmap(&e->tex, m->min_f >= 4 ? GPU_LINEAR : GPU_NEAREST);
    if (e->lv > 1) {   /* GX min/max LOD (1/16) to whole PICA levels, bias (1/32) as is */
        int lo = m->min_lod / 16, hi = (m->max_lod + 15) / 16;
        if (hi > e->lv - 1) hi = e->lv - 1;
        if (lo > hi) lo = hi;
        e->tex.minLevel = (u8)lo; e->tex.maxLevel = (u8)hi;
        C3D_TexSetLodBias(&e->tex, m->lod_bias / 32.0f);
    }
    if ((e->tex.param != param || e->tex.lodParam != lodp) && pend_n) {   /* a pending draw may use the old parameters */
        u32 nw = e->tex.param, nl = e->tex.lodParam; e->tex.param = param; e->tex.lodParam = lodp; pend_flush(); e->tex.param = nw; e->tex.lodParam = nl;
    }
    return &e->tex;
}

/* ---------------------------------------------------------------- TEV -> texenv */
static int unit_of_map[8];

/* cst: a per-draw constant (rgb in v[0..2] for the colour side, v[3] for alpha),
 * placed in the PICA stage constant or the combiner buffer's initial colour */
typedef struct { GPU_TEVSRC src; int opnd; int cst; u8 v[4]; } Src;   /* opnd: 0 color, 1 alpha, 2 one-minus color, 3-5 R/G/B */

/* GX TEV registers on PICA, per side (0 rgb, 1 alpha): a register is a per-draw
 * constant until a stage writes it, then lives in the PICA previous output; when
 * another register is written while it is still read later, it moves to the
 * combiner buffer (update bit on its last stage, readable two stages later) */
enum { L_CONST, L_PREV, L_BUF, L_LOST };
static struct { int loc[4], prev_reg, buf_reg, buf_ready, buf_mask, bufc_used; u8 bufc[4]; } tr[2];

static int konst_value(int sel, int alpha, u8 rgba[4]) {
    if (sel < 8) { u8 v = (u8)((8 - sel) * 255 / 8); rgba[0] = rgba[1] = rgba[2] = rgba[3] = v; return 1; }
    if (!alpha && sel >= 0x0c && sel <= 0x0f) { memcpy(rgba, gx.kcolor[sel - 0x0c], 4); return 1; }
    if (sel >= 0x10) { u8 v = gx.kcolor[sel & 3][(sel - 0x10) >> 2]; rgba[0] = rgba[1] = rgba[2] = rgba[3] = v; return 1; }
    return 0;
}

/* COLOR1/ALPHA1 are not computed by the vertex shader: their unlit part
 * (ambient * material when lit, material otherwise) */
static void chan1_value(u8 v[4]) {
    for (int i = 0; i < 4; i++) {
        const Chan *c = &gx.chan[i < 3 ? 1 : 3];
        int m = gx.mat[1][i], a = gx.amb[1][i];
        if (gx.n_chans < 2) { v[i] = 0; continue; }
        if (c->mat_src || (c->enable && c->amb_src)) { ONCE("COLOR1 vertex colour taken as white\n"); if (c->mat_src) m = 255; if (c->amb_src) a = 255; }
        if (c->enable && c->light_mask) ONCE("COLOR1 lighting not emulated (ambient only)\n");
        v[i] = (u8)(c->enable ? a * m / 255 : m);
    }
}

static Src konst_src(const u8 v[4], int opnd, int alpha) {
    Src s = {GPU_CONSTANT, alpha, 1, {0, 0, 0, 0}};
    if (alpha) s.v[3] = v[opnd ? 3 : 0];
    else if (opnd == 1) s.v[0] = s.v[1] = s.v[2] = v[3];
    else memcpy(s.v, v, 3);
    return s;
}

/* GX colour/alpha arg -> PICA source */
static Src color_src(const TevStage *t, int arg, int alpha) {
    Src s = {GPU_PREVIOUS, alpha, 0, {0, 0, 0, 0}};
    u8 v[4] = {0, 0, 0, 0};
    int map_unit = t->map < 8 ? unit_of_map[t->map] : -1;
    static const u8 alpha_arg[8] = {1, 3, 5, 7, 9, 11, 14, 15};   /* GX_CA_* -> GX_CC_* */
    if (alpha) arg = alpha_arg[arg & 7];
    switch (arg) {
    case 0: case 1: case 2: case 3: case 4: case 5: case 6: case 7: {
        int r = arg >> 1, opnd = alpha ? 1 : (arg & 1), side = alpha || (arg & 1);
        switch (tr[side].loc[r]) {
        case L_PREV: s.src = GPU_PREVIOUS; s.opnd = opnd; return s;
        case L_BUF: s.src = GPU_PREVIOUS_BUFFER; s.opnd = opnd; s.cst = -1 - side; return s;   /* cst < 0: buffer read, checked when placed */
        case L_LOST: ONCE("TEV register read after it was overwritten in PICA\n"); break;
        }
        return konst_src(gx.tev_reg[r], opnd, alpha);
    }
    case 8: case 9:
        if (map_unit < 0) { ONCE("TEXC without texture\n"); v[0] = v[1] = v[2] = v[3] = 255; return konst_src(v, 0, alpha); }
        s.src = (GPU_TEVSRC)(GPU_TEXTURE0 + map_unit); s.opnd = alpha ? 1 : (arg & 1);
        if (!alpha && !(arg & 1) && t->tex_swap) s.opnd = 2 + t->tex_swap;   /* default swap tables 1-3: RRR, GGG, BBB */
        if (t->tex_swap) ONCE("TEV texture swap %d (alpha kept)\n", t->tex_swap);
        return s;
    case 10: case 11:
        if (t->chan == 1 || t->chan == 3 || t->chan == 5) { chan1_value(v); return konst_src(v, alpha ? 1 : (arg & 1), alpha); }
        if (t->chan == 6) return konst_src(v, 0, alpha);
        s.src = GPU_PRIMARY_COLOR; s.opnd = alpha ? 1 : (arg & 1); return s;
    case 12: v[0] = v[1] = v[2] = v[3] = 255; return konst_src(v, 0, alpha);
    case 13: v[0] = v[1] = v[2] = v[3] = 128; return konst_src(v, 0, alpha);
    case 14: konst_value(alpha ? t->ksel_a : t->ksel_c, alpha, v); return konst_src(v, alpha, alpha);
    default: return konst_src(v, 0, alpha);
    }
}

/* one GX side (color or alpha) -> up to 3 PICA ops */
typedef struct { GPU_COMBINEFUNC f; Src s[3]; int scale; } POp;

static int cst_is(Src s, int alpha, int val) {
    if (s.cst <= 0) return 0;
    return alpha ? s.v[3] == val : s.v[0] == val && s.v[1] == val && s.v[2] == val;
}

/* the combiner buffer's initial colour is free to hold a second constant */
static int bufc_free(int alpha) { return !tr[alpha].buf_mask && !tr[alpha].bufc_used; }

static int side_ops(const TevStage *t, int alpha, POp *ops) {
    const TevSide *g = alpha ? &t->a : &t->c;
    Src A = color_src(t, g->a, alpha), B = color_src(t, g->b, alpha);
    Src C = color_src(t, g->c, alpha), D = color_src(t, g->d, alpha);
    int z_a = cst_is(A, alpha, 0), z_b = cst_is(B, alpha, 0), z_c = cst_is(C, alpha, 0), z_d = cst_is(D, alpha, 0);
    int one_c = cst_is(C, alpha, 255);
    int n = 0, scale = g->scale == 1 ? 1 : g->scale == 2 ? 2 : 0;
    Src prev = {GPU_PREVIOUS, alpha, 0, {0, 0, 0, 0}};
    if (g->scale == 3) ONCE("TEV scale 1/2 not supported\n");
    if (g->op > 1) ONCE("TEV compare op %d not supported\n", g->op);
    if (g->bias) ONCE("TEV bias %d not supported\n", g->bias);
    /* lerp part: a*(1-c) + b*c */
    int lerp_zero = (z_a && z_b) || (z_a && z_c) || (z_b && one_c);
    if (lerp_zero) {   /* d +- 0 */
        ops[n++] = (POp){GPU_REPLACE, {D, D, D}, scale};
        return n;
    }
    POp l;
    if (z_a) l = one_c ? (POp){GPU_REPLACE, {B, B, B}} : (POp){GPU_MODULATE, {B, C, C}};
    else if (one_c) l = (POp){GPU_REPLACE, {B, B, B}};
    else if (z_c) l = (POp){GPU_REPLACE, {A, A, A}};
    else if (A.cst > 0 && C.cst > 0 && B.cst <= 0 && !bufc_free(alpha)) {
        /* two constants and no free buffer colour: b*c, then + a*(1-c) folded on the CPU */
        Src K = A;
        for (int i = 0; i < 4; i++) K.v[i] = (u8)(A.v[i] * (255 - C.v[i]) / 255);
        ops[n++] = (POp){GPU_MODULATE, {B, C, C}, 0};
        l = (POp){GPU_ADD, {prev, K, K}};
    } else l = (POp){GPU_INTERPOLATE, {B, A, C}};
    if (z_d) { l.scale = scale; if (g->op == 1) ONCE("TEV 0 - x\n"); ops[n++] = l; return n; }
    if (n == 0 && g->op == 0 && l.f == GPU_MODULATE) { ops[n++] = (POp){GPU_MULTIPLY_ADD, {B, C, D}, scale}; return n; }
    if (n == 0 && l.f == GPU_REPLACE) {
        ops[n++] = (POp){g->op ? GPU_SUBTRACT : GPU_ADD, {g->op ? D : l.s[0], g->op ? l.s[0] : D, D}, scale};
        return n;
    }
    ops[n++] = l;
    if (D.src == GPU_PREVIOUS) ONCE("TEV lerp + CPREV needs the combiner buffer\n");
    ops[n++] = (POp){g->op ? GPU_SUBTRACT : GPU_ADD, {g->op ? D : prev, g->op ? prev : D, D}, scale};
    return n;
}

static int opnd_rgb(int o) {
    static const int m[6] = {GPU_TEVOP_RGB_SRC_COLOR, GPU_TEVOP_RGB_SRC_ALPHA, GPU_TEVOP_RGB_SRC_COLOR, GPU_TEVOP_RGB_SRC_R, GPU_TEVOP_RGB_SRC_G, GPU_TEVOP_RGB_SRC_B};
    return m[o < 6 ? o : 0];
}

/* is side x of register r read by a stage after si before being rewritten? (PREV is the output) */
static int tev_live(int r, u32 si, int x) {
    for (u32 s = si + 1; s < gx.n_stages && s < 16; s++) {
        const TevStage *t = &gx.tev[s];
        const u8 *ca = &t->c.a, *aa = &t->a.a;
        for (int i = 0; i < 4; i++) {
            if (!x && ca[i] == 2 * r) return 1;
            if (x && (ca[i] == 2 * r + 1 || aa[i] == r)) return 1;
        }
        if ((x ? t->a.out : t->c.out) == r) return 0;
    }
    return r == 0;
}

/* places the constants of one PICA stage: stage constant first, then the buffer colour */
static void place_srcs(Src *s, int alpha, int pstage, u8 sc[4], int *scu) {
    for (int i = 0; i < 3; i++) {
        if (s[i].cst < 0) {
            if (tr[-1 - s[i].cst].buf_ready > pstage) ONCE("TEV register read from the combiner buffer too early\n");
            continue;
        }
        if (!s[i].cst) continue;
        const u8 *v = s[i].v + (alpha ? 3 : 0);
        u8 *d = sc + (alpha ? 3 : 0), *b = tr[alpha].bufc + (alpha ? 3 : 0);
        int n = alpha ? 1 : 3;
        if (!*scu || !memcmp(d, v, n)) { memcpy(d, v, n); *scu = 1; s[i].src = GPU_CONSTANT; }
        else if (!tr[alpha].buf_mask && (!tr[alpha].bufc_used || !memcmp(b, v, n))) {
            memcpy(b, v, n); tr[alpha].bufc_used = 1; s[i].src = GPU_PREVIOUS_BUFFER;
        } else { ONCE("three different constants in one PICA stage\n"); s[i].src = GPU_CONSTANT; }
    }
}

/* everything setup_tev reads: unchanged since the last draw -> the texenv is too */
static int tev_changed(void) {
    static TevStage tev[16];
    static u8 reg[16], kc[16], mat[4], amb[4];
    static s16 s10[16];
    static int uom[8];
    static Chan chan[4];
    static u32 n, nch;
    static int valid;
    u32 ns = gx.n_stages < 16 ? gx.n_stages : 16;
    if (valid && ns == n && gx.n_chans == nch && !memcmp(tev, gx.tev, ns * sizeof(TevStage)) &&
        !memcmp(reg, gx.tev_reg, 16) && !memcmp(s10, gx.tev_s10, 32) && !memcmp(kc, gx.kcolor, 16) && !memcmp(uom, unit_of_map, sizeof uom) &&
        !memcmp(chan, gx.chan, sizeof chan) && !memcmp(mat, gx.mat[1], 4) && !memcmp(amb, gx.amb[1], 4)) return 0;
    n = ns; nch = gx.n_chans;
    memcpy(tev, gx.tev, ns * sizeof(TevStage)); memcpy(reg, gx.tev_reg, 16); memcpy(s10, gx.tev_s10, 32); memcpy(kc, gx.kcolor, 16);
    memcpy(uom, unit_of_map, sizeof uom); memcpy(chan, gx.chan, sizeof chan); memcpy(mat, gx.mat[1], 4); memcpy(amb, gx.amb[1], 4);
    valid = 1;
    return 1;
}

/* The THP movie's YUV -> RGB program (4 GX stages, maps Y/U/V as I8): signed intermediates
 * (GXSetTevColorS10 REG1 = -90, 0, -114) that PICA's 0..1 clamped stages cannot hold. Per channel
 * the program is out = Y + c0 + cu*U + cv*V (red/blue from the colour side, green from the alpha
 * side, picked by stage 3's KONST 255/0 lerp); the coefficients are read from the registers.
 * PICA computes it at half scale with only additions before one final subtraction:
 * 0.5*Y + pos offsets, + |c|/2*X for c > 0, + |c|/2*(1-X) for c < 0 (moving -|c|/2 into the
 * offset), then - neg offsets with scale 2. A sum clamped at 1 early means the result saturates
 * anyway, since everything subtracted later is the final offset only. */
static int tev_yuv(void) {
    const TevStage *t = gx.tev;
    if (gx.n_stages != 4) return 0;
    for (int s = 0; s < 3; s++) if (t[s].map > 7 || t[s].tex_swap || t[s].c.bias || t[s].a.bias || t[s].c.out || t[s].a.out) return 0;
    /* 0: d + U*K (colour), d -+ U*K (alpha); 1: (prev + V*K) * scale; 2: Y + prev; 3: lerp(APREV, CPREV, K) */
    if (t[0].c.a != 15 || t[0].c.b != 8 || t[0].c.c != 14 || (t[0].c.d & 1) || t[0].c.d > 6 || t[0].c.op || t[0].c.scale) return 0;
    if (t[0].a.a != 7 || t[0].a.b != 4 || t[0].a.c != 6 || t[0].a.d > 3 || t[0].a.op > 1 || t[0].a.scale) return 0;
    if (t[1].c.a != 15 || t[1].c.b != 8 || t[1].c.c != 14 || t[1].c.d != 0 || t[1].c.op || t[1].c.scale > 2) return 0;
    if (t[1].a.a != 7 || t[1].a.b != 4 || t[1].a.c != 6 || t[1].a.d != 0 || t[1].a.op > 1 || t[1].a.scale > 2) return 0;
    if (t[2].c.a != 15 || t[2].c.b != 8 || t[2].c.c != 12 || t[2].c.d != 0 || t[2].c.op || t[2].c.scale) return 0;
    if (t[2].a.a != 4 || t[2].a.b != 7 || t[2].a.c != 7 || t[2].a.d != 0 || t[2].a.op || t[2].a.scale) return 0;
    if (t[3].c.a != 1 || t[3].c.b != 0 || t[3].c.c != 14 || t[3].c.d != 15 || t[3].c.op || t[3].c.scale || t[3].c.out) return 0;
    if (t[3].a.a != 7 || t[3].a.b != 7 || t[3].a.c != 7 || t[3].a.d != 7) return 0;
    u8 k0[4], k1[4], k3[4], ka0[4], ka1[4];
    if (!konst_value(t[0].ksel_c, 0, k0) || !konst_value(t[1].ksel_c, 0, k1) || !konst_value(t[3].ksel_c, 0, k3) ||
        !konst_value(t[0].ksel_a, 1, ka0) || !konst_value(t[1].ksel_a, 1, ka1)) return 0;
    for (int i = 0; i < 3; i++) if (k3[i] != 0 && k3[i] != 255) return 0;
    float c0[3], cu[3], cv[3];
    float sc = t[1].c.scale == 1 ? 2.f : t[1].c.scale == 2 ? 4.f : 1.f, sa = t[1].a.scale == 1 ? 2.f : t[1].a.scale == 2 ? 4.f : 1.f;
    float su = t[0].a.op ? -1.f : 1.f, sv = t[1].a.op ? -1.f : 1.f;
    for (int i = 0; i < 3; i++) {
        if (k3[i]) { c0[i] = sc * gx.tev_s10[t[0].c.d >> 1][i] / 255.f; cu[i] = sc * k0[i] / 255.f; cv[i] = sc * k1[i] / 255.f; }
        else { c0[i] = sa * gx.tev_s10[t[0].a.d][3] / 255.f; cu[i] = sa * su * ka0[0] / 255.f; cv[i] = sa * sv * ka1[0] / 255.f; }
        if (cu[i] > 2 || cu[i] < -2 || cv[i] > 2 || cv[i] < -2) return 0;
    }
    int ty = unit_of_map[t[2].map], tu = unit_of_map[t[0].map], tv = unit_of_map[t[1].map];
    if (ty < 0 || tu < 0 || tv < 0) return 0;
    float off[3];
    u8 kpos[4] = {0, 0, 0, 0}, kneg[4] = {0, 0, 0, 0};
    struct { int unit, inv; u8 k[4]; } term[4];
    int nt = 0;
    for (int x = 0; x < 2; x++)
        for (int inv = 0; inv < 2; inv++) {
            const float *c = x ? cv : cu;
            u8 k[4] = {0, 0, 0, 0}; int any = 0;
            for (int i = 0; i < 3; i++) if (inv ? c[i] < 0 : c[i] > 0) { float h = (c[i] < 0 ? -c[i] : c[i]) / 2; k[i] = (u8)(h * 255 + 0.5f); any = 1; }
            if (!any) continue;
            term[nt].unit = x ? tv : tu; term[nt].inv = inv; memcpy(term[nt].k, k, 4); nt++;
        }
    for (int i = 0; i < 3; i++) {
        off[i] = c0[i] / 2;
        if (cu[i] < 0) off[i] += cu[i] / 2;
        if (cv[i] < 0) off[i] += cv[i] / 2;
        float v = off[i] < 0 ? -off[i] : off[i];
        u8 b = v > 1 ? 255 : (u8)(v * 255 + 0.5f);
        if (off[i] < 0) kneg[i] = b; else kpos[i] = b;
    }
    int pstage = 0;
    C3D_TexEnv *env = C3D_GetTexEnv(pstage++);   /* 0.5 * Y + positive offsets (buffer colour) */
    C3D_TexEnvInit(env);
    C3D_TexEnvSrc(env, C3D_RGB, (GPU_TEVSRC)(GPU_TEXTURE0 + ty), GPU_CONSTANT, GPU_PREVIOUS_BUFFER);
    C3D_TexEnvFunc(env, C3D_RGB, GPU_MULTIPLY_ADD);
    C3D_TexEnvSrc(env, C3D_Alpha, GPU_CONSTANT, 0, 0);
    C3D_TexEnvFunc(env, C3D_Alpha, GPU_REPLACE);
    C3D_TexEnvColor(env, 128 | 128 << 8 | 128 << 16);
    for (int k = 0; k < nt; k++) {
        env = C3D_GetTexEnv(pstage++);
        C3D_TexEnvInit(env);
        C3D_TexEnvSrc(env, C3D_RGB, (GPU_TEVSRC)(GPU_TEXTURE0 + term[k].unit), GPU_CONSTANT, GPU_PREVIOUS);
        C3D_TexEnvOpRgb(env, term[k].inv ? GPU_TEVOP_RGB_ONE_MINUS_SRC_COLOR : GPU_TEVOP_RGB_SRC_COLOR, GPU_TEVOP_RGB_SRC_COLOR, GPU_TEVOP_RGB_SRC_COLOR);
        C3D_TexEnvFunc(env, C3D_RGB, GPU_MULTIPLY_ADD);
        C3D_TexEnvSrc(env, C3D_Alpha, GPU_CONSTANT, 0, 0);
        C3D_TexEnvFunc(env, C3D_Alpha, GPU_REPLACE);
        C3D_TexEnvColor(env, term[k].k[0] | term[k].k[1] << 8 | term[k].k[2] << 16);
    }
    env = C3D_GetTexEnv(pstage++);   /* (sum - negative offsets) * 2 */
    C3D_TexEnvInit(env);
    C3D_TexEnvSrc(env, C3D_RGB, GPU_PREVIOUS, GPU_CONSTANT, 0);
    C3D_TexEnvFunc(env, C3D_RGB, GPU_SUBTRACT);
    C3D_TexEnvScale(env, C3D_RGB, GPU_TEVSCALE_2);
    C3D_TexEnvSrc(env, C3D_Alpha, GPU_CONSTANT, 0, 0);
    C3D_TexEnvFunc(env, C3D_Alpha, GPU_REPLACE);
    C3D_TexEnvColor(env, kneg[0] | kneg[1] << 8 | kneg[2] << 16);
    C3D_TexEnvBufUpdate(C3D_RGB, 0);
    C3D_TexEnvBufUpdate(C3D_Alpha, 0);
    C3D_TexEnvBufColor(kpos[0] | kpos[1] << 8 | kpos[2] << 16);
    for (int i = pstage; i < 6; i++) {
        env = C3D_GetTexEnv(i);
        C3D_TexEnvInit(env);
        C3D_TexEnvSrc(env, C3D_Both, GPU_PREVIOUS, 0, 0);
        C3D_TexEnvFunc(env, C3D_Both, GPU_REPLACE);
    }
    ONCE("TEV YUV -> RGB program (THP) as %d PICA stages\n", pstage);
    return 1;
}

static void setup_tev(void) {
    int pstage = 0;
    if (!tev_changed()) return;
    pend_flush();
    if (tev_yuv()) return;
    memset(tr, 0, sizeof tr);
    for (int x = 0; x < 2; x++) { tr[x].prev_reg = tr[x].buf_reg = -1; tr[x].buf_ready = 99; }
    for (u32 si = 0; si < gx.n_stages && si < 16; si++) {
        const TevStage *t = &gx.tev[si];
        POp co[3], ao[3];
        int nc = side_ops(t, 0, co), na = side_ops(t, 1, ao);
        int n = nc > na ? nc : na;
        for (int x = 0; x < 2; x++) {   /* the register in PICA previous is about to be replaced: keep it if still read */
            int w = x ? t->a.out : t->c.out, h = tr[x].prev_reg;
            if (h < 0 || h == w) continue;
            tr[x].loc[h] = L_LOST;
            if (!tev_live(h, si, x)) continue;
            if (tr[x].buf_reg >= 0 && tev_live(tr[x].buf_reg, si, x)) ONCE("two TEV registers live besides the one being written\n");
            else if (pstage - 1 > 3) ONCE("TEV register kept after PICA stage 3\n");
            else { tr[x].buf_mask |= 1 << (pstage - 1); tr[x].loc[h] = L_BUF; tr[x].buf_reg = h; tr[x].buf_ready = pstage + 1; }
        }
        for (int k = 0; k < n; k++) {
            if (pstage >= 6) { ONCE("more than 6 PICA texenv stages needed\n"); break; }
            C3D_TexEnv *env = C3D_GetTexEnv(pstage);
            C3D_TexEnvInit(env);
            POp c = k < nc ? co[k] : (POp){GPU_REPLACE, {{GPU_PREVIOUS, 0, 0, {0}}}}, a = k < na ? ao[k] : (POp){GPU_REPLACE, {{GPU_PREVIOUS, 1, 0, {0}}}};
            u8 sc[4] = {0, 0, 0, 0}; int scc = 0, sca = 0;
            place_srcs(c.s, 0, pstage, sc, &scc);
            place_srcs(a.s, 1, pstage, sc, &sca);
            C3D_TexEnvSrc(env, C3D_RGB, c.s[0].src, c.s[1].src, c.s[2].src);
            C3D_TexEnvOpRgb(env, opnd_rgb(c.s[0].opnd), opnd_rgb(c.s[1].opnd), opnd_rgb(c.s[2].opnd));
            C3D_TexEnvFunc(env, C3D_RGB, c.f);
            C3D_TexEnvScale(env, C3D_RGB, (GPU_TEVSCALE)(k + 1 == nc ? c.scale : 0));
            C3D_TexEnvSrc(env, C3D_Alpha, a.s[0].src, a.s[1].src, a.s[2].src);
            C3D_TexEnvOpAlpha(env, GPU_TEVOP_A_SRC_ALPHA, GPU_TEVOP_A_SRC_ALPHA, GPU_TEVOP_A_SRC_ALPHA);
            C3D_TexEnvFunc(env, C3D_Alpha, a.f);
            C3D_TexEnvScale(env, C3D_Alpha, (GPU_TEVSCALE)(k + 1 == na ? a.scale : 0));
            C3D_TexEnvColor(env, sc[0] | sc[1] << 8 | sc[2] << 16 | (u32)sc[3] << 24);
            pstage++;
        }
        for (int x = 0; x < 2; x++) {
            int w = x ? t->a.out : t->c.out;
            if (tr[x].buf_reg == w) tr[x].buf_reg = -1;
            tr[x].loc[w] = L_PREV; tr[x].prev_reg = w;
        }
    }
    C3D_TexEnvBufUpdate(C3D_RGB, tr[0].buf_mask);
    C3D_TexEnvBufUpdate(C3D_Alpha, tr[1].buf_mask);
    C3D_TexEnvBufColor(tr[0].bufc[0] | tr[0].bufc[1] << 8 | tr[0].bufc[2] << 16 | (u32)tr[1].bufc[3] << 24);
    if (!pstage) {   /* no stages: pass the rasterized color */
        C3D_TexEnv *env = C3D_GetTexEnv(0);
        C3D_TexEnvInit(env);
        C3D_TexEnvSrc(env, C3D_Both, GPU_PRIMARY_COLOR, 0, 0);
        C3D_TexEnvFunc(env, C3D_Both, GPU_REPLACE);
        pstage = 1;
    }
    for (int i = pstage; i < 6; i++) {
        C3D_TexEnv *env = C3D_GetTexEnv(i);
        C3D_TexEnvInit(env);
        C3D_TexEnvSrc(env, C3D_Both, GPU_PREVIOUS, 0, 0);
        C3D_TexEnvFunc(env, C3D_Both, GPU_REPLACE);
    }
}

static GPU_TESTFUNC test_func(int f) {
    static const GPU_TESTFUNC m[8] = {GPU_NEVER, GPU_LESS, GPU_EQUAL, GPU_LEQUAL, GPU_GREATER, GPU_NOTEQUAL, GPU_GEQUAL, GPU_ALWAYS};
    return m[f & 7];
}
static GPU_BLENDFACTOR blend_factor(int f, int dst) {
    switch (f) {
    case 0: return GPU_ZERO; case 1: return GPU_ONE;
    case 2: return dst ? GPU_SRC_COLOR : GPU_DST_COLOR;
    case 3: return dst ? GPU_ONE_MINUS_SRC_COLOR : GPU_ONE_MINUS_DST_COLOR;
    case 4: return GPU_SRC_ALPHA; case 5: return GPU_ONE_MINUS_SRC_ALPHA;
    case 6: return GPU_DST_ALPHA; default: return GPU_ONE_MINUS_DST_ALPHA;
    }
}

/* PICA state set only when its arguments changed (citro3d re-sends every state
 * that was set, changed or not, at the next draw) */
#define SET_IF_CHANGED(call, ...) do { static u32 last_[16]; static int valid_; \
    const u32 cur_[] = {__VA_ARGS__}; \
    if (!valid_ || memcmp(last_, cur_, sizeof cur_)) { memcpy(last_, cur_, sizeof cur_); valid_ = 1; pend_flush(); call; } } while (0)
static u32 fbits(float f) { u32 u; memcpy(&u, &f, 4); return u; }

static void setup_pixel(void) {
    u32 mask = (gx.color_update ? GPU_WRITE_COLOR : 0) | (gx.alpha_update ? GPU_WRITE_ALPHA : 0);
    int zt = gx.z_enable, zf = zt ? test_func(gx.z_func) : GPU_ALWAYS;
    if (zt && gx.z_update) mask |= GPU_WRITE_DEPTH;
    SET_IF_CHANGED(C3D_DepthTest(zt, (GPU_TESTFUNC)zf, (GPU_WRITEMASK)mask), zt, zf, mask);
    /* GX window depth = far + (far - near) * z/w, clamped to [0, 1] by the depth buffer */
    float zs = gx.vp[5] - gx.vp[4], zo = gx.vp[5];
    SET_IF_CHANGED(C3D_DepthMap(true, zs, zo), fbits(zs), fbits(zo));
    GPU_BLENDEQUATION eq = GPU_BLEND_ADD;
    GPU_BLENDFACTOR bs = GPU_ONE, bd = GPU_ZERO;
    if (gx.blend_type == 1) { bs = blend_factor(gx.blend_src, 0); bd = blend_factor(gx.blend_dst, 1); }
    else if (gx.blend_type == 3) { eq = GPU_BLEND_REVERSE_SUBTRACT; bs = bd = GPU_ONE; }
    else if (gx.blend_type == 2) ONCE("logic op blend %d\n", gx.logic_op);
    SET_IF_CHANGED(C3D_AlphaBlend(eq, eq, bs, bd, bs, bd), eq, bs, bd);
    /* alpha compare: PICA has one test; take whichever GX comparison is not ALWAYS */
    u8 *ac = gx.alpha_cmp;
    if (ac[0] != 7 && ac[3] != 7 && !(ac[2] == 0 && (ac[0] == 7 || ac[3] == 7))) ONCE("two-sided alpha compare %d %d op %d\n", ac[0], ac[3], ac[2]);
    int at = 1, af = GPU_ALWAYS, ar = 0;
    if (ac[0] != 7) { af = test_func(ac[0]); ar = ac[1]; }
    else if (ac[3] != 7) { af = test_func(ac[3]); ar = ac[4]; }
    else at = 0;
    SET_IF_CHANGED(C3D_AlphaTest(at, (GPU_TESTFUNC)af, ar), at, af, ar);
    /* GX cull 1 front / 2 back, front = clockwise in window space (y down). set_clip
     * maps window space to PICA clip space with a mirror (det < 0); the mapping below
     * was checked with framebuffer dumps (unknowns "Cull mode"). 3 (all) has no PICA
     * mode: drawn unculled, logged. */
    GPU_CULLMODE cm = GPU_CULL_NONE;
    if (gx.cull == 1) cm = GPU_CULL_BACK_CCW;
    else if (gx.cull == 2) cm = GPU_CULL_FRONT_CCW;
    else if (gx.cull == 3) ONCE("cull mode 3 (all) drawn without culling\n");
    SET_IF_CHANGED(C3D_CullFace(cm), cm);
    /* scissor: GX EFB rect -> screen -> framebuffer (240 wide, 400 tall, rotated) */
    static s32 sk[8]; static int sk_valid;
    const s32 cur[8] = {(s32)gx.sc[0], (s32)gx.sc[1], (s32)gx.sc[2], (s32)gx.sc[3], (s32)gx.sc_off[0], (s32)gx.sc_off[1], (s32)gx.efb_w, (s32)gx.efb_h};
    if (sk_valid && !scissor_lost && !memcmp(sk, cur, sizeof cur)) return;
    memcpy(sk, cur, sizeof cur); sk_valid = 1;
    float s = efb_scale(), x0 = efb_x0(), y0 = efb_y0();
    float l = x0 + (gx.sc[0] - gx.sc_off[0]) * s, t = y0 + (gx.sc[1] - gx.sc_off[1]) * s;
    float r = l + gx.sc[2] * s, b = t + gx.sc[3] * s;
    l = l < x0 ? x0 : l;
    t = t < y0 ? y0 : t;
    r = r > 400 - x0 ? 400 - x0 : r < l ? l : r;
    b = b > 240 - y0 ? 240 - y0 : b < t ? t : b;
    u32 s0 = (u32)(240 - b), s1 = (u32)(400 - r), s2 = (u32)(240 - t), s3 = (u32)(400 - l);
    /* GPU test: sdmc:/cr_gpu_quarter.txt clips drawing to the centre quarter of the screen (same vertex
     * work, 1/4 of the pixels); if gpu ms drops ~4x the frame is fill-bound, if not, vertex-bound */
    static int quarter = 2;
    if (quarter == 2) { FILE *f = fopen("sdmc:/cr_gpu_quarter.txt", "rb"); quarter = f != NULL; if (f) fclose(f);
        recomp_log("# gx: gpu quarter-screen test %s\n", quarter ? "on (cr_gpu_quarter.txt)" : "off"); }
    if (quarter) {
        s0 = s0 < 60 ? 60 : s0; s2 = s2 > 180 ? 180 : s2; s1 = s1 < 100 ? 100 : s1; s3 = s3 > 300 ? 300 : s3;
        if (s2 < s0) s2 = s0;
        if (s3 < s1) s3 = s1;
    }
    pend_flush();
    C3D_SetScissor(GPU_SCISSOR_NORMAL, s0, s1, s2, s3);
    scissor_lost = 0;
}

/* ---------------------------------------------------------------- vertices */

static const u8 comp_size[5] = {1, 1, 2, 2, 4};
static const u8 clr_size[6] = {2, 3, 4, 2, 3, 4};

static int attr_size_in(const u8 *desc, VtxFmt (*vat)[A_MAX], int fmt, int a) {
    int d = desc[a];
    if (!d) return 0;
    if (a <= 8) return 1;
    if (d >= 2) return d == 2 ? 1 : 2;
    const VtxFmt *f = &vat[fmt][a];
    switch (a) {
    case A_POS: return comp_size[f->type > 4 ? 4 : f->type] * (f->cnt ? 3 : 2);
    case A_NRM: return comp_size[f->type > 4 ? 4 : f->type] * (f->cnt ? 9 : 3);
    case A_CLR0: case A_CLR1: return clr_size[f->type > 5 ? 5 : f->type];
    default: return comp_size[f->type > 4 ? 4 : f->type] * (f->cnt ? 2 : 1);
    }
}
static int attr_size(int fmt, int a) { return attr_size_in(gx.desc, gx.vat, fmt, a); }
static int vtx_size(int fmt) { int n = 0; for (int a = 0; a <= 20; a++) n += attr_size(fmt, a); return n; }
/* game thread: the same from its copy of the vertex format (FIFO assembly) */
static u8 vsz_g_ok;      /* bit fmt: vsz_g[fmt] is current (cleared when gs.desc / gs.vat change) */
static u16 vsz_g[8];
static u8 desc_dirty;    /* lazy vertex descriptor: gs.desc changed since the last desc_sync (sdk_GXBegin's section) */
static void desc_sync(void);
static u32 st_desc_calls, st_desc_recs, st_gq_recs, st_gq_words;   /* game thread, per 300-frame "# gx queue" line */
static int vtx_size_g(int fmt) {
    if (desc_dirty) desc_sync();
    if (!(vsz_g_ok >> fmt & 1)) {
        int n = 0; for (int a = 0; a <= 20; a++) n += attr_size_in(gs.desc, gs.vat, fmt, a);
        vsz_g[fmt] = n; vsz_g_ok |= 1 << fmt;
    }
    return vsz_g[fmt];
}

static void read_color(const u8 *p, int type, u8 *c) {
    switch (type) {
    case 0: { u32 v = rgb565(p[0] << 8 | p[1]); c[0] = v; c[1] = v >> 8; c[2] = v >> 16; c[3] = 255; break; }
    case 1: c[0] = p[0]; c[1] = p[1]; c[2] = p[2]; c[3] = 255; break;
    case 2: c[0] = p[0]; c[1] = p[1]; c[2] = p[2]; c[3] = 255; break;
    case 3: c[0] = (p[0] >> 4) * 17; c[1] = (p[0] & 15) * 17; c[2] = (p[1] >> 4) * 17; c[3] = (p[1] & 15) * 17; break;
    case 4: { u32 v = p[0] << 16 | p[1] << 8 | p[2];
        for (int i = 0; i < 4; i++) { u32 x = (v >> (18 - 6 * i)) & 63; c[i] = x << 2 | x >> 4; } break; }
    default: c[0] = p[0]; c[1] = p[1]; c[2] = p[2]; c[3] = p[3]; break;
    }
}

/* One plan per draw: where each present attribute's bytes come from. rd reads
 * n components (big-endian, fixed point scaled) into out. */
typedef struct APlan APlan;
struct APlan { s8 attr, d, type, cs, n; u8 off, size, dst, kind, dpos, tco[3]; u16 imax; float scale; const u8 *base; u32 stride; void (*rd)(const u8 *, const APlan *, float *); };   /* tco: packed, byte offset of each further texcoord unit in dst */

static inline u16 be16(const u8 *p) { return (u16)(p[0] << 8 | p[1]); }
static void rd_u8(const u8 *p, const APlan *a, float *o) { for (int i = 0; i < a->n; i++) o[i] = p[i] * a->scale; }
static void rd_s8(const u8 *p, const APlan *a, float *o) { for (int i = 0; i < a->n; i++) o[i] = (s8)p[i] * a->scale; }
static void rd_u16(const u8 *p, const APlan *a, float *o) { for (int i = 0; i < a->n; i++) o[i] = be16(p + 2 * i) * a->scale; }
static void rd_s16(const u8 *p, const APlan *a, float *o) { for (int i = 0; i < a->n; i++) o[i] = (s16)be16(p + 2 * i) * a->scale; }
static void rd_f32(const u8 *p, const APlan *a, float *o) {
    for (int i = 0; i < a->n; i++) { u32 v = (u32)p[4 * i] << 24 | p[4 * i + 1] << 16 | p[4 * i + 2] << 8 | p[4 * i + 3]; o[i] = u2f(v); }
}
static void rd_s16x3(const u8 *p, const APlan *a, float *o) {
    float s = a->scale; o[0] = (s16)be16(p) * s; o[1] = (s16)be16(p + 2) * s; o[2] = (s16)be16(p + 4) * s;
}
static void rd_s16x2(const u8 *p, const APlan *a, float *o) { float s = a->scale; o[0] = (s16)be16(p) * s; o[1] = (s16)be16(p + 2) * s; }
static void rd_s8x3(const u8 *p, const APlan *a, float *o) { float s = a->scale; o[0] = (s8)p[0] * s; o[1] = (s8)p[1] * s; o[2] = (s8)p[2] * s; }
static void rd_f32x3(const u8 *p, const APlan *a, float *o) {
    (void)a; for (int i = 0; i < 3; i++, p += 4) o[i] = u2f((u32)p[0] << 24 | p[1] << 16 | p[2] << 8 | p[3]);
}
static void rd_f32x2(const u8 *p, const APlan *a, float *o) {
    (void)a; for (int i = 0; i < 2; i++, p += 4) o[i] = u2f((u32)p[0] << 24 | p[1] << 16 | p[2] << 8 | p[3]);
}

static int make_plan(int fmt, APlan *pl, int *vsize) {
    int np = 0, off = 0;
    for (int a = 0; a <= 20; a++) {
        int d = gx.desc[a];
        if (!d) continue;
        APlan *q = &pl[np++];
        memset(q, 0, sizeof *q);
        q->attr = a; q->d = d; q->off = off;
        if (a <= 8) { off += 1; continue; }
        const VtxFmt *f = &gx.vat[fmt][a];
        gx.desc[a] = 1; q->size = attr_size(fmt, a); gx.desc[a] = d;
        off += d == 1 ? q->size : d == 2 ? 1 : 2;
        if (d >= 2) {
            q->base = ram + (GUEST(gx.array_base[a - A_POS]) & RAM_MASK);
            q->stride = gx.array_stride[a - A_POS];
        }
        q->type = f->type > 4 ? 4 : f->type;
        q->cs = comp_size[(int)q->type];
        int frac = f->frac;
        if (a == A_POS) q->n = f->cnt ? 3 : 2;
        else if (a == A_NRM) { q->n = 3; frac = f->type == 1 ? 6 : f->type == 3 ? 14 : 0; }
        else if (a >= A_TEX0) q->n = f->cnt ? 2 : 1;
        if (a == A_CLR0 || a == A_CLR1) q->type = f->type > 5 ? 5 : f->type;
        q->scale = 1.0f / (float)(1 << frac);
        if (a != A_CLR0 && a != A_CLR1) {
            static void (*const gen[5])(const u8 *, const APlan *, float *) = {rd_u8, rd_s8, rd_u16, rd_s16, rd_f32};
            q->rd = gen[(int)q->type];
            if (q->type == 3 && q->n == 3) q->rd = rd_s16x3;
            else if (q->type == 3 && q->n == 2) q->rd = rd_s16x2;
            else if (q->type == 1 && q->n == 3) q->rd = rd_s8x3;
            else if (q->type == 4 && q->n == 3) q->rd = rd_f32x3;
            else if (q->type == 4 && q->n == 2) q->rd = rd_f32x2;
        }
    }
    *vsize = off;
    return np;
}

/* Keeps only what the PICA vertex needs: PNMTXIDX, POS, NRM, CLR0 when a
 * material component comes from it, and texcoords a unit reads. dst = PVtx tc
 * unit mask for TEXn, and POS/NRM also feeding texgen. */
static u8 tc_fed;   /* PICA tc units written by some attribute (trim_plan) */

static int trim_plan(APlan *pl, int np, const s8 tsrc[3], int drop_nrm) {   /* drop_nrm: packed, light loop off */
    int k = 0, clr = mat_vtx[0] || mat_vtx[3];
    tc_fed = 0;
    for (int i = 0; i < np; i++) {
        APlan q = pl[i];
        q.dst = 0;
        for (int u = 0; u < 3; u++)
            if ((q.attr == A_POS && tsrc[u] == 1) || (q.attr == A_NRM && tsrc[u] == 2) || (q.attr >= A_TEX0 && tsrc[u] == 3 + q.attr - A_TEX0)) q.dst |= 1 << u;
        tc_fed |= q.dst;
        if (q.attr == A_PNMTX || q.attr == A_POS || (q.attr == A_NRM && (!drop_nrm || q.dst)) || (q.attr == A_CLR0 && clr) || (q.attr >= A_TEX0 && q.dst)) pl[k++] = q;
    }
    return k;
}

enum { K_PNMTX, K_FLOAT, K_CLR, K_RAW, K_CONV };   /* K_RAW: packed, components byte-swapped as they are; K_CONV: packed, rd -> floats */
static u32 pmtx_row(void) {
    u32 pn = gx.cur_mtx;
    if (pn % 3 || pn >= 30) { ONCE("position matrix row %lu not a PNMTXn\n", (unsigned long)pn); pn = 0; }
    return pn;
}
/* packed: a source attribute -> PICA format, component count (*cnt) and whether it is copied raw.
 * home: 0 position, 2 normal (lit), 3.. texcoord (texgen source; c2: its texm column 2 is zero) */
static int pk_fmt(const APlan *a, int home, int c2, u8 *cnt) {
    int t = a->type, tex = a->attr >= A_TEX0, raw = t == 0 || t == 1 || t == 3 || t == 4;
    if (tex && a->n < 2) raw = 0;   /* (s, 0, 1): as PVtx */
    if (!raw || t == 4) {
        *cnt = home == 0 ? 4 : tex && a->n >= 2 && c2 ? 2 : 3;
        return GPU_FLOAT | (raw ? 0x100 : 0);
    }
    *cnt = tex && t == 3 && c2 ? 2 : 4;
    return (t == 3 ? GPU_SHORT : t == 1 ? GPU_BYTE : GPU_UNSIGNED_BYTE) | 0x100;
}
static int fmt_size(int f) { return f == GPU_FLOAT ? 4 : f == GPU_SHORT ? 2 : 1; }
static void put_comp(u8 *d, int f, int v) {   /* integer v in format f */
    if (f == GPU_FLOAT) { float x = (float)v; memcpy(d, &x, 4); }
    else if (f == GPU_SHORT) { s16 x = (s16)v; memcpy(d, &x, 2); }
    else *d = (u8)v;
}
/* PICA vertex: see runtime/vshader.v.pica */
/* Per draw: tm = the vertex with every field an attribute does not write
 * (pos/nrm 0, texcoords (0, 0, 1), register material, current matrix). Per
 * attribute the destination is fixed (dpos = float offset in PVtx; packed: byte
 * offset), so the vertex loop has no per-attribute decisions beyond its kind.
 * Indexed attributes record the largest index they read (imax, for the cache).
 * L = the vertex layout: lay_full unless pk (then lit and c2z, texm column 2 zero
 * per unit, decide it). */
static u32 pow2_frac(float f) { return (127 - (fbits(f) >> 23 & 0xff)) & 31; }   /* f = 2^-frac */
static void plan_finish(APlan *pl, int np, PVtx *tm, VLay *L, int pk, int lit, u8 c2z) {
    memset(tm, 0, sizeof *tm);
    u32 pn = pmtx_row();
    for (int i = 0; i < np; i++) pl[i].imax = 0;
    if (!pk) {
        *L = lay_full;
        for (int u = 0; u < 3; u++) tm->tc[u][2] = 1;
        tm->pos[3] = (float)pn;
        for (int i = 0; i < 4; i++) tm->mat[i] = mat_vtx[i] ? 0 : mat_reg[i];
        for (int i = 0; i < np; i++) {
            APlan *a = &pl[i];
            if (a->attr == A_PNMTX) { a->kind = K_PNMTX; continue; }
            if (a->attr == A_CLR0) { a->kind = K_CLR; continue; }
            a->kind = K_FLOAT;
            int u0 = a->dst ? __builtin_ctz(a->dst) : -1;
            if (a->attr == A_POS) a->dpos = offsetof(PVtx, pos) / 4;
            else if (a->attr == A_NRM) a->dpos = offsetof(PVtx, nrm) / 4;
            else { a->dpos = offsetof(PVtx, tc[u0]) / 4; a->dst &= ~(1 << u0); }
        }
        return;
    }
    /* packed: each attribute's home register (position 0, normal 2 when lit, else its first
     * texcoord unit), the format it decides; further texcoord units it feeds get copies */
    memset(L, 0, sizeof *L);
    L->pk = 1; L->c2z = c2z; L->vs[0] = L->vs[1] = 1;
    for (int u = 0; u < 3; u++) L->ts[u] = 1;
    s8 src[6] = {-1, -1, -1, -1, -1, -1}, home[21];
    u8 rf[6] = {0}, rc[6] = {0}, roff[6] = {0}, israw[6] = {0};
    for (int i = 0; i < np; i++) {
        APlan *a = &pl[i];
        home[i] = -1;
        if (a->attr == A_PNMTX || a->attr == A_CLR0) continue;
        int h = a->attr == A_POS ? 0 : a->attr == A_NRM && lit ? 2 : a->dst ? 3 + __builtin_ctz(a->dst) : -1;
        home[i] = (s8)h;
        if (h >= 0) src[h] = (s8)i;
        for (int u = 0; u < 3; u++) if (a->dst >> u & 1) src[3 + u] = (s8)i;
    }
    int off = 0;
    for (int r = 0; r < 6; r++) {
        if (r == 1) { rf[r] = GPU_UNSIGNED_BYTE; rc[r] = 4; }
        else if (src[r] >= 0) {
            const APlan *a = &pl[(int)src[r]];
            int f = pk_fmt(a, r == 0 ? 0 : r == 2 ? 2 : 3, r >= 3 && a->attr >= A_TEX0 && (c2z >> (r - 3) & 1), &rc[r]);
            rf[r] = (u8)(f & 0xff); israw[r] = f >> 8;
            if (a->attr == A_POS && r >= 3) rc[r] = rf[r] == GPU_FLOAT ? 3 : 4;   /* texgen from position: w = 1 */
            if (israw[r] && rf[r] != GPU_FLOAT) {
                if (r == 0) L->vs[0] = a->scale;
                else if (r == 2) L->vs[1] = a->scale;
                else { L->ts[r - 3] = a->scale; L->tn[r - 3] = a->attr >= A_TEX0 ? 2 : 3; }
            }
        } else continue;
        roff[r] = (u8)off;
        L->reg[L->nl] = (u8)r; L->fmt[L->nl] = rf[r]; L->cnt[L->nl++] = rc[r];
        off += rc[r] * fmt_size(rf[r]);
    }
    L->stride = (u8)off;
    L->id = lay_id(L);
    /* scales are 2^-frac, frac 0..31: 5 bits each, plus 2 bits of tn per unit (exact, 31 bits) */
    L->ush = pow2_frac(L->vs[0]) | pow2_frac(L->vs[1]) << 5;
    for (int u = 0; u < 3; u++) L->ush |= pow2_frac(L->ts[u]) << (10 + 5 * u) | (u32)(L->tn[u] & 3) << (25 + 2 * u);
    /* template: position row, register material, texcoord z / w = 1 */
    u8 *b = (u8 *)tm;
    if (src[0] >= 0) put_comp(b + roff[0] + 3 * fmt_size(rf[0]), rf[0], (int)pn);
    for (int i = 0; i < 4; i++) b[roff[1] + i] = mat_vtx[i] ? 0 : mat_reg[i];
    for (int u = 0; u < 3; u++) {
        int r = 3 + u;
        if (src[r] < 0) continue;
        int fs = fmt_size(rf[r]), tex = pl[(int)src[r]].attr >= A_TEX0;
        if (rc[r] == 4) put_comp(b + roff[r] + 3 * fs, rf[r], 1);
        if (tex && rc[r] >= 3) put_comp(b + roff[r] + 2 * fs, rf[r], 1);
    }
    for (int i = 0; i < np; i++) {
        APlan *a = &pl[i];
        if (a->attr == A_PNMTX) { a->kind = K_PNMTX; a->cs = (s8)fmt_size(rf[0]); a->dpos = (u8)(roff[0] + 3 * a->cs); continue; }
        if (a->attr == A_CLR0) { a->kind = K_CLR; a->dpos = roff[1]; continue; }
        int h = home[i];
        a->dpos = roff[h];
        a->kind = israw[h] ? K_RAW : K_CONV;
        if (h >= 3) a->dst &= ~(1 << (h - 3));
        for (int u = 0; u < 3; u++) a->tco[u] = a->dst >> u & 1 ? roff[3 + u] : 0;
    }
}

/* attribute use per stats frame (frames % 300 == 0), weighted by vertices drawn:
 * path (0 vbuf, 1 vcache, 2 dl cache), texcoord sets fed, lit, source types */
static u32 va_v[3], va_tc[4], va_lit, va_pos[5], va_nrm[5], va_tcs[5], va_tg[3], va_frac[3][17];
static void va_count(const APlan *pl, int np, int lit, int path, u32 nv) {
    if (!va_on) return;
    va_v[path] += nv; va_lit += lit ? nv : 0;
    int ntc = 0;
    for (int i = 0; i < np; i++) {
        const APlan *a = &pl[i];
        if (a->attr == A_POS) { va_pos[(int)a->type] += nv; }
        if (a->attr == A_NRM && lit) va_nrm[(int)a->type] += nv;
        /* plan_finish dropped a texcoord's own unit from dst: dst = further units only */
        int nu = __builtin_popcount(a->dst) + (a->attr >= A_TEX0);
        if (a->attr >= A_TEX0) va_tcs[(int)a->type] += nv;
        ntc += nu; va_tg[a->attr == A_POS ? 0 : a->attr == A_NRM ? 1 : 2] += nu * nv;
        if (a->kind != K_PNMTX && a->kind != K_CLR && a->scale > 0) {
            int fr = 0; for (float s = a->scale; s < 1 && fr < 16; s *= 2) fr++;
            int w = a->attr == A_POS ? 0 : a->attr == A_NRM ? 1 : 2;
            if (a->type != 4 && (w != 1 || lit)) va_frac[w][fr] += nv;
        }
    }
    va_tc[ntc > 3 ? 3 : ntc] += nv;
}
static void va_log(void) {
    char fr[3][128];
    for (int w = 0; w < 3; w++) { int p = 0; fr[w][0] = 0; for (int f = 0; f <= 16; f++) if (va_frac[w][f]) p += snprintf(fr[w] + p, sizeof fr[w] - p, " %d:%lu", f, (unsigned long)va_frac[w][f]); }
    recomp_log("# gx vattr frame %lu: verts vbuf %lu vcache %lu dl %lu; tc sets 0/1/2/3 %lu/%lu/%lu/%lu; lit %lu; pos u8/s8/u16/s16/f32 %lu/%lu/%lu/%lu/%lu; nrm(lit) %lu/%lu/%lu/%lu/%lu; tc %lu/%lu/%lu/%lu/%lu; tc from pos/nrm/tex %lu/%lu/%lu; frac pos%s nrm%s tc%s\n",
        (unsigned long)frames, (unsigned long)va_v[0], (unsigned long)va_v[1], (unsigned long)va_v[2],
        (unsigned long)va_tc[0], (unsigned long)va_tc[1], (unsigned long)va_tc[2], (unsigned long)va_tc[3], (unsigned long)va_lit,
        (unsigned long)va_pos[0], (unsigned long)va_pos[1], (unsigned long)va_pos[2], (unsigned long)va_pos[3], (unsigned long)va_pos[4],
        (unsigned long)va_nrm[0], (unsigned long)va_nrm[1], (unsigned long)va_nrm[2], (unsigned long)va_nrm[3], (unsigned long)va_nrm[4],
        (unsigned long)va_tcs[0], (unsigned long)va_tcs[1], (unsigned long)va_tcs[2], (unsigned long)va_tcs[3], (unsigned long)va_tcs[4],
        (unsigned long)va_tg[0], (unsigned long)va_tg[1], (unsigned long)va_tg[2], fr[0], fr[1], fr[2]);
    memset(va_v, 0, sizeof va_v); memset(va_tc, 0, sizeof va_tc); va_lit = 0; memset(va_pos, 0, sizeof va_pos);
    memset(va_nrm, 0, sizeof va_nrm); memset(va_tcs, 0, sizeof va_tcs); memset(va_tg, 0, sizeof va_tg); memset(va_frac, 0, sizeof va_frac);
}

static void decode_vtx(const u8 *p, APlan *pl, int np, const PVtx *tm, PVtx *o) {
    *o = *tm;
    for (int i = 0; i < np; i++) {
        APlan *a = &pl[i];
        const u8 *src = p + a->off;
        if (a->kind == K_PNMTX) {
            u32 pn = *src;
            if (pn % 3 || pn >= 30) { ONCE("position matrix row %lu not a PNMTXn\n", (unsigned long)pn); pn = 0; }
            o->pos[3] = (float)pn;
            continue;
        }
        if (a->d >= 2) {
            u32 ix = a->d == 2 ? src[0] : (u32)(src[0] << 8 | src[1]);
            if (ix > a->imax) a->imax = ix;
            src = a->base + ix * a->stride;
        }
        if (a->kind == K_CLR) {
            u8 clr[4];
            read_color(src, a->type, clr);
            for (int c = 0; c < 4; c++) if (mat_vtx[c]) o->mat[c] = clr[c];
            continue;
        }
        float *d = (float *)o + a->dpos;
        a->rd(src, a, d);
        if (a->dst) for (int u = 0; u < 3; u++) if (a->dst >> u & 1) memcpy(o->tc[u], d, 12);
    }
}

/* packed vertex (plan_finish with pk): template, then each attribute at its byte offset */
static void decode_pk(const u8 *p, APlan *pl, int np, const u8 *tm, u32 stride, u8 *o) {
    memcpy(o, tm, stride);
    for (int i = 0; i < np; i++) {
        APlan *a = &pl[i];
        const u8 *src = p + a->off;
        if (a->kind == K_PNMTX) {
            u32 pn = *src;
            if (pn % 3 || pn >= 30) { ONCE("position matrix row %lu not a PNMTXn\n", (unsigned long)pn); pn = 0; }
            put_comp(o + a->dpos, a->cs == 4 ? GPU_FLOAT : a->cs == 2 ? GPU_SHORT : GPU_BYTE, (int)pn);
            continue;
        }
        if (a->d >= 2) {
            u32 ix = a->d == 2 ? src[0] : (u32)(src[0] << 8 | src[1]);
            if (ix > a->imax) a->imax = ix;
            src = a->base + ix * a->stride;
        }
        u8 *d = o + a->dpos;
        if (a->kind == K_CLR) {
            u8 clr[4];
            read_color(src, a->type, clr);
            for (int c = 0; c < 4; c++) if (mat_vtx[c]) d[c] = clr[c];
            continue;
        }
        int nb;
        if (a->kind == K_RAW) {
            nb = a->n * a->cs;
            if (a->cs == 4) for (int k = 0; k < nb; k += 4) { u32 w; memcpy(&w, src + k, 4); w = __builtin_bswap32(w); memcpy(d + k, &w, 4); }
            else if (a->cs == 2) for (int k = 0; k < nb; k += 2) { u16 w; memcpy(&w, src + k, 2); w = __builtin_bswap16(w); memcpy(d + k, &w, 2); }
            else memcpy(d, src, nb);
        } else {
            float f[3];
            a->rd(src, a, f);
            nb = a->n * 4;
            memcpy(d, f, nb);
        }
        if (a->dst) for (int u = 0; u < 3; u++) if (a->dst >> u & 1) memcpy(o + a->tco[u], d, nb);
    }
}

/* n vertices of a draw into o (PVtx or packed per L) */
static void decode_run(const u8 *data, u32 n, int vs, APlan *pl, int np, const PVtx *tm, const VLay *L, u8 *o) {
    if (!L->pk) { PVtx *v = (PVtx *)o; for (u32 i = 0; i < n; i++) decode_vtx(data + i * vs, pl, np, tm, &v[i]); }
    else for (u32 i = 0; i < n; i++) decode_pk(data + i * vs, pl, np, (const u8 *)tm, L->stride, o + i * L->stride);
    st_dec_v += n; st_dec_b += n * L->stride;
}

/* ---------------------------------------------------------------- vertex cache
 * Decoded vertices of display-list draws, keyed by the vertex data address,
 * count, and a signature of everything decode reads besides guest memory (plan,
 * template vertex). Valid while a hash of the draw's vertex bytes and of each
 * indexed array (see arr_hash) still matches. Entries live in linear memory
 * the GPU reads directly (no copy into vbuf or ibuf): vertices in vc_arena,
 * index lists in vc_ixa, each allocated from a ring (ring_alloc). */
#define VC_N 16384
#define CACHE_WAYS 4   /* both caches: sets of 4 slots, least recently used way replaced. 2-way
                        * thrashed: ~250 display-list keys a battle frame in 2048 sets put 3 in
                        * one set about every other build (all of them missing every frame) */
#define VC_ARR 6
#define VC_NONE 0xffffffffu
typedef struct {
    const u8 *data; u32 n, sig, dhash, ahash, narr, used, gen, off, checked;
    u32 ixoff, ixn, prim;   /* the draw's index list in vc_ixa (ixn 0: none yet) */
    u32 vrec, irec;         /* ring records of off and ixoff (index + 1, 0: none) */
    const u8 *arr[VC_ARR]; u32 alen[VC_ARR], ah[VC_ARR];
} VCache;
static VCache vcache[VC_N];
static VCache *vc_last;   /* entry of the last vc_draw, NULL if not kept */
/* display-list cache entries (see dl_draw) */
#define DL_N 4096   /* CACHE_WAYS-way */
#define DL_ARR 12
typedef struct {
    const u8 *list; u32 n, nv, sig, gen, slow, lhash, ahash, checked, used, narr, off, ixoff, ixn, prims, tris, vep, vrec, irec;
    const u8 *arr[DL_ARR]; u32 alen[DL_ARR], ah[DL_ARR];
} DlCache;
static DlCache dlcache[DL_N];

/* Ring allocation (vring: vertices, iring: index lists). Each allocation has a record; the
 * records of live allocations are queued in arena order starting at the ring's pointer (top),
 * so the next allocation in the way is always the queue's front. An allocation takes the space
 * at top: allocations in the way are evicted (their owner entry dropped: it misses and decodes
 * again) unless drawn in this GX frame (record last == frames), since the GPU may still read
 * those; top then moves past them. The previous frame's are free to go: frame_begin's
 * C3D_FrameBegin waits for its GPU queue. Second chance: allocations drawn in the last 3 frames
 * are also passed over until a whole lap found no room (a plain ring evicted entries drawn every
 * frame, which were decoded again the next frame and evicted the next ones: a wave of decodes
 * the size of the new allocation, every frame). Was: bump allocation, and the whole arena
 * emptied at a frame start when 3/4 full (every entry decoded again in one frame).
 * Entries hold their records as index + 1; a record's owner is cleared (OWN_NONE) when the
 * entry gets a new allocation, and that space is reused once top passes it. */
#define OWN_NONE 0xffffffffu
#define OWN_DL 0x80000000u
static u32 own_vc(const VCache *e) { return (u32)(e - vcache); }
static u32 own_dl(const DlCache *e) { return OWN_DL | (u32)(e - dlcache); }
static void ring_flush(VcRing *g) {
    if (g->dhi > g->dlo) GSPGPU_FlushDataCache(g->base + g->dlo * g->unit, (g->dhi - g->dlo) * g->unit);
    g->dlo = 0xffffffffu; g->dhi = 0;
}
static void ring_rotate(VcRing *g) {   /* the front record goes behind top: last in arena order */
    g->q[(g->qh + g->qn) % VC_REC] = g->q[g->qh];
    g->qh = (g->qh + 1) % VC_REC;
}
static void ring_evict(VcRing *g, u32 ri) {   /* the record at the front: its owner loses it */
    u32 o = g->rec[ri].owner;
    if (o != OWN_NONE && (o & OWN_DL)) {
        DlCache *e = &dlcache[o & ~OWN_DL];
        u32 *rp = g == &vring ? &e->vrec : &e->irec;
        if (*rp == ri + 1) { *rp = 0; e->gen = 0; }
    } else if (o != OWN_NONE) {
        VCache *e = &vcache[o];
        if (g == &vring && e->vrec == ri + 1) { e->vrec = 0; e->gen = 0; }
        if (g == &iring && e->irec == ri + 1) { e->irec = 0; e->ixn = 0; }   /* the vertices stay */
    }
    g->qh = (g->qh + 1) % VC_REC; g->qn--;
    g->freel[g->nfree++] = (u16)ri;
    g->live -= g->rec[ri].len;
    g->evicted++;
}
/* n units for owner, starting at a multiple of al units; *rec gets its record (index + 1).
 * VC_NONE when this frame's draws leave no room or no record is free. */
static u32 ring_alloc(VcRing *g, u32 n, u32 al, u32 owner, u32 *rec) {
    *rec = 0;
    if (!n || n > g->size || !g->nfree) { g->fails++; return VC_NONE; }
    u32 moved = 0;   /* units top went past without allocating: a lap, then recent ones go too; 2.5 laps: no room */
    for (;;) {
        if (moved > 2 * g->size + g->size / 2) { g->fails++; return VC_NONE; }
        u32 t = (g->top + al - 1) / al * al;   /* the allocation's start: [top, t) stays unused */
        if (t + n > g->size) {   /* too short before the end: wrap, the allocations there stay */
            for (u32 k = g->qn; k && g->rec[g->q[g->qh]].off >= g->top; k--) ring_rotate(g);
            moved += g->size - g->top; g->top = 0;
            continue;
        }
        if (!g->qn) break;
        u32 ri = g->q[g->qh];
        const VcRec *r = &g->rec[ri];
        if (r->off < g->top || r->off >= t + n) break;   /* nothing (else) in the way */
        /* drawn this frame: never evicted; drawn in the last 3 frames: only once a lap found nothing older */
        if (r->last == frames || (moved < g->size && frames - r->last < 4)) {
            moved += r->off + r->len - g->top; g->top = r->off + r->len; ring_rotate(g); continue;
        }
        ring_evict(g, ri);
    }
    u32 ri = g->freel[--g->nfree];
    VcRec *r = &g->rec[ri];
    r->off = (g->top + al - 1) / al * al; r->len = n; r->last = frames; r->owner = owner;
    g->q[(g->qh + g->qn++) % VC_REC] = (u16)ri;
    if (g->dhi > g->dlo && (r->off < g->dhi || r->off >= g->dhi + al)) ring_flush(g);   /* not contiguous with what was written (up to the alignment gap): flush that now */
    if (g->dhi <= g->dlo) g->dlo = r->off;
    g->dhi = r->off + n;
    g->top = r->off + n; g->live += n;
    *rec = ri + 1;
    return r->off;
}
/* the entry gets a new allocation (or none): its old one becomes unowned, reused once top passes it */
static void ring_detach(VcRing *g, u32 *rec) {
    if (*rec) g->rec[*rec - 1].owner = OWN_NONE;
    *rec = 0;
}
static void ring_used(VcRing *g, u32 rec) { if (rec) g->rec[rec - 1].last = frames; }   /* drawn this frame: kept until the next */


static u32 hash_mem(const u8 *p, u32 n, u32 h) {
    u32 w;
    if (n >= 32) {   /* four independent lanes: the single chain is bound by multiply latency */
        u32 h1 = h ^ 0x85ebca6bu, h2 = h ^ 0xc2b2ae35u, h3 = h ^ 0x27d4eb2fu, w1, w2, w3;
        for (; n >= 16; n -= 16, p += 16) {
            memcpy(&w, p, 4); memcpy(&w1, p + 4, 4); memcpy(&w2, p + 8, 4); memcpy(&w3, p + 12, 4);
            h = (h ^ w) * 0x9e3779b1u; h1 = (h1 ^ w1) * 0x9e3779b1u; h2 = (h2 ^ w2) * 0x9e3779b1u; h3 = (h3 ^ w3) * 0x9e3779b1u;
            h ^= h >> 15; h1 ^= h1 >> 15; h2 ^= h2 >> 15; h3 ^= h3 >> 15;
        }
        h = ((h * 0x85ebca6bu ^ h1) * 0xc2b2ae35u ^ h2) * 0x27d4eb2fu ^ h3;
    }
    for (; n >= 4; n -= 4, p += 4) { memcpy(&w, p, 4); h = (h ^ w) * 0x9e3779b1u; h ^= h >> 15; }
    for (; n; n--, p++) { h = (h ^ *p) * 0x9e3779b1u; h ^= h >> 15; }
    return h;
}

/* Cache keys use guest offsets, never host pointers (ram[] and the rd_ functions move with
 * every build, which moved entries between sets: per-build conflict misses). rd follows from
 * type and n, which are hashed. */
static inline u32 gofs(const u8 *p) { return (u32)(p - ram); }

static u32 plan_sig(const APlan *pl, int np, int vs, const PVtx *tm, const VLay *L) {
    u32 h = hash_mem((const u8 *)tm, sizeof *tm, (u32)vs * 0x85ebca6bu + (u32)np);
    h = hash_mem((const u8 *)L, sizeof *L, h);
    for (int i = 0; i < np; i++) {
        const APlan *a = &pl[i];
        u32 f[8] = {(u32)a->attr | (u32)(u8)a->d << 8 | (u32)(u8)a->type << 16 | (u32)(u8)a->n << 24,
                    (u32)a->off | (u32)a->size << 8 | (u32)a->dst << 16 | (u32)a->dpos << 24,
                    fbits(a->scale), a->base ? gofs(a->base) : 0, a->stride,
                    (u32)a->tco[0] | (u32)a->tco[1] << 8 | (u32)a->tco[2] << 16 | (u32)(u8)a->cs << 24, (u32)a->kind, 0};
        h = hash_mem((const u8 *)f, sizeof f, h);
    }
    return h;
}

/* Display-list draws: the trimmed, finished plan, template vertex and signature memoised on
 * everything make_plan, trim_plan, plan_finish and the signature read (vertex format of the
 * attributes present, array base/stride of the indexed ones, current matrix, material sources,
 * texgen sources), compared in full on a hit. A position matrix row that is not a PNMTXn is
 * never memoised (plan_finish logs it per call). */
typedef struct {
    u8 desc[21], fmt; s8 tsrc[3]; u8 mv[4], mr[4], pk, lit, c2z, pad; VtxFmt vat[12];
    u32 cur_mtx, base[12], stride[12];
} PlanKey;
#define PM_N 512   /* 2-way: the packed key (lit, texm column 2) put 10 of ~250 battle calls a frame on colliding slots of a 256-entry direct-mapped memo */
#define PM_MAX 8   /* trimmed plans hold PNMTXIDX, POS, NRM, CLR0 and texcoords feeding 3 units */
typedef struct { PlanKey k; u32 valid, used, kh, sig, vs, np; APlan pl[PM_MAX]; PVtx tm; VLay lay; } PlanMemo;   /* kh: hash of k */
static PlanMemo pmemo[PM_N];
/* NULL: build it per call (not memoised) */
static const PlanMemo *plan_get(int fmt, const s8 tsrc[3], int pk, int lit, u8 c2z) {
    if (gx.cur_mtx % 3 || gx.cur_mtx >= 30) return NULL;
    PlanKey k;
    memset(&k, 0, sizeof k);
    memcpy(k.desc, gx.desc, 21); k.fmt = (u8)fmt; memcpy(k.tsrc, tsrc, 3);
    memcpy(k.mv, mat_vtx, 4); memcpy(k.mr, mat_reg, 4); k.cur_mtx = gx.cur_mtx;
    if (pk) { k.pk = 1; k.lit = (u8)lit; k.c2z = c2z; }   /* only a packed layout depends on them */
    for (int a = A_POS; a <= 20; a++) {
        if (!gx.desc[a]) continue;
        k.vat[a - A_POS] = gx.vat[fmt][a];
        if (gx.desc[a] >= 2) { k.base[a - A_POS] = gx.array_base[a - A_POS]; k.stride[a - A_POS] = gx.array_stride[a - A_POS]; }
    }
    u32 kh = hash_mem((const u8 *)&k, sizeof k, 0x811c9dc5u);
    PlanMemo *set = &pmemo[(kh & (PM_N / 2 - 1)) * 2];
    for (int i = 0; i < 2; i++) if (set[i].valid && set[i].kh == kh && !memcmp(&set[i].k, &k, sizeof k)) { set[i].used = frames; pm_hits++; return &set[i]; }
    pm_misses++;
    PlanMemo *m = !set[0].valid ? &set[0] : !set[1].valid ? &set[1] : set[0].used <= set[1].used ? &set[0] : &set[1];
    m->used = frames; m->kh = kh;
    APlan pl[21];
    int vs, np = make_plan(fmt, pl, &vs);
    np = trim_plan(pl, np, tsrc, pk && !lit);
    if (np > PM_MAX) return NULL;
    m->valid = 0;
    plan_finish(pl, np, &m->tm, &m->lay, pk, lit, c2z);
    m->sig = plan_sig(pl, np, vs, &m->tm, &m->lay) ^ ((u32)fmt << 28 | (u32)mat_vtx[0] | (u32)mat_vtx[3] << 1) * 0x85ebca6bu;
    memcpy(m->pl, pl, np * sizeof *pl);
    m->np = np; m->vs = vs; m->k = k; m->valid = 1;
    return m;
}

/* Each vertex array (base address) is hashed once per frame over the extent
 * any cached draw read from it (a larger extent rehashes, invalidating older
 * entries once). Assumes the game does not rewrite a vertex array between
 * two display-list draws of the same frame that read it. */
static u32 arr_hash(const u8 *p, u32 len) {
    static struct { const u8 *p; u32 len, frame, h; } memo[1024];
    u32 k = (u32)(uintptr_t)p * 0x9e3779b1u >> 22;
    if (memo[k].p != p) { memo[k].p = p; memo[k].len = 0; memo[k].frame = 0; }
    if (len > memo[k].len) { memo[k].len = len; memo[k].frame = 0; }
    if (memo[k].frame != frames + 1) { memo[k].frame = frames + 1; memo[k].h = hash_mem(p, memo[k].len, 0x811c9dc5u); }
    return memo[k].h;
}

static u32 vc_arrays_hash(VCache *e) {
    u32 h = 0;
    for (u32 i = 0; i < e->narr; i++) { u32 a = arr_hash(e->arr[i], e->alen[i]); e->ah[i] = a; h = (h ^ a) * 0x9e3779b1u; }
    return h;
}

/* Display-list draw: returns the arena vertex offset of its decoded vertices
 * (decoding them there on a miss), or VC_NONE when the caller decodes into vbuf */
static u32 vc_draw(const u8 *data, int n, int vs, APlan *pl, int np, const PVtx *tm, const VLay *L) {
    vc_last = NULL;
    if (!vc_arena || n > 0x8000) return VC_NONE;
    u32 sig = plan_sig(pl, np, vs, tm, L);
    u32 slot = (gofs(data) * 0x9e3779b1u ^ sig ^ (u32)n * 0x85ebca6bu) >> 18 & (VC_N - 1);
    VCache *set = &vcache[slot & ~(CACHE_WAYS - 1u)], *e = NULL, *v = set;
    for (int i = 0; i < CACHE_WAYS; i++) {
        VCache *c = set + i;
        if (c->gen && c->data == data && c->n == (u32)n && c->sig == sig) { e = c; break; }
        if (v->gen && (!c->gen || c->used < v->used)) v = c;   /* replace the empty or least recently used way */
    }
    if (!e) e = v;
    /* the list bytes and arrays are checked once per frame per entry (same
     * assumption as arr_hash: not rewritten between two draws of a frame) */
    int same = e->gen && e->data == data && e->n == (u32)n && e->sig == sig;
    if (same && e->checked == frames + 1) {
        e->used = frames; ring_used(&vring, e->vrec);
        vc_hits++;
        vc_last = e;
        return e->off;
    }
    u32 dh = hash_mem(data, (u32)n * vs, 0x811c9dc5u);
    if (same && e->dhash == dh && vc_arrays_hash(e) == e->ahash) {
        e->checked = frames + 1;
        e->used = frames; ring_used(&vring, e->vrec);
        vc_hits++;
        vc_last = e;
        return e->off;
    }
    vc_misses++;
    if (frames % 300 == 0) {   /* stats frames: why */
        static u32 logged, lframe = 0xffffffffu;
        if (lframe != frames) { lframe = frames; logged = 0; }
        if (logged++ < 32) {
            char a[96] = "";
            if (same && e->dhash == dh)
                for (u32 i = 0, p = 0; i < e->narr && p < sizeof a - 20; i++)
                    if (arr_hash(e->arr[i], e->alen[i]) != e->ah[i]) p += snprintf(a + p, sizeof a - p, " %08lx+%lx", (unsigned long)(e->arr[i] - ram + 0x80000000u), (unsigned long)e->alen[i]);
            recomp_log("# gx vc miss frame %lu: data %08lx n %d %s%s\n", (unsigned long)frames, (unsigned long)(data - ram + 0x80000000u), n,
                       !same ? "new" : e->dhash != dh ? "vertex bytes" : "arrays", a);
        }
    }
    e->gen = 0; e->ixn = 0;
    ring_detach(&vring, &e->vrec); ring_detach(&iring, &e->irec);
    u32 vr, off = ring_alloc(&vring, n * L->stride / 4, L->stride / 4, own_vc(e), &vr);
    if (off == VC_NONE) return VC_NONE;
    decode_run(data, n, vs, pl, np, tm, L, vc_arena + off * 4);
    u32 narr = 0;
    for (int i = 0; i < np; i++) if (pl[i].d >= 2 && pl[i].kind != K_PNMTX) {
        if (narr == VC_ARR) { ring_detach(&vring, &vr); return off; }   /* drawn from the arena, not kept */
        e->arr[narr] = pl[i].base;
        e->alen[narr++] = pl[i].imax * pl[i].stride + pl[i].size;
    }
    e->data = data; e->n = n; e->sig = sig; e->dhash = dh; e->narr = narr; e->off = off;
    e->ahash = vc_arrays_hash(e);
    e->used = frames; e->gen = 1; e->vrec = vr; e->checked = frames + 1;
    vc_last = e;
    return off;
}

/* ---------------------------------------------------------------- uniforms */
static void unif_set(int loc, const float (*v)[4], int n) {
    if (!memcmp(ushadow[loc], v, n * 16)) return;
    pend_flush();
    memcpy(ushadow[loc], v, n * 16);
    C3D_FVec *p = C3D_FVUnifWritePtr(GPU_VERTEX_SHADER, loc, n);
    for (int i = 0; i < n; i++) { p[i].x = v[i][0]; p[i].y = v[i][1]; p[i].z = v[i][2]; p[i].w = v[i][3]; }
}

/* eye space -> clip space of the rotated 3DS framebuffer: GX projection, then the
 * viewport (ndc -> EFB pixel), EFB -> 400x240 and the 90-degree rotation, all
 * affine in (cx, cy, cw) so one 4x4 matrix */
/* memo of a uniform setter's inputs: true when they equal the previous call's */
static int same_inputs(void *last, int *valid, const void *key, size_t n) {
    if (*valid && !memcmp(last, key, n)) return 1;
    memcpy(last, key, n); *valid = 1;
    return 0;
}

static void set_clip(void) {
    struct { float proj[6], vp[4]; u32 ortho; s32 sc_off[2], w, h; } key;
    static u8 last[sizeof key]; static int valid;
    memcpy(key.proj, gx.proj, sizeof key.proj); memcpy(key.vp, gx.vp, sizeof key.vp);
    key.ortho = gx.proj_ortho; key.sc_off[0] = gx.sc_off[0]; key.sc_off[1] = gx.sc_off[1]; key.w = gx.efb_w; key.h = gx.efb_h;
    if (same_inputs(last, &valid, &key, sizeof key)) return;
    const float *p = gx.proj;
    float P[4][4] = {{0}}, m[4][4];
    if (gx.proj_ortho) { P[0][0] = p[0]; P[0][3] = p[1]; P[1][1] = p[2]; P[1][3] = p[3]; P[2][2] = p[4]; P[2][3] = p[5]; P[3][3] = 1; }
    else { P[0][0] = p[0]; P[0][2] = p[1]; P[1][1] = p[2]; P[1][2] = p[3]; P[2][2] = p[4]; P[2][3] = p[5]; P[3][2] = -1; }
    float s = efb_scale();
    float ay = s * gx.vp[3] * 0.5f / 120.0f, by = 1 - (efb_y0() + s * (gx.vp[1] + gx.vp[3] * 0.5f - gx.sc_off[1])) / 120.0f;
    float ax = s * gx.vp[2] * 0.5f / 200.0f, bx = 1 - (efb_x0() + s * (gx.vp[0] + gx.vp[2] * 0.5f - gx.sc_off[0])) / 200.0f;
    for (int c = 0; c < 4; c++) {
        m[0][c] = ay * P[1][c] + by * P[3][c];
        m[1][c] = -ax * P[0][c] + bx * P[3][c];
        m[2][c] = P[2][c];   /* GX and PICA both clip z to [-w, 0]; depth map in setup_pixel */
        m[3][c] = P[3][c];
    }
    unif_set(u_clip, m, 4);
}

static u32 xf_gen = 1;   /* bumped on every XF matrix write */
static u32 xf_row_gen[0x180];   /* xf_gen of the last write per XF row (4 floats; load_mtx) */
static u32 light_gen = 1;   /* bumped on every GXLoadLightObjImm */

/* rows of the posm / nrmm uniforms written since their last upload (load_mtx,
 * GXLoadNrmMtxImm); only runs of these rows are compared and uploaded */
static u32 pos_dirty = 0x3fffffff, nrm_dirty = 0x3fffffff;
static void set_mtx(int lit) {
    float m[30][4];
    while (pos_dirty) {
        int r0 = __builtin_ctz(pos_dirty), r1 = r0;
        while (r1 < 30 && pos_dirty >> r1 & 1) r1++;
        pos_dirty &= ~((1u << r1) - (1u << r0));
        unif_set(u_pos + r0, (const float (*)[4])&gx.xf[r0 * 4], r1 - r0);
    }
    if (!lit) return;
    while (nrm_dirty) {
        int r0 = __builtin_ctz(nrm_dirty), r1 = r0;
        while (r1 < 30 && nrm_dirty >> r1 & 1) r1++;
        nrm_dirty &= ~((1u << r1) - (1u << r0));
        for (int r = r0; r < r1; r++) { const float *n = &gx.xf[0x400 + r * 3]; m[r][0] = n[0]; m[r][1] = n[1]; m[r][2] = n[2]; m[r][3] = 0; }
        unif_set(u_nrm + r0, &m[r0], r1 - r0);
    }
}

/* texgen for PICA unit u: rows R0, R1, R2 with (s, t) = (R0.x / R2.x, R1.x / R2.x).
 * Kept in tex_m (tex_c2z: units whose column 2 is zero) and uploaded by set_vtx_unif with
 * the draw's packed scales folded in. */
static float tex_m[9][4];
static u8 tex_c2z;
static u32 tex_m_gen = 1;   /* bumped when tex_m is recomputed */
static void set_texgen(const float tscale[3][2], const int tcoord[3], s8 tsrc[3], const u8 *v0, const APlan *pl, int np) {
    struct { float tscale[3][2]; int tcoord[3]; u32 n, xf[3][3]; TexGen g[8]; } key;
    static u8 last[sizeof key]; static int valid; static s8 last_src[3];
    int per_vtx = 0;
    for (int u = 0; u < 3; u++) if (tcoord[u] >= 0 && tcoord[u] < 8 && gx.desc[1 + tcoord[u]]) per_vtx = 1;
    memset(&key, 0, sizeof key);
    memcpy(key.tscale, tscale, sizeof key.tscale); memcpy(key.tcoord, tcoord, sizeof key.tcoord);
    key.n = gx.n_texgens;
    for (int u = 0; u < 3; u++) {   /* the matrix rows read below (was: any matrix write since) */
        int tci = tcoord[u];
        if (tci < 0 || tci >= (int)gx.n_texgens) continue;
        const TexGen *g = &gx.texgen[tci];
        if (g->func <= 1 && g->mtx != 60) for (int r = 0; r < 3; r++) key.xf[u][r] = xf_row_gen[(g->mtx & 63) + r];
    }
    for (int i = 0; i < 8; i++) { key.g[i].func = gx.texgen[i].func; key.g[i].src = gx.texgen[i].src; key.g[i].normalize = gx.texgen[i].normalize; key.g[i].mtx = gx.texgen[i].mtx; }
    if (per_vtx) valid = 0;   /* reads the first vertex */
    else if (same_inputs(last, &valid, &key, sizeof key)) { memcpy(tsrc, last_src, 3); return; }
    float m[9][4];
    memset(m, 0, sizeof m);
    for (int u = 0; u < 3; u++) {
        float (*row)[4] = &m[u * 3];
        float g3[3][4] = {{1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 0, 1}};
        tsrc[u] = 0;
        row[2][3] = 1;
        int tci = tcoord[u];
        if (tci < 0 || tci >= (int)gx.n_texgens) continue;
        const TexGen *g = &gx.texgen[tci];
        if (g->src == 0) tsrc[u] = 1;
        else if (g->src == 1) tsrc[u] = 2;
        else if (g->src >= 4 && g->src <= 11) tsrc[u] = 3 + g->src - 4;
        else { ONCE("texgen src %d not supported\n", g->src); tsrc[u] = 0; }
        u32 mid = g->mtx;
        if (gx.desc[1 + tci]) {   /* per-vertex TEXnMTXIDX: the first vertex's */
            for (int i = 0; i < np; i++) if (pl[i].attr == 1 + tci) mid = v0[pl[i].off];
            ONCE("per-vertex texture matrix index: first vertex's used for the draw\n");
        }
        if (g->func <= 1 && mid != 60) {
            const float *x = &gx.xf[(mid & 63) * 4];
            for (int r = 0; r < (g->func == 0 ? 3 : 2); r++) memcpy(g3[r], x + r * 4, 16);
            if (g->src == 1) for (int r = 0; r < 3; r++) g3[r][3] = 0;   /* normal: w = 0 */
            if (g->func == 0) ONCE("projective texgen divided per vertex\n");
        } else if (g->func > 1) ONCE("texgen func %d not supported\n", g->func);
        for (int c = 0; c < 4; c++) {
            row[0][c] = g3[0][c] * tscale[u][0];
            row[1][c] = g3[2][c] - g3[1][c] * tscale[u][1];   /* PICA v = 1 is memory row 0 */
            row[2][c] = g3[2][c];
        }
    }
    memcpy(tex_m, m, sizeof m); tex_m_gen++;
    tex_c2z = 0;
    for (int u = 0; u < 3; u++) if (m[u * 3][2] == 0 && m[u * 3 + 1][2] == 0 && m[u * 3 + 2][2] == 0) tex_c2z |= 1 << u;
    memcpy(last_src, tsrc, 3);
}

/* the draw's texm (tex_m, columns of a packed integer texcoord scaled: exact, powers of 2) and vscale */
static void set_vtx_unif(const VLay *L) {
    static u32 last_gen, last_ush;
    if (last_gen == tex_m_gen && last_ush == L->ush) return;
    last_gen = tex_m_gen; last_ush = L->ush;
    st_vu++;
    float m[9][4], v[1][4] = {{L->vs[0], L->vs[1], 0, 0}};
    memcpy(m, tex_m, sizeof m);
    for (int u = 0; u < 3; u++)
        for (int r = 0; r < 3; r++) for (int c = 0; c < L->tn[u]; c++) m[u * 3 + r][c] *= L->ts[u];
    unif_set(u_tex, m, 9);
    unif_set(u_vsc, v, 1);
}

static void set_blit(int on, int n) {   /* light loop on/off and its count */
    static int last_on = -1, last_n = -1;
    if (on == last_on && (!on || n == last_n)) return;
    pend_flush();
    C3D_BoolUnifSet(GPU_VERTEX_SHADER, u_blit, on);
    if (on) C3D_IVUnifSet(GPU_VERTEX_SHADER, u_iloop, n - 1, 0, 5, 0);
    last_on = on; if (on) last_n = n;
}

static int set_lighting_inner(const Chan *c0, const Chan *c2, int e0, int e2);
/* COLOR0/ALPHA0 lighting; returns whether the shader's light loop runs */
static int set_lighting(void) {
    const Chan *c0 = &gx.chan[0], *c2 = &gx.chan[2];
    int e0 = gx.n_chans && c0->enable, e2 = gx.n_chans && c2->enable;
    mat_vtx[0] = mat_vtx[1] = mat_vtx[2] = gx.n_chans && c0->mat_src;
    mat_vtx[3] = gx.n_chans && c2->mat_src;
    for (int i = 0; i < 4; i++) mat_reg[i] = gx.n_chans ? gx.mat[0][i] : 0;
    struct { Chan c[4]; u32 n, lg; u8 amb[4]; } key;
    static u8 last[sizeof key]; static int valid, last_lit;
    memset(&key, 0, sizeof key);
    for (int i = 0; i < 4; i++) key.c[i] = gx.chan[i];
    key.n = gx.n_chans; key.lg = light_gen; memcpy(key.amb, gx.amb[0], 4);
    if (same_inputs(last, &valid, &key, sizeof key)) return last_lit;
    last_lit = set_lighting_inner(c0, c2, e0, e2);
    return last_lit;
}
static int set_lighting_inner(const Chan *c0, const Chan *c2, int e0, int e2) {
    if ((e0 && c0->amb_src) || (e2 && c2->amb_src)) ONCE("vertex ambient colour: register used\n");
    float amb[1][4];
    for (int i = 0; i < 4; i++) amb[0][i] = (i < 3 ? e0 : e2) ? gx.amb[0][i] / 255.0f : 1;
    unif_set(u_amb, amb, 1);
    u8 mask = (e0 ? c0->light_mask : 0) | (e2 ? c2->light_mask : 0);
    if (!mask) { set_blit(0, 0); return 0; }
    const Chan *c = e0 ? c0 : c2;
    if (e0 && e2 && (c0->attn_fn != c2->attn_fn || c0->diff_fn != c2->diff_fn)) ONCE("COLOR0/ALPHA0 light functions differ: COLOR0's used\n");
    if (c->attn_fn == 0) ONCE("specular light attenuation approximated\n");
    float cfg[1][4] = {{c->diff_fn ? 1 : 0, c->diff_fn ? 0 : 1, c->diff_fn == 2 ? 0 : -1e30f, 0}};
    unif_set(u_lcfg, cfg, 1);
    float L[15][4];
    memset(L, 0, sizeof L);
    int n = 0;
    for (int l = 0; l < 8; l++) {
        if (!(mask >> l & 1)) continue;
        if (n == 3) { ONCE("more than 3 lights in one draw\n"); break; }
        const Light *s = &gx.light[l];
        float (*d)[4] = &L[n++ * 5];
        memcpy(d[0], s->pos, 12); memcpy(d[1], s->dir, 12);
        if (c->attn_fn == 1) { memcpy(d[2], s->a, 12); memcpy(d[3], s->k, 12); }
        else { d[2][0] = 1; d[3][0] = 1; }
        for (int i = 0; i < 4; i++) {
            int on = i < 3 ? e0 && (c0->light_mask >> l & 1) : e2 && (c2->light_mask >> l & 1);
            d[4][i] = on ? s->color[i] / 255.0f : 0;
        }
    }
    unif_set(u_lights, L, n * 5);
    set_blit(1, n);
    return 1;
}

static int in_dl;   /* inside GXCallDisplayList */

/* a full GPU command buffer is a libctru svcBreak(PANIC). The frame's buffer
 * is one block (a C3D_FrameSplit only hands back the rest of it), so when
 * under 1024 words (~9 draws) remain, the frame's remaining draws are skipped */
static int cmdbuf_room(int prim, int n) {
    u32 *cb_addr, cb_size, cb_off;
    GPUCMD_GetBuffer(&cb_addr, &cb_size, &cb_off);   /* words; C3D_GetCmdBufUsage is only updated per frame */
    if (!kick_off && cb_off >= KICK_WORDS) { gpu_kick(); GPUCMD_GetBuffer(&cb_addr, &cb_size, &cb_off); }   /* offset restarts after a split */
    recomp_gx_dbg[0] = cb_off; recomp_gx_dbg[1] = prim; recomp_gx_dbg[2] = n;
    if (cb_off + 1024 > cb_size) {
        cb_full = 1;
        if (!st_split++) recomp_log("# gx: GPU command buffer full at frame %lu: rest of the frame's draws skipped\n", (unsigned long)frames);
        return 0;
    }
    return 1;
}

/* textures: GX texmaps used by the TEV stages -> PICA units 0..2; then TEV and
 * pixel state. Returns the number of units. */
static int setup_draw_state(int tcoord[3], float tscale[3][2]) {
    int nunits = 0;
    for (int i = 0; i < 8; i++) unit_of_map[i] = -1;
    for (u32 si = 0; si < gx.n_stages; si++) {
        const TevStage *t = &gx.tev[si];
        if (t->map >= 8 || unit_of_map[t->map] >= 0) continue;
        if (nunits >= 3) { ONCE("more than 3 texmaps in one draw\n"); tex_miss("4th texmap in a draw", &gx.map[t->map]); continue; }
        C3D_Tex *tex = tex_get(&gx.map[t->map], &tscale[nunits][0], &tscale[nunits][1]);
        if (!tex) continue;
        static const C3D_Tex *bound[3]; static u32 bound_param[3]; static void *bound_data[3];
        if (bound[nunits] != tex || bound_param[nunits] != tex->param || bound_data[nunits] != tex->data) {
            pend_flush();
            C3D_TexBind(nunits, tex);
            bound[nunits] = tex; bound_param[nunits] = tex->param; bound_data[nunits] = tex->data;
        }
        tcoord[nunits] = t->coord < 8 ? t->coord : -1;
        unit_of_map[t->map] = nunits++;
    }
    setup_tev();
    setup_pixel();
    t_mark(PROF_GX_STATE);   /* the caller's t_start: texture binds, TEV and pixel state */
    return nunits;
}

static int tri_count(int prim, int n) { return prim == 0x80 ? n / 4 * 2 : prim == 0x90 ? n / 3 : n >= 3 ? n - 2 : 0; }

/* triangle list of one primitive whose vertices start at index v0; returns the index count */
static int gen_ix(int prim, int ntri, u32 v0, u16 *ix) {
    int k = 0;
    for (int t = 0; t < ntri; t++) {
        int a, b, c;
        if (prim == 0x80) { int q = (t / 2) * 4; if (t & 1) { a = q; b = q + 2; c = q + 3; } else { a = q; b = q + 1; c = q + 2; } }
        else if (prim == 0x90) { a = 3 * t; b = a + 1; c = a + 2; }
        else if (prim == 0x98) { a = t; b = t + 1; c = t + 2; if (t & 1) { int x = a; a = b; b = x; } }
        else { a = 0; b = t + 1; c = t + 2; }
        ix[k++] = v0 + a; ix[k++] = v0 + b; ix[k++] = v0 + c;
    }
    return k;
}

/* indices at ix (ibuf + ib_used, or a list in vc_ixa: cached) appended to the pending draw or issued */
static int emit_ix(u16 *ix, int k, int cached) {
    int merge = last_end == ix && pend_n + k <= 0xfff0;
    if (merge && pend_n) pend_n += k;
    else { pend_issue(); pend_ix = ix; pend_n = k; }
    if (batch_off) pend_issue();
    last_end = ix + k;
    if (!cached) ib_used += k;
    return merge;
}

static void draw_inner(int prim, int fmt, const u8 *data, int n);
static void draw(int prim, int fmt, const u8 *data, int n) { in_draw++; draw_inner(prim, fmt, data, n); in_draw--; }   /* recomp_first_draw: set when the draw is recorded */
static void draw_inner(int prim, int fmt, const u8 *data, int n) {
    if (prim >= 0xa8) { ONCE("lines/points (0x%x) not drawn\n", prim); return; }
    if (in_dl) st_dl_prims++;
    frame_begin();
    if (!cmdbuf_room(prim, n)) return;
    int tcoord[3] = {-1, -1, -1};
    float tscale[3][2] = {{1, 1}, {1, 1}, {1, 1}};
    t_start();
    int nunits = setup_draw_state(tcoord, tscale);
    int ntri = tri_count(prim, n);
    if (!ntri) return;
    static u32 full_logged;
    if (ib_used + ntri * 3 > IB_INDICES) {
        if (full_logged < 8) { full_logged++; recomp_log("# gx: vertex buffer full at frame %lu (verts %lu + %d, indices %lu)\n",
            (unsigned long)frames, (unsigned long)vb_used, n, (unsigned long)ib_used); }
        return;
    }
    u64 tv = t_last;   /* start of the vertex part (PROF_VX_DL) */
    APlan pl[21];
    int vs, np = make_plan(fmt, pl, &vs);
    s8 tsrc[3];
    t_mark(PROF_GX_VTX);
    set_clip();
    int lit = set_lighting();
    set_mtx(lit);
    set_texgen(tscale, tcoord, tsrc, data, pl, np);
    t_mark2(PROF_GX_VTX, PROF_VX_UNIF);
    int pk = in_dl && !recomp_vtxpack_off && vc_arena && n <= 0x8000;   /* vbuf draws stay PVtx */
    np = trim_plan(pl, np, tsrc, pk && !lit);
    PVtx tm;
    VLay L;
    plan_finish(pl, np, &tm, &L, pk, lit, tex_c2z);
    va_count(pl, np, lit, in_dl ? 1 : 0, n);
    u32 co = in_dl ? vc_draw(data, n, vs, pl, np, &tm, &L) : VC_NONE, v0, vadd = 0;
    if (co != VC_NONE) { u32 a = L.stride / 4, w = co / (a * 0x8000) * (a * 0x8000); use_buf(vc_arena + w * 4, &L); v0 = (co - w) / a; }
    else {
        if (vb_used + n > VB_VERTS) {
            if (full_logged < 8) { full_logged++; recomp_log("# gx: vertex buffer full at frame %lu (verts %lu + %d)\n", (unsigned long)frames, (unsigned long)vb_used, n); }
            return;
        }
        if (pk) {   /* no room in the vertex cache: drawn from vbuf as PVtx */
            np = make_plan(fmt, pl, &vs);
            np = trim_plan(pl, np, tsrc, 0);
            plan_finish(pl, np, &tm, &L, 0, lit, 0);
        }
        if (vb_used + n - vb_base > 0x10000) vb_base = vb_used;
        use_buf(vbuf[vb_half] + vb_base, &L);
        decode_run(data, n, vs, pl, np, &tm, &L, (u8 *)(vbuf[vb_half] + vb_used));
        v0 = vb_used - vb_base; vadd = n;
    }
    set_vtx_unif(&L);
    t_mark2(PROF_GX_VTX, PROF_VX_DEC);
    /* a cached entry's index list is kept in vc_ixa and drawn from there */
    u16 *ix;
    int k = 0, own = 1;
    VCache *ve = co != VC_NONE ? vc_last : NULL;
    if (ve && ve->ixn && ve->prim == (u32)prim) { ix = vc_ixa + ve->ixoff; k = ve->ixn; ring_used(&iring, ve->irec); }
    else {
        u32 io = VC_NONE;
        if (ve) { ve->ixn = 0; ring_detach(&iring, &ve->irec); io = ring_alloc(&iring, ntri * 3, 1, own_vc(ve), &ve->irec); }
        if (io != VC_NONE) { ix = vc_ixa + io; ve->ixoff = io; ve->prim = prim; ve->ixn = k = gen_ix(prim, ntri, v0, ix); }
        else { ix = ibuf[vb_half] + ib_used; k = gen_ix(prim, ntri, v0, ix); own = 0; }
    }
    ntri = k / 3;
    st_draws++; st_tris += ntri; if (nunits) st_tex++;
    if (emit_ix(ix, k, own)) st_merged++;
    vb_used += vadd;
    t_mark(PROF_GX_VTX);
    if (in_dl) recomp_prof[PROF_VX_DL] += t_last - tv;
}

/* ---------------------------------------------------------------- display-list cache
 * parse_cmds does not emulate CP/XF/BP writes, so all primitives of a display
 * list without nested calls share the GX state of the GXCallDisplayList. A
 * whole list is set up once (textures, TEV, pixel, uniforms, vertex plan),
 * keyed by (list address, size, plan signature); its primitives are decoded
 * into the vertex cache arena once and their merged triangle list kept in
 * vc_ixa. A hit draws that list in place as one draw. Valid while the entry
 * keeps its allocations (ring_alloc), the list bytes and the indexed arrays
 * (checked once per frame, as in the vertex cache) match. Lists with nested calls, other commands, mixed vertex
 * formats, lines/points or per-vertex texture matrices take the per-primitive
 * path (entry marked slow). sdmc:/cr_dlcache_off.txt disables it. */
static int dl_off;   /* read in recomp_gx_start, like dl_clean_off */
static u32 dl_logged, dl_logframe = 0xffffffffu;

static int dl_clean_off;   /* sdmc:/cr_dlclean_off.txt = always hash */
static u32 dl_arrays_hash(DlCache *e, int keep) {
    u32 h = 0;
    for (u32 i = 0; i < e->narr; i++) { u32 a = arr_hash(e->arr[i], e->alen[i]); if (keep) e->ah[i] = a; h = (h ^ a) * 0x9e3779b1u; }
    return h;
}

/* No guest write since the entry was last hashed: every 1 KB page of the list and of each array
 * has an epoch older than e->vep (a write sets the epoch to the current one, also in the epoch of
 * the hash itself, so it is only ever re-hashed too often). A 4-byte store can straddle into the
 * next page, hence one page before each range. Ranges near the stack are always hashed (stores
 * through r1/r11 do not mark pages; r1 as recorded with the draw). */
static int range_clean(const u8 *p, u32 n, u32 vep) {
    u32 a = (u32)(p - ram) & RAM_MASK;
    u32 sp = cur_r1 & RAM_MASK;
    if (a + n > sp - 0x10000 && a < sp + 0x10000) return 0;
    u32 lo = (a >> WR_PAGE_SHIFT) - (a >= (1u << WR_PAGE_SHIFT)), hi = (a + n - 1) >> WR_PAGE_SHIFT;
    /* a store's mark (ppc.h WR_SET) = written in the current epoch; the sweep stamps ram_wr[] before it
     * clears a mark, so marks are read first */
    for (u32 k = lo; k <= hi; k++) if (ram_dirty[k]) return 0;
    __dmb();
    for (u32 k = lo; k <= hi; k++) if (ram_wr[k] >= vep) return 0;
    return 1;
}

static int dl_clean(const DlCache *e, const u8 *list, u32 len) {
    if (!e->vep || dl_clean_off) return 0;
    if (!range_clean(list, len, e->vep)) return 0;
    for (u32 i = 0; i < e->narr; i++) if (!range_clean(e->arr[i], e->alen[i], e->vep)) return 0;
    return 1;
}

static void dl_log_miss(const u8 *list, u32 n, const char *why, DlCache *e) {
    if (frames % 300) return;   /* stats frames only */
    if (dl_logframe != frames) { dl_logframe = frames; dl_logged = 0; }
    if (dl_logged++ >= 32) return;
    char a[96] = "";
    if (e && !strcmp(why, "arrays"))
        for (u32 i = 0, p = 0; i < e->narr && p < sizeof a - 20; i++)
            if (arr_hash(e->arr[i], e->alen[i]) != e->ah[i]) p += snprintf(a + p, sizeof a - p, " %08lx+%lx", (unsigned long)(e->arr[i] - ram + 0x80000000u), (unsigned long)e->alen[i]);
    recomp_log("# gx dl miss frame %lu: list %08lx size %lu %s%s\n", (unsigned long)frames, (unsigned long)(list - ram + 0x80000000u), (unsigned long)n, why, a);
}

/* 1 = drawn (or skipped) here; 0 = the caller parses the list per primitive */
static int dl_draw(const u8 *list, u32 len) {
    if (dl_off || !vc_arena || len > 0x40000) return 0;   /* (no vc_arena: no vc_ixa either) */
    u32 p0 = 0;
    while (p0 < len && !list[p0]) p0++;
    if (len - p0 < 3) return 0;
    u8 op0 = list[p0];
    if (op0 < 0x80 || op0 >= 0xa8) return 0;
    int fmt = op0 & 7;
    frame_begin();
    if (!cmdbuf_room(op0 & 0xf8, 0)) return 1;
    int tcoord[3] = {-1, -1, -1};
    float tscale[3][2] = {{1, 1}, {1, 1}, {1, 1}};
    t_start();
    int nunits = setup_draw_state(tcoord, tscale);
    for (int u = 0; u < 3; u++) if (tcoord[u] >= 0 && tcoord[u] < 8 && gx.desc[1 + tcoord[u]]) return 0;   /* per-vertex texture matrix */
    u64 tp = t_last;   /* start of the vertex part (PROF_VX_DL) */
    s8 tsrc[3];
    set_clip();
    int lit = set_lighting();
    set_mtx(lit);
    set_texgen(tscale, tcoord, tsrc, NULL, NULL, 0);   /* no per-vertex texture matrix here: the plan is not read */
    t_mark2(PROF_GX_VTX, PROF_VX_UNIF);
    APlan pl[21];
    PVtx tm;
    VLay L;
    int vs, np, pk = !recomp_vtxpack_off;
    u32 sig;
    const PlanMemo *pm = plan_get(fmt, tsrc, pk, lit, tex_c2z);
    if (pm) { vs = pm->vs; np = pm->np; sig = pm->sig; L = pm->lay; }   /* pl and tm are copied on a miss (decode) */
    else {
        np = make_plan(fmt, pl, &vs);
        np = trim_plan(pl, np, tsrc, pk && !lit);
        plan_finish(pl, np, &tm, &L, pk, lit, tex_c2z);
        sig = plan_sig(pl, np, vs, &tm, &L) ^ ((u32)fmt << 28 | (u32)mat_vtx[0] | (u32)mat_vtx[3] << 1) * 0x85ebca6bu;
    }
    u32 slot = (gofs(list) * 0x9e3779b1u ^ sig ^ len * 0x85ebca6bu) >> 19 & (DL_N - 1);
    DlCache *set = &dlcache[slot & ~(CACHE_WAYS - 1u)], *e = NULL, *v = set;
    for (int i = 0; i < CACHE_WAYS; i++) {
        DlCache *c = set + i;
        if (c->gen && c->list == list && c->n == len && c->sig == sig) { e = c; break; }
        if (v->gen && (!c->gen || c->used < v->used)) v = c;
    }
    if (!e) e = v;
    int same = e->gen && e->list == list && e->n == len && e->sig == sig;
    const char *why = same ? NULL : "new";
    if (same && e->checked != frames + 1) {
        u32 ep = epoch_now();
        if (dl_clean(e, list, len)) e->checked = frames + 1;
        else if (hash_mem(list, len, 0x811c9dc5u) != e->lhash) why = "list bytes";
        else if (!e->slow && dl_arrays_hash(e, 0) != e->ahash) why = "arrays";
        else { e->checked = frames + 1; e->vep = ep; }
    }
    u32 w;
    if (!why) {   /* hit */
        e->used = frames;
        if (e->slow) { t_mark(PROF_GX_VTX); st_dl_slow++; return 0; }
        st_dl_hits++;
        if (!e->ixn) { t_mark(PROF_GX_VTX); return 1; }
        ring_used(&vring, e->vrec); ring_used(&iring, e->irec);
    } else {
        st_dl_misses++;
        dl_log_miss(list, len, why, same ? e : NULL);
        e->gen = 0;
        ring_detach(&vring, &e->vrec); ring_detach(&iring, &e->irec);
        if (pm) { memcpy(pl, pm->pl, np * sizeof *pl); tm = pm->tm; }   /* decode updates imax: a copy (memoised at 0) */
        u32 ep = epoch_now();   /* before the list and arrays are read for the entry */
        /* walk: one vertex format, triangle primitives and NOPs only */
        u32 pos = p0, nv = 0, nix = 0, slow = 0;
        while (pos < len) {
            u8 op = list[pos];
            if (!op) { pos++; continue; }
            if (op < 0x80 || op >= 0xa8 || (op & 7) != fmt || len - pos < 3) { slow = 1; break; }
            u32 n = list[pos + 1] << 8 | list[pos + 2];
            if (len - pos < 3 + n * vs) { slow = 1; break; }
            nv += n; nix += tri_count(op & 0xf8, n) * 3;
            pos += 3 + n * vs;
        }
        if (nv > 0x8000) slow = 1;
        e->list = list; e->n = len; e->sig = sig; e->used = frames; e->checked = frames + 1;
        e->lhash = hash_mem(list, len, 0x811c9dc5u); e->vep = ep;
        if (slow) { e->slow = 1; e->gen = 1; t_mark(PROF_GX_VTX); st_dl_slow++; return 0; }
        e->slow = 0; e->ixn = 0;
        if (!nix) { e->narr = 0; e->gen = 1; t_mark(PROF_GX_VTX); return 1; }   /* no triangles: nothing to draw */
        u32 a = L.stride / 4, off = ring_alloc(&vring, nv * a, a, own_dl(e), &e->vrec), io = VC_NONE;
        if (off != VC_NONE) io = ring_alloc(&iring, nix, 1, own_dl(e), &e->irec);
        if (io == VC_NONE) {   /* no room this frame: per primitive (vc_draw, or vbuf) */
            if (e->vrec) { vring.rec[e->vrec - 1].last = frames - 1; ring_detach(&vring, &e->vrec); }   /* not drawn: free to go */
            t_mark(PROF_GX_VTX); return 0;
        }
        w = off / (a * 0x8000) * (a * 0x8000);
        t_mark(PROF_GX_VTX);
        u8 *o = vc_arena + off * 4;
        u16 *ix = vc_ixa + io;
        u32 v = 0, k = 0, prims = 0;
        for (pos = p0; pos < len;) {
            u8 op = list[pos];
            if (!op) { pos++; continue; }
            u32 n = list[pos + 1] << 8 | list[pos + 2];
            const u8 *d = list + pos + 3;
            decode_run(d, n, vs, pl, np, &tm, &L, o + v * L.stride);
            int nt = tri_count(op & 0xf8, n);
            if (nt) { k += gen_ix(op & 0xf8, nt, (off - w) / a + v, ix + k); prims++; }
            v += n;
            pos += 3 + n * vs;
        }
        t_mark2(PROF_GX_VTX, PROF_VX_DEC);
        u32 narr = 0;
        for (int i = 0; i < np; i++) if (pl[i].d >= 2 && pl[i].kind != K_PNMTX) {
            u32 al = pl[i].imax * pl[i].stride + pl[i].size, j;
            for (j = 0; j < narr && e->arr[j] != pl[i].base; j++) ;
            if (j < narr) { if (al > e->alen[j]) e->alen[j] = al; continue; }
            if (narr == DL_ARR) { narr = DL_ARR + 1; break; }
            e->arr[narr] = pl[i].base; e->alen[narr++] = al;
        }
        e->ixoff = io; e->ixn = k; e->nv = nv; e->prims = prims; e->tris = k / 3; e->off = off;
        if (narr <= DL_ARR) { e->narr = narr; e->ahash = dl_arrays_hash(e, 1); e->gen = 1; }   /* else drawn once, not kept: */
        else { ring_detach(&vring, &e->vrec); ring_detach(&iring, &e->irec); }                   /* the space is reused after this frame */
    }
    va_count(pm ? pm->pl : pl, np, lit, 2, e->nv);
    u32 a = L.stride / 4;
    w = e->off / (a * 0x8000) * (a * 0x8000);
    use_buf(vc_arena + w * 4, &L);
    set_vtx_unif(&L);
    u32 prims = e->prims, tris = e->tris;
    st_dl_prims += prims; st_draws += prims; st_tris += tris; if (nunits) st_tex += prims;
    st_merged += prims - 1 + emit_ix(vc_ixa + e->ixoff, e->ixn, 1);
    t_mark(PROF_GX_VTX);
    recomp_prof[PROF_VX_DL] += t_last - tp;
    return 1;
}

/* ---------------------------------------------------------------- framebuffer dump
 * sdmc:/cr_fbdump.txt = GX frame numbers (copies), one per line: after that
 * frame the render target is written to sdmc:/cr_fb_<frame>.bmp (240x400
 * as stored, i.e. rotated), for deterministic before/after image compares. */
static u32 want[64], nwant;   /* read in recomp_gx_start */
static volatile int fb_next;  /* recomp_gx_fb_dump_next (memory dump): the next frame too */
void recomp_gx_fb_dump_next(void) { fb_next = 1; }
static void fb_dump_one(C3D_RenderTarget *t, const char *suffix);
static void fb_dump(void) {
    u32 i;
    for (i = 0; i < nwant && want[i] != frames; i++) ;
    if (i == nwant && !fb_next) return;
    fb_next = 0;
    fb_dump_one(target, "");
    if (st_r_done) fb_dump_one(target_r, "_r");   /* right eye of a stereo frame */
}
static void fb_dump_one(C3D_RenderTarget *t, const char *suffix) {
    static u8 *buf;
    if (!buf && !(buf = linearAlloc(240 * 400 * 3))) return;
    C3D_SyncDisplayTransfer((u32 *)t->frameBuf.colorBuf, GX_BUFFER_DIM(240, 400), (u32 *)buf, GX_BUFFER_DIM(240, 400),
        GX_TRANSFER_FLIP_VERT(0) | GX_TRANSFER_OUT_TILED(0) | GX_TRANSFER_RAW_COPY(0) |
        GX_TRANSFER_IN_FORMAT(GX_TRANSFER_FMT_RGBA8) | GX_TRANSFER_OUT_FORMAT(GX_TRANSFER_FMT_RGB8) | GX_TRANSFER_SCALING(GX_TRANSFER_SCALE_NO));
    GSPGPU_InvalidateDataCache(buf, 240 * 400 * 3);
    char name[64];
    snprintf(name, sizeof name, "sdmc:/cr_fb_%lu%s.bmp", (unsigned long)frames, suffix);
    FILE *f = fopen(name, "wb");
    if (!f) return;
    u8 h[54] = {'B', 'M'};
    u32 sz = 54 + 240 * 400 * 3, w = 240, hh = 400;
    memcpy(h + 2, &sz, 4); h[10] = 54; h[14] = 40; memcpy(h + 18, &w, 4); memcpy(h + 22, &hh, 4); h[26] = 1; h[28] = 24;
    fwrite(h, 1, 54, f); fwrite(buf, 1, 240 * 400 * 3, f); fclose(f);
}

/* ---------------------------------------------------------------- GX thread
 * docs/recomp_unknowns.md "GX thread". Every GX call that changes GX state or draws
 * is a record in gq: its handler (x_*) and the handler's inputs, either r3.. (GQ_REGS:
 * the handler reads r[3]..) or a copy of what the call reads from guest memory. With
 * core 2 the GX thread runs the handlers, in order, on its own gx state; otherwise (or
 * sdmc:/cr_gxthread_off.txt) gq_end runs each record at once on the game thread.
 * One producer (game thread), one consumer; indices are free-running word counts.
 * Records are published at draws, frame ends and waits, not per setter. */
typedef void (*GqFn)(const u32 *);
#define GQ_WORDS (1u << 18)       /* 1 MB; the largest record (a full FIFO) is 64 K words */
#define GQ_REGS 0x80000000u       /* header flag: the payload is r3.. */
#define GQ_SPIN 200000            /* GX thread: ~1 ms of polling for work before it sleeps */
#define GQ_SPIN_GAME 20000        /* game thread: ~0.1 ms of polling before it sleeps in a wait */
static u32 gq[GQ_WORDS];
static volatile u32 gq_w, gq_r;   /* published end; end of the records the GX thread has finished */
static u32 gq_wl, gq_rc, gq_cur, gq_copy_prev;   /* game thread: end written, gq_r last read, open record size, end of the previous GXCopyDisp */
static volatile u32 gq_sleeping, gq_waiting, gq_wait_end, gq_quit;
static LightEvent gq_ev, gq_done_ev;   /* work for a sleeping GX thread; the game thread's wait is over */
int recomp_gx_thread;

static void gq_run(const u32 *rec) {
    GqFn fn = (GqFn)(uintptr_t)rec[1];
    if (rec[0] & GQ_REGS) { u32 r[11]; memcpy(r + 3, rec + 2, ((rec[0] & 0xffffff) - 2) * 4); fn(r); }
    else fn(rec + 2);
}

static void gq_publish(void) {
    if (!recomp_gx_thread || gq_w == gq_wl) return;
    __dmb();   /* the records, and the guest writes before them, before the index */
    gq_w = gq_wl;
    __dmb();
    if (gq_sleeping) LightEvent_Signal(&gq_ev);
}

/* game thread: until the GX thread has finished every record before end */
static void gq_wait(u32 end) {
    if ((s32)(gq_r - end) >= 0) return;
    gq_publish();
    u64 t0 = recomp_ticks();
    for (u32 i = 0; i < GQ_SPIN_GAME && (s32)(gq_r - end) < 0; i++) ;
    while ((s32)(gq_r - end) < 0) {
        gq_wait_end = end; gq_waiting = 1;
        __dmb();
        if ((s32)(gq_r - end) < 0) LightEvent_WaitTimeout(&gq_done_ev, 1000000);
    }
    gq_waiting = 0;
    __dmb();   /* the GX thread's work before gq_r, before what the game thread does next */
    recomp_prof[PROF_GX_WAIT] += recomp_ticks() - t0;
}

/* a record of n payload words: returns the payload (inline mode: gq is scratch) */
static u32 *gq_begin(GqFn fn, u32 n, u32 flags) {
    u32 need = n + 2;
    st_gq_recs++; st_gq_words += need;
    if (!recomp_gx_thread) { gq[0] = need | flags; gq[1] = (u32)(uintptr_t)fn; return gq + 2; }
    u32 pos = gq_wl & (GQ_WORDS - 1), pad = pos + need > GQ_WORDS ? GQ_WORDS - pos : 0;
    if (gq_wl + pad + need - gq_rc > GQ_WORDS) {   /* full as last seen: look again, wait while it still is */
        gq_rc = gq_r;
        if (gq_wl + pad + need - gq_rc > GQ_WORDS) { gq_wait(gq_wl + pad + need - GQ_WORDS); gq_rc = gq_r; }
        __dmb();   /* the GX thread is done reading what gets overwritten */
    }
    if (pad) { gq[pos] = 0; gq_wl += pad; pos = 0; }   /* header 0: the rest of the ring is padding */
    gq[pos] = need | flags; gq[pos + 1] = (u32)(uintptr_t)fn;
    gq_cur = need;
    return gq + pos + 2;
}
static void gq_end(void) {
    if (!recomp_gx_thread) { gq_run(gq); return; }
    gq_wl += gq_cur;
}
static void gq_regs(GqFn fn, u32 n) { u32 *p = gq_begin(fn, n, GQ_REGS); memcpy(p, &R[3], n * 4); gq_end(); }

/* core 2, priority below the helper threads there (they preempt its polling) */
static void gx_thread(void *arg) {
    (void)arg;
    recomp_thread_crash_log();
    recomp_log("# fpscr: gx thread %08lx\n", recomp_fpscr());
    u32 r = 0;
    while (!gq_quit) {
        u32 w = gq_w;
        if (w == r) {
            for (u32 i = 0; i < GQ_SPIN && gq_w == r && !gq_quit; i++) ;
            if (gq_w != r) continue;
            gq_sleeping = 1;
            __dmb();
            if (gq_w == r && !gq_quit) LightEvent_WaitTimeout(&gq_ev, 2000000);
            gq_sleeping = 0;
            continue;
        }
        __dmb();   /* the records up to w are visible */
        u64 t0 = recomp_ticks();
        while (r != w) {
            u32 pos = r & (GQ_WORDS - 1), h = gq[pos];
            if (!h) r += GQ_WORDS - pos;
            else { gq_run(gq + pos); r += h & 0xffffff; }
            __dmb();   /* done with the record before the game thread may reuse its words */
            gq_r = r;
            __dmb();
            if (gq_waiting && (s32)(r - gq_wait_end) >= 0) LightEvent_Signal(&gq_done_ev);
        }
        recomp_prof[PROF_GX_BUSY] += recomp_ticks() - t0;
    }
}

/* game thread: every record so far handled (GXDrawDone, before aptMainLoop, at exit) */
void recomp_gx_sync(void) { if (recomp_gx_thread) gq_wait(gq_wl); }

/* A bottom-screen tap (runtime/pad.c) switches between the GX thread and inline. It is
 * applied at the next retrace, after the sync: every record so far has been handled, so
 * the other thread carries on with the same gx and citro3d state. */
static int gq_ok;              /* the GX thread exists (core 2) */
static volatile int gq_req;    /* a tap since the last retrace */
static int gq_fliptest;        /* sdmc:/cr_gxthread_fliptest.txt: a switch every 300 retraces (tests the switch) */
u32 recomp_gx_thread_flips;    /* switches so far */
void recomp_gx_thread_toggle(void) { gq_req = 1; }

int recomp_gx_thread_ok(void) { return gq_ok; }

void recomp_gx_retrace(void) {
    recomp_gx_sync();
    if (gq_fliptest && recomp_retrace_count() % 300 == 150) gq_req = 2;
    if (gq_req) {
        const char *why = gq_req == 2 ? "cr_gxthread_fliptest.txt" : "bottom-screen tap";
        gq_req = 0;
        if (!gq_ok) recomp_log("# gx r%lu: GX thread toggle ignored (no core 2)\n", (unsigned long)recomp_retrace_count());
        else {
            recomp_gx_thread ^= 1;
            recomp_gx_thread_flips++;
            recomp_log("# gx r%lu: GX thread %s (%s)\n", (unsigned long)recomp_retrace_count(), recomp_gx_thread ? "on" : "off", why);
        }
    }
    recomp_menu_retrace();
    recomp_memdump_retrace();
}

void recomp_gx_stop(void) {
    if (!gq_ok) return;
    recomp_gx_sync();
    gq_quit = 1;
    __dmb();
    LightEvent_Signal(&gq_ev);
}

/* libctru's linear heap has no lock, and both threads allocate (ax.c's ndsp buffers on the
 * game thread, citro3d textures on the GX thread): linked with --wrap (Makefile) */
void *__real_linearAlloc(size_t size);
void *__real_linearMemAlign(size_t size, size_t alignment);
void __real_linearFree(void *mem);
/* lin_lock (declared above frame_end): zero = unlocked (libctru reads a 0 LightLock as unlocked) */
void *__wrap_linearAlloc(size_t size) { RecursiveLock_Lock(&lin_lock); void *p = __real_linearAlloc(size); RecursiveLock_Unlock(&lin_lock); return p; }
void *__wrap_linearMemAlign(size_t size, size_t alignment) {
    RecursiveLock_Lock(&lin_lock); void *p = __real_linearMemAlign(size, alignment); RecursiveLock_Unlock(&lin_lock); return p;
}
void __wrap_linearFree(void *mem) { RecursiveLock_Lock(&lin_lock); __real_linearFree(mem); RecursiveLock_Unlock(&lin_lock); }

/* sdmc flag files, read once before the game starts (a lazy fopen stalled 4 s at r13 in
 * hw runs 17b/19b); then the GX thread, when there is a core 2 */
static int flag_file(const char *name) {
    char p[48]; snprintf(p, sizeof p, "sdmc:/%s.txt", name);
    FILE *f = fopen(p, "rb");
    if (f) fclose(f);
    return f != NULL;
}
void recomp_gx_start(void) {
    vsync_on = flag_file("cr_vsync_on");
    kick_off = flag_file("cr_gpukick_off");
    batch_off = flag_file("cr_batch_off");
    dl_off = flag_file("cr_dlcache_off");
    dl_clean_off = flag_file("cr_dlclean_off");
    vramtex_off = flag_file("cr_vramtex_off");   /* sdmc:/cr_vramtex_off.txt: textures stay in linear memory */
    recomp_vt_on = !vramtex_off;
    if (flag_file("cr_vcache_small")) {   /* test: rings at 5/8 so that replays evict (ring_alloc) */
        vring.size = vring.size / 8 * 5; iring.size = VC_IXN / 8 * 5;
        recomp_log("# gx: vertex cache rings at 5/8 (cr_vcache_small.txt): %lu KB of vertices, %lu indices\n", (unsigned long)(vring.size * 4 >> 10), (unsigned long)iring.size);
    }
    recomp_vtxpack_off = flag_file("cr_vtxpack_off");
    recomp_menu_bind("Vertex pack", &recomp_vtxpack_off, NULL);
    recomp_lazydesc_off = flag_file("cr_lazydesc_off");
    recomp_menu_bind("Lazy vtx desc", &recomp_lazydesc_off, NULL);
    recomp_log("# gx: vertex descriptor %s\n", recomp_lazydesc_off ? "eager (cr_lazydesc_off.txt): a GX-queue record per GXSetVtxDesc/GXClearVtxDesc" : "lazy (one record before a draw, when it changed)");
    recomp_log("# gx: vertex packing %s\n", recomp_vtxpack_off ? "off (cr_vtxpack_off.txt): every vertex 68 bytes" : "on (display-list draws: per-draw layout)");
    FILE *sf = fopen("sdmc:/cr_stereo_force.txt", "rb");
    if (sf) {
        float v[3] = {1, st_sep, st_conv};
        int n = fscanf(sf, "%f %f %f", &v[0], &v[1], &v[2]);
        fclose(sf);
        st_force = v[0]; st_sep = v[1]; st_conv = v[2];
        recomp_log("# gx: stereo forced (cr_stereo_force.txt, %d values): slider %.2f sep %.3f conv %.1f\n", n < 0 ? 0 : n, st_force, st_sep, st_conv);
    }
    FILE *f = fopen("sdmc:/cr_fbdump.txt", "rb");
    int fbd = f != NULL;
    if (f) { unsigned long v; while (nwant < 64 && fscanf(f, "%lu", &v) == 1) want[nwant++] = v; fclose(f); }
    recomp_log("# gx: frame pacing %s\n", vsync_on ? "vblank wait every frame (cr_vsync_on.txt)" : "60 Hz cap only");
    if (kick_off) recomp_log("# gx: mid-frame GPU submit off (cr_gpukick_off.txt)\n");
    else recomp_log("# gx: mid-frame GPU submit on (every %d command words, at most %d per frame)\n", KICK_WORDS, KICK_MAX);
    recomp_gx_kick_on = !kick_off;   /* frame_begin keeps it; set here for the status line */
    recomp_log("# gx: draw batching %s\n", batch_off ? "off (cr_batch_off.txt)" : "on");
    recomp_log("# gx: display-list cache %s\n", dl_off ? "off (cr_dlcache_off.txt)" : "on");
    recomp_log("# gx: display-list write tracking %s\n", dl_clean_off ? "off (cr_dlclean_off.txt)" : "on");
    recomp_log("# gx: VRAM textures %s (VRAM free %lu KB after the render target, reserve %u KB, at most %d moves per frame)\n",
               vramtex_off ? "off (cr_vramtex_off.txt): linear only" : "on", (unsigned long)(vramSpaceFree() >> 10), VT_RESERVE >> 10, VT_COPIES);
    if (fbd) recomp_log("# gx: framebuffer dump at %lu frames\n", (unsigned long)nwant);
    int off = flag_file("cr_gxthread_off");   /* only the starting mode: the thread exists either way */
    gq_fliptest = flag_file("cr_gxthread_fliptest");
    if (gq_fliptest) recomp_log("# gx: GX thread switch test: on/off every 300 retraces (cr_gxthread_fliptest.txt)\n");
    if (recomp_aux_core() == 2) {
        LightEvent_Init(&gq_ev, RESET_ONESHOT);
        LightEvent_Init(&gq_done_ev, RESET_ONESHOT);
        s32 prio = 0x30;
        svcGetThreadPriority(&prio, CUR_THREAD_HANDLE);
        gq_ok = recomp_aux_thread(gx_thread, 0x10000, prio + 2) != NULL;
    }
    recomp_gx_thread = gq_ok && !off;
    recomp_log("# gx: GX thread %s\n", !gq_ok ? "off (no core 2): GX runs on the game thread"
               : recomp_gx_thread ? "on core 2 (GX state and draws handled off the game thread; its bottom-screen button toggles it)"
               : "on core 2, starting off (cr_gxthread_off.txt): GX runs on the game thread until its bottom-screen button is tapped");
}

/* ---------------------------------------------------------------- FIFO
 * The write-gather pipe is assembled on the game thread (sdk_GXBegin pushes the draw
 * command, then the game writes the vertices to 0xCC008000 itself). Each run of
 * complete commands becomes one record (x_fifo: r1, byte count, bytes), which the GX
 * thread parses and draws (parse_cmds). */
u8 recomp_fifo[0x40000];
u32 recomp_fifo_n, recomp_fifo_lim;   /* ppc.h gx_fifo_put: the inline WGPIPE path */
#define fifo recomp_fifo
#define fifo_n recomp_fifo_n
static u32 fifo_need;   /* bytes of the primitive GXBegin started (header + n * vtx_size), 0 = unknown */
/* gx_fifo_put appends inline while fifo_n + size < recomp_fifo_lim: the primitive GXBegin started is
 * still incomplete after the write (no flush due) and the buffer has room; 0 = every write comes here */
static void fifo_lim(void) { recomp_fifo_lim = fifo_need < sizeof fifo ? fifo_need : sizeof fifo; }

/* size of the command at b (avail bytes there; fewer than the result = not complete yet),
 * 0 = unknown opcode. vsize: the vertex size from the calling thread's vertex format */
static u32 cmd_size(const u8 *b, u32 avail, int (*vsize)(int)) {
    u8 op = b[0];
    if (op >= 0x80) return avail < 3 ? 3 : 3 + (u32)(b[1] << 8 | b[2]) * vsize(op & 7);
    switch (op) {
    case 0x00: case 0x48: return 1;
    case 0x08: return 6;
    case 0x10: return avail < 5 ? 5 : 5 + ((b[1] << 8 | b[2]) + 1) * 4;
    case 0x20: case 0x28: case 0x30: case 0x38: case 0x61: return 5;
    case 0x40: return 9;
    default: return 0;
    }
}

/* GX command stream: draws are decoded; CP/XF/BP register writes are not
 * emulated (the GX calls above keep the state) and are logged once. Returns
 * the bytes consumed; an incomplete command at the end is left. */
static void call_display_list(u32 addr, u32 n);

static u32 parse_cmds(const u8 *buf, u32 len) {
    u32 pos = 0;
    while (pos < len) {
        u8 op = buf[pos];
        u32 need = cmd_size(buf + pos, len - pos, vtx_size);
        if (!need) { ONCE("unknown GX command %02x, rest of stream dropped\n", op); return len; }
        if (len - pos < need) break;
        if (op >= 0x80) draw(op & 0xf8, op & 7, buf + pos + 3, buf[pos + 1] << 8 | buf[pos + 2]);
        else if (op == 0x40) {
            const u8 *q = buf + pos + 1;
            call_display_list((u32)q[0] << 24 | q[1] << 16 | q[2] << 8 | q[3], (u32)q[4] << 24 | q[5] << 16 | q[6] << 8 | q[7]);
        } else if (op && op != 0x48) ONCE("raw GX command %02x not emulated\n", op);
        pos += need;
    }
    return pos;
}

static void call_display_list(u32 addr, u32 n) {
    static int depth;
    if (depth > 2 || n > 0x1000000) { ONCE("display list nesting/size\n"); return; }
    const u8 *list = ram + (GUEST(addr) & RAM_MASK);
    if (!depth) {
        st_dl_calls++;
        in_draw++;
        int done = dl_draw(list, n);
        in_draw--;
        if (done) return;
    }
    depth++; in_dl++;
    u32 used = parse_cmds(list, n);
    if (used < n) ONCE("display list with a truncated command\n");
    depth--; in_dl--;
}

static void x_call_dl(const u32 *p) { cur_r1 = p[2]; call_display_list(p[0], p[1]); }
void sdk_GXCallDisplayList(void) {
    LOG();
    if (!recomp_first_draw) recomp_first_draw = recomp_ticks();
    if (desc_dirty) desc_sync();
    u32 *p = gq_begin(x_call_dl, 3, 0);
    p[0] = R[3]; p[1] = R[4]; p[2] = R[1];
    gq_end();
    gq_publish();
}

/* game thread: bytes of the complete commands at the front of the FIFO, as parse_cmds
 * will consume them; *drw: a primitive or display-list call among them */
static u32 fifo_complete(int *drw) {
    u32 pos = 0;
    while (pos < fifo_n) {
        u32 need = cmd_size(fifo + pos, fifo_n - pos, vtx_size_g);
        if (!need) return fifo_n;   /* parse_cmds logs it and drops the rest */
        if (fifo_n - pos < need) break;
        if (fifo[pos] >= 0x80 || fifo[pos] == 0x40) *drw = 1;
        pos += need;
    }
    return pos;
}

static void x_fifo(const u32 *p) { cur_r1 = p[0]; parse_cmds((const u8 *)(p + 2), p[1]); }
static void fifo_flush(void) {
    int drw = 0;
    u32 pos = fifo_complete(&drw);
    if (pos) {
        if (drw && !recomp_first_draw) recomp_first_draw = recomp_ticks();
        if (desc_dirty) desc_sync();
        u32 *p = gq_begin(x_fifo, 2 + (pos + 3) / 4, 0);
        p[0] = R[1]; p[1] = pos;
        memcpy(p + 2, fifo, pos);
        gq_end();
        if (drw) gq_publish();
    }
    memmove(fifo, fifo + pos, fifo_n - pos);
    fifo_n -= pos;
    fifo_need = 0;
    fifo_lim();
}

static void fifo_put(u32 v, int size) {
    if (fifo_n + size > sizeof fifo) { ONCE_G("FIFO overflow\n"); fifo_n = 0; }
    for (int i = size - 1; i >= 0; i--) fifo[fifo_n++] = (u8)(v >> (8 * i));
}

/* write-gather pipe (hw_write 0xCC008000) */
void recomp_gx_fifo_write(u32 v, int size) {
    fifo_put(v, size);
    if (fifo_need) {
        if (fifo_n >= fifo_need) fifo_flush();
    } else if (fifo_n >= 3 && fifo[0] >= 0x80) {
        u32 n = fifo[1] << 8 | fifo[2];
        if (fifo_n >= 3 + n * vtx_size_g(fifo[0] & 7)) fifo_flush();
    } else if (fifo_n && fifo[0] < 0x80) fifo_flush();
    fifo_lim();
}

/* ---------------------------------------------------------------- draw-done
 * The GPU reads display lists, vertex arrays and textures after the call that names
 * them, and GXDrawDone is where the game may reuse them: it waits for the GX thread. */
static u32 draw_done_cb;
static int draw_done_pending;

static void deliver_draw_done(void) {
    if (!draw_done_pending) return;
    draw_done_pending = 0;
    if (draw_done_cb) recomp_call_guest(draw_done_cb, 0);
}

void sdk_GXSetDrawDoneCallback(void) { LOG(); u32 old = draw_done_cb; draw_done_cb = R[3]; R[3] = old; }
void sdk_GXWaitDrawDone(void) { LOG(); recomp_gx_sync(); deliver_draw_done(); }
void sdk_GXDrawDone(void) { LOG(); draw_done_pending = 1; recomp_gx_sync(); deliver_draw_done(); }

/* ---------------------------------------------------------------- vertex format
 * desc/vat are kept on both threads: gs (game, FIFO assembly) and gx */
void sdk_GXBegin(void) {
    LOG();
    fifo_need = 0;
    if (fifo_n) { ONCE_G("GXBegin with %lu bytes of the previous primitive missing\n", (unsigned long)fifo_n); fifo_n = 0; }
    fifo_put((R[3] & 0xf8) | (R[4] & 7), 1);
    fifo_put(R[5] & 0xffff, 2);
    int vs = vtx_size_g(R[4] & 7);
    if ((R[5] & 0xffff) == 0 || vs == 0) fifo_n = 0;
    else fifo_need = 3 + (R[5] & 0xffff) * vs;
    fifo_lim();
}
/* Lazy vertex descriptor (sdmc:/cr_lazydesc_off.txt = a record per call; bottom-screen button
 * "Lazy vtx desc"). Only the draw records (x_fifo, x_call_dl) read gx.desc, so the setters only
 * change gs.desc; before a draw record is queued (and when the game thread needs a vertex size),
 * desc_sync sends the whole descriptor as one record if it differs from the last one sent. The
 * particle code clears and sets the same descriptor before every quad (docs/recomp_unknowns.md
 * "Lazy vertex descriptor"). desc_sent is what gx.desc holds once the queued records have run, in
 * both modes, so the button can switch at any call. */
int recomp_lazydesc_off;
static u8 desc_sent[A_MAX];
static void x_set_desc(const u32 *p) { memcpy(gx.desc, p, A_MAX); }
static void desc_sync(void) {
    desc_dirty = 0;
    if (!memcmp(desc_sent, gs.desc, A_MAX)) return;
    memcpy(desc_sent, gs.desc, A_MAX);
    vsz_g_ok = 0;
    st_desc_recs++;
    u32 *p = gq_begin(x_set_desc, (A_MAX + 3) / 4, 0);
    memcpy(p, gs.desc, A_MAX);
    gq_end();
}
static void x_GXSetVtxDesc(const u32 *r) { if (r[3] < A_MAX) gx.desc[r[3]] = r[4] & 3; else ONCE("GXSetVtxDesc attr %lu\n", (unsigned long)r[3]); }
void sdk_GXSetVtxDesc(void) {
    LOG();
    st_desc_calls++;
    if (!recomp_lazydesc_off && R[3] < A_MAX) { gs.desc[R[3]] = R[4] & 3; desc_dirty = 1; return; }
    if (desc_dirty) desc_sync();
    if (R[3] < A_MAX) gs.desc[R[3]] = desc_sent[R[3]] = R[4] & 3;
    vsz_g_ok = 0; st_desc_recs++;
    gq_regs(x_GXSetVtxDesc, 2);
}
static void x_GXClearVtxDesc(const u32 *r) { (void)r; memset(gx.desc, 0, sizeof gx.desc); gx.desc[A_POS] = 1; }
void sdk_GXClearVtxDesc(void) {
    LOG();
    st_desc_calls++;
    memset(gs.desc, 0, sizeof gs.desc); gs.desc[A_POS] = 1;
    if (!recomp_lazydesc_off) { desc_dirty = 1; return; }
    desc_dirty = 0; memcpy(desc_sent, gs.desc, A_MAX);
    vsz_g_ok = 0; st_desc_recs++;
    gq_regs(x_GXClearVtxDesc, 0);
}
static void x_GXSetVtxAttrFmt(const u32 *r) { if ((r[3] & 7) == r[3] && r[4] < A_MAX) gx.vat[r[3]][r[4]] = (VtxFmt){(u8)r[5], (u8)r[6], (u8)r[7]}; }
void sdk_GXSetVtxAttrFmt(void) {
    LOG();
    if ((R[3] & 7) == R[3] && R[4] < A_MAX) gs.vat[R[3]][R[4]] = (VtxFmt){(u8)R[5], (u8)R[6], (u8)R[7]};
    vsz_g_ok = 0;
    gq_regs(x_GXSetVtxAttrFmt, 5);
}
static void x_GXSetArray(const u32 *r) { if (r[3] >= A_POS && r[3] <= 20) { gx.array_base[r[3] - A_POS] = r[4]; gx.array_stride[r[3] - A_POS] = r[5] & 0xff; } }
void sdk_GXSetArray(void) { LOG(); gq_regs(x_GXSetArray, 3); }
void sdk_GXInvalidateVtxCache(void) { LOG(); }

/* ---------------------------------------------------------------- transform
 * matrix records: XF address, rows, columns, then the words read at the call */
static void x_load_mtx(const u32 *p) {
    u32 addr = p[0]; int rows = p[1], cols = p[2];
    xf_gen++;
    for (u32 r = addr / 4; r <= (addr + rows * cols - 1) / 4 && r < 30; r++) pos_dirty |= 1u << r;
    for (u32 r = addr / 4; r <= (addr + rows * cols - 1) / 4 && r < 0x180; r++) xf_row_gen[r] = xf_gen;
    for (int r = 0; r < rows; r++) for (int c = 0; c < cols; c++)
        if (addr + r * cols + c < 0x600) gx.xf[addr + r * cols + c] = u2f(p[3 + r * cols + c]);
}
static void load_mtx(u32 src, u32 addr, int rows, int cols) {
    u32 *p = gq_begin(x_load_mtx, 3 + rows * cols, 0);
    p[0] = addr; p[1] = rows; p[2] = cols;
    for (int k = 0; k < rows * cols; k++) p[3 + k] = rd32(src + k * 4);
    gq_end();
}
void sdk_GXLoadPosMtxImm(void) { LOG(); load_mtx(R[3], R[4] * 4, 3, 4); }
static void x_GXLoadNrmMtxImm(const u32 *p) {   /* matrix id, then the 3x3 part */
    u32 id = p[0];
    xf_gen++;
    for (u32 r = id; r < id + 3 && r < 30; r++) nrm_dirty |= 1u << r;
    if (id * 3 + 9 > 0x200) { ONCE("normal matrix %lu out of range\n", (unsigned long)id); return; }
    for (int r = 0; r < 3; r++) for (int c = 0; c < 3; c++) gx.xf[0x400 + id * 3 + r * 3 + c] = u2f(p[1 + r * 3 + c]);
}
void sdk_GXLoadNrmMtxImm(void) {   /* source is a 3x4 matrix, the 3x3 part is used */
    LOG();
    u32 *p = gq_begin(x_GXLoadNrmMtxImm, 10, 0);
    p[0] = R[4];
    if (R[4] * 3 + 9 <= 0x200) for (int r = 0; r < 3; r++) for (int c = 0; c < 3; c++) p[1 + r * 3 + c] = rd32(R[3] + (r * 4 + c) * 4);
    gq_end();
}
static void x_tex_mtx_ignored(const u32 *r) { ONCE("post-transform texture matrix %lu ignored\n", (unsigned long)r[4]); }
void sdk_GXLoadTexMtxImm(void) {
    LOG();
    if (R[4] >= 64) { gq_regs(x_tex_mtx_ignored, 2); return; }
    load_mtx(R[3], R[4] * 4, R[5] == 1 ? 2 : 3, 4);
}
static void x_GXSetCurrentMtx(const u32 *r) { gx.cur_mtx = r[3]; }
void sdk_GXSetCurrentMtx(void) { LOG(); gq_regs(x_GXSetCurrentMtx, 1); }
/* projection and viewport: computed into gs on the game thread (the getters read it),
 * the records carry the values */
static void x_proj(const u32 *p) { gx.proj_ortho = p[0]; for (int i = 0; i < 6; i++) gx.proj[i] = u2f(p[1 + i]); }
static void rec_proj(void) {
    u32 *p = gq_begin(x_proj, 7, 0);
    p[0] = gs.proj_ortho;
    for (int i = 0; i < 6; i++) p[1 + i] = f2u(gs.proj[i]);
    gq_end();
}
void sdk_GXSetProjection(void) {
    LOG();
    u32 m = R[3];
    gs.proj_ortho = R[4] == 1;
    gs.proj[0] = u2f(rd32(m + 0x00));
    gs.proj[1] = u2f(rd32(m + (gs.proj_ortho ? 0x0c : 0x08)));
    gs.proj[2] = u2f(rd32(m + 0x14));
    gs.proj[3] = u2f(rd32(m + (gs.proj_ortho ? 0x1c : 0x18)));
    gs.proj[4] = u2f(rd32(m + 0x28));
    gs.proj[5] = u2f(rd32(m + 0x2c));
    rec_proj();
}
void sdk_GXSetProjectionv(void) {
    LOG();
    u32 p = R[3];
    gs.proj_ortho = u2f(rd32(p)) != 0;
    for (int i = 0; i < 6; i++) gs.proj[i] = u2f(rd32(p + 4 + i * 4));
    rec_proj();
}
void sdk_GXGetProjectionv(void) {
    LOG();
    wr32(R[3], f2u(gs.proj_ortho ? 1.0f : 0.0f));
    for (int i = 0; i < 6; i++) wr32(R[3] + 4 + i * 4, f2u(gs.proj[i]));
}
static void x_vp(const u32 *p) { for (int i = 0; i < 6; i++) gx.vp[i] = u2f(p[i]); }
void sdk_GXSetViewport(void) {
    LOG();
    u32 *p = gq_begin(x_vp, 6, 0);
    for (int i = 0; i < 6; i++) { gs.vp[i] = (float)F[1 + i].ps0; p[i] = f2u(gs.vp[i]); }
    gq_end();
}
void sdk_GXGetViewportv(void) { LOG(); for (int i = 0; i < 6; i++) wr32(R[3] + i * 4, f2u(gs.vp[i])); }
static void x_GXSetScissor(const u32 *r) { gx.sc[0] = r[3]; gx.sc[1] = r[4]; gx.sc[2] = r[5]; gx.sc[3] = r[6]; }
void sdk_GXSetScissor(void) { LOG(); gq_regs(x_GXSetScissor, 4); }
static void x_GXSetScissorBoxOffset(const u32 *r) { gx.sc_off[0] = (s32)r[3]; gx.sc_off[1] = (s32)r[4]; }
void sdk_GXSetScissorBoxOffset(void) { LOG(); gq_regs(x_GXSetScissorBoxOffset, 2); }
void sdk_GXSetClipMode(void) { LOG(); }
static void x_GXSetCullMode(const u32 *r) { gx.cull = r[3]; }
void sdk_GXSetCullMode(void) { LOG(); gq_regs(x_GXSetCullMode, 1); }

/* ---------------------------------------------------------------- texgen / channels / lights */
static void x_GXSetNumTexGens(const u32 *r) { gx.n_texgens = r[3] & 15; }
void sdk_GXSetNumTexGens(void) { LOG(); gq_regs(x_GXSetNumTexGens, 1); }
static void x_GXSetTexCoordGen2(const u32 *r) { if (r[3] < 8) gx.texgen[r[3]] = (TexGen){(u8)r[4], (u8)r[5], (u8)r[7], r[6]}; }
void sdk_GXSetTexCoordGen2(void) { LOG(); gq_regs(x_GXSetTexCoordGen2, 5); }
static void x_GXSetNumChans(const u32 *r) { gx.n_chans = r[3] & 3; }
void sdk_GXSetNumChans(void) { LOG(); gq_regs(x_GXSetNumChans, 1); }
static void x_GXSetChanCtrl(const u32 *r) {
    Chan c = {(u8)r[4], (u8)r[5], (u8)r[6], (u8)r[9], (u8)r[8], (u8)r[7]};
    u32 id = r[3];
    if (id == 4 || id == 5) { gx.chan[id - 4] = c; gx.chan[id - 2] = c; }
    else if (id < 4) gx.chan[id] = c;
}
void sdk_GXSetChanCtrl(void) { LOG(); gq_regs(x_GXSetChanCtrl, 7); }
static void set_color(u8 *dst, u32 ptr) { for (int i = 0; i < 4; i++) dst[i] = rd8(ptr + i); }
/* colour records carry the four bytes read at the call */
static u32 color_word(u32 ptr) { u8 c[4]; u32 w; set_color(c, ptr); memcpy(&w, c, 4); return w; }
static void x_GXSetChanMatColor(const u32 *p) { u32 id = p[0]; u8 c[4]; memcpy(c, p + 1, 4);
    if (id == 4 || id == 5) memcpy(gx.mat[id - 4], c, 4); else if (id < 2) memcpy(gx.mat[id], c, 3); else if (id < 4) gx.mat[id - 2][3] = c[3]; }
void sdk_GXSetChanMatColor(void) { LOG(); u32 *p = gq_begin(x_GXSetChanMatColor, 2, 0); p[0] = R[3]; p[1] = color_word(R[4]); gq_end(); }
static void x_GXSetChanAmbColor(const u32 *p) { u32 id = p[0]; u8 c[4]; memcpy(c, p + 1, 4);
    if (id == 4 || id == 5) memcpy(gx.amb[id - 4], c, 4); else if (id < 2) memcpy(gx.amb[id], c, 3); else if (id < 4) gx.amb[id - 2][3] = c[3]; }
void sdk_GXSetChanAmbColor(void) { LOG(); u32 *p = gq_begin(x_GXSetChanAmbColor, 2, 0); p[0] = R[3]; p[1] = color_word(R[4]); gq_end(); }

/* GXLightObj, SDK layout (0x40 bytes, big-endian): +0xc color, +0x10 a0 a1 a2
 * k0 k1 k2, +0x28 position, +0x34 direction */
static float lf(u32 o, int i) { return u2f(rd32(o + i * 4)); }
static void lset(u32 o, int i, float v) { wr32(o + i * 4, f2u(v)); }
void sdk_GXInitLightPos(void) { LOG(); for (int i = 0; i < 3; i++) lset(R[3] + 0x28, i, F[1 + i].ps0); }
void sdk_GXInitLightDir(void) { LOG(); for (int i = 0; i < 3; i++) lset(R[3] + 0x34, i, -F[1 + i].ps0); }  /* SDK stores -dir */
void sdk_GXInitLightColor(void) { LOG(); wr32(R[3] + 0xc, rd32(R[4])); }
void sdk_GXInitLightAttn(void) { LOG(); for (int i = 0; i < 6; i++) lset(R[3] + 0x10, i, F[1 + i].ps0); }
void sdk_GXInitLightSpot(void) {
    LOG();
    float cut = F[1].ps0, a0, a1, a2; int fn = R[4];
    if (cut <= 0 || cut > 90) fn = 0;
    float cr = cosf(cut * 3.14159265f / 180), d = (1 - cr) * (1 - cr);
    switch (fn) {   /* GXInitLightSpot formulas (SDK) */
    case 1: a0 = -1000 * cr; a1 = 1000; a2 = 0; break;
    case 2: a0 = -cr / (1 - cr); a1 = 1 / (1 - cr); a2 = 0; break;
    case 3: a0 = 0; a1 = -cr / (1 - cr); a2 = 1 / (1 - cr); break;
    case 4: a0 = cr * (cr - 2) / d; a1 = 2 / d; a2 = -1 / d; break;
    case 5: a0 = -4 * cr / d; a1 = 4 * (1 + cr) / d; a2 = -4 / d; break;
    case 6: a0 = 1 - 2 * cr * cr / d; a1 = 4 * cr / d; a2 = -2 / d; break;
    default: a0 = 1; a1 = a2 = 0; break;
    }
    lset(R[3] + 0x10, 0, a0); lset(R[3] + 0x10, 1, a1); lset(R[3] + 0x10, 2, a2);
}
void sdk_GXInitLightDistAttn(void) {
    LOG();
    float ref_d = F[1].ps0, ref_b = F[2].ps0, k0 = 1, k1 = 0, k2 = 0; int fn = R[4];
    if (ref_d < 0 || ref_b <= 0 || ref_b >= 1) fn = 0;
    switch (fn) {
    case 1: k1 = (1 - ref_b) / (ref_b * ref_d); break;
    case 2: k1 = 0.5f * (1 - ref_b) / (ref_b * ref_d); k2 = 0.5f * (1 - ref_b) / (ref_b * ref_d * ref_d); break;
    case 3: k2 = (1 - ref_b) / (ref_b * ref_d * ref_d); break;
    }
    lset(R[3] + 0x10, 3, k0); lset(R[3] + 0x10, 4, k1); lset(R[3] + 0x10, 5, k2);
}
static void x_GXLoadLightObjImm(const u32 *p) { light_gen++; memcpy(&gx.light[p[0]], p + 1, sizeof(Light)); }   /* light index, the Light read at the call */
void sdk_GXLoadLightObjImm(void) {
    LOG();
    for (int i = 0; i < 8; i++) if (R[4] == (1u << i)) {
        Light l; u32 o = R[3];
        for (int k = 0; k < 3; k++) { l.a[k] = lf(o + 0x10, k); l.k[k] = lf(o + 0x1c, k); l.pos[k] = lf(o + 0x28, k); l.dir[k] = -lf(o + 0x34, k); }
        for (int k = 0; k < 4; k++) l.color[k] = rd8(o + 0xc + k);
        u32 *p = gq_begin(x_GXLoadLightObjImm, 1 + (sizeof l + 3) / 4, 0);
        p[0] = i; memcpy(p + 1, &l, sizeof l);
        gq_end();
        return;
    }
}

/* ---------------------------------------------------------------- textures */
/* GXTexObj, SDK layout (0x20 bytes, big-endian): +0 mode0 (wrap s/t, mag, min,
 * LOD bias s2.5 in bits 9-16), +4 mode1 (min LOD, max LOD u4.4), +8 image0 (w-1, h-1, fmt), +0xc image3 (address >> 5), +0x10 user data,
 * +0x14 fmt, +0x18 tlut name, +0x1f flags (1 mipmap, 2 CI).
 * GXTlutObj (0xc): +0 fmt << 10, +4 address >> 5 | 0x64 << 24, +8 entries. */
static const u8 min_hw[6] = {0, 4, 1, 5, 2, 6};   /* GXTexFilter -> TX_SETMODE0 encoding */
static void texobj_init(u32 o, u32 img, u32 w, u32 h, u32 fmt, u32 ws, u32 wt, u32 mip, int ci, u32 tlut) {
    /* as the SDK's GXInitTexObj: mipmaps = LIN_MIP_LIN (LIN_MIP_NEAR for CI), max LOD log2 of the larger side */
    u32 lg = 0;
    for (u32 v = w > h ? w : h; v > 1; v >>= 1) lg++;
    wr32(o + 0, (ws & 3) | (wt & 3) << 2 | 1 << 4 | (mip ? (fmt >= 8 && fmt <= 10 ? 5u : 6u) : 4u) << 5);
    wr32(o + 4, mip ? (lg * 16 & 0xff) << 8 : 0);
    wr32(o + 8, ((w - 1) & 0x3ff) | ((h - 1) & 0x3ff) << 10 | (fmt & 15) << 20);
    wr32(o + 0xc, (img & RAM_MASK) >> 5);
    wr32(o + 0x14, fmt);
    wr32(o + 0x18, tlut);
    wr8(o + 0x1f, (mip ? 1 : 0) | (ci ? 2 : 0));
}
static u32 lod16(float v) { return v < 0 ? 0 : v > 10 ? 160 : (u32)(v * 16); }
static void texobj_lod(u32 o, u32 min_f, u32 mag_f, float min_lod, float max_lod, float bias) {
    s32 b = bias < -4 ? -128 : bias > 3.99f ? 127 : (s32)(bias * 32);
    wr32(o, (rd32(o) & ~0x1fff0u) | (mag_f & 1) << 4 | min_hw[min_f < 6 ? min_f : 1] << 5 | ((u32)b & 0xff) << 9);
    wr32(o + 4, lod16(min_lod) | lod16(max_lod) << 8);
}
static TexMap texobj_read(u32 o) {
    u32 m0 = rd32(o), i0 = rd32(o + 8), hw_min = (m0 >> 5) & 7, min_f = 1;
    for (int i = 0; i < 6; i++) if (min_hw[i] == hw_min) min_f = i;
    return (TexMap){0x80000000u | rd32(o + 0xc) << 5, (u16)((i0 & 0x3ff) + 1), (u16)(((i0 >> 10) & 0x3ff) + 1),
                    (u8)((i0 >> 20) & 15), (u8)(m0 & 3), (u8)((m0 >> 2) & 3), (u8)min_f, (u8)((m0 >> 4) & 1),
                    (u8)((rd8(o + 0x1f) & 2) != 0), rd32(o + 0x18), (u8)rd32(o + 4), (u8)(rd32(o + 4) >> 8), (s8)(m0 >> 9)};
}
static void tlutobj_init(u32 o, u32 lut, u32 fmt, u32 n) {
    wr32(o, (fmt & 3) << 10);
    wr32(o + 4, (lut & RAM_MASK) >> 5 | 0x64u << 24);
    wr16(o + 8, (u16)n);
}
void sdk_GXInitTexObj(void) { LOG(); texobj_init(R[3], R[4], R[5], R[6], R[7], R[8], R[9], R[10] & 0xff, 0, 0); }
void sdk_GXInitTexObjCI(void) { LOG(); texobj_init(R[3], R[4], R[5], R[6], R[7], R[8], R[9], R[10] & 0xff, 1, rd32(R[1] + 8)); }
void sdk_GXInitTexObjLOD(void) { LOG(); texobj_lod(R[3], R[4], R[5], (float)F[1].ps0, (float)F[2].ps0, (float)F[3].ps0); }   /* clamp/edge/aniso not kept */
static void x_GXLoadTexObj(const u32 *p) { memcpy(&gx.map[p[0]], p + 1, sizeof(TexMap)); }   /* texmap id, the TexMap read at the call */
void sdk_GXLoadTexObj(void) {
    LOG();
    if (R[4] >= 8) return;
    TexMap m = texobj_read(R[3]);
    u32 *p = gq_begin(x_GXLoadTexObj, 1 + (sizeof m + 3) / 4, 0);
    p[0] = R[4]; memcpy(p + 1, &m, sizeof m);
    gq_end();
}
void sdk_GXInitTlutObj(void) { LOG(); tlutobj_init(R[3], R[4], R[5], R[6] & 0xffff); }
static void x_GXLoadTlut(const u32 *p) {   /* tlut name, the Tlut read at the call */
    if (p[0] >= 20) { ONCE("tlut name %lu\n", (unsigned long)p[0]); return; }
    memcpy(&gx.tlut[p[0]], p + 1, sizeof(Tlut));
}
void sdk_GXLoadTlut(void) {
    LOG();
    Tlut t = {0, 0, 0};
    if (R[4] < 20) t = (Tlut){0x80000000u | (rd32(R[3] + 4) & 0xffffff) << 5, (u8)((rd32(R[3]) >> 10) & 3), rd16(R[3] + 8)};
    u32 *p = gq_begin(x_GXLoadTlut, 1 + (sizeof t + 3) / 4, 0);
    p[0] = R[4]; memcpy(p + 1, &t, sizeof t);
    gq_end();
}
static u32 tex_region_cb, tlut_region_cb;
void sdk_GXSetTexRegionCallback(void) { LOG(); u32 o = tex_region_cb; tex_region_cb = R[3]; R[3] = o; }
void sdk_GXSetTlutRegionCallback(void) { LOG(); u32 o = tlut_region_cb; tlut_region_cb = R[3]; R[3] = o; }

/* TPL helpers (TPL library, SDK): descriptors are {image header*, palette header*};
 * image header: +0 h, +2 w, +4 fmt, +8 data, +0xc/+0x10 wrap s/t, +0x14/+0x18 min/mag,
 * +0x1c LOD bias (f32), +0x21/+0x22 min/max LOD (mipmap when they differ) */
void sdk_TPLGet(void) { LOG(); R[3] = rd32(R[3] + 8) + R[4] * 8; }
static void tpl_texobj(u32 obj, u32 img, int ci, u32 tlut) {
    texobj_init(obj, rd32(img + 8), rd16(img + 2), rd16(img), rd32(img + 4), rd32(img + 0xc), rd32(img + 0x10),
                rd8(img + 0x21) != rd8(img + 0x22), ci, tlut);
    u32 bb = rd32(img + 0x1c); float bias; memcpy(&bias, &bb, 4);
    texobj_lod(obj, rd32(img + 0x14), rd32(img + 0x18), rd8(img + 0x21), rd8(img + 0x22), bias);
}
void sdk_TPLGetGXTexObjFromPalette(void) {
    LOG();
    u32 d = rd32(R[3] + 8) + R[5] * 8;
    tpl_texobj(R[4], rd32(d), 0, 0);
}
void sdk_TPLGetGXTexObjFromPaletteCI(void) {
    LOG();
    u32 d = rd32(R[3] + 8) + R[7] * 8, pal = rd32(d + 4);
    tlutobj_init(R[5], rd32(pal + 8), rd32(pal + 4), rd16(pal));
    tpl_texobj(R[4], rd32(d), 1, R[6]);
}

/* ---------------------------------------------------------------- TEV */
static TevSide *side(const u32 *r, int alpha) { return r[3] < 16 ? (alpha ? &gx.tev[r[3]].a : &gx.tev[r[3]].c) : NULL; }
static void x_GXSetNumTevStages(const u32 *r) { gx.n_stages = r[3] & 31; }
void sdk_GXSetNumTevStages(void) { LOG(); gq_regs(x_GXSetNumTevStages, 1); }
static void x_GXSetTevOrder(const u32 *r) { if (r[3] < 16) { TevStage *t = &gx.tev[r[3]]; t->coord = r[4]; t->map = r[5] == 0xff ? 0xff : r[5] & 7; t->chan = r[6]; } }
void sdk_GXSetTevOrder(void) { LOG(); gq_regs(x_GXSetTevOrder, 4); }
static void x_GXSetTevColorIn(const u32 *r) { TevSide *s = side(r, 0); if (s) { s->a = r[4]; s->b = r[5]; s->c = r[6]; s->d = r[7]; } }
void sdk_GXSetTevColorIn(void) { LOG(); gq_regs(x_GXSetTevColorIn, 5); }
static void x_GXSetTevAlphaIn(const u32 *r) { TevSide *s = side(r, 1); if (s) { s->a = r[4]; s->b = r[5]; s->c = r[6]; s->d = r[7]; } }
void sdk_GXSetTevAlphaIn(void) { LOG(); gq_regs(x_GXSetTevAlphaIn, 5); }
static void x_GXSetTevColorOp(const u32 *r) { TevSide *s = side(r, 0); if (s) { s->op = r[4]; s->bias = r[5]; s->scale = r[6]; s->clamp = r[7]; s->out = r[8]; } }
void sdk_GXSetTevColorOp(void) { LOG(); gq_regs(x_GXSetTevColorOp, 6); }
static void x_GXSetTevAlphaOp(const u32 *r) { TevSide *s = side(r, 1); if (s) { s->op = r[4]; s->bias = r[5]; s->scale = r[6]; s->clamp = r[7]; s->out = r[8]; } }
void sdk_GXSetTevAlphaOp(void) { LOG(); gq_regs(x_GXSetTevAlphaOp, 6); }
static void x_GXSetTevOp(const u32 *r) {
    if (r[3] >= 16) return;
    TevStage *t = &gx.tev[r[3]];
    int c = r[3] ? 0 : 10, a = r[3] ? 0 : 5;   /* CPREV/APREV after stage 0, RASC/RASA at stage 0 */
    static const u8 cin[5][4] = {{15, 8, 99, 15}, {99, 8, 9, 15}, {99, 12, 8, 15}, {15, 15, 15, 8}, {15, 15, 15, 99}};
    static const u8 ain[5][4] = {{7, 4, 99, 7}, {7, 7, 7, 99}, {7, 4, 99, 7}, {7, 7, 7, 4}, {7, 7, 7, 99}};
    int m = r[4] < 5 ? r[4] : 0;
    u8 *cp = &t->c.a, *ap = &t->a.a;
    for (int i = 0; i < 4; i++) { cp[i] = cin[m][i] == 99 ? c : cin[m][i]; ap[i] = ain[m][i] == 99 ? a : ain[m][i]; }
    t->c.op = t->c.bias = t->c.scale = t->c.out = 0; t->c.clamp = 1;
    t->a.op = t->a.bias = t->a.scale = t->a.out = 0; t->a.clamp = 1;
}
void sdk_GXSetTevOp(void) { LOG(); gq_regs(x_GXSetTevOp, 2); }
static void x_GXSetTevColor(const u32 *p) { u32 id = p[0]; memcpy(gx.tev_reg[id], p + 1, 4); for (int i = 0; i < 4; i++) gx.tev_s10[id][i] = gx.tev_reg[id][i]; }
void sdk_GXSetTevColor(void) { LOG(); if (R[3] < 4) { u32 *p = gq_begin(x_GXSetTevColor, 2, 0); p[0] = R[3]; p[1] = color_word(R[4]); gq_end(); } }
static void x_GXSetTevColorS10(const u32 *p) {   /* register, then the four s16 read at the call */
    u32 id = p[0];
    for (int i = 0; i < 4; i++) { s16 v = (s16)p[1 + i]; gx.tev_s10[id][i] = v; gx.tev_reg[id][i] = v < 0 ? 0 : v > 255 ? 255 : v; }
}
void sdk_GXSetTevColorS10(void) {
    LOG();
    if (R[3] >= 4) return;
    u32 *p = gq_begin(x_GXSetTevColorS10, 5, 0);
    p[0] = R[3];
    for (int i = 0; i < 4; i++) p[1 + i] = rd16(R[4] + i * 2);
    gq_end();
}
static void x_GXSetTevKColor(const u32 *p) { memcpy(gx.kcolor[p[0]], p + 1, 4); }
void sdk_GXSetTevKColor(void) { LOG(); if (R[3] < 4) { u32 *p = gq_begin(x_GXSetTevKColor, 2, 0); p[0] = R[3]; p[1] = color_word(R[4]); gq_end(); } }
static void x_GXSetTevKColorSel(const u32 *r) { if (r[3] < 16) gx.tev[r[3]].ksel_c = r[4]; }
void sdk_GXSetTevKColorSel(void) { LOG(); gq_regs(x_GXSetTevKColorSel, 2); }
static void x_GXSetTevKAlphaSel(const u32 *r) { if (r[3] < 16) gx.tev[r[3]].ksel_a = r[4]; }
void sdk_GXSetTevKAlphaSel(void) { LOG(); gq_regs(x_GXSetTevKAlphaSel, 2); }
static void x_GXSetTevSwapMode(const u32 *r) { if (r[3] < 16) { gx.tev[r[3]].ras_swap = r[4] & 3; gx.tev[r[3]].tex_swap = r[5] & 3; if (r[4] & 3) ONCE("TEV raster swap %lu ignored\n", (unsigned long)(r[4] & 3)); } }
void sdk_GXSetTevSwapMode(void) { LOG(); gq_regs(x_GXSetTevSwapMode, 3); }
static void x_GXSetAlphaCompare(const u32 *r) { gx.alpha_cmp[0] = r[3]; gx.alpha_cmp[1] = r[4]; gx.alpha_cmp[2] = r[5]; gx.alpha_cmp[3] = r[6]; gx.alpha_cmp[4] = r[7]; }
void sdk_GXSetAlphaCompare(void) { LOG(); gq_regs(x_GXSetAlphaCompare, 5); }

/* ---------------------------------------------------------------- pixel engine */
static void x_GXSetBlendMode(const u32 *r) { gx.blend_type = r[3]; gx.blend_src = r[4]; gx.blend_dst = r[5]; gx.logic_op = r[6]; }
void sdk_GXSetBlendMode(void) { LOG(); gq_regs(x_GXSetBlendMode, 4); }
static void x_GXSetZMode(const u32 *r) { gx.z_enable = r[3]; gx.z_func = r[4]; gx.z_update = r[5]; }
void sdk_GXSetZMode(void) { LOG(); gq_regs(x_GXSetZMode, 3); }
static void x_GXSetColorUpdate(const u32 *r) { gx.color_update = r[3]; }
void sdk_GXSetColorUpdate(void) { LOG(); gq_regs(x_GXSetColorUpdate, 1); }
static void x_GXSetAlphaUpdate(const u32 *r) { gx.alpha_update = r[3]; }
void sdk_GXSetAlphaUpdate(void) { LOG(); gq_regs(x_GXSetAlphaUpdate, 1); }

/* ---------------------------------------------------------------- copy */
static void x_GXSetDispCopySrc(const u32 *r) { if (r[5] && r[6]) { gx.efb_w = r[5]; gx.efb_h = r[6]; } }
void sdk_GXSetDispCopySrc(void) { LOG(); gq_regs(x_GXSetDispCopySrc, 4); }
static void x_GXSetCopyClear(const u32 *p) { memcpy(gx.clear_color, p, 4); gx.clear_z = p[1]; }   /* colour, z */
void sdk_GXSetCopyClear(void) { LOG(); u32 *p = gq_begin(x_GXSetCopyClear, 2, 0); p[0] = color_word(R[3]); p[1] = R[4] & 0xffffff; gq_end(); }
static void x_GXCopyTex(const u32 *r) {
    u32 a = GUEST(r[3]);
    for (int i = 0; i < 8; i++) if (copy_dst[i] == a) return;
    for (int i = 0; i < 8; i++) if (!copy_dst[i]) { copy_dst[i] = a; return; }
    ONCE("more than 8 EFB copy destinations\n");
}
void sdk_GXCopyTex(void) { LOG(); gq_regs(x_GXCopyTex, 1); }

static void x_GXCopyDisp(const u32 *r) { frame_end(); if (r[4]) clear_pending = 1; }
void sdk_GXCopyDisp(void) {
    LOG();
    gq_regs(x_GXCopyDisp, 2);   /* r4: clear */
    game_frames++;
    if (game_frames % 300 == 0) {
        extern u32 recomp_wr_sweep_pages; extern u64 recomp_wr_sweep_ticks;   /* cpu.c */
        recomp_log("# gx queue frame %lu: per frame %lu records, %lu words; vertex descriptor %s: %lu setter calls, %lu records; "
                   "write sweep %lu pages, %.1f us\n", (unsigned long)game_frames,
                   (unsigned long)(st_gq_recs / 300), (unsigned long)(st_gq_words / 300), recomp_lazydesc_off ? "eager" : "lazy",
                   (unsigned long)(st_desc_calls / 300), (unsigned long)(st_desc_recs / 300),
                   (unsigned long)(recomp_wr_sweep_pages / 300), recomp_wr_sweep_ticks * 1e6 / SYSCLOCK_ARM11 / 300);
        st_gq_recs = st_gq_words = st_desc_calls = st_desc_recs = 0;
        recomp_wr_sweep_pages = 0; recomp_wr_sweep_ticks = 0;
        extern u32 recomp_libm_pairs, recomp_libm_pair_waste;   /* libm.c */
        recomp_log("# libm frame %lu: per frame %.1f cos calls served by a sin/cos pair, %.1f pairs unused\n", (unsigned long)game_frames,
                   recomp_libm_pairs / 300.0, recomp_libm_pair_waste / 300.0);
        recomp_libm_pairs = recomp_libm_pair_waste = 0;
    }
    recomp_wr_sweep();   /* the frame's marked pages get the epoch that is ending */
    __dmb();   /* the frame's guest writes before the epoch that ends it (epoch_now) */
    recomp_wr_epoch = game_frames + 1;
    if (recomp_gx_thread) { gq_publish(); gq_wait(gq_copy_prev); gq_copy_prev = gq_wl; }   /* at most one frame queued */
}
