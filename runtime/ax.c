/* AX: the voice mixer, done on the host the way Dolphin's HLE AX ucode does
 * it, 5 ms per frame (160 samples at 32 kHz), output through ndsp.
 *
 * AXInit is not run, so the SDK's own AXVPB pool and free list (0x8041fa20)
 * stay empty; voices are handed out from guest RAM above the GameCube's
 * 24 MB (0x81800000.., unused by the game). The voice setters are hand-
 * written and store into the AXVPB at the SDK's own offsets (checked against
 * the disassembly, see sdk_functions.md); the AXPB itself starts at +0x138
 * (Dolphin's AXPB field order). The mixer reads it there every frame and
 * writes back what the DSP would: state, current address, ADPCM history,
 * sample-rate fraction, envelope and mixer volumes.
 *
 * The AX frame callback (AXRegisterCallback 0x80145fb8) is the sound
 * driver's tick: 200 per second, run at the VIWaitForRetrace wait point.
 * Interactive play paces it by host time (as the DSP interrupt does on the
 * GameCube), autotests by 10 per 3 retraces (deterministic replays); see
 * recomp_ax_service. As in the SDK's AX interrupt, each frame first
 * consumes the voices' sync flags and renders them (__AXSyncPBs, DSP), then
 * calls the callback; what it changes is heard from the next frame. synth.c
 * asserts this order (AX_SYNC_USER_VE clear in its tick). */
#include <3ds.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "ppc.h"

#define LOG() RECOMP_LOG()

#define VPB_BASE 0x81800000u
#define VPB_SIZE 0x200u
#define VPB_COUNT 64
#define FRAME 160                  /* samples per 5 ms frame at 32 kHz */

/* AXVPB offsets */
#define V_PRIO 0x0c
#define V_SYNC 0x1c
#define V_DEPOP 0x20
#define PB 0x138
#define P_SRCSEL (PB + 0x08)
#define P_MIXCTRL (PB + 0x0c)
#define P_STATE (PB + 0x0e)
#define P_TYPE (PB + 0x10)
#define P_MIX (PB + 0x12)          /* vL, dL, vR, dR, auxA L/R, auxB L/R, S.. */
#define P_ITD (PB + 0x36)
#define P_VE (PB + 0x64)           /* volume, delta */
#define P_ADDR (PB + 0x6e)         /* loop flag, format, loop, end, current */
#define P_ADPCM (PB + 0x7e)        /* coefs[16], gain, pred_scale, yn1, yn2 */
#define P_SRC (PB + 0xa6)          /* ratio hi/lo, cur frac, last samples[4] */
#define P_ADPCMLOOP (PB + 0xb4)    /* pred_scale, yn1, yn2 */

static u32 ax_cb;
static u8 vpb_used[VPB_COUNT];

void sdk_AXRegisterCallback(void) { LOG(); u32 old = ax_cb; ax_cb = R[3]; R[3] = old; }

/* ---- voices ---- */

/* one log line per voice event, capped */
static void vlog(const char *what, u32 v, u32 val) {
    static u32 n;
    if (++n > 400) return;
    recomp_log("# AX: r%lu v%d %s %lx cur %lx loop %lx end %lx fmt %x lp %x\n", (unsigned long)recomp_retrace_count(),
               (int)((v - VPB_BASE) / VPB_SIZE), what, (unsigned long)val, (unsigned long)rd32(v + P_ADDR + 12),
               (unsigned long)rd32(v + P_ADDR + 4), (unsigned long)rd32(v + P_ADDR + 8), rd16(v + P_ADDR + 2), rd16(v + P_ADDR));
}

void sdk_AXAcquireVoice(void) {
    LOG();
    for (u32 i = 0; i < VPB_COUNT; i++) {
        if (vpb_used[i]) continue;
        u32 v = VPB_BASE + i * VPB_SIZE;
        vpb_used[i] = 1;
        for (u32 k = 0; k < VPB_SIZE; k += 4) wr32(v + k, 0);
        wr32(v + V_PRIO, R[3]); wr32(v + 0x10, R[4]); wr32(v + 0x14, R[5]); wr32(v + 0x18, i);
        R[3] = v;
        return;
    }
    recomp_log("# AX: out of voices\n");
    R[3] = 0;
}

static int voice_index(u32 v) {
    u32 i = (v - VPB_BASE) / VPB_SIZE;
    return v >= VPB_BASE && i < VPB_COUNT ? (int)i : -1;
}

void sdk_AXFreeVoice(void) {
    LOG();
    int i = voice_index(R[3]);
    if (i < 0) return;
    vlog("free", R[3], 0);
    vpb_used[i] = 0;
    wr16(R[3] + P_STATE, 0);
    wr32(R[3] + V_PRIO, 0);
}

void sdk_AXSetVoicePriority(void) { LOG(); wr32(R[3] + V_PRIO, R[4]); }

static void sync(u32 v, u32 bits) { wr32(v + V_SYNC, rd32(v + V_SYNC) | bits); }
static void copy16(u32 dst, u32 src, u32 n) { for (u32 k = 0; k < n; k += 2) wr16(dst + k, rd16(src + k)); }

static u32 voices_started;

/* type: 0 polyphase 8 tap, 1 polyphase 4 tap, 2 linear, 3/4 none (coef sel) */
void sdk_AXSetVoiceSrcType(void) {
    LOG();
    static const u16 sel[5][2] = { {2, 0}, {1, 0}, {0, 0}, {0, 1}, {0, 2} };
    u32 t = R[4] < 5 ? R[4] : 0;
    wr16(R[3] + P_SRCSEL, sel[t][0]);
    if (t >= 2) wr16(R[3] + P_SRCSEL + 2, sel[t][1]);
    sync(R[3], 0x1);
}

void sdk_AXSetVoiceState(void) {
    LOG();
    vlog("state", R[3], R[4]);
    if (R[4] == 1 && rd16(R[3] + P_STATE) != 1) voices_started++;
    wr16(R[3] + P_STATE, R[4]);
    if (R[4] == 0) wr32(R[3] + V_DEPOP, 1);
    sync(R[3], 0x4);
}

void sdk_AXSetVoiceMix(void) {
    LOG();
    static const u16 bit[18] = { 0x1, 0x9, 0x2, 0xa, 0x10, 0x50, 0x20, 0x60, 0x200, 0xa00, 0x400, 0xc00,
                                 0x1000, 0x3000, 0x4, 0xc, 0x80, 0x180 };
    u32 v = R[3], m = R[4], ctrl = 0;
    for (u32 k = 0; k < 18; k++) {
        u16 x = rd16(m + k * 2);
        wr16(v + P_MIX + k * 2, x);
        if (x) ctrl |= bit[k];
    }
    wr16(v + P_MIXCTRL, ctrl);
    sync(v, 0x12);
}

void sdk_AXSetVoiceItdOn(void) {
    LOG();
    wr16(R[3] + P_ITD, 1);
    for (u32 k = 6; k < 14; k += 2) wr16(R[3] + P_ITD + k, 0);
    wr32(R[3] + V_SYNC, (rd32(R[3] + V_SYNC) & ~0x40u) | 0x20);
}
void sdk_AXSetVoiceItdTarget(void) { LOG(); wr16(R[3] + P_ITD + 10, R[4]); wr16(R[3] + P_ITD + 12, R[5]); sync(R[3], 0x40); }
void sdk_AXSetVoiceVe(void) { LOG(); copy16(R[3] + P_VE, R[4], 4); sync(R[3], 0x200); }
void sdk_AXSetVoiceVeDelta(void) { LOG(); wr16(R[3] + P_VE + 2, R[4]); sync(R[3], 0x400); }

void sdk_AXSetVoiceAddr(void) {
    LOG();
    u32 v = R[3];
    copy16(v + P_ADDR, R[4], 16);
    vlog("addr", v, 0);
    u16 fmt = rd16(v + P_ADDR + 2);
    if (fmt == 0x0a || fmt == 0x19) {         /* PCM16 / PCM8: clear ADPCM block, gain */
        for (u32 k = 0; k < 0x28; k += 2) wr16(v + P_ADPCM + k, 0);
        wr16(v + P_ADPCM + 0x20, fmt == 0x0a ? 0x0800 : 0x0100);
    }
    wr32(v + V_SYNC, (rd32(v + V_SYNC) & ~0x1e000u) | 0x21000);
}
void sdk_AXSetVoiceLoop(void) { LOG(); vlog("loopflag", R[3], R[4]); wr16(R[3] + P_ADDR, R[4]); sync(R[3], 0x2000); }
void sdk_AXSetVoiceLoopAddr(void) { LOG(); vlog("loopaddr", R[3], R[4]); wr16(R[3] + P_ADDR + 4, R[4] >> 16); wr16(R[3] + P_ADDR + 6, R[4]); sync(R[3], 0x4000); }
void sdk_AXSetVoiceEndAddr(void) { LOG(); vlog("endaddr", R[3], R[4]); wr16(R[3] + P_ADDR + 8, R[4] >> 16); wr16(R[3] + P_ADDR + 10, R[4]); sync(R[3], 0x8000); }
void sdk_AXSetVoiceCurrentAddr(void) { LOG(); vlog("curaddr", R[3], R[4]); wr16(R[3] + P_ADDR + 12, R[4] >> 16); wr16(R[3] + P_ADDR + 14, R[4]); sync(R[3], 0x10000); }
void sdk_AXSetVoiceAdpcm(void) { LOG(); copy16(R[3] + P_ADPCM, R[4], 0x28); sync(R[3], 0x20000); }
void sdk_AXSetVoiceSrc(void) { LOG(); copy16(R[3] + P_SRC, R[4], 14); wr32(R[3] + V_SYNC, (rd32(R[3] + V_SYNC) & ~0x80000u) | 0x40000); }
void sdk_AXSetVoiceSrcRatio(void) {
    LOG();
    float f = (float)F[1].ps0 * 65536.0f;
    u32 r = f <= 0 ? 0 : f >= (float)0x40000 ? 0x40000 : (u32)f;
    wr16(R[3] + P_SRC, r >> 16); wr16(R[3] + P_SRC + 2, r);
    sync(R[3], 0x80000);
}
void sdk_AXSetVoiceAdpcmLoop(void) { LOG(); copy16(R[3] + P_ADPCMLOOP, R[4], 6); sync(R[3], 0x100000); }

/* aux effect callbacks: no aux bus mixing (reverb/chorus), see recomp_unknowns.md */
void sdk_AXRegisterAuxACallback(void) { LOG(); }
void sdk_AXRegisterAuxBCallback(void) { LOG(); }

/* ---- mixer ---- */

static s32 mixL[FRAME], mixR[FRAME];
static struct { u32 frames, peak, active_max, ended, looped, queued, dropped, underruns, skipped, trimmed, hash; u64 sumsq; } st;

static inline s16 clamp16(s32 x) { return x < -32768 ? -32768 : x > 32767 ? 32767 : x; }

/* one source sample at the current address; advances the address, handles
 * end/loop; returns 0 and stops the voice past a non-looping end */
typedef struct { u32 cur, end, loop; u16 fmt, looping, ps; s16 yn1, yn2; s16 coef[16]; int stopped, is_stream; u32 v; } Src;

static s16 fetch(Src *s) {
    if (s->stopped) return 0;
    s16 out;
    if (s->fmt == 0) {                        /* DSP-ADPCM, nibble addresses */
        if ((s->cur & 15) == 0) { s->ps = recomp_aram_read8(s->cur >> 1); s->cur += 2; }
        u8 b = recomp_aram_read8(s->cur >> 1);
        s32 nib = (s->cur & 1) ? (b & 15) : (b >> 4);
        if (nib >= 8) nib -= 16;
        s32 c1 = s->coef[(s->ps >> 4) * 2 & 15], c2 = s->coef[((s->ps >> 4) * 2 + 1) & 15];
        s32 val = ((nib << (s->ps & 15)) << 11) + c1 * s->yn1 + c2 * s->yn2;
        out = clamp16((val + 1024) >> 11);
        s->yn2 = s->yn1; s->yn1 = out;
    } else if (s->fmt == 0x0a) {              /* PCM16, word addresses */
        out = (s16)((recomp_aram_read8(s->cur * 2) << 8) | recomp_aram_read8(s->cur * 2 + 1));
    } else {                                  /* PCM8 */
        out = (s16)((s8)recomp_aram_read8(s->cur) << 8);
    }
    if (s->cur == s->end) {
        if (s->looping) {
            s->cur = s->loop;
            if (s->fmt == 0) {
                s->ps = rd16(s->v + P_ADPCMLOOP);
                if (!s->is_stream) { s->yn1 = rd16(s->v + P_ADPCMLOOP + 2); s->yn2 = rd16(s->v + P_ADPCMLOOP + 4); }
            }
            st.looped++;
        } else {
            s->stopped = 1;
            st.ended++;
            vlog("ended", s->v, s->cur);
        }
    } else s->cur++;
    return out;
}

/* what a frame's decode did besides the samples: loop wraps, a stop (and the address it stopped at, for
 * vlog), and with track the lowest and highest ARAM byte it read */
typedef struct { u32 looped, stop_cur, alo, ahi; u8 stopped; } DecEv;

/* n source samples into out[], the same values and state changes as n fetch() calls, with the
 * ARAM bytes read directly and the state in locals. lp = the PB's loop predictor/history (ps, yn1, yn2).
 * The caller applies ev (st counters, vlog), so this also runs on the mix worker. */
static inline __attribute__((always_inline)) void decode_t(Src *s, const u8 *A, s16 *out, u32 n, const s16 *lp, DecEv *ev, int track) {
    const u32 M = 0x00ffffffu;   /* ARAM_SIZE - 1, as recomp_aram_read8 masks */
    u32 cur = s->cur, end = s->end, i = 0, lo = ~0u, hi = 0;
    ev->looped = 0; ev->stopped = 0;
    if (s->stopped) goto zero;
    if (s->fmt == 0) {                        /* DSP-ADPCM */
        u32 ps = s->ps;
        s32 yn1 = s->yn1, yn2 = s->yn2;
        s32 c1 = s->coef[(ps >> 4) * 2 & 15], c2 = s->coef[((ps >> 4) * 2 + 1) & 15];
        for (; i < n; i++) {
            if ((cur & 15) == 0) {
                if (track) { u32 a = (cur >> 1) & M; lo = a < lo ? a : lo; hi = a > hi ? a : hi; }
                ps = A[(cur >> 1) & M]; cur += 2;
                c1 = s->coef[(ps >> 4) * 2 & 15]; c2 = s->coef[((ps >> 4) * 2 + 1) & 15];
            }
            if (track) { u32 a = (cur >> 1) & M; lo = a < lo ? a : lo; hi = a > hi ? a : hi; }
            u32 b = A[(cur >> 1) & M];
            s32 nib = (cur & 1) ? (b & 15) : (b >> 4);
            if (nib >= 8) nib -= 16;
            s32 val = ((nib << (ps & 15)) << 11) + c1 * yn1 + c2 * yn2;
            s16 o = clamp16((val + 1024) >> 11);
            yn2 = yn1; yn1 = o;
            out[i] = o;
            if (cur == end) {
                if (s->looping) {
                    cur = s->loop;
                    ps = (u16)lp[0];
                    if (!s->is_stream) { yn1 = lp[1]; yn2 = lp[2]; }
                    c1 = s->coef[(ps >> 4) * 2 & 15]; c2 = s->coef[((ps >> 4) * 2 + 1) & 15];
                    ev->looped++;
                } else {
                    s->stopped = 1; ev->stopped = 1; ev->stop_cur = cur;
                    i++; break;
                }
            } else cur++;
        }
        s->ps = ps; s->yn1 = yn1; s->yn2 = yn2;
    } else {                                  /* PCM16 (word addresses) / PCM8 */
        int w = s->fmt == 0x0a;
        for (; i < n; i++) {
            if (track) {
                u32 a0 = (w ? cur * 2 : cur) & M, a1 = w ? (cur * 2 + 1) & M : a0;
                lo = a0 < lo ? a0 : lo; hi = a0 > hi ? a0 : hi; lo = a1 < lo ? a1 : lo; hi = a1 > hi ? a1 : hi;
            }
            out[i] = w ? (s16)((A[(cur * 2) & M] << 8) | A[(cur * 2 + 1) & M]) : (s16)((s8)A[cur & M] << 8);
            if (cur == end) {
                if (s->looping) { cur = s->loop; ev->looped++; }
                else { s->stopped = 1; ev->stopped = 1; ev->stop_cur = cur; i++; break; }
            } else cur++;
        }
    }
    s->cur = cur;
zero:
    for (; i < n; i++) out[i] = 0;
    ev->alo = lo; ev->ahi = hi;
}
static void decode(Src *s, const u8 *A, s16 *out, u32 n, const s16 *lp, DecEv *ev) { decode_t(s, A, out, n, lp, ev, 0); }
static void decode_track(Src *s, const u8 *A, s16 *out, u32 n, const s16 *lp, DecEv *ev) { decode_t(s, A, out, n, lp, ev, 1); }

/* ---- mix jobs. The game thread decodes each voice's samples for the frame and writes back the PB state
 * the DSP would (what the sound driver's tick reads); the resampling, volumes, sum and output then run from
 * a job: on a worker thread on core 2 (New 3DS) while the game goes on, else at once. The samples are the
 * same either way: every input of a voice's mix is in its job. ---- */
#define POOL 8192                  /* decoded samples per frame (4 history + the frame's fetches per voice) */
typedef struct { u32 ratio, frac, off; s32 ve, vl, vr; s16 ved, dl, dr; } VJob;
typedef struct {
    u32 nv, used, fb;              /* voices, pool samples used, fbL/fbR hold slow-path voices */
    VJob v[VPB_COUNT];
    s16 pool[POOL];
    s32 fbL[FRAME], fbR[FRAME];    /* voices mixed by mix_voice (AXFAST=0, no ARAM, too many samples) */
    u32 report;                    /* the 1000-frame line after this frame, with the game thread's counts */
    u32 frames, started, active_max, ended, looped, underruns, skipped, trimmed, rt, spec_hits, spec_tries;
    u32 fi, npost;                /* AX frame number; voices still playing after it, with their decode block: */
    struct { u8 vi; u16 type; u8 blk[76]; } post[VPB_COUNT];
} FJob;

/* ---- decode predictions (mix worker). A voice's decode for one AX frame depends only on the 76 bytes of its
 * PB at +0x6e..+0xba (looping, format, loop/end/current address, ADPCM coefs/gain/predictor/history, SRC ratio,
 * fraction and history, loop predictor/history; BLK), on P_TYPE and on the ARAM bytes it reads. After each mix
 * job the worker decodes the next SPEC_K frames of every voice still playing from the block the frame left,
 * chaining each prediction's resulting block into the next. The game thread takes a prediction only when the
 * voice's block and type equal the prediction's key and no ARAM DMA since it was made touched the range it read;
 * then the samples and the state it writes are the ones its own decode would make. ---- */
#define BLK (PB + 0x6e)
#define BLK_N 76
#define SPEC_K 4                   /* AX frames predicted ahead per voice */
#define SPEC_R 8                   /* ring slots per voice: more than SPEC_K, so a slot is not rewritten while read */
#define SPEC_MAXS 512              /* decoded samples per prediction (ratio up to 3.2) */
typedef struct {
    volatile u32 seq;              /* odd while the worker writes it */
    u32 frame, gen;                /* AX frame it is for; aram_gen before its decode read ARAM */
    u16 type; u8 key[BLK_N], next[BLK_N];   /* the block it was made from; the block it leaves */
    u32 total; DecEv ev;
    s16 h[4 + SPEC_MAXS];          /* SRC history, then the decoded samples */
} SpecEnt;
static SpecEnt spec[VPB_COUNT][SPEC_R];
static int job_worker;             /* the job ring below: 1 = a worker thread */
static u32 spec_last[VPB_COUNT];   /* worker: last frame predicted per voice (0 = none) */
static u32 spec_hits, spec_tries;  /* game thread, for the 5 s line */
#ifdef RECOMP_AX_NO_SPEC
static const int spec_on = 0;      /* make AXSPEC=0 */
#else
static const int spec_on = 1;
#endif

/* ARAM writes (ar.c, game thread): a ring of recent DMA ranges, newest generation aram_gen */
#define DMA_LOG 64
static struct { u32 gen, lo, hi; } dma_log[DMA_LOG];
static volatile u32 aram_gen;
void recomp_aram_written(u32 a, u32 n) {
    if (!n) return;
    u32 g = aram_gen + 1;
    dma_log[g % DMA_LOG].gen = g; dma_log[g % DMA_LOG].lo = a; dma_log[g % DMA_LOG].hi = a + n - 1;
    __dmb();   /* the bytes (written before the call) and the record before the generation */
    aram_gen = g;
}
static int aram_clean(u32 gen, u32 lo, u32 hi) {
    u32 now = aram_gen;
    if (now - gen >= DMA_LOG) return now == gen;
    for (u32 g = gen + 1; g != now + 1; g++)
        if (dma_log[g % DMA_LOG].lo <= hi && dma_log[g % DMA_LOG].hi >= lo) return 0;
    return 1;
}

static inline u32 be16(const u8 *p) { return (u32)p[0] << 8 | p[1]; }
static inline u32 be32(const u8 *p) { return (u32)p[0] << 24 | (u32)p[1] << 16 | (u32)p[2] << 8 | p[3]; }
static inline void put16(u8 *p, u32 v) { p[0] = (u8)(v >> 8); p[1] = (u8)v; }

/* worker: the prediction for frame f from block key (type t); 0 if it cannot be made */
static int spec_make(SpecEnt *e, u32 f, const u8 *key, u16 t, const u8 *A) {
    u32 ratio = be32(key + 0x38), frac = be16(key + 0x3c);
    u64 total = ((u64)frac + (u64)ratio * FRAME) >> 16;
    if (total > SPEC_MAXS) return 0;
    e->seq++;
    __dmb();
    e->gen = aram_gen;
    __dmb();   /* the generation before the ARAM reads */
    Src s;
    s.v = 0; s.looping = be16(key); s.fmt = be16(key + 2);
    s.loop = be32(key + 4); s.end = be32(key + 8); s.cur = be32(key + 12);
    for (int k = 0; k < 16; k++) s.coef[k] = (s16)be16(key + 0x10 + k * 2);
    s.ps = be16(key + 0x32); s.yn1 = (s16)be16(key + 0x34); s.yn2 = (s16)be16(key + 0x36);
    s.is_stream = t != 0; s.stopped = 0;
    s16 lp[3] = { (s16)be16(key + 0x46), (s16)be16(key + 0x48), (s16)be16(key + 0x4a) };
    for (int k = 0; k < 4; k++) e->h[k] = (s16)be16(key + 0x3e + k * 2);
    decode_track(&s, A, e->h + 4, (u32)total, lp, &e->ev);
    e->frame = f; e->type = t; e->total = (u32)total;
    memcpy(e->key, key, BLK_N); memcpy(e->next, key, BLK_N);
    put16(e->next + 12, s.cur >> 16); put16(e->next + 14, s.cur);
    put16(e->next + 0x32, s.ps); put16(e->next + 0x34, (u16)s.yn1); put16(e->next + 0x36, (u16)s.yn2);
    put16(e->next + 0x3c, (u32)(frac + (u64)ratio * FRAME));
    for (int k = 0; k < 4; k++) put16(e->next + 0x3e + k * 2, (u16)e->h[total + k]);
    __dmb();
    e->seq++;
    return 1;
}

/* worker, after job j: keep SPEC_K frames predicted for each voice still playing; a chain that matches the
 * block the frame really left is extended, any other starts again from that block */
static void spec_extend(const FJob *j) {
    const u8 *A = recomp_aram();
    if (!A) return;
    for (u32 p = 0; p < j->npost; p++) {
        u32 i = j->post[p].vi, f = j->fi + 1, last = j->fi + SPEC_K;
        u16 t = j->post[p].type;
        const u8 *key = j->post[p].blk;
        const SpecEnt *e = &spec[i][f % SPEC_R];
        if (spec_last[i] >= f && e->frame == f && e->type == t && !memcmp(e->key, key, BLK_N)) {
            const SpecEnt *le = &spec[i][spec_last[i] % SPEC_R];   /* the chain holds: go on from its last prediction */
            if (le->ev.stopped) continue;
            f = spec_last[i] + 1;
            key = le->next;
        }
        for (; f <= last; f++) {
            SpecEnt *n = &spec[i][f % SPEC_R];
            u8 k[BLK_N];
            memcpy(k, key, BLK_N);
            if (!spec_make(n, f, k, t, A)) { spec_last[i] = 0; break; }
            spec_last[i] = f;
            if (n->ev.stopped) break;   /* nothing to predict after a stop */
            key = n->next;
        }
    }
}

static void mix_voice(u32 v, s32 *outL, s32 *outR);

static void slow_voice(u32 v, FJob *j) {
    if (!j->fb) { memset(j->fbL, 0, sizeof j->fbL); memset(j->fbR, 0, sizeof j->fbR); j->fb = 1; }
    mix_voice(v, j->fbL, j->fbR);
}

/* game thread: the prediction for voice i's frame fi if it still holds: its samples copied to h, its
 * events and resulting state to *ev / *s (all copied before the slot's sequence is checked again) */
static int spec_take(u32 i, u32 v, u32 fi, s16 *h, u32 room, DecEv *ev, Src *s) {
    const SpecEnt *e = &spec[i][fi % SPEC_R];
    u32 q = e->seq;
    __dmb();
    if ((q & 1) || e->frame != fi || e->total + 4 > room) return 0;
    if (e->type != rd16(v + P_TYPE) || memcmp(e->key, ram + ((v + BLK) & RAM_MASK), BLK_N)) return 0;
    *ev = e->ev;
    if (!aram_clean(e->gen, ev->alo, ev->ahi)) return 0;
    memcpy(h, e->h, (e->total + 4) * sizeof *h);
    s->cur = be32(e->next + 12); s->ps = be16(e->next + 0x32);
    s->yn1 = (s16)be16(e->next + 0x34); s->yn2 = (s16)be16(e->next + 0x36);
    s->stopped = ev->stopped;
    u32 tot = e->total;
    __dmb();
    return e->seq == q && tot + 4 <= room;
}

/* the game thread's part of a voice's frame: its source samples into the job, its PB state as after the
 * frame (as mix_voice leaves it) */
static void voice_frame(u32 i, u32 v, FJob *j) {
#ifdef RECOMP_AX_REF
    (void)i;
    slow_voice(v, j);
#else
    const u8 *A = recomp_aram();
    u32 ratio = rd32(v + P_SRC);
    u32 frac = rd16(v + P_SRC + 4);
    u64 total = ((u64)frac + (u64)ratio * FRAME) >> 16;
    if (!A || total + 4 > POOL - j->used) { slow_voice(v, j); return; }
    Src s;
    DecEv ev;
    VJob *vj = &j->v[j->nv++];
    s16 *h = j->pool + j->used;               /* last[0..3], then the samples fetched this frame */
    vj->off = j->used; j->used += (u32)total + 4;
    int hit = 0;
    if (spec_on && job_worker == 1) { spec_tries++; hit = spec_take(i, v, j->fi, h, POOL - vj->off, &ev, &s); }
    if (hit) spec_hits++;   /* the samples are in h, the state the decode would leave in s */
    else {
        s.v = v;
        s.looping = rd16(v + P_ADDR);
        s.fmt = rd16(v + P_ADDR + 2);
        s.loop = rd32(v + P_ADDR + 4);
        s.end = rd32(v + P_ADDR + 8);
        s.cur = rd32(v + P_ADDR + 12);
        for (int k = 0; k < 16; k++) s.coef[k] = rd16(v + P_ADPCM + k * 2);
        s.ps = rd16(v + P_ADPCM + 0x22);
        s.yn1 = rd16(v + P_ADPCM + 0x24);
        s.yn2 = rd16(v + P_ADPCM + 0x26);
        s.is_stream = rd16(v + P_TYPE) != 0;
        s.stopped = 0;
        s16 lp[3] = { (s16)rd16(v + P_ADPCMLOOP), (s16)rd16(v + P_ADPCMLOOP + 2), (s16)rd16(v + P_ADPCMLOOP + 4) };
        for (int k = 0; k < 4; k++) h[k] = rd16(v + P_SRC + 6 + k * 2);
        decode(&s, A, h + 4, (u32)total, lp, &ev);
    }
    st.looped += ev.looped;
    if (ev.stopped) { st.ended++; vlog("ended", v, ev.stop_cur); }
    vj->ratio = ratio; vj->frac = frac;
    vj->ve = rd16(v + P_VE); vj->ved = rd16(v + P_VE + 2);
    vj->vl = rd16(v + P_MIX); vj->vr = rd16(v + P_MIX + 4);
    vj->dl = rd16(v + P_MIX + 2); vj->dr = rd16(v + P_MIX + 6);
    /* after FRAME steps of "v += d, clamped to 0..0xffff" from 0 <= v <= 0xffff: one clamp of the sum
     * (the sequence is monotone, so it stays at a bound once it reaches it) */
    s32 ve = vj->ve + FRAME * vj->ved, vl = vj->vl + FRAME * vj->dl, vr = vj->vr + FRAME * vj->dr;
    ve = ve < 0 ? 0 : ve > 0xffff ? 0xffff : ve;
    vl = vl < 0 ? 0 : vl > 0xffff ? 0xffff : vl;
    vr = vr < 0 ? 0 : vr > 0xffff ? 0xffff : vr;

    if (s.stopped) wr16(v + P_STATE, 0);
    wr16(v + P_ADDR + 12, s.cur >> 16); wr16(v + P_ADDR + 14, s.cur);
    wr16(v + P_ADPCM + 0x22, s.ps); wr16(v + P_ADPCM + 0x24, s.yn1); wr16(v + P_ADPCM + 0x26, s.yn2);
    wr16(v + P_SRC + 4, (u32)(frac + (u64)ratio * FRAME));
    for (int k = 0; k < 4; k++) wr16(v + P_SRC + 6 + k * 2, h[total + k]);
    wr16(v + P_VE, ve);
    wr16(v + P_MIX, vl); wr16(v + P_MIX + 4, vr);
    if (spec_on && job_worker == 1 && !s.stopped) {   /* the block this frame left, for the worker's predictions */
        u32 n = j->npost++;
        j->post[n].vi = (u8)i; j->post[n].type = rd16(v + P_TYPE);
        memcpy(j->post[n].blk, ram + ((v + BLK) & RAM_MASK), BLK_N);
    }
#endif
}

/* the job's part: mix_voice's sum, from the decoded samples */
static void job_mix(const FJob *j) {
    if (j->fb) { memcpy(mixL, j->fbL, sizeof mixL); memcpy(mixR, j->fbR, sizeof mixR); }
    else { memset(mixL, 0, sizeof mixL); memset(mixR, 0, sizeof mixR); }
    for (u32 n = 0; n < j->nv; n++) {
        const VJob *vj = &j->v[n];
        const s16 *h = j->pool + vj->off;
        u32 ratio = vj->ratio, f = vj->frac, k = 0;
        s32 ve = vj->ve, vl = vj->vl, vr = vj->vr;
        s32 ved = vj->ved, dl = vj->dl, dr = vj->dr;
        if (ved == 0 && dl == 0 && dr == 0) {     /* constant volumes: the clamps leave them as they are */
            if ((vl == 0 && vr == 0) || ve == 0) continue;   /* adds 0 to every sample */
            for (int i = 0; i < FRAME; i++) {
                f += ratio; k += f >> 16; f &= 0xffff;
                s32 a = h[k + 2], b = h[k + 3];
                s32 x = a + (((s32)(b - a) * (s32)(f >> 1)) >> 15);
                x = (x * ve) >> 15;
                mixL[i] += (x * vl) >> 15;
                mixR[i] += (x * vr) >> 15;
            }
        } else for (int i = 0; i < FRAME; i++) {
            f += ratio; k += f >> 16; f &= 0xffff;
            s32 a = h[k + 2], b = h[k + 3];
            s32 x = a + (((s32)(b - a) * (s32)(f >> 1)) >> 15);
            x = (x * ve) >> 15;
            mixL[i] += (x * vl) >> 15;
            mixR[i] += (x * vr) >> 15;
            ve += ved; ve = ve < 0 ? 0 : ve > 0xffff ? 0xffff : ve;
            vl += dl; vr += dr;
            vl = vl < 0 ? 0 : vl > 0xffff ? 0xffff : vl;
            vr = vr < 0 ? 0 : vr > 0xffff ? 0xffff : vr;
        }
    }
}

static void mix_voice(u32 v, s32 *outL, s32 *outR) {
    Src s;
    s.v = v;
    s.looping = rd16(v + P_ADDR);
    s.fmt = rd16(v + P_ADDR + 2);
    s.loop = rd32(v + P_ADDR + 4);
    s.end = rd32(v + P_ADDR + 8);
    s.cur = rd32(v + P_ADDR + 12);
    for (int k = 0; k < 16; k++) s.coef[k] = rd16(v + P_ADPCM + k * 2);
    s.ps = rd16(v + P_ADPCM + 0x22);
    s.yn1 = rd16(v + P_ADPCM + 0x24);
    s.yn2 = rd16(v + P_ADPCM + 0x26);
    s.is_stream = rd16(v + P_TYPE) != 0;
    s.stopped = 0;

    u32 ratio = rd32(v + P_SRC);
    u32 frac = rd16(v + P_SRC + 4);
    s16 last[4];
    for (int k = 0; k < 4; k++) last[k] = rd16(v + P_SRC + 6 + k * 2);
    s32 ve = rd16(v + P_VE);
    s16 ved = rd16(v + P_VE + 2);
    s32 vl = rd16(v + P_MIX), vr = rd16(v + P_MIX + 4);
    s16 dl = rd16(v + P_MIX + 2), dr = rd16(v + P_MIX + 6);

    for (int i = 0; i < FRAME; i++) {
        frac += ratio;
        while (frac >= 0x10000) {             /* linear interpolation between last[2] and last[3] */
            frac -= 0x10000;
            last[0] = last[1]; last[1] = last[2]; last[2] = last[3];
            last[3] = fetch(&s);
        }
        s32 x = last[2] + (((s32)(last[3] - last[2]) * (s32)(frac >> 1)) >> 15);
        x = (x * ve) >> 15;
        outL[i] += (x * vl) >> 15;
        outR[i] += (x * vr) >> 15;
        ve += ved; ve = ve < 0 ? 0 : ve > 0xffff ? 0xffff : ve;
        vl += dl; vr += dr;
        vl = vl < 0 ? 0 : vl > 0xffff ? 0xffff : vl;
        vr = vr < 0 ? 0 : vr > 0xffff ? 0xffff : vr;
    }

    if (s.stopped) wr16(v + P_STATE, 0);
    wr16(v + P_ADDR + 12, s.cur >> 16); wr16(v + P_ADDR + 14, s.cur);
    wr16(v + P_ADPCM + 0x22, s.ps); wr16(v + P_ADPCM + 0x24, s.yn1); wr16(v + P_ADPCM + 0x26, s.yn2);
    wr16(v + P_SRC + 4, frac);
    for (int k = 0; k < 4; k++) wr16(v + P_SRC + 6 + k * 2, last[k]);
    wr16(v + P_VE, ve);
    wr16(v + P_MIX, vl); wr16(v + P_MIX + 4, vr);
}

/* ---- output: ndsp + a WAV dump of the first WAV_SECONDS for checking ---- */

#define NBUF 32
#define BUF_FRAMES 4               /* 20 ms per ndsp buffer; 640 ms ring: real-time pacing mixes a whole game frame at once, ring >= 2 * MAX_DUE (lead + burst) */
#define WAV_SECONDS 60
static int ndsp_ok = -1;
static ndspWaveBuf wbuf[NBUF];
static s16 *wdata;
static int wfill, wcur, wqueued;
static FILE *wav;
static u32 wav_samples;

static void out_init(void) {
    Result r = ndspInit();
    ndsp_ok = R_SUCCEEDED(r);
    recomp_log("# AX: ndspInit %08lx%s\n", (unsigned long)r, ndsp_ok ? "" : " (no sdmc:/3ds/dspfirm.cdc? mixing to the WAV only)");
    if (ndsp_ok) {
        ndspSetOutputMode(NDSP_OUTPUT_STEREO);
        ndspChnReset(0);
        ndspChnSetInterp(0, NDSP_INTERP_LINEAR);
        ndspChnSetRate(0, 32000.0f);
        ndspChnSetFormat(0, NDSP_FORMAT_STEREO_PCM16);
        float mix[12] = { 1.0f, 1.0f };
        ndspChnSetMix(0, mix);
        wdata = linearAlloc(NBUF * BUF_FRAMES * FRAME * 4);
        memset(wbuf, 0, sizeof wbuf);
        for (int i = 0; i < NBUF; i++) {
            wbuf[i].data_vaddr = wdata + i * BUF_FRAMES * FRAME * 2;
            wbuf[i].nsamples = BUF_FRAMES * FRAME;
            wbuf[i].status = NDSP_WBUF_DONE;
        }
    }
    /* opt-in (autotest.ps1 writes the flag): on hardware, small SD writes every
     * 5 ms cost ~170 ms per retrace with multi-second stalls */
    FILE *flag = fopen("sdmc:/cr_audio_dump.txt", "r");
    if (flag) {
        fclose(flag);
        wav = fopen("sdmc:/cr_audio.wav", "wb");
        if (wav) { static u8 hdr[44]; setvbuf(wav, NULL, _IOFBF, 256 * 1024); fwrite(hdr, 1, 44, wav); }
    }
}

static void wav_close(void) {
    if (!wav) return;
    u32 bytes = wav_samples * 4;
    u8 h[44] = "RIFF....WAVEfmt \x10\0\0\0\x01\0\x02\0\0}\0\0\0\xf4\x01\0\x04\0\x10\0data....";
    u32 riff = bytes + 36;
    memcpy(h + 4, &riff, 4); memcpy(h + 40, &bytes, 4);
    fseek(wav, 0, SEEK_SET); fwrite(h, 1, 44, wav); fclose(wav); wav = NULL;
}

static void out_frame(void) {
    s16 pcm[FRAME * 2];
    for (int i = 0; i < FRAME; i++) {
        s16 l = clamp16(mixL[i]), r = clamp16(mixR[i]);
        pcm[i * 2] = l; pcm[i * 2 + 1] = r;
        u32 a = l < 0 ? -l : l; if (a > st.peak) st.peak = a;
        st.sumsq += (s32)l * l;
        st.hash = (st.hash * 31 + (u16)l) * 31 + (u16)r;
    }
    if (wav) {
        fwrite(pcm, 2, FRAME * 2, wav);     /* little-endian host = WAV order */
        if ((wav_samples += FRAME) >= 32000u * WAV_SECONDS) wav_close();
    }
    if (ndsp_ok != 1) return;
    ndspWaveBuf *b = &wbuf[wcur];
    if (b->status != NDSP_WBUF_DONE && b->status != NDSP_WBUF_FREE) { st.dropped++; return; }   /* output behind: drop */
    memcpy((s16 *)b->data_vaddr + wfill * FRAME * 2, pcm, sizeof pcm);
    if (++wfill == BUF_FRAMES) {
        DSP_FlushDataCache(b->data_vaddr, BUF_FRAMES * FRAME * 4);
        ndspChnWaveBufAdd(0, b); st.queued++; wqueued = 1;
        wfill = 0;
        wcur = (wcur + 1) % NBUF;
    }
}

/* ---- the job ring: the game thread fills jobs[job_w % NJOB], the worker (core 2) mixes and outputs
 * them in order; job_r = jobs done ---- */
#define NJOB 16
static FJob jobs[NJOB];
static volatile u32 job_w, job_r;
static volatile int job_sleeping, job_waiting, job_quit;
static LightEvent job_ev, job_done_ev;
static int job_worker = -1;        /* -1 not started yet, 0 none (jobs run at once), 1 worker thread */

/* the job's output: the 5 s line (before this frame's output, as it always was), ndsp and the WAV */
static void job_out(const FJob *j) {
    if (j->report) {
        recomp_log("# AX: frame %lu voices started %lu max active %lu ended %lu loops %lu peak %lu ndsp bufs %lu dropped frames %lu rms %lu, hash %08lx; pacing %s, underruns %lu, skipped %lu, trimmed %lu; decode predictions %lu of %lu\n",
                   (unsigned long)j->frames, (unsigned long)j->started, (unsigned long)j->active_max,
                   (unsigned long)j->ended, (unsigned long)j->looped, (unsigned long)st.peak, (unsigned long)st.queued, (unsigned long)st.dropped,
                   (unsigned long)__builtin_sqrt((double)st.sumsq / (FRAME * 1000)), (unsigned long)st.hash, j->rt ? "rt" : "retrace",
                   (unsigned long)j->underruns, (unsigned long)j->skipped, (unsigned long)j->trimmed,
                   (unsigned long)j->spec_hits, (unsigned long)j->spec_tries);
        st.peak = st.queued = st.dropped = 0; st.sumsq = 0;
    }
    out_frame();
}

static void job_worker_fn(void *arg) {
    (void)arg;
    for (;;) {
        u32 r = job_r;
        while (job_w == r) {
            if (job_quit) return;   /* recomp_ax_close: every job done */
            job_sleeping = 1;
            __dmb();
            if (job_w == r) LightEvent_WaitTimeout(&job_ev, 2000000);
            job_sleeping = 0;
        }
        __dmb();   /* the job's contents before using them */
        const FJob *j = &jobs[r % NJOB];
        job_mix(j);
        job_out(j);
        if (spec_on) spec_extend(j);
        __dmb();   /* done with the job before the game thread may refill it */
        job_r = r + 1;
        __dmb();
        if (job_waiting) LightEvent_Signal(&job_done_ev);
    }
}

/* wait until the worker has done every job before job number n */
static void job_drain(u32 n) {
    while ((s32)(job_r - n) < 0) {
        job_waiting = 1;
        __dmb();
        if ((s32)(job_r - n) < 0) LightEvent_WaitTimeout(&job_done_ev, 1000000);
        job_waiting = 0;
    }
    __dmb();
}

static void job_start(void) {
#ifndef RECOMP_AX_NO_WORKER
    if (recomp_aux_core() == 2) {
        LightEvent_Init(&job_ev, RESET_ONESHOT);
        LightEvent_Init(&job_done_ev, RESET_ONESHOT);
        s32 prio = 0x30;
        svcGetThreadPriority(&prio, CUR_THREAD_HANDLE);
        job_worker = recomp_aux_thread(job_worker_fn, 0x4000, prio + 1) != NULL;   /* ahead of the GX thread (prio + 2): a job is short, and a full ring stops the game thread */
    } else
#endif
        job_worker = 0;
    recomp_log("# AX: mix jobs %s\n", job_worker ? "on a worker thread (core 2)" : "run at once (no worker: AXWORKER=0 or no core 2)");
}

/* the game thread's part of one AX frame: consume the sync flags, decode and update every playing voice
 * into the next job (waiting for a free one) */
static FJob *ax_frame(void) {
    if (job_worker == 1 && job_w - job_r >= NJOB) job_drain(job_w - NJOB + 1);
    FJob *j = &jobs[job_w % NJOB];
    j->nv = j->used = j->fb = 0;
    j->fi = st.frames; j->npost = 0;
    u32 active = 0;
    for (u32 i = 0; i < VPB_COUNT; i++) {
        u32 v = VPB_BASE + i * VPB_SIZE;
        if (!vpb_used[i]) continue;
        wr32(v + V_SYNC, 0);                /* user changes consumed (__AXSyncPBs) */
        if (rd16(v + P_STATE) != 1) continue;
        active++;
        voice_frame(i, v, j);
    }
    if (active > st.active_max) st.active_max = active;
    j->report = ++st.frames % 1000 == 0;   /* every 5 s of audio */
    if (j->report) {
        j->frames = st.frames; j->started = voices_started; j->active_max = st.active_max; j->ended = st.ended;
        j->looped = st.looped; j->underruns = st.underruns; j->skipped = st.skipped; j->trimmed = st.trimmed;
        j->rt = recomp_ax_rt; j->spec_hits = spec_hits; j->spec_tries = spec_tries;
        st.active_max = st.ended = st.looped = st.underruns = st.skipped = st.trimmed = 0;
        spec_hits = spec_tries = 0;
    }
    return j;
}

/* ---- AI DMA chain (runtime/ai.c): while the game has its own AI callback in front of
 * AX's (the THP movie), each frame runs the chain and plays the buffer it leaves ---- */
#define AX_OUT 0x81808000u                 /* 3 AX output buffers, after the VPBs */
static u32 out_idx;

/* 0x80145a00 __AXOutAiCallback, reached through the guest's AI callback: the frame just
 * mixed becomes the next DMA buffer, big-endian R, L pairs (Dolphin's HLE AX output order) */
void sdk___AXOutAiCallback(void) {
    LOG();
    u32 b = AX_OUT + out_idx * FRAME * 4;
    out_idx = (out_idx + 1) % 3;
    for (int i = 0; i < FRAME; i++) { wr16(b + i * 4, clamp16(mixR[i])); wr16(b + i * 4 + 2, clamp16(mixL[i])); }
    recomp_ai_dma_addr = b & 0x03ffffe0u; recomp_ai_dma_len = FRAME * 4;
}

static void ai_frame(void) {
    recomp_call_guest(recomp_ai_dma_cb, 0);
    u32 a = 0x80000000u | recomp_ai_dma_addr, n = recomp_ai_dma_len / 4 < FRAME ? recomp_ai_dma_len / 4 : FRAME;
    for (u32 i = 0; i < FRAME; i++) {
        mixR[i] = i < n ? (s16)rd16(a + i * 4) : 0;
        mixL[i] = i < n ? (s16)rd16(a + i * 4 + 2) : 0;
    }
}

/* ---- pacing ----
 * Retrace pacing (autotests, sdmc:/cr_audio_rt_off.txt): 10 AX frames per 3 retraces, so
 * a replay runs the same callbacks at the same retraces. Below 60 fps that is less audio
 * than ndsp plays (34.9 fps battle: 58%), so the sound stutters.
 * Real-time pacing (interactive play; sdmc:/cr_audio_rt_on.txt forces it in autotests):
 * the AX frames due are the host time since the last retrace, 200 per second, mixed at
 * once at the wait point. The ndsp queue must then cover the longest gap between retraces:
 * it starts PREBUF frames ahead, each underrun (ndsp ran dry) adds PREBUF more, and a lead
 * that stayed above SLACK for ~1 s is trimmed one frame per retrace (one frame fewer mixed:
 * the sound driver's clock slips 5 ms, nothing is cut). At most MAX_DUE frames per retrace:
 * the rest of a longer stall (HOME menu, SD stall) is skipped. */
#define PREBUF 8                   /* 40 ms */
#define SLACK 8                    /* lead kept above the worst gap (the playing buffer counts whole: up to 4 too many) */
#define MAX_DUE 48                 /* 240 ms per retrace (movie on hardware: ~111 ms frames) */
int recomp_ax_rt = -1;
u32 recomp_ax_flips, recomp_ax_underruns, recomp_ax_skipped;
static int rt_req;
static u64 rt_last, rt_acc;        /* rt_acc: ticks * 200, one AX frame = SYSCLOCK_ARM11 */
static u32 rt_acc3;                /* retrace pacing: 10 per retrace, 3 per frame */
static u32 lead_min = ~0u, lead_t, trim;

static int flag(const char *p) { FILE *f = fopen(p, "rb"); if (f) fclose(f); return f != NULL; }

void recomp_ax_pacing_init(void) {
    if (recomp_ax_rt >= 0) return;
    const char *why;
    if (flag("sdmc:/cr_audio_rt_off.txt")) { recomp_ax_rt = 0; why = "cr_audio_rt_off"; }
    else if (flag("sdmc:/cr_audio_rt_on.txt")) { recomp_ax_rt = 1; why = "cr_audio_rt_on"; }
    else { recomp_ax_rt = !recomp_autotest_run(); why = recomp_ax_rt ? "interactive" : "autotest"; }
    recomp_log("# AX: pacing %s (%s)\n", recomp_ax_rt ? "real-time" : "per retrace", why);
    printf("audio %s (%s)\n", recomp_ax_rt ? "real-time" : "per retrace", why);   /* bottom screen */
}

void recomp_ax_pacing_toggle(void) { rt_req = 1; }

/* AX frames queued in ndsp or mixed into the unqueued buffer (the playing one counts whole) */
static u32 ndsp_lead(void) {
    u32 n = wfill + (job_w - job_r);
    for (int i = 0; i < NBUF; i++) if (wbuf[i].status == NDSP_WBUF_QUEUED || wbuf[i].status == NDSP_WBUF_PLAYING) n += BUF_FRAMES;
    return n;
}

static void rt_start(u64 now) { rt_last = now; rt_acc = (u64)PREBUF * SYSCLOCK_ARM11; lead_min = ~0u; lead_t = 0; trim = 0; }

/* sdmc:/cr_ax_stall.txt = "<retrace> <ms> [sleep]" (Citra test, no button): a stall of <ms>
 * before that retrace's AX frames, like a HOME/Rosalina pause: added to the real-time pacing
 * clock, or with "sleep" the game thread really sleeps (OS time moves too). Real-time pacing
 * then skips all but MAX_DUE frames; hw run 30's snr_t_06 hang does not come from that
 * (recomp_unknowns.md "Run 30 text hang") */
static u32 stall_r = ~0u, stall_ms, stall_sleep, stall_read;
static void stall_init(void) {
    stall_read = 1;
    FILE *f = fopen("sdmc:/cr_ax_stall.txt", "rb");
    if (!f) return;
    char mode[16] = "";
    unsigned long r, ms;
    if (fscanf(f, "%lu %lu %15s", &r, &ms, mode) >= 2) { stall_r = r; stall_ms = ms; stall_sleep = !strcmp(mode, "sleep"); }
    fclose(f);
    recomp_log("# AX: injected stall at r%lu: %lu ms (%s)\n", (unsigned long)stall_r, (unsigned long)stall_ms, stall_sleep ? "sleep" : "pacing clock");
}

void recomp_ax_service(void) {
    u64 t0 = recomp_ticks();
    if (!stall_read) stall_init();
    if (recomp_retrace_count() == stall_r) {
        stall_r = ~0u;
        recomp_log("# AX r%lu: injected stall %lu ms\n", (unsigned long)recomp_retrace_count(), (unsigned long)stall_ms);
        if (stall_sleep) { svcSleepThread((s64)stall_ms * 1000000); t0 = recomp_ticks(); }
        else if (rt_last) rt_last -= (u64)stall_ms * (SYSCLOCK_ARM11 / 1000);
    }
    if (ndsp_ok < 0) out_init();
    if (recomp_ax_rt < 0) recomp_ax_pacing_init();
    if (rt_req) {                        /* touch screen (runtime/pad.c), applied between retraces */
        rt_req = 0; recomp_ax_rt ^= 1; recomp_ax_flips++;
        if (recomp_ax_rt) rt_start(t0); else rt_acc3 = 0;
        recomp_log("# AX r%lu: pacing %s (touch)\n", (unsigned long)recomp_retrace_count(), recomp_ax_rt ? "real-time" : "per retrace");
        printf("r%lu audio %s\n", (unsigned long)recomp_retrace_count(), recomp_ax_rt ? "real-time" : "per retrace");
    }
    u32 lead = ndsp_ok == 1 ? ndsp_lead() : ~0u;
    int dry = wqueued && lead == 0;
    if (dry) { st.underruns++; recomp_ax_underruns++; }
    u32 due;
    if (recomp_ax_rt) {
        if (!rt_last) rt_start(t0);
        rt_acc += (t0 - rt_last) * 200; rt_last = t0;
        if (dry) rt_acc += (u64)PREBUF * SYSCLOCK_ARM11;
        u64 n = rt_acc / SYSCLOCK_ARM11;
        rt_acc -= n * SYSCLOCK_ARM11;
        if (n > MAX_DUE) { st.skipped += n - MAX_DUE; recomp_ax_skipped += n - MAX_DUE; n = MAX_DUE; }
        /* latency trim: the lowest lead seen before mixing over ~1 s */
        if (lead != ~0u && !dry) {
            if (lead < lead_min) lead_min = lead;
            if ((lead_t += (u32)n) >= 200) {
                trim = lead_min > SLACK ? lead_min - SLACK : 0;
                lead_min = ~0u; lead_t = 0;
            }
        }
        if (trim && n) { n--; trim--; st.trimmed++; }
        due = (u32)n;
    } else {
        rt_acc3 += 10;
        due = rt_acc3 / 3;
        rt_acc3 -= due * 3;
        rt_last = 0;
    }
    if (job_worker < 0) job_start();
    for (u32 k = 0; k < due; k++) {
        FJob *j = ax_frame();
        if (recomp_ai_dma_cb && recomp_ai_dma_cb != AX_AI_CALLBACK) {   /* the movie's AI chain reads this frame's mix now */
            job_drain(job_w);
            job_mix(j);
            u64 c0 = recomp_ticks(); ai_frame(); recomp_prof[PROF_AUDIO_CB] += recomp_ticks() - c0;
            job_out(j);
        } else if (job_worker == 1) {
            __dmb();   /* the job's contents before its index */
            job_w++;
            __dmb();
            if (job_sleeping) LightEvent_Signal(&job_ev);
        } else { job_mix(j); job_out(j); }
        if (ax_cb) { u64 c0 = recomp_ticks(); recomp_call_guest(ax_cb, 0); recomp_prof[PROF_AUDIO_CB] += recomp_ticks() - c0; }
    }
    recomp_prof[PROF_AUDIO] += recomp_ticks() - t0;
}

void recomp_ax_close(void) {
    if (job_worker == 1) {   /* the worker ends once every job is out, before ndsp and the log close */
        job_drain(job_w);
        job_quit = 1;
        __dmb();
        LightEvent_Signal(&job_ev);
    }
    wav_close();
    if (ndsp_ok == 1) { ndspExit(); ndsp_ok = 0; }
}
