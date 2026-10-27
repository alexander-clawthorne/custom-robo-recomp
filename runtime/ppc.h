/* Gekko CPU state and helpers used by the translated code in gen/.
 * Semantics follow Dolphin's interpreter (the oracle); deviations are listed
 * in docs/recomp_unknowns.md. */
#pragma once
#include <stdint.h>
#include <string.h>
#include <math.h>

typedef uint8_t u8; typedef uint16_t u16; typedef uint32_t u32; typedef uint64_t u64;
typedef int8_t s8; typedef int16_t s16; typedef int32_t s32; typedef int64_t s64;

typedef struct { double ps0, ps1; } PS;

typedef struct {
    u32 gpr[32];
    PS  fpr[32];
    u32 cr;                 /* PPC order: cr0 in bits 31..28 */
    u32 lr, ctr;
    u32 xer_so, xer_ov, xer_ca, xer_bc;
    u32 fpscr;
    u32 gqr[8];
    u32 msr;
    u32 spr[1024];          /* everything not given its own field */
} CPU;

extern CPU cpu;
extern u8 ram[];            /* 0x80000000 mirror, RAM_MASK+1 bytes (+8 pad) */
#define RAM_MASK 0x01FFFFFFu
#define STACK_TOP 0x80462ac0u     /* initial r1 (crt0 table) = linker __ArenaLo */
/* the 16 KB locked cache (0xE0000000 on the GameCube) lives in RAM above the guest's 24 MB:
 * THPInit hands the THP decoders this address instead (runtime/thp.c, os.c LC*) */
#define LC_BASE 0x81E00000u

#define R cpu.gpr
#define F cpu.fpr

/* ---- memory (big-endian guest) ---- */
u32  hw_read(u32 a, int size);
void hw_write(u32 a, u32 v, int size);
#define IS_HW(a) (((a) & 0xFE000000u) == 0xCC000000u)
/* Loads and the hardware range (make HWCHECK=..): by default a guest load does not check for a hardware
 * address and reads RAM through the mask. No translated load has reached a hardware register in any log
 * (Citra replays, hardware runs 31-34), hw_read returns 0 anyway (register reads are not emulated), and the
 * check cost ~3 hot ARM instructions per load plus a never-run hw_read call (2026-10-26, unknowns "Loads
 * without the hardware check"). HWCHECK=1 (RECOMP_HWCHECK) = the old checked loads; HWCHECK=verify
 * (RECOMP_HWCHECK_VERIFY) = checked loads that log each hardware load ("# hwcheck load", cpu.c), to prove a
 * replay never makes one. Stores keep their check: the GX write-gather pipe is a hardware address. */
#if defined(RECOMP_HWCHECK) || defined(RECOMP_HWCHECK_VERIFY)
#define LOAD_HW(a) __builtin_expect(IS_HW(a), 0)
#else
#define LOAD_HW(a) 0
#endif
/* Stores (D3, 2026-10-26): a gen/ file that recomp.py writes with RECOMP_STORE_FAST (every function that never sets
 * a hardware address, 0xCC/0xCD......) only takes the hardware path for an address GCC knows to be hardware; its
 * other stores write RAM without the test. The functions that set one (lis 0xCC01 for the GX write-gather pipe;
 * recomp_hw_*.c), runtime/ and `make STORECHECK=1` / HWCHECK=1 / HWCHECK=verify test every store. No replay
 * or hardware log has had a store reach hw_write (make HWCHECK=verify counts them per site). */
#if defined(RECOMP_STORECHECK) || defined(RECOMP_HWCHECK) || defined(RECOMP_HWCHECK_VERIFY)
#define RECOMP_STORE_MODE "checked in every function"
#else
#define RECOMP_STORE_MODE "checked only in functions that set a hardware address (recomp_hw_*.c)"
#endif
#if defined(RECOMP_STORE_FAST) && !defined(RECOMP_STORECHECK) && !defined(RECOMP_HWCHECK) && !defined(RECOMP_HWCHECK_VERIFY)
#define STORE_HW(a) (__builtin_constant_p(a) && IS_HW(a))
#else
#define STORE_HW(a) __builtin_expect(IS_HW(a), 0)
#endif
#ifdef RECOMP_HWCHECK_VERIFY
u32 hw_read_verify(u32 a, int size);
#define HW_READ(a, n) hw_read_verify(a, n)
#define RECOMP_HWCHECK_MODE "verify (loads checked, each hardware load logged)"
#elif defined(RECOMP_HWCHECK)
#define HW_READ(a, n) hw_read(a, n)
#define RECOMP_HWCHECK_MODE "checked (HWCHECK=1)"
#else
#define HW_READ(a, n) hw_read(a, n)
#define RECOMP_HWCHECK_MODE "unchecked (default)"
#endif
/* guest-write epochs per 1 KB page (wr8/16/32 and the runtime's bulk writers): the GX display-list cache
 * skips re-hashing a list/array range whose pages were not written since it was last hashed. Stores
 * through r1/r11 (wr*_ram) are not marked; the cache hashes ranges near the stack.
 * A guest store sets its page's byte in ram_dirty[] to 1 (one strb with the page index as a shifted
 * register offset, no epoch load); recomp_wr_sweep (sdk_GXCopyDisp, end of each game frame) stamps the
 * marked pages with the frame's epoch in ram_wr[] and clears the marks. A marked page counts as written
 * in the current epoch (gx.c range_clean). `make WRMAP=0`: stores write the epoch into ram_wr[] directly
 * (the old way, for an A/B). Bulk writers (recomp_mark_ram) always write ram_wr[]. */
#define WR_PAGE_SHIFT 10
extern u32 ram_wr[];   /* (RAM_MASK + 1) >> WR_PAGE_SHIFT entries, guard space around them (cpu.c) */
extern u8 ram_dirty[];   /* same indices, same guard */
extern u32 recomp_wr_epoch;
#ifdef RECOMP_WR_EPOCH_STORES
#define WR_SET(i) (ram_wr[i] = recomp_wr_epoch)
#define RECOMP_WRMAP_MODE "stores write the epoch (WRMAP=0)"
#else
#define WR_SET(i) (ram_dirty[i] = 1)
#define RECOMP_WRMAP_MODE "stores mark a byte map, swept per frame (default)"
#endif
#define WR_MARK(a) WR_SET(((a) & RAM_MASK) >> WR_PAGE_SHIFT)
void recomp_wr_sweep(void);   /* game thread, before the epoch advances */
void recomp_align_report(u32 retrace);   /* cpu.c, make ALIGNVERIFY=1 */
void recomp_mark_ram(u32 a, u32 n);   /* bulk writers */
void recomp_gx_fifo_write(u32 v, int size); /* WGPIPE 0xCC008000 */
/* hw_write for an address GCC knows: WGPIPE stores (GXPosition3f32 & co. inlined in game code, lis
 * 0xCC01 / -0x8000) call the FIFO directly, as hw_write would */
#define HW_WRITE(a, v, n) ((__builtin_constant_p(a) && ((a) & ~7u) == 0xCC008000u) ? gx_fifo_put(v, n) : hw_write(a, v, n))
/* WGPIPE store inline: the bytes go straight into gx.c's FIFO (big-endian) while the primitive GXBegin
 * started is still incomplete after them (recomp_fifo_lim, gx.c fifo_lim); the store that completes it
 * (or any other) goes through recomp_gx_fifo_write, which flushes */
extern unsigned char recomp_fifo[];
extern u32 recomp_fifo_n, recomp_fifo_lim;
static inline __attribute__((always_inline)) void gx_fifo_put(u32 v, int n) {
    u32 f = recomp_fifo_n;
    if (__builtin_expect(f + n < recomp_fifo_lim, 1)) {
        if (n == 4) { u32 b = __builtin_bswap32(v); memcpy(recomp_fifo + f, &b, 4); }
        else if (n == 2) { u16 b = __builtin_bswap16((u16)v); memcpy(recomp_fifo + f, &b, 2); }
        else recomp_fifo[f] = (u8)v;
        recomp_fifo_n = f + n;
    } else recomp_gx_fifo_write(v, n);
}

/* forced inline: in gen/'s large functions GCC stopped inlining these (37.5k out-of-line calls) */
#define RI static inline __attribute__((always_inline))
/* make ALIGNVERIFY=1 (RECOMP_ALIGN_VERIFY): every 16/32-bit guest access whose address is not a multiple of its size
 * is logged with its host site (cpu.c recomp_align_miss: "# align site" lines, "# align r<n>" counts). Measures
 * whether guest RAM could hold native-endian words (unknowns "Misaligned guest accesses"); no effect otherwise. */
#ifdef RECOMP_ALIGN_VERIFY
void recomp_align_miss(u32 a, int n) __attribute__((noinline));
#define ALN(a, n) (__builtin_expect(((a) & ((n) - 1)) != 0, 0) ? recomp_align_miss((a), (n)) : (void)0)
#elif defined(RECOMP_RANGE_VERIFY)
/* make RANGEVERIFY=1: every 8/16/32-bit guest access outside 0x80000000-0x81ffffff goes to the same log (cpu.c),
 * to see how often the uncached mirror or other ranges are used (unknowns "Guest address ranges") */
void recomp_align_miss(u32 a, int n) __attribute__((noinline));
#define ALN(a, n) (__builtin_expect(((a) >> 25) != 0x40u, 0) ? recomp_align_miss((a), (n)) : (void)0)
#else
#define ALN(a, n) ((void)0)
#endif
RI u8  rd8 (u32 a){ ALN(a, 1); if (LOAD_HW(a)) return HW_READ(a,1); return ram[a & RAM_MASK]; }
RI u16 rd16(u32 a){ ALN(a, 2); if (LOAD_HW(a)) return HW_READ(a,2); u16 v; memcpy(&v, ram + (a & RAM_MASK), 2); return __builtin_bswap16(v); }
RI u32 rd32(u32 a){ ALN(a, 4); if (LOAD_HW(a)) return HW_READ(a,4); u32 v; memcpy(&v, ram + (a & RAM_MASK), 4); return __builtin_bswap32(v); }
RI u64 rd64(u32 a){ return ((u64)rd32(a) << 32) | rd32(a + 4); }
RI void wr8 (u32 a, u8 v){ ALN(a, 1); if (STORE_HW(a)) { HW_WRITE(a,v,1); return; } WR_MARK(a); ram[a & RAM_MASK] = v; }
RI void wr16(u32 a, u16 v){ ALN(a, 2); if (STORE_HW(a)) { HW_WRITE(a,v,2); return; } WR_MARK(a); v = __builtin_bswap16(v); memcpy(ram + (a & RAM_MASK), &v, 2); }
RI void wr32(u32 a, u32 v){ ALN(a, 4); if (STORE_HW(a)) { HW_WRITE(a,v,4); return; } WR_MARK(a); v = __builtin_bswap32(v); memcpy(ram + (a & RAM_MASK), &v, 4); }
RI void wr64(u32 a, u64 v){ wr32(a, (u32)(v >> 32)); wr32(a + 4, (u32)v); }

/* r1-relative (stack) accesses never reach hardware registers: no IS_HW check */
RI u8  rd8_ram (u32 a){ return ram[a & RAM_MASK]; }
RI u16 rd16_ram(u32 a){ ALN(a, 2); u16 v; memcpy(&v, ram + (a & RAM_MASK), 2); return __builtin_bswap16(v); }
RI u32 rd32_ram(u32 a){ ALN(a, 4); u32 v; memcpy(&v, ram + (a & RAM_MASK), 4); return __builtin_bswap32(v); }
RI u64 rd64_ram(u32 a){ return ((u64)rd32_ram(a) << 32) | rd32_ram(a + 4); }
RI void wr8_ram (u32 a, u8 v){ ram[a & RAM_MASK] = v; }
RI void wr16_ram(u32 a, u16 v){ ALN(a, 2); v = __builtin_bswap16(v); memcpy(ram + (a & RAM_MASK), &v, 2); }
RI void wr32_ram(u32 a, u32 v){ ALN(a, 4); v = __builtin_bswap32(v); memcpy(ram + (a & RAM_MASK), &v, 4); }
RI void wr64_ram(u32 a, u64 v){ wr32_ram(a, (u32)(v >> 32)); wr32_ram(a + 4, (u32)v); }

/* gen/'s r1/r11-relative accesses with a constant offset (recomp.py stack_access): ram + (b & RAM_MASK)
 * + k, so GCC masks each r1 value once and folds k into the load/store (rd32_ram(b + k) masked every
 * address, often precomputed and spilled). Same bytes as rd32_ram(b + k) while b & RAM_MASK is more
 * than 32 KB from both ends of the mirror: the stack lies below STACK_TOP (recomp_call_guest checks). */
RI u8 *sp_at(u32 b, s32 k){ return ram + (b & RAM_MASK) + k; }
RI u8  rd8_sp (u32 b, s32 k){ return *sp_at(b, k); }
RI u16 rd16_sp(u32 b, s32 k){ ALN(b + k, 2); u16 v; memcpy(&v, sp_at(b, k), 2); return __builtin_bswap16(v); }
RI u32 rd32_sp(u32 b, s32 k){ ALN(b + k, 4); u32 v; memcpy(&v, sp_at(b, k), 4); return __builtin_bswap32(v); }
RI u64 rd64_sp(u32 b, s32 k){ return ((u64)rd32_sp(b, k) << 32) | rd32_sp(b, k + 4); }
RI void wr8_sp (u32 b, s32 k, u8 v){ *sp_at(b, k) = v; }
RI void wr16_sp(u32 b, s32 k, u16 v){ ALN(b + k, 2); v = __builtin_bswap16(v); memcpy(sp_at(b, k), &v, 2); }
RI void wr32_sp(u32 b, s32 k, u32 v){ ALN(b + k, 4); v = __builtin_bswap32(v); memcpy(sp_at(b, k), &v, 4); }
RI void wr64_sp(u32 b, s32 k, u64 v){ wr32_sp(b, k, (u32)(v >> 32)); wr32_sp(b, k + 4, (u32)v); }

/* gen/'s rA + k accesses for other base registers (recomp.py base_access): the hardware check and the
 * mask work on the base b, k is the load/store offset, so GCC checks and masks a pointer once for all
 * the fields read through it. Same result as rd32(b + k) unless b + k crosses into or out of the
 * hardware range or b & RAM_MASK is within 32 KB of either end of the mirror: ram[], ram_wr[] and ram_dirty[]
 * have guard space on both sides (cpu.c), so such an access lands there instead of wrapping. */
#define RAM_GUARD 0x8000u
RI u8 *b_at(u32 b, s32 k){ return ram + (b & RAM_MASK) + k; }
#define WR_MARK_B(b, k) WR_SET((s32)(((b) & RAM_MASK) + (k)) >> WR_PAGE_SHIFT)
RI u8  rd8_b (u32 b, s32 k){ ALN(b + k, 1); if (LOAD_HW(b)) return HW_READ(b + k,1); return *b_at(b, k); }
RI u16 rd16_b(u32 b, s32 k){ ALN(b + k, 2); if (LOAD_HW(b)) return HW_READ(b + k,2); u16 v; memcpy(&v, b_at(b, k), 2); return __builtin_bswap16(v); }
RI u32 rd32_b(u32 b, s32 k){ ALN(b + k, 4); if (LOAD_HW(b)) return HW_READ(b + k,4); u32 v; memcpy(&v, b_at(b, k), 4); return __builtin_bswap32(v); }
RI u64 rd64_b(u32 b, s32 k){ return ((u64)rd32_b(b, k) << 32) | rd32_b(b, k + 4); }
RI void wr8_b (u32 b, s32 k, u8 v){ ALN(b + k, 1); if (STORE_HW(b)) { HW_WRITE(b + k,v,1); return; } WR_MARK_B(b, k); *b_at(b, k) = v; }
RI void wr16_b(u32 b, s32 k, u16 v){ ALN(b + k, 2); if (STORE_HW(b)) { HW_WRITE(b + k,v,2); return; } WR_MARK_B(b, k); v = __builtin_bswap16(v); memcpy(b_at(b, k), &v, 2); }
RI void wr32_b(u32 b, s32 k, u32 v){ ALN(b + k, 4); if (STORE_HW(b)) { HW_WRITE(b + k,v,4); return; } WR_MARK_B(b, k); v = __builtin_bswap32(v); memcpy(b_at(b, k), &v, 4); }
RI void wr64_b(u32 b, s32 k, u64 v){ wr32_b(b, k, (u32)(v >> 32)); wr32_b(b, k + 4, (u32)v); }

RI float  u2f(u32 v){ float f; memcpy(&f, &v, 4); return f; }
RI u32    f2u(float f){ u32 v; memcpy(&v, &f, 4); return v; }
RI double u2d(u64 v){ double d; memcpy(&d, &v, 8); return d; }
RI u64    d2u(double d){ u64 v; memcpy(&v, &d, 8); return v; }

/* ---- condition register ---- */
#define CRBIT(n) ((cpu.cr >> (31 - (n))) & 1u)
static inline void setcrbit(int n, u32 v){ cpu.cr = (cpu.cr & ~(1u << (31 - n))) | ((v & 1u) << (31 - n)); }
static inline void setcrf(int f, u32 v){ int sh = 28 - 4 * f; cpu.cr = (cpu.cr & ~(0xFu << sh)) | (v << sh); }
static inline void cmps(int f, s32 a, s32 b){ setcrf(f, (a < b ? 8 : a > b ? 4 : 2) | cpu.xer_so); }
static inline void cmpu(int f, u32 a, u32 b){ setcrf(f, (a < b ? 8 : a > b ? 4 : 2) | cpu.xer_so); }
#define CR0(x) cmps(0, (s32)(x), 0)
static inline void fcmp(int f, double a, double b){ setcrf(f, (a < b) ? 8 : (a > b) ? 4 : (a == b) ? 2 : 1); }
void ppc_fp_record(void);   /* Rc=1 float ops: cr1 <- FPSCR[FX..OX] */
/* gen/ keeps each CR field in a local (cr0..cr7, low 4 bits) and XER[CA] in `ca` */
static inline u32 cmp_s(s32 a, s32 b){ return a < b ? 8 : a > b ? 4 : 2; }
static inline u32 cmp_u(u32 a, u32 b){ return a < b ? 8 : a > b ? 4 : 2; }
static inline u32 fcmp_v(double a, double b){ return (a < b) ? 8 : (a > b) ? 4 : (a == b) ? 2 : 1; }

/* ---- integer helpers ---- */
static inline u32 rotl(u32 v, int n){ n &= 31; return n ? (v << n) | (v >> (32 - n)) : v; }
static inline u32 cntlzw(u32 v){ return v ? __builtin_clz(v) : 32; }
static inline u32 slw(u32 a, u32 b){ return (b & 0x20) ? 0 : a << (b & 31); }
static inline u32 srw(u32 a, u32 b){ return (b & 0x20) ? 0 : a >> (b & 31); }
static inline u32 sraw(u32 a, u32 b){
    s32 s = (s32)a;
    if (b & 0x20) { cpu.xer_ca = s < 0; return s < 0 ? 0xFFFFFFFFu : 0; }
    b &= 31; cpu.xer_ca = s < 0 && b && (a & ((1u << b) - 1)); return (u32)(s >> b);
}
static inline u32 srawi(u32 a, int n){
    s32 s = (s32)a; cpu.xer_ca = s < 0 && n && (a & ((1u << n) - 1)); return (u32)(s >> n);
}
static inline u32 addc(u32 a, u32 b, u32 c){ u64 s = (u64)a + b + c; cpu.xer_ca = (u32)(s >> 32); return (u32)s; }
static inline u32 addc_l(u32 *ca, u32 a, u32 b, u32 c){ u64 s = (u64)a + b + c; *ca = (u32)(s >> 32); return (u32)s; }
static inline u32 sraw_l(u32 *ca, u32 a, u32 b){
    s32 s = (s32)a;
    if (b & 0x20) { *ca = s < 0; return s < 0 ? 0xFFFFFFFFu : 0; }
    b &= 31; *ca = s < 0 && b && (a & ((1u << b) - 1)); return (u32)(s >> b);
}
static inline u32 srawi_l(u32 *ca, u32 a, int n){
    s32 s = (s32)a; *ca = s < 0 && n && (a & ((1u << n) - 1)); return (u32)(s >> n);
}
/* Dolphin: divide by zero / overflow results */
static inline u32 divw(u32 a, u32 b){
    s32 sa = (s32)a, sb = (s32)b;
    if (sb == 0 || (a == 0x80000000u && sb == -1)) return sa < 0 ? 0xFFFFFFFFu : 0;
    return (u32)(sa / sb);
}
static inline u32 divwu(u32 a, u32 b){ return b ? a / b : 0; }
static inline u32 mulhw(u32 a, u32 b){ return (u32)(((s64)(s32)a * (s32)b) >> 32); }
static inline u32 mulhwu(u32 a, u32 b){ return (u32)(((u64)a * b) >> 32); }

static inline u32 get_xer(void){ return (cpu.xer_so << 31) | (cpu.xer_ov << 30) | (cpu.xer_ca << 29) | cpu.xer_bc; }
static inline void set_xer(u32 v){ cpu.xer_so = v >> 31; cpu.xer_ov = (v >> 30) & 1; cpu.xer_ca = (v >> 29) & 1; cpu.xer_bc = v & 0x7F; }

/* ---- floating point (Dolphin's ForceSingle / Force25Bit) ---- */
static inline double fsgl(double d){ return (double)(float)d; }
static inline double f25(double d){ u64 b = d2u(d); b = (b & 0xFFFFFFFFF8000000ull) + (b & 0x8000000ull); return u2d(b); }
static inline double fnoneg(double d){ return isnan(d) ? d : -d; }
/* fneg/fabs/fnabs (and ps_): the sign bit only. -d and fabs() are VNEG/VABS on ARM, which change only
 * the sign bit (no NaN processing, no flush-to-zero), and stay in VFP registers; the integer form
 * (d2u ^ sign) went through core registers and the C stack. */
RI double fneg_d(double d){ return -d; }
RI double fabs_d(double d){ return __builtin_fabs(d); }
RI double fnabs_d(double d){ return -__builtin_fabs(d); }
/* fctiw/fctiwz result bits (Dolphin): saturated to s32, NaN -> 0x80000000, -0 quirk in the upper
 * word. ARM: vcvt (toward zero) / vcvtr (FPSCR mode, nearest like rint) saturate the same way and
 * give 0 for NaN, so only NaN is special-cased; was libm trunc + an out-of-line call per site. */
RI u64 fctiw_bits(double b, int tz){
    s32 r;
#ifdef __arm__
    float t;
    if (tz) __asm__("vcvt.s32.f64 %0, %P1" : "=t"(t) : "w"(b));
    else __asm__("vcvtr.s32.f64 %0, %P1" : "=t"(t) : "w"(b));
    memcpy(&r, &t, 4);
#else
    double t = tz ? trunc(b) : rint(b);
    if (t <= -2147483648.0) r = (s32)0x80000000; else if (t >= 2147483647.0) r = 0x7FFFFFFF; else r = (s32)t;
#endif
    if (isnan(b)) r = (s32)0x80000000;
    u64 v = 0xFFF8000000000000ull | (u32)r;
    if (r == 0 && signbit(b)) v |= 0x100000000ull;
    return v;
}
/* fctiw of a float (recomp.py fp_single): vcvt from single, the same integer as from its double */
RI u64 fctiw_bits_f(float b, int tz){
#ifdef __arm__
    s32 r;
    float t;
    if (tz) __asm__("vcvt.s32.f32 %0, %1" : "=t"(t) : "t"(b));
    else __asm__("vcvtr.s32.f32 %0, %1" : "=t"(t) : "t"(b));
    memcpy(&r, &t, 4);
    if (isnan(b)) r = (s32)0x80000000;
    u64 v = 0xFFF8000000000000ull | (u32)r;
    if (r == 0 && signbit(b)) v |= 0x100000000ull;
    return v;
#else
    return fctiw_bits(b, tz);
#endif
}
double ppc_fres(double b);      /* Gekko table estimates; see recomp_unknowns.md */
double ppc_frsqrte(double b);
/* single-precision madd (fmadds, ps_madd...): a is single, c is cut to 25 bits (f25), so
 * a*c is exact in a double and a*c + b rounds once, like Gekko's fused madd. */
static inline double fmadd_s(double a, double c, double b){ return a * c + b; }
/* double fmadd/fmsub/fnmadd/fnmsub: a*c + b with two roundings. Gekko fuses them (one rounding);
 * the 3DS has no fused multiply-add and newlib's fma() is itself a non-fused vmla.f64 (checked in
 * devkitARM's libm.a 2026-10-22), so gen/ always rounded twice here: this is the same result inline,
 * without the libm call (recomp_unknowns.md "Native MSL sin/cos"). */
RI double fmadd_d(double a, double c, double b){ return a * c + b; }
static inline double fsel(double a, double b, double c){ return (a >= -0.0) ? b : c; }
/* single-precision ops as gen/ emits them (fadds..fnmsubs, ps_add..ps_madds1), result rounded
 * to single. Default: Dolphin's double + fsgl with c cut by f25 and one rounding per madd.
 * make FLOATOPS=1 (RECOMP_FLOATOPS): mul/madd/msub in float (inputs rounded to float first,
 * no f25, madd rounds twice). add/sub/div stay double + fsgl: the int->float idiom subtracts
 * a double magic constant (fsubs), and rounding those inputs broke it. recomp_unknowns.md "Float ops". */
static inline double SADD(double a, double b){ return fsgl(a + b); }
static inline double SSUB(double a, double b){ return fsgl(a - b); }
static inline double SDIV(double a, double b){ return fsgl(a / b); }
#ifdef RECOMP_FLOATOPS
static inline double SMUL(double a, double c){ return (float)a * (float)c; }
static inline double SMADD(double a, double c, double b){ return (float)a * (float)c + (float)b; }
static inline double SMSUB(double a, double c, double b){ return (float)a * (float)c - (float)b; }
#else
static inline double SMUL(double a, double c){ return fsgl(a * f25(c)); }
static inline double SMADD(double a, double c, double b){ return fsgl(fmadd_s(a, f25(c), b)); }
static inline double SMSUB(double a, double c, double b){ return fsgl(fmadd_s(a, f25(c), -b)); }
#endif
/* the same ops when every input is a float (recomp.py fp_single: FPR halves that provably hold a
 * single). add/sub/mul/div in float are the double op + fsgl bit for bit: a double has 53 >= 2*24+2
 * bits, so rounding twice gives the correctly rounded float (any rounding mode; a float's f25 is
 * itself). madd/msub as SMADD/SMSUB: FLOATOPS float, else Dolphin's one double rounding then fsgl.
 * recomp_unknowns.md "Single-precision FPR halves". */
RI float FADD(float a, float b){ return a + b; }
RI float FSUB(float a, float b){ return a - b; }
RI float FMUL(float a, float c){ return a * c; }
RI float FDIV(float a, float b){ return a / b; }
#ifdef RECOMP_FLOATOPS
RI float FMADD(float a, float c, float b){ return a * c + b; }
RI float FMSUB(float a, float c, float b){ return a * c - b; }
#else
RI float FMADD(float a, float c, float b){ return (float)fmadd_s(a, c, b); }
RI float FMSUB(float a, float c, float b){ return (float)fmadd_s(a, c, -(double)b); }
#endif
RI float fnoneg_f(float d){ return isnan(d) ? d : -d; }
RI float fnabs_f(float d){ return -__builtin_fabsf(d); }
RI float fsel_f(double a, float b, float c){ return (a >= -0.0) ? b : c; }

/* ---- paired-single quantization (GQR) ---- */
static inline double dequant(u32 a, int type, int scale, int *size){
    float s = ldexpf(1.0f, -((scale << 26) >> 26));
    switch (type) {
    case 4: *size = 1; return (double)((float)rd8(a) * s);
    case 5: *size = 2; return (double)((float)rd16(a) * s);
    case 6: *size = 1; return (double)((float)(s8)rd8(a) * s);
    case 7: *size = 2; return (double)((float)(s16)rd16(a) * s);
    default: *size = 4; return (double)u2f(rd32(a));
    }
}
static inline void psq_load(int fd, u32 a, int w, int i){
    u32 g = cpu.gqr[i]; int type = (g >> 16) & 7, scale = (g >> 24) & 0x3F, sz;
    double p0 = dequant(a, type, scale, &sz);
    double p1 = w ? 1.0 : dequant(a + sz, type, scale, &sz);
    F[fd].ps0 = p0; F[fd].ps1 = p1;
}
static inline void quant(u32 a, double v, int type, int scale, int *size){
    float f = (float)v * u2f((u32)(127 + ((scale << 26) >> 26)) << 23);
    switch (type) {
    case 4: *size = 1; wr8 (a, (u8) (f <= 0 ? 0 : f >= 255.0f ? 255 : (int)f)); return;
    case 5: *size = 2; wr16(a, (u16)(f <= 0 ? 0 : f >= 65535.0f ? 65535 : (int)f)); return;
    case 6: *size = 1; wr8 (a, (u8) (s8) (f <= -128.0f ? -128 : f >= 127.0f ? 127 : (int)f)); return;
    case 7: *size = 2; wr16(a, (u16)(s16)(f <= -32768.0f ? -32768 : f >= 32767.0f ? 32767 : (int)f)); return;
    default: *size = 4; wr32(a, f2u((float)v)); return;
    }
}
static inline void psq_store(int fs, u32 a, int w, int i){
    u32 g = cpu.gqr[i]; int type = g & 7, scale = (g >> 8) & 0x3F, sz;
    quant(a, F[fs].ps0, type, scale, &sz);
    if (!w) quant(a + sz, F[fs].ps1, type, scale, &sz);
}

/* by value for gen/ (FPRs are C locals there). 2^n scale built from float bits:
 * ldexpf was a libm call per element */
static inline float qscale(int n){ return u2f((u32)(127 + n) << 23); }
static inline PS psq_ld(u32 a, int w, int i){
    u32 g = cpu.gqr[i]; int type = (g >> 16) & 7;
    PS r;
    if (type == 0) { r.ps0 = u2f(rd32(a)); r.ps1 = w ? 1.0 : u2f(rd32(a + 4)); return r; }
    float s = qscale(-((((int)(g >> 24) & 0x3F) << 26) >> 26));
    switch (type) {
    case 4: r.ps0 = (float)rd8(a) * s;      r.ps1 = w ? 1.0 : (float)rd8(a + 1) * s; break;
    case 5: r.ps0 = (float)rd16(a) * s;     r.ps1 = w ? 1.0 : (float)rd16(a + 2) * s; break;
    case 6: r.ps0 = (float)(s8)rd8(a) * s;  r.ps1 = w ? 1.0 : (float)(s8)rd8(a + 1) * s; break;
    case 7: r.ps0 = (float)(s16)rd16(a) * s; r.ps1 = w ? 1.0 : (float)(s16)rd16(a + 2) * s; break;
    default: r.ps0 = u2f(rd32(a)); r.ps1 = w ? 1.0 : u2f(rd32(a + 4)); break;
    }
    return r;
}
static inline void psq_st(PS v, u32 a, int w, int i){
    u32 g = cpu.gqr[i]; int type = g & 7, sz;
    if (type == 0) { wr32(a, f2u((float)v.ps0)); if (!w) wr32(a + 4, f2u((float)v.ps1)); return; }
    int scale = (g >> 8) & 0x3F;
    quant(a, v.ps0, type, scale, &sz);
    if (!w) quant(a + sz, v.ps1, type, scale, &sz);
}

/* psq_l/psq_st with GQR0 (I = 0: MWCC's f14-f31 saves, PSMTX-style code): GQR0 is 0 (float, no
 * scale), so no GQR load or type switch; _ram = r1-relative (no IS_HW check, no WR_MARK, like
 * wr32_ram). recomp.py guards the assumption: a translated mtspr GQR0 makes funcs.h define
 * RECOMP_GQR0_WRITTEN (the general path below), check_gqr_writes() fails on runtime/ writers. */
#ifndef RECOMP_GQR0_WRITTEN
RI PS psq_ld0(u32 a, int w){ PS r; r.ps0 = u2f(rd32(a)); r.ps1 = w ? 1.0 : u2f(rd32(a + 4)); return r; }
RI PS psq_ld0_ram(u32 a, int w){ PS r; r.ps0 = u2f(rd32_ram(a)); r.ps1 = w ? 1.0 : u2f(rd32_ram(a + 4)); return r; }
RI void psq_st0(PS v, u32 a, int w){ wr32(a, f2u((float)v.ps0)); if (!w) wr32(a + 4, f2u((float)v.ps1)); }
RI void psq_st0_ram(PS v, u32 a, int w){ wr32_ram(a, f2u((float)v.ps0)); if (!w) wr32_ram(a + 4, f2u((float)v.ps1)); }
RI PS psq_ld0_sp(u32 b, s32 k, int w){ PS r; r.ps0 = u2f(rd32_sp(b, k)); r.ps1 = w ? 1.0 : u2f(rd32_sp(b, k + 4)); return r; }
RI void psq_st0_sp(PS v, u32 b, s32 k, int w){ wr32_sp(b, k, f2u((float)v.ps0)); if (!w) wr32_sp(b, k + 4, f2u((float)v.ps1)); }
/* psq_l of a stack word pair held in C locals (recomp.py promote_stack_slots): a, b = the words' bits */
RI PS psq_ld0_v(u32 a, u32 b, int w){ PS r; r.ps0 = u2f(a); r.ps1 = w ? 1.0 : u2f(b); return r; }
RI PS psq_ld0_b(u32 b, s32 k, int w){ PS r; r.ps0 = u2f(rd32_b(b, k)); r.ps1 = w ? 1.0 : u2f(rd32_b(b, k + 4)); return r; }
RI void psq_st0_b(PS v, u32 b, s32 k, int w){ wr32_b(b, k, f2u((float)v.ps0)); if (!w) wr32_b(b, k + 4, f2u((float)v.ps1)); }
#else
#define psq_ld0_b(b, k, w) psq_ld((b) + (k), w, 0)
#define psq_st0_b(v, b, k, w) psq_st(v, (b) + (k), w, 0)
#define psq_ld0(a, w) psq_ld(a, w, 0)
#define psq_ld0_ram(a, w) psq_ld(a, w, 0)
#define psq_st0(v, a, w) psq_st(v, a, w, 0)
#define psq_st0_ram(v, a, w) psq_st(v, a, w, 0)
#define psq_ld0_sp(b, k, w) psq_ld((b) + (k), w, 0)
#define psq_st0_sp(v, b, k, w) psq_st(v, (b) + (k), w, 0)
#endif

#ifdef RECOMP_FNPROF               /* make FNPROF=1: sampler counts the running guest function */
extern volatile u32 recomp_cur_fn;
#define FNPROF(x) (recomp_cur_fn = (x))
#else
#define FNPROF(x) ((void)0)
#endif

/* ---- system ---- */
u32  ppc_mfspr(int n);
void ppc_mtspr(int n, u32 v);
u32  ppc_mftb(int tbr);
void ppc_mtmsr(u32 v);
void ppc_sc(u32 pc);
void ppc_rfi(u32 pc);
void ppc_trap(u32 pc);
void ppc_mtfsf(u32 mask, u64 v);
static inline void dcbz(u32 a){ a &= ~31u; for (int k = 0; k < 32; k += 4) wr32(a + k, 0); }

/* ---- control flow ---- */
typedef void (*GuestFn)(void);
void call_indirect(u32 target);       /* function table lookup */
void recomp_bad(u32 pc, const char *why);
void recomp_skipped(u32 addr);        /* MetroTRK and other skipped code */
void recomp_sdk_call(const char *name); /* logs an SDK call that runtime/ does not implement */

/* ---- runtime services (runtime/log.c, runtime/dol.c) ---- */
void recomp_log(const char *fmt, ...) __attribute__((format(printf, 1, 2)));
u32 recomp_fpscr(void);   /* main.c: this thread's VFP FPSCR */
void recomp_sdk_log(const char *name);  /* same report for hand-written SDK functions */
/* LOG() in the runtime/ files: recomp_sdk_log inline, with the name's count slot looked up once per function
 * (static); the first RECOMP_SDK_PER_NAME calls of a name write the log line (log.c) */
#define RECOMP_SDK_PER_NAME 20
extern const char *volatile recomp_sdk_ring[32];
extern volatile u32 recomp_sdk_ring_pos;
u32 *recomp_sdk_count(const char *name);
void recomp_sdk_log_line(const char *name);
static inline __attribute__((always_inline)) void recomp_sdk_log_at(const char *name, u32 **slot) {
    FNPROF((u32)(uintptr_t)name);   /* make FNPROF=1: SDK time under the SDK name */
    u32 *n = *slot;
    if (__builtin_expect(!n, 0)) *slot = n = recomp_sdk_count(name);
    recomp_sdk_ring[recomp_sdk_ring_pos++ & 31] = name;
    if (__builtin_expect(++*n <= RECOMP_SDK_PER_NAME, 0)) recomp_sdk_log_line(name);
}
#define RECOMP_LOG() do { static u32 *lc_; recomp_sdk_log_at(__func__ + 4, &lc_); } while (0)
void recomp_done(const char *why) __attribute__((noreturn)); /* ends the guest run */
int  recomp_load_dol(void);             /* main.dol + boot info into ram; 0 on success */
void recomp_call_guest(u32 fn, u32 arg); /* interrupt-style call: guest state saved around it */
void recomp_call_guest2(u32 fn, u32 arg, u32 arg2);
void recomp_dvd_service(void);          /* runs completed DVD callbacks */
void recomp_ar_service(void);           /* runs completed ARQ callbacks */
extern volatile u32 recomp_event_pending; /* an ARQ completion is in flight */
void recomp_poll_events(void);          /* delivers due events if MSR[EE] */
#define RECOMP_POLL() do { if (__builtin_expect(recomp_event_pending, 0)) recomp_poll_events(); } while (0)
u32  recomp_retrace_count(void);         /* frames so far (VI retraces) */
void recomp_ax_service(void);           /* AX frame callbacks due since the last retrace */
int  recomp_autotest_run(void);         /* cr_pad_script.txt or cr_pad_replay.txt exists (runtime/vi.c) */
void recomp_ax_pacing_init(void);       /* audio pacing mode from flags, printed at boot (runtime/ax.c) */
void recomp_ax_pacing_toggle(void);     /* touch screen: real-time <-> per-retrace audio from the next retrace */
extern int recomp_ax_rt;                /* audio pacing now: 1 real-time, 0 per retrace (for the # time lines) */
extern u32 recomp_ax_flips, recomp_ax_underruns, recomp_ax_skipped;   /* toggles, ndsp ran dry, rt frames skipped (cumulative) */
#define AX_AI_CALLBACK 0x80145a00u        /* __AXOutAiCallback: AX's AI DMA callback */
extern u32 recomp_ai_dma_cb, recomp_ai_dma_addr, recomp_ai_dma_len;   /* AI DMA (runtime/ai.c) */
/* frame timing (runtime/log.c): ticks spent per bucket, logged by VI */
enum { PROF_GX_VTX, PROF_GX_TEX, PROF_GX_END, PROF_AUDIO, PROF_GX_STATE,
       PROF_VX_UNIF, PROF_VX_DEC, PROF_VX_DL, PROF_LOG, PROF_PREDRAW, PROF_GX_SYNC, PROF_GPU, PROF_GPU_SPAN, PROF_DVD,
       PROF_GX_WAIT, PROF_GX_BUSY, PROF_GPU_DRAWS, PROF_GPU_IDX, PROF_GPU_VBYTES, PROF_AUDIO_CB, PROF_N };   /* parts of PROF_GX_VTX: uniforms, decode loop, draws from display lists; PROF_LOG: recomp_log writes (inside the others); PROF_GX_SYNC: C3D_FrameBegin (vblank cap + GPU queue wait); PROF_GPU: GPU queue time from C3D_FrameEnd to queue done (C3D_GetDrawingTime; with mid-frame submits only the tail); PROF_GPU_SPAN: first mid-frame submit (else C3D_FrameEnd) to queue done; PROF_DVD: game thread waiting for the DVD reader thread (inside the others); with the GX thread (recomp_gx_thread) the PROF_GX_* and PROF_VX_* buckets are GX-thread time, PROF_GX_WAIT is the game thread waiting for it, PROF_GX_BUSY the GX thread handling records; PROF_GPU_DRAWS/IDX/VBYTES: counts, not ticks (C3D_DrawElements calls, indices, indices x vertex size); PROF_AUDIO_CB: the guest AX/AI callbacks (sound driver tick), inside PROF_AUDIO */
extern u64 recomp_prof[PROF_N];
extern volatile u32 recomp_logw_since;  /* low word of the tick the log writer's fwrite+fflush began, 0 = idle */
extern volatile u32 recomp_logw_n;      /* log writes finished */
struct Thread_tag *recomp_aux_thread(void (*f)(void *), size_t stack, int prio);   /* thread off the game's core (log.c) */
int  recomp_aux_core(void);             /* the core recomp_aux_thread uses: 2 (New 3DS), 1 or 0 */
void recomp_thread_crash_log(void);     /* crashes on the calling thread end the log too (log.c) */
extern u64 recomp_first_draw;   /* tick of the frame's first GX draw (0 = none yet); PROF_PREDRAW: frame time before it */
extern u32 recomp_run_frames;
u64 recomp_ticks(void);
void recomp_prof_frame(u32 retrace);
void recomp_prof_sampler_start(void);  /* cpu.lr histogram thread (log.c) */
void recomp_ax_close(void);             /* flushes the WAV dump, stops ndsp */
u32 recomp_retrace_count(void);
u8 recomp_aram_read8(u32 a);            /* ARAM byte (runtime/ar.c) */
void recomp_aram_written(u32 a, u32 n);  /* ax.c: n ARAM bytes at a were just written (ar.c DMA, game thread) */
void recomp_wait_point(void);           /* pending "interrupts": DVD, then ARQ */
u64  recomp_timebase(void);             /* Gekko timebase, 40.5 MHz */
void recomp_gx_init(void);             /* citro3d, render target, shader */
void recomp_gx_start(void);            /* flag files, then the GX thread on core 2 (after recomp_log_open) */
void recomp_gx_sync(void);             /* game thread: wait until the GX thread handled every GX call so far */
void recomp_gx_retrace(void);          /* VIWaitForRetrace: sync, then apply a requested GX thread on/off */
void recomp_gx_thread_toggle(void);    /* bottom-screen tap: GX thread on/off at the next retrace */
void recomp_gx_stop(void);             /* at exit: sync, then end the GX thread */
extern int recomp_gx_thread;           /* GX calls are handled on the GX thread now (else inline on the game thread) */
extern u32 recomp_gx_thread_flips;     /* GX thread switches so far (for the # time lines) */
#define RECOMP_STAT_ROWS 9              /* bottom screen's stats rows (log.c) */
void recomp_status_row(int row, const char *s);   /* stats row 0..RECOMP_STAT_ROWS-1, redrawn only when changed (menu.c; game thread) */
void recomp_menu_init(void);           /* bottom screen console + layout (menu.c) */
void recomp_menu_header(const char *build, const char *flags);   /* build id + sdmc flags rows */
int  recomp_gx_thread_ok(void);        /* the GX thread exists (core 2) */
extern volatile u32 recomp_gx_stat[4];  /* GX frames, draws, triangles (running totals), linear heap free (bytes); written by whoever runs GX */
extern u32 recomp_dvd_slow_n, recomp_dvd_late_n;   /* dvd.c */
void recomp_menu_touch(int px, int py); /* PADRead: a new touch, in pixels */
void recomp_menu_retrace(void);        /* recomp_gx_retrace after the sync: apply bound-flag toggles, redraw buttons */
void recomp_menu_bind(const char *name, int *var, void (*changed)(void));   /* live-toggle a flag variable from its button (menu.c btn[]) */
extern u32 recomp_menu_flips;          /* bound-flag toggles so far */
extern int recomp_libm_off;            /* libm.c: sin/cos/sinf/cosf run the translated bodies (sdmc:/cr_libm_off.txt) */
extern u32 recomp_libm_flips;
extern int recomp_vt_on;               /* gx.c: cached textures move to VRAM (sdmc:/cr_vramtex_off.txt = 0) */
extern int recomp_vtxpack_off;          /* gx.c: display-list vertices packed per draw (sdmc:/cr_vtxpack_off.txt = 1) */
extern int recomp_lazydesc_off;         /* gx.c: a GX-queue record per vertex descriptor call (sdmc:/cr_lazydesc_off.txt = 1) */
extern int recomp_ptr_extent;           /* main.c: gen/ f_X with a second body runs fp_X (pointer-extent stack promotion; sdmc:/cr_ptrext_off.txt = 0) */
extern int recomp_inline_sites;         /* main.c: the INLINE_SITES bl sites run their inline copy (sdmc:/cr_inline_off.txt = 0) */
extern u32 recomp_codegen_flips;        /* main.c: toggles of the two above (# time split: "mixed") */
extern int recomp_perfctr_on;           /* perfctr.c: ARM11 counters per frame, "# perf" lines (sdmc:/cr_perfctr.txt = 1, button "Perf counters") */
extern u32 recomp_perfctr_flips;        /* perfctr.c: toggles so far (# time split: "mixed") */
void recomp_perfctr_init(void);         /* main.c at boot: flag file, button, enable */
void recomp_perfctr_frame(u32 retrace); /* recomp_prof_frame (game thread): read, rotate, log every 60 retraces */
void recomp_hwcheck_report(u32 retrace); /* cpu.c, make HWCHECK=verify: stores that reached hw_write, per host site */
void recomp_perfctr_exit(void);         /* main.c at exit: release the counters */
void recomp_memdump_request(void);    /* bottom-screen button "Dump RAM": snapshot at the next retrace (memdump.c) */
void recomp_memdump_retrace(void);     /* recomp_gx_retrace after the sync: button / sdmc:/cr_memdump.txt dumps */
extern u32 recomp_memdump_n;           /* dumps written */
const u8 *recomp_aram(void);           /* ar.c: ARAM (16 MB), NULL before ARInit */
void recomp_dvd_describe(void *file);  /* dvd.c: reads in flight to a FILE *, for a memory dump */
void recomp_gx_fb_dump_next(void);     /* gx.c: write the next GX frame to sdmc:/cr_fb_<frame>.bmp */
void recomp_vt_toggle(void);           /* bottom-screen button: flip it at the next frame start (texture cache invalidated) */
extern u32 recomp_vt_flips;            /* VRAM texture toggles so far */
extern volatile u32 recomp_vt_stat[6]; /* textures in VRAM, KB; in linear, KB (every 30 GX frames); moves, fallbacks (running totals) */
extern int recomp_gx_kick_on;          /* mid-frame GPU submit mode now (for the # time lines) */
extern u32 recomp_gx_kick_flips;       /* ZL toggles so far */
extern u32 recomp_gx_dbg[3];           /* last draw: gpucmd offset at entry, prim, vertex count */
void recomp_gx_fifo_write(u32 v, int size); /* WGPIPE 0xCC008000 */
void recomp_log_open(void);
void recomp_log_close(void);
