/* MSL (fdlibm) sin/cos and their float wrappers replaced natively (recomp.py BODY_NATIVE):
 *   0x80168f38 sin, 0x801689d0 cos, 0x801693dc sinf (= frsp(sin)), 0x80169400 cosf (= frsp(cos)).
 * In the 2v2 battle they are ~2400 calls per frame, each going through __ieee754_rem_pio2
 * (0x801671f4) and __kernel_sin / __kernel_cos (0x801684dc / 0x80167594) as separate translated
 * calls with stack round trips for the high word.
 *
 * Bit-identical to the translated code, not to Gekko: the bodies below follow the guest's
 * instruction order with the same operations as gen/ (double fmadd/fmsub/fnmsub as fmadd_d(a, c, +-b),
 * two roundings like newlib's non-fused fma() that gen/ called before, unlike Gekko's fused
 * fmadd; see recomp_unknowns.md "Native MSL sin/cos"). They also leave the registers the
 * translated calls leave in `cpu` for a later reader: r3, r4, r6, f1-f4 (ps0 only, as lfd and
 * double ops keep ps1) and XER[CA] (srawi in rem_pio2's careful path). Not written: r0, f0,
 * f5-f7, cr0 (no translated return stores them) and the guest stack frames below r1.
 * |x| > 2^19 * pi/2 (0x413921fb), inf and NaN run the translated body (rem_pio2's large path
 * calls __kernel_rem_pio2).
 * sdmc:/cr_libm_off.txt (or the bottom-screen button "Native sin/cos", live) routes every call to
 * the translated bodies; sdmc:/cr_bench.txt adds
 * startup "# bench libm" lines (native vs translated on random inputs, all those registers). */
#include <3ds.h>
#include <stdio.h>
#include "ppc.h"

#ifdef RECOMP_FNPROF
void recomp_fnprof_native(const char *name);
#define NATIVE_PROF(name) recomp_fnprof_native(name)
#else
#define NATIVE_PROF(name) ((void)0)
#endif

void f_80168f38(void); void f_801689d0(void); void f_801693dc(void); void f_80169400(void);

/* .sdata2 constants (r2 = 0x804581a0), as gen/ folds them */
#define K(b) u2d(b##ull)
#define S1 K(0xbfc5555555555549)   /* __kernel_sin, r2-0x5e68 .. -0x5e90 */
#define S2 K(0x3f8111111110f8a6)
#define S3 K(0xbf2a01a019c161d5)
#define S4 K(0x3ec71de357b1fe7d)
#define S5 K(0xbe5ae5e68a2b9ceb)
#define S6 K(0x3de5d93a5acfd57c)
#define C1 K(0x3fa555555555554c)   /* __kernel_cos, r2-0x5f10 .. -0x5ee8 */
#define C2 K(0xbf56c16c16c15177)
#define C3 K(0x3efa01a019cb1590)
#define C4 K(0xbe927e4f809c52ad)
#define C5 K(0x3e21ee9ebdb4b1c4)
#define C6 K(0xbda8fae9be8838d4)
#define QX K(0x3fd2000000000000)   /* 0.28125, r2-0x5ed8 */
#define HALF K(0x3fe0000000000000)
#define ONE K(0x3ff0000000000000)
#define INVPIO2 K(0x3fe45f306dc9c883)   /* __ieee754_rem_pio2, r2-0x5f40 .. -0x5f20 */
#define PIO2_1 K(0x3ff921fb54400000)
#define PIO2_1T K(0x3dd0b4611a626331)
#define PIO2_2 K(0x3dd0b4611a600000)
#define PIO2_2T K(0x3ba3198a2e037073)
#define PIO2_3 K(0x3ba3198a2e000000)
#define PIO2_3T K(0x397b839a252049c1)
#define MAGIC K(0x4330000080000000)   /* 2^52 + 2^31: (double)(s32) idiom */
#define NPIO2_HW 0x801c4c58u          /* npio2_hw[32] (lis -0x7fe4; addi 0x4c58) */
static const u32 npio2_hw[32] = {
    0x3ff921fb, 0x400921fb, 0x4012d97c, 0x401921fb, 0x401f6a7a, 0x4022d97c, 0x4025fdbb, 0x402921fb,
    0x402c463a, 0x402f6a7a, 0x4031475c, 0x4032d97c, 0x40346b9c, 0x4035fdbb, 0x40378fdb, 0x403921fb,
    0x403ab41b, 0x403c463a, 0x403dd85a, 0x403f6a7a, 0x40407e4c, 0x4041475c, 0x4042106c, 0x4042d97c,
    0x4043a28c, 0x40446b9c, 0x404534ac, 0x4045fdbb, 0x4046c6cb, 0x40478fdb, 0x404858eb, 0x404921fb };

static inline u32 hi(double x) { return (u32)(d2u(x) >> 32); }

/* guest registers a call leaves for later readers */
typedef struct { u32 r3, r4, r6; double f2, f3, f4; u8 w3, w4, w6, ca0; } St;

/* 0x801684dc __kernel_sin(x f1, y f2, iy r3) */
static double k_sin(St *s, double x, double y, u32 iy) {
    u32 ix = hi(x) & 0x7fffffffu;
    s->r4 = ix;
    if ((s32)ix < 0x3e400000 && (u32)fctiw_bits(x, 1) == 0) return x;
    double z = x * x;
    double f4 = fmadd_d(S6, z, S5);
    double v = z * x;
    double f3 = fmadd_d(z, f4, S4);
    double f1 = fmadd_d(z, f3, S3);
    f1 = fmadd_d(z, f1, S2);
    s->f4 = f4; s->w4 = 1; s->w3 = 1;
    if (iy == 0) { s->f3 = f3; return fmadd_d(v, fmadd_d(z, f1, S1), x); }
    double f0 = v * f1;
    f0 = fmadd_d(HALF, y, -f0);
    f0 = fmadd_d(z, f0, -y);
    f0 = fnoneg(fmadd_d(S1, v, -f0));
    s->f3 = S1;
    return x - f0;
}

/* 0x80167594 __kernel_cos(x f1, y f2) */
static double k_cos(St *s, double x, double y) {
    u32 hx = hi(x), ix = hx & 0x7fffffffu;
    s->r3 = hx; s->r4 = ix;
    if ((s32)ix < 0x3e400000 && (u32)fctiw_bits(x, 1) == 0) return ONE;
    s->r3 = 0x3fd30000u;
    double z = x * x;
    double f5 = fmadd_d(C6, z, C5);
    double f4 = fmadd_d(z, f5, C4);
    double f3 = fmadd_d(z, f4, C3);
    double f1 = fmadd_d(z, f3, C2);
    double f0 = fmadd_d(z, f1, C1);
    f4 = z * f0;
    s->f4 = f4; s->w4 = 1; s->w3 = 1;
    if ((s32)ix < 0x3fd33333) {
        f0 = x * y;
        f0 = fmadd_d(z, f4, -f0);
        f0 = fmadd_d(HALF, z, -f0);
        s->f2 = ONE; s->f3 = f3;
        return ONE - f0;
    }
    double qx;
    if ((s32)ix > 0x3fe90000) qx = QX;
    else { s->r3 = ix - 0x00200000u; qx = u2d((u64)s->r3 << 32); }
    f0 = x * y;
    f1 = fmadd_d(HALF, z, -qx);
    f0 = fmadd_d(z, f4, -f0);
    double f2 = ONE - qx;
    f0 = f1 - f0;
    s->f2 = f2; s->f3 = qx;
    return f2 - f0;
}

/* 0x801671f4 __ieee754_rem_pio2(x f1, y r3) for 0x3fe921fb < |hi| <= 0x413921fb: returns n,
 * y[0], y[1]; f1 = the last tail, f2-f4 as the guest leaves them */
static s32 rem_pio2(St *s, double x, double *y0, double *y1) {
    u32 hx = hi(x), ix = hx & 0x7fffffffu;
    s->r6 = ix; s->w6 = 1; s->r4 = 0x3fe90000u;
    double f1, f2;
    if ((s32)ix < 0x4002d97c) {   /* |x| < 3pi/4: n = +-1 */
        if ((s32)hx > 0) {
            f2 = x - PIO2_1;
            if (ix - 0x3ff90000u != 0x21fb) f1 = PIO2_1T;
            else { f2 = f2 - PIO2_2; f1 = PIO2_2T; }
            *y0 = f2 - f1; *y1 = (f2 - *y0) - f1;
            s->r3 = 1;
        } else {
            f2 = PIO2_1 + x;
            if (ix - 0x3ff90000u != 0x21fb) f1 = PIO2_1T;
            else { f2 = f2 + PIO2_2; f1 = PIO2_2T; }
            *y0 = f1 + f2; *y1 = f1 + (f2 - *y0);
            s->r3 = (u32)-1;
        }
        s->f2 = f2;
        return (s32)s->r3;
    }
    /* medium: |x| <= 2^19 * pi/2 */
    double t = fabs_d(x);
    s32 n = (s32)(u32)fctiw_bits(fmadd_d(INVPIO2, t, HALF), 1);
    f2 = u2d((u64)0x43300000u << 32 | ((u32)n ^ 0x80000000u));
    double f3 = MAGIC;
    double fn = f2 - f3;
    double r = fnoneg(fmadd_d(PIO2_1, fn, -t));
    double w = PIO2_1T * fn;
    double y;
    if (n < 32 && ix != npio2_hw[n - 1]) {
        s->r4 = NPIO2_HW + 4 * (u32)n;
        y = r - w;
    } else {
        y = r - w;
        u32 j = ix >> 20;   /* srawi of ix >= 0: CA = 0 */
        s->r4 = j; s->ca0 = 1;
        if ((s32)(j - (hi(y) >> 20 & 0x7ff)) > 16) {   /* 2nd iteration, good to 118 bits */
            double tt = r;
            f2 = PIO2_2 * fn;
            r = r - f2;
            w = fmadd_d(PIO2_2T, fn, -((tt - r) - f2));
            y = r - w;
            f3 = tt;
            if ((s32)(j - (hi(y) >> 20 & 0x7ff)) > 49) {   /* 3rd iteration, 151 bits */
                double t2 = r;
                f3 = PIO2_3 * fn;
                r = r - f3;
                w = fmadd_d(PIO2_3T, fn, -((t2 - r) - f3));
                y = r - w;
                f2 = t2;
            }
        }
    }
    double yl = (r - y) - w;
    if ((s32)hx < 0) { n = -n; y = -y; yl = -yl; }
    *y0 = y; *y1 = yl;
    s->r3 = (u32)n;
    s->f2 = f2; s->f3 = f3; s->f4 = r; s->w3 = 1; s->w4 = 1;
    return n;
}

static void put(const St *s, double f1) {
    F[1].ps0 = f1;
    R[3] = s->r3; R[4] = s->r4;
    F[2].ps0 = s->f2;
    if (s->w3) F[3].ps0 = s->f3;
    if (s->w4) F[4].ps0 = s->f4;
    if (s->w6) R[6] = s->r6;
    if (s->ca0) cpu.xer_ca = 0;
}

/* the result and registers of sin (c 0) or cos (c 1) of x for 0x3fe921fb < ix <= 0x413921fb, from
 * rem_pio2's n, y0, y1 and the registers it left in *s */
static inline __attribute__((always_inline)) double from_rem(St *s, s32 n, double y0, double y1) {
    switch (n & 3) {
    case 0: s->r3 = 1; return k_sin(s, y0, y1, 1);
    case 1: return k_cos(s, y0, y1);
    case 2: s->r3 = 1; return fneg_d(k_sin(s, y0, y1, 1));
    default: return fneg_d(k_cos(s, y0, y1));
    }
}

/* sin/cos pairs: the game mostly calls sin(a) then cos(a) (0x8019fc14: three angles), ~900 of each per 2v2
 * frame. When the last sin was followed by cos of the same argument, a sin call also computes cos (sharing
 * rem_pio2, the two kernels independent) and keeps its result and registers; the cos call then only writes
 * them. sin and cos are pure functions of f1, so the cos call leaves exactly what its own evaluation would. */
static u64 memo_x;
static St memo_s;
static double memo_r;
static int memo_ok, pair_hint = 1, sin_pending;
static u64 sin_x;
u32 recomp_libm_pairs, recomp_libm_pair_waste;   /* cos calls served from a pair; pairs not used */

/* 0x80168f38 sin / 0x801689d0 cos; 0: done, 1: translated body needed */
static int nat_sincos(int c) {
    double x = F[1].ps0, res;
    u64 xb = d2u(x);
    if (c && memo_ok && xb == memo_x) { memo_ok = 0; recomp_libm_pairs++; put(&memo_s, memo_r); return 0; }
    if (memo_ok) { memo_ok = 0; pair_hint = 0; recomp_libm_pair_waste++; }
    if (c && sin_pending && xb == sin_x) pair_hint = 1;   /* a pair the last sin did not compute */
    sin_pending = 0;
    u32 ix = hi(x) & 0x7fffffffu;
    St s = {0};
    if ((s32)ix > 0x413921fb) return 1;
    if (!c && pair_hint) {
        St sc;
        double rc;
        if ((s32)ix <= 0x3fe921fb) {
            s.f2 = 0.0; sc = s;
            s.r3 = 0; res = k_sin(&s, x, 0.0, 0);
            sc.r3 = ix; rc = k_cos(&sc, x, 0.0);
        } else {
            double y0, y1;
            s32 n = rem_pio2(&s, x, &y0, &y1);
            s.f2 = y1; sc = s;
            res = from_rem(&s, n, y0, y1);
            rc = from_rem(&sc, n + 1, y0, y1);   /* cos(x) = sin(x + pi/2) */
        }
        memo_x = xb; memo_s = sc; memo_r = rc; memo_ok = 1;
        put(&s, res);
        return 0;
    }
    if (!c) { sin_pending = 1; sin_x = xb; }
    if ((s32)ix <= 0x3fe921fb) {
        s.f2 = 0.0;
        if (c) { s.r3 = ix; res = k_cos(&s, x, 0.0); }
        else { s.r3 = 0; res = k_sin(&s, x, 0.0, 0); }
    } else {
        double y0, y1;
        s32 n = rem_pio2(&s, x, &y0, &y1) + c;
        s.f2 = y1;
        res = from_rem(&s, n, y0, y1);
    }
    put(&s, res);
    return 0;
}

int recomp_libm_off;   /* ppc.h; sdmc:/cr_libm_off.txt, bottom-screen button "Native sin/cos" */
u32 recomp_libm_flips;
static void libm_changed(void) { recomp_libm_flips++; }
#define libm_off recomp_libm_off
static void nat_sin(void) { if (nat_sincos(0)) f_80168f38(); }
static void nat_cos(void) { if (nat_sincos(1)) f_801689d0(); }
/* sinf/cosf: frsp of the double result into ps0 and ps1 */
static void nat_sinf(void) { if (nat_sincos(0)) { f_801693dc(); return; } double v = fsgl(F[1].ps0); F[1].ps0 = v; F[1].ps1 = v; }
static void nat_cosf(void) { if (nat_sincos(1)) { f_80169400(); return; } double v = fsgl(F[1].ps0); F[1].ps0 = v; F[1].ps1 = v; }

void sdk_sin(void) { NATIVE_PROF("sin"); if (libm_off) f_80168f38(); else nat_sin(); }
void sdk_cos(void) { NATIVE_PROF("cos"); if (libm_off) f_801689d0(); else nat_cos(); }
void sdk_sinf(void) { NATIVE_PROF("sinf"); if (libm_off) f_801693dc(); else nat_sinf(); }
void sdk_cosf(void) { NATIVE_PROF("cosf"); if (libm_off) f_80169400(); else nat_cosf(); }

/* ---- sdmc:/cr_bench.txt: native vs translated ---- */
#define B_STACK 0x8170f000u   /* scratch stack (mtx.c's bench area), saved and restored */
static u32 rng = 4242;
static u32 rnd32(void) { rng = rng * 1664525u + 1013904223u; return rng; }
static double rnd_u(void) { return (rnd32() >> 8) * (1.0 / 16777216.0); }   /* [0, 1) */

static double input(int i) {
    double pio2 = 1.5707963267948966;
    double sg = (rnd32() & 1) ? -1.0 : 1.0;
    switch (i % 9) {
    case 0: return sg * rnd_u() * 4.0;                                  /* small, both kernels */
    case 1: return sg * rnd_u() * 1000.0;                               /* medium reduction */
    case 2: return sg * rnd_u() * 900000.0;                             /* up to and past 2^19 pi/2 */
    case 3: return (float)(sg * rnd_u() * 200.0);                      /* float arguments (sinf/cosf callers) */
    case 4: return sg * ((rnd32() % 64) * pio2 + (rnd_u() - 0.5) * 1e-6);   /* near n pi/2: more iterations */
    case 5: return sg * ((rnd32() % 64) * pio2) * (1.0 + (rnd_u() - 0.5) * 1e-15);
    case 6: return sg * rnd_u() * 1e-8;                                 /* tiny: kernel early returns */
    case 7: return u2d(((u64)(0x3fe921f0u + rnd32() % 32) << 32) | rnd32());   /* pi/4 boundary */
    default: {
        static const u64 sp[] = { 0x7ff0000000000000ull, 0xfff0000000000000ull, 0x7ff8000000000000ull, 0x0ull,
                                  0x8000000000000000ull, 0x4002d97c00000000ull, 0x413921fb00000000ull,
                                  0x413921fc00000000ull, 0x3ff921fb54442d18ull, 0x400921fb54442d18ull };
        return u2d(sp[rnd32() % (sizeof sp / sizeof *sp)]);
    }
    }
}

/* the registers compared: r3, r4, r6, f1-f4 (ps0 and ps1), CA. Not r1: the translated call can leave
 * its own frame's r1 in cpu (stored before its inner calls), which no caller reads */
static int same(const CPU *a, const CPU *b) {
    for (int k = 3; k <= 6; k++) if (k != 5 && a->gpr[k] != b->gpr[k]) return 0;
    for (int k = 1; k <= 4; k++)
        if (d2u(a->fpr[k].ps0) != d2u(b->fpr[k].ps0) || d2u(a->fpr[k].ps1) != d2u(b->fpr[k].ps1)) return 0;
    return a->xer_ca == b->xer_ca;
}

void recomp_libm_bench(void) {
    static void (*const tr[4])(void) = { f_80168f38, f_801689d0, f_801693dc, f_80169400 };
    static void (*const nat[4])(void) = { nat_sin, nat_cos, nat_sinf, nat_cosf };
    static const char *const name[4] = { "sin", "cos", "sinf", "cosf" };
    static u8 save_ram[0x1000]; static CPU save;
    memcpy(save_ram, ram + ((B_STACK - 0x1000) & RAM_MASK), sizeof save_ram);
    save = cpu;
    for (int f = 0; f < 4; f++) {
        u32 bad = 0, fb = 0; const int N = 20000;
        for (int i = 0; i < N; i++) {
            CPU c = save;   /* poison every register compared, ps1 included */
            c.gpr[1] = B_STACK; c.gpr[3] = rnd32(); c.gpr[4] = rnd32(); c.gpr[6] = rnd32(); c.xer_ca = rnd32() & 1;
            for (int k = 1; k <= 4; k++) { c.fpr[k].ps0 = rnd_u() - 0.5; c.fpr[k].ps1 = rnd_u() + 7.0; }
            c.fpr[1].ps0 = input(i); c.lr = 0x80001234u;
            CPU a, b;
            cpu = c; tr[f](); a = cpu;
            cpu = c; nat[f](); b = cpu;
            fb += (s32)(hi(c.fpr[1].ps0) & 0x7fffffffu) > 0x413921fb;
            if (!same(&a, &b)) {
                if (bad++ < 4)
                    recomp_log("# bench libm %s mismatch x %016llx: f1 %016llx/%016llx r3 %08lx/%08lx r4 %08lx/%08lx\n", name[f],
                               d2u(c.fpr[1].ps0), d2u(a.fpr[1].ps0), d2u(b.fpr[1].ps0), a.gpr[3], b.gpr[3], a.gpr[4], b.gpr[4]);
            }
        }
        recomp_log("# bench libm %s native-vs-translated: %d inputs (%lu via the translated body), %lu differ\n", name[f], N, fb, bad);
    }
    for (int f = 0; f < 2; f++) {   /* pairs: native sin(f) then cos(f) of the same input vs the translated cos(f) */
        u32 bad = 0, hits = 0; const int N = 20000;
        for (int i = 0; i < N; i++) {
            CPU c = save;
            c.gpr[1] = B_STACK; c.gpr[3] = rnd32(); c.gpr[4] = rnd32(); c.gpr[6] = rnd32(); c.xer_ca = rnd32() & 1;
            for (int k = 1; k <= 4; k++) { c.fpr[k].ps0 = rnd_u() - 0.5; c.fpr[k].ps1 = rnd_u() + 7.0; }
            c.fpr[1].ps0 = input(i); c.lr = 0x80001234u;
            CPU a, b;
            cpu = c; tr[1 + 2 * f](); a = cpu;
            cpu = c; pair_hint = 1; nat[2 * f]();
            u32 p0 = recomp_libm_pairs;
            cpu = c; nat[1 + 2 * f](); b = cpu;
            hits += recomp_libm_pairs != p0;
            if (!same(&a, &b) && bad++ < 4)
                recomp_log("# bench libm pair %s mismatch x %016llx: f1 %016llx/%016llx\n", name[1 + 2 * f], d2u(c.fpr[1].ps0),
                           d2u(a.fpr[1].ps0), d2u(b.fpr[1].ps0));
        }
        recomp_log("# bench libm pair %s then %s: %d inputs, %lu served by the pair, %lu differ\n", name[2 * f], name[1 + 2 * f], N, hits, bad);
    }
    for (int f = 0; f < 2; f++) {   /* timings on medium arguments */
        u64 ts[2];
        for (int m = 0; m < 2; m++) {
            const int N = 20000;
            u64 t0 = svcGetSystemTick();
            for (int i = 0; i < N; i++) { cpu.gpr[1] = B_STACK; cpu.fpr[1].ps0 = 0.5 + (i & 255) * 0.03; (m ? nat : tr)[f](); }
            ts[m] = (svcGetSystemTick() - t0) / N;
        }
        recomp_log("# bench time libm %s translated %.0f ns native %.0f ns per call\n", name[f],
                   ts[0] * 1e9 / SYSCLOCK_ARM11, ts[1] * 1e9 / SYSCLOCK_ARM11);
    }
    cpu = save;
    memcpy(ram + ((B_STACK - 0x1000) & RAM_MASK), save_ram, sizeof save_ram);
}

void recomp_libm_init(void) {
    FILE *f = fopen("sdmc:/cr_libm_off.txt", "rb");
    if (f) fclose(f);
    libm_off = f != NULL;
    recomp_log("# libm: sin/cos/sinf/cosf %s (sdmc:/cr_libm_off.txt)\n", libm_off ? "translated" : "native");
    recomp_menu_bind("Native sin/cos", &recomp_libm_off, libm_changed);
    f = fopen("sdmc:/cr_bench.txt", "rb");
    if (f) { fclose(f); recomp_libm_bench(); }
}
