/* CPU state, guest RAM and the hardware-layer hooks the translated code
 * calls. Phase 2: logging placeholders only; later phases replace them. */
#include <3ds.h>
#include <stdio.h>
#include <stdlib.h>
#include "ppc.h"

CPU cpu;
/* static: its address is a link-time constant in gen/, not a pointer reloaded after
 * every byte store (gen/ is built with -fno-strict-aliasing) */
/* ram[] and ram_wr[] sit RAM_GUARD bytes (32 pages) inside larger blocks: a base + offset access
 * (ppc.h *_b) whose base is near either end of the mirror lands in the guard, not outside the array */
u8 recomp_ram_block[RAM_GUARD + RAM_MASK + 1 + 8 + RAM_GUARD] __attribute__((aligned(64)));
#define WR_ENTRIES ((RAM_GUARD >> WR_PAGE_SHIFT) * 2 + ((RAM_MASK + 1) >> WR_PAGE_SHIFT))
u32 recomp_ram_wr_block[WR_ENTRIES];
u8 recomp_ram_dirty_block[WR_ENTRIES] __attribute__((aligned(32)));
__asm__(".global ram\n.set ram, recomp_ram_block + 0x8000\n"
        ".global ram_wr\n.set ram_wr, recomp_ram_wr_block + 4 * (0x8000 >> 10)\n"
        ".global ram_dirty\n.set ram_dirty, recomp_ram_dirty_block + (0x8000 >> 10)");
u32 recomp_wr_epoch = 1;
void recomp_mark_ram(u32 a, u32 n) {
    if (!n) return;
    for (u32 p = (a & RAM_MASK) >> WR_PAGE_SHIFT, e = ((a + n - 1) & RAM_MASK) >> WR_PAGE_SHIFT; p <= e; p++) ram_wr[p] = recomp_wr_epoch;
}
/* ppc.h WR_SET: the pages guest stores marked in ram_dirty[] get the current epoch in ram_wr[], then the
 * marks are cleared. The GX thread may test pages meanwhile (gx.c range_clean reads the marks, a barrier,
 * then ram_wr[]), so every epoch is written before any mark is cleared. Guard entries are swept too. */
u32 recomp_wr_sweep_pages; u64 recomp_wr_sweep_ticks;   /* for the "# gx queue" line */
void recomp_wr_sweep(void) {
#ifdef RECOMP_WR_EPOCH_STORES
    return;   /* make WRMAP=0: no marks */
#endif
    u64 t0 = svcGetSystemTick();
    /* 32 marks per test: a 2v2 frame marks ~700 pages of 32832, in clusters */
    u32 *d = (u32 *)recomp_ram_dirty_block, e = recomp_wr_epoch, n = 0;
    static u16 hit[WR_ENTRIES / 32];
    u32 nh = 0;
    for (u32 i = 0; i < WR_ENTRIES / 4; i += 8) {
        if (!(d[i] | d[i + 1] | d[i + 2] | d[i + 3] | d[i + 4] | d[i + 5] | d[i + 6] | d[i + 7])) continue;
        hit[nh++] = i;
        for (u32 w = i; w < i + 8; w++) {
            u32 m = d[w], *p = recomp_ram_wr_block + w * 4;
            if (!m) continue;
            if (m & 0xffu) { p[0] = e; n++; }
            if (m & 0xff00u) { p[1] = e; n++; }
            if (m & 0xff0000u) { p[2] = e; n++; }
            if (m & 0xff000000u) { p[3] = e; n++; }
        }
    }
    if (nh) {
        __dmb();
        for (u32 h = 0; h < nh; h++) memset(d + hit[h], 0, 32);
    }
    recomp_wr_sweep_pages += n;
    recomp_wr_sweep_ticks += svcGetSystemTick() - t0;
}

void cpu_init(void) {
    memset(&cpu, 0, sizeof cpu);
    R[1] = STACK_TOP;
    R[2] = 0x804581a0u;
    R[13] = 0x804543c0u;
}

void recomp_skipped(u32 addr) { recomp_log("recomp_skipped %08lx lr=%08lx\n", (unsigned long)addr, (unsigned long)cpu.lr); }

/* hardware registers: not emulated in phase 3, each address logged a few times */
static int hw_seen(u32 a) {
    static u32 seen[256];
    static u8 n[256];
    u32 h = (a >> 1) & 255;
    if (seen[h] != a) { seen[h] = a; n[h] = 0; }
    return n[h] < 4 ? ++n[h] : 0;
}
u32 hw_read(u32 a, int size) {
    if (hw_seen(a)) recomp_log("hw_read %08lx/%d lr=%08lx\n", (unsigned long)a, size, (unsigned long)cpu.lr);
    return 0;
}
/* make HWCHECK=verify: a load that reached a hardware address (ppc.h LOAD_HW). Every one is counted and the
 * first 64 are logged with the guest lr, so a replay without "# hwcheck load" lines made none */
u32 recomp_hwcheck_loads;
u32 hw_read_verify(u32 a, int size) {
    if (recomp_hwcheck_loads++ < 64)
        recomp_log("# hwcheck load %08lx/%d lr=%08lx (load %lu)\n", (unsigned long)a, size, (unsigned long)cpu.lr,
                   (unsigned long)recomp_hwcheck_loads);
    return hw_read(a, size);
}
#if defined(RECOMP_ALIGN_VERIFY) || defined(RECOMP_RANGE_VERIFY)
/* make ALIGNVERIFY=1: misaligned 16/32-bit guest accesses per host site (the inlined helper in a gen/ f_X or a
 * runtime/ function; map with addr2line), with the first address and guest lr; counts every 600 retraces.
 * make RANGEVERIFY=1: the same lines for 8/16/32-bit accesses outside 0x80000000-0x81ffffff. */
static struct { const void *pc; u32 a, lr, n; u8 size; } al_site[512];
static u32 al_sites, al_total[5], al_lost;
void recomp_align_miss(u32 a, int n) {
    const void *pc = __builtin_return_address(0);
    al_total[n]++;
    for (u32 i = 0; i < al_sites; i++) if (al_site[i].pc == pc) { al_site[i].n++; return; }
    if (al_sites == 512) { al_lost++; return; }
    al_site[al_sites].pc = pc; al_site[al_sites].a = a; al_site[al_sites].lr = cpu.lr; al_site[al_sites].size = n;
    al_site[al_sites++].n = 1;
    recomp_log("# align site %lu: host %p addr %08lx size %d lr=%08lx\n", (unsigned long)al_sites, pc, (unsigned long)a, n,
               (unsigned long)cpu.lr);
}
void recomp_align_report(u32 retrace) {
    if (retrace % 600) return;
    recomp_log("# align r%lu: 8-bit %lu, 16-bit %lu, 32-bit %lu; %lu sites (%lu over the table)\n", (unsigned long)retrace,
               (unsigned long)al_total[1], (unsigned long)al_total[2], (unsigned long)al_total[4], (unsigned long)al_sites, (unsigned long)al_lost);
    for (u32 i = 0; i < al_sites; i++)
        recomp_log("# align count %p size %d %lu addr %08lx lr=%08lx\n", al_site[i].pc, al_site[i].size, (unsigned long)al_site[i].n,
                   (unsigned long)al_site[i].a, (unsigned long)al_site[i].lr);
}
#else
void recomp_align_report(u32 retrace) { (void)retrace; }
#endif
#ifdef RECOMP_HWCHECK_VERIFY
/* make HWCHECK=verify: every store that reaches hw_write, per calling host site (the inlined wr* in a gen/ f_X;
 * map with nm), with its first address and guest lr. recomp_hwcheck_report lists them every 600 retraces. */
static struct { const void *pc; u32 a, lr, n; } st_site[256];
static u32 st_sites, st_total, st_lost;
static void store_site(const void *pc, u32 a) {
    st_total++;
    for (u32 i = 0; i < st_sites; i++) if (st_site[i].pc == pc) { st_site[i].n++; return; }
    if (st_sites == 256) { st_lost++; return; }
    st_site[st_sites].pc = pc; st_site[st_sites].a = a; st_site[st_sites].lr = cpu.lr; st_site[st_sites++].n = 1;
    recomp_log("# hwcheck store site %lu: host %p addr %08lx lr=%08lx\n", (unsigned long)st_sites, pc, (unsigned long)a, (unsigned long)cpu.lr);
}
void recomp_hwcheck_report(u32 retrace) {
    if (retrace % 600) return;
    recomp_log("# hwcheck stores r%lu: %lu total, %lu sites (%lu stores at sites over the table)\n", (unsigned long)retrace,
               (unsigned long)st_total, (unsigned long)st_sites, (unsigned long)st_lost);
    for (u32 i = 0; i < st_sites; i++)
        recomp_log("# hwcheck store count %p %lu addr %08lx lr=%08lx\n", st_site[i].pc, (unsigned long)st_site[i].n,
                   (unsigned long)st_site[i].a, (unsigned long)st_site[i].lr);
}
#else
void recomp_hwcheck_report(u32 retrace) { (void)retrace; }
#endif
void hw_write(u32 a, u32 v, int size) {
#ifdef RECOMP_HWCHECK_VERIFY
    store_site(__builtin_return_address(0), a);
#endif
    if ((a & ~7u) == 0xCC008000u) { recomp_gx_fifo_write(v, size); return; }   /* GX write-gather pipe */
    if (hw_seen(a)) recomp_log("hw_write %08lx/%d = %08lx lr=%08lx\n", (unsigned long)a, size, (unsigned long)v, (unsigned long)cpu.lr);
}
u32 ppc_mfspr(int n) { return cpu.spr[n & 1023]; }
void ppc_mtspr(int n, u32 v) { cpu.spr[n & 1023] = v; }
u32 ppc_mftb(int tbr) { u64 t = recomp_timebase(); return tbr == 269 ? (u32)(t >> 32) : (u32)t; }

/* an interrupt-style guest call (VI retrace callback etc.): the interrupted
 * code's registers are saved and restored around it, as OSContext does */
void recomp_wait_point(void) { recomp_dvd_service(); recomp_ar_service(); }
void recomp_call_guest(u32 fn, u32 arg) { recomp_call_guest2(fn, arg, 0); }
void recomp_call_guest2(u32 fn, u32 arg, u32 arg2) {
    CPU saved;   /* on the host stack: a callback may reach another wait point */
    static int r1_said;   /* gen/'s *_sp stack accesses (ppc.h) assume r1 well inside the RAM mirror */
    u32 o = R[1] & RAM_MASK;
    if (!r1_said && (o < 0x10000u || o > RAM_MASK - 0x10000u)) {
        r1_said = 1;
        recomp_log("# r1 %08lx within 64 KB of the RAM mirror's end: *_sp stack accesses are wrong\n", (unsigned long)R[1]);
    }
    memcpy(&saved, &cpu, sizeof cpu);
    R[3] = arg;
    R[4] = arg2;
    R[1] = (R[1] - 0x200) & ~7u;
    cpu.lr = 0;
    call_indirect(fn);
    memcpy(&cpu, &saved, sizeof cpu);
}
void ppc_mtmsr(u32 v) { cpu.msr = v; }
void ppc_sc(u32 pc) { recomp_bad(pc, "sc"); }
void ppc_rfi(u32 pc) { recomp_bad(pc, "rfi"); }
void ppc_trap(u32 pc) { recomp_bad(pc, "trap"); }
void ppc_mtfsf(u32 mask, u64 v) {
    u32 m = 0;
    for (int i = 0; i < 8; i++) if (mask & (0x80 >> i)) m |= 0xFu << (28 - 4 * i);
    cpu.fpscr = (cpu.fpscr & ~m) | ((u32)v & m);
}
void ppc_fp_record(void) { setcrf(1, cpu.fpscr >> 28); }
/* TODO(phase 7): Dolphin's table-based estimates; see docs/recomp_unknowns.md */
double ppc_fres(double b) { return 1.0 / b; }
double ppc_frsqrte(double b) { return 1.0 / sqrt(b); }
