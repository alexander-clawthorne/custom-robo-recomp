/* AR / ARQ: auxiliary RAM (16 MB) as a host buffer. DMA copies at once;
 * an ARQ request's callback runs like the ARAM DMA interrupt would: at the
 * next wait point, or at a loop back-edge in translated code (RECOMP_POLL)
 * once the modelled DMA time has passed and MSR[EE] is set. Audio itself is phase 6; this only lets the game move
 * data to and from ARAM.
 * ARQRequest: next 0, owner 4, type 8, priority 0xc, source 0x10, dest 0x14,
 * length 0x18, callback 0x1c. Type 0 = main RAM -> ARAM, 1 = ARAM -> main RAM. */
#include <stdlib.h>
#include "ppc.h"

#define LOG() RECOMP_LOG()
#define ARAM_SIZE 0x01000000u
#define ARAM_RESERVED 0x4000u      /* ARInit keeps the first 16 KB for the OS/DSP */
#define MAXPENDING 64

static u8 *aram;
static u32 ar_top = ARAM_RESERVED;

static void dma(u32 type, u32 mram, u32 aaddr, u32 len) {
    if (!aram) aram = calloc(1, ARAM_SIZE);
    aaddr &= ARAM_SIZE - 1;
    if (aaddr + len > ARAM_SIZE) { recomp_log("# AR: DMA past ARAM end %lx+%lx\n", (unsigned long)aaddr, (unsigned long)len); return; }
    if (type == 0) { memcpy(aram + aaddr, ram + (mram & RAM_MASK), len); recomp_aram_written(aaddr, len); }   /* ax.c decode predictions */
    else { recomp_mark_ram(mram, len); memcpy(ram + (mram & RAM_MASK), aram + aaddr, len); }
}

void sdk_ARInit(void) { LOG(); if (!aram) aram = calloc(1, ARAM_SIZE); R[3] = ar_top; }
void sdk_ARAlloc(void) { LOG(); u32 p = ar_top; ar_top += R[3]; R[3] = p; }
void sdk_ARGetSize(void) { LOG(); R[3] = ARAM_SIZE; }
void sdk_ARGetDMAStatus(void) { LOG(); R[3] = 0; }
void sdk_ARStartDMA(void) { LOG(); dma(R[3], R[4], R[5], R[6]); }
void sdk_ARQInit(void) { LOG(); }

static u32 pending[MAXPENDING];
static u64 due[MAXPENDING];        /* timebase when the DMA would be done */
static int npending;
volatile u32 recomp_event_pending;

/* ARAM DMA latency: ~10 us setup plus ~80 MB/s (timebase 40.5 MHz) */
static u64 dma_ticks(u32 len) { return 400 + (u64)len / 2; }

void sdk_ARQPostRequest(void) {
    LOG();
    u32 req = R[3], type = R[5], src = R[7], dst = R[8], len = R[9];
    wr32(req + 4, R[4]); wr32(req + 8, type); wr32(req + 0xc, R[6]);
    wr32(req + 0x10, src); wr32(req + 0x14, dst); wr32(req + 0x18, len); wr32(req + 0x1c, R[10]);
    if (type == 0) dma(0, src, dst, len); else dma(1, dst, src, len);
    if (npending == MAXPENDING) recomp_ar_service();
    if (npending == MAXPENDING) { recomp_log("# ARQ: pending queue full\n"); return; }
    due[npending] = recomp_timebase() + dma_ticks(len);
    pending[npending++] = req;
    recomp_event_pending = 1;
}

/* runs callbacks in order; only_due: stop at the first request still in flight */
static void service(int only_due) {
    static int busy;
    if (busy) return;
    busy = 1;
    u64 now = only_due ? recomp_timebase() : 0;
    int n = 0;
    while (n < npending && (!only_due || due[n] <= now)) {
        u32 req = pending[n++];
        u32 cb = rd32(req + 0x1c);
        if (cb) recomp_call_guest(cb, req);
    }
    /* a callback may have posted more requests behind these */
    memmove(pending, pending + n, (npending - n) * sizeof pending[0]);
    memmove(due, due + n, (npending - n) * sizeof due[0]);
    npending -= n;
    recomp_event_pending = npending != 0;
    busy = 0;
}
void recomp_ar_service(void) { service(0); }

/* RECOMP_POLL at a loop back-edge: deliver due ARQ completions if MSR[EE] */
void recomp_poll_events(void) {
    if (cpu.msr & 0x8000u) service(1);
}

const u8 *recomp_aram(void) { return aram; }

u8 recomp_aram_read8(u32 a) { return aram ? aram[a & (ARAM_SIZE - 1)] : 0; }
