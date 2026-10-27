#---------------------------------------------------------------------------------
# custom-robo-recomp: runtime/ (hand-written SDK layer) + gen/ (translated main.dol)
# gen/ is produced by: PYTHONUTF8=1 python tools/recomp/recomp.py
#---------------------------------------------------------------------------------
.SUFFIXES:
ifeq ($(strip $(DEVKITARM)),)
$(error "Please set DEVKITARM in your environment. export DEVKITARM=<path to>devkitARM")
endif
TOPDIR ?= $(CURDIR)
include $(DEVKITARM)/3ds_rules

TARGET		:=	custom-robo-recomp
BUILD		:=	build
SOURCES		:=	runtime gen
DATA		:=
INCLUDES	:=	runtime gen
ROMFS		:=	romfs

APP_TITLE	:=	Custom Robo Recomp
APP_DESCRIPTION	:=	Static recompilation of Custom Robo
APP_AUTHOR	:=	custom-robo-recomp

ARCH	:=	-march=armv6k -mtune=mpcore -mfloat-abi=hard -mtp=soft
# gen/ is goto-heavy and has unused labels/values by construction
CFLAGS	:=	-g -Wall -Wno-unused-label -Wno-unused-but-set-variable -O2 -mword-relocations \
			-ffunction-sections -fno-strict-aliasing $(ARCH)
CFLAGS	+=	$(INCLUDE) -D__3DS__
ifeq ($(FNPROF),1)
CFLAGS	+=	-DRECOMP_FNPROF
endif
FLOATOPS ?= 1
ifeq ($(FLOATOPS),1)
CFLAGS	+=	-DRECOMP_FLOATOPS
endif
# GENOPT: extra flags for the translated gen/recomp_*.c only (e.g. GENOPT=-O3); LTO=1: link-time optimisation
GENOPT ?=
LTO ?= 0
ifeq ($(LTO),1)
CFLAGS	+=	-flto
endif
# HWCHECK: guest loads skip the hardware-address check by default (runtime/ppc.h LOAD_HW); HWCHECK=1 keeps it,
# HWCHECK=verify keeps it and logs every load that reaches a hardware address ("# hwcheck load"). Switching
# needs `make clean` (make does not track CFLAGS)
HWCHECK ?= 0
ifeq ($(HWCHECK),1)
CFLAGS	+=	-DRECOMP_HWCHECK
endif
ifeq ($(HWCHECK),verify)
CFLAGS	+=	-DRECOMP_HWCHECK_VERIFY
endif
# STORECHECK=1: every translated store tests for a hardware address again (by default only the functions that set
# one, gen/recomp_hw_*.c, do: runtime/ppc.h STORE_HW); needs `make clean` too
STORECHECK ?= 0
ifeq ($(STORECHECK),1)
CFLAGS	+=	-DRECOMP_STORECHECK
endif
# WRMAP=0: guest stores write the write-tracking epoch per page directly instead of marking the byte map
# that sdk_GXCopyDisp sweeps (runtime/ppc.h WR_SET); needs `make clean` too
WRMAP ?= 1
ifeq ($(WRMAP),0)
CFLAGS	+=	-DRECOMP_WR_EPOCH_STORES
endif
# AXFAST=0: the AX mixer fetches each source sample through recomp_aram_read8 (the old mixer, runtime/ax.c
# mix_voice) instead of decoding a frame's samples first (mix_voice_fast); needs `make clean` too
AXFAST ?= 1
ifeq ($(AXFAST),0)
CFLAGS	+=	-DRECOMP_AX_REF
endif
# AXSPEC=0: no decode predictions on the AX mix worker (runtime/ax.c spec_extend); needs `make clean` too
AXSPEC ?= 1
ifeq ($(AXSPEC),0)
CFLAGS	+=	-DRECOMP_AX_NO_SPEC
endif
# ALIGNVERIFY=1: every misaligned 16/32-bit guest access is logged per host site (runtime/ppc.h ALN, "# align" lines)
ALIGNVERIFY ?= 0
ifeq ($(ALIGNVERIFY),1)
CFLAGS	+=	-DRECOMP_ALIGN_VERIFY
endif
# RANGEVERIFY=1: every 8/16/32-bit guest access outside 0x80000000-0x81ffffff is logged the same way
RANGEVERIFY ?= 0
ifeq ($(RANGEVERIFY),1)
CFLAGS	+=	-DRECOMP_RANGE_VERIFY
endif
# AXWORKER=0: the AX mix jobs (resampling, volumes, output) run on the game thread at once instead of on a
# worker thread on core 2 (runtime/ax.c job_start); needs `make clean` too
AXWORKER ?= 1
ifeq ($(AXWORKER),0)
CFLAGS	+=	-DRECOMP_AX_NO_WORKER
endif
CXXFLAGS	:= $(CFLAGS) -fno-rtti -fno-exceptions -std=gnu++11
ASFLAGS	:=	-g $(ARCH)
# --wrap=linear*: a lock around libctru's linear heap, which the game and GX threads both use (runtime/gx.c)
LDFLAGS	=	-specs=3dsx.specs -g $(ARCH) -Wl,-Map,$(notdir $*.map) -Wl,--wrap=svcBreak \
			-Wl,--wrap=linearAlloc,--wrap=linearMemAlign,--wrap=linearFree
ifeq ($(LTO),1)
LDFLAGS	+=	-flto=20
endif
LIBS	:= -lcitro3d -lctru -lm
LIBDIRS	:= $(CTRULIB)

ifneq ($(BUILD),$(notdir $(CURDIR)))
export OUTPUT	:=	$(CURDIR)/$(TARGET)
export TOPDIR	:=	$(CURDIR)
export VPATH	:=	$(foreach dir,$(SOURCES),$(CURDIR)/$(dir))
export DEPSDIR	:=	$(CURDIR)/$(BUILD)
CFILES		:=	$(foreach dir,$(SOURCES),$(notdir $(wildcard $(dir)/*.c)))
export LD	:=	$(CC)
PICAFILES	:=	$(foreach dir,$(SOURCES),$(notdir $(wildcard $(dir)/*.v.pica)))
export OFILES_SRC	:=	$(CFILES:.c=.o)
export OFILES	:=	$(PICAFILES:.v.pica=.shbin.o) $(OFILES_SRC)
export HFILES	:=	$(PICAFILES:.v.pica=_shbin.h)
export INCLUDE	:=	$(foreach dir,$(INCLUDES),-I$(CURDIR)/$(dir)) \
			$(foreach dir,$(LIBDIRS),-I$(dir)/include) -I$(CURDIR)/$(BUILD)
export LIBPATHS	:=	$(foreach dir,$(LIBDIRS),-L$(dir)/lib)
export _3DSXDEPS	:=	$(OUTPUT).smdh
export APP_ICON := $(DEVKITPRO)/libctru/default_icon.png
export _3DSXFLAGS += --smdh=$(CURDIR)/$(TARGET).smdh

.PHONY: all clean
all: $(BUILD)
	@sh tools/buildinfo.sh $(BUILD)/buildinfo.h "$(strip $(if $(filter 1,$(LTO)),lto) $(if $(filter 1,$(HWCHECK)),hwcheck) $(if $(filter verify,$(HWCHECK)),hwverify) $(if $(filter 0,$(WRMAP)),wrepoch) $(if $(filter 1,$(STORECHECK)),storecheck) $(if $(filter 0,$(AXFAST)),axref) $(if $(filter 0,$(AXWORKER)),axsync) $(if $(filter 0,$(AXSPEC)),axnospec)$(if $(filter 1,$(ALIGNVERIFY)),alignverify) $(if $(filter 1,$(RANGEVERIFY)),rangeverify) $(GENOPT))"
	@$(MAKE) --no-print-directory -C $(BUILD) -f $(CURDIR)/Makefile
	@python3 tools/check_3dsx.py $(TARGET).3dsx
$(BUILD):
	@mkdir -p $@
clean:
	@rm -fr $(BUILD) $(TARGET).3dsx $(OUTPUT).smdh $(TARGET).elf
else
$(OUTPUT).3dsx	:	$(OUTPUT).elf $(_3DSXDEPS)
$(OUTPUT).elf	:	$(OFILES)
$(OFILES_SRC)	:	$(HFILES)
# gen/: IVOPTS folds `ram + (p & RAM_MASK)` into `ram - 0x80000000 + k` literals, which the
# 3dsx format cannot relocate (tools/check_3dsx.py, run after every build)
recomp_%.o	:	CFLAGS += -fno-ivopts $(GENOPT)
# vertex shaders: picasso, then embedded with bin2s (symbols <name>_shbin, _shbin_size)
%.shbin.o %_shbin.h : %.v.pica
	@echo $(notdir $<)
	@picasso -o $*.shbin $<
	@bin2s -a 4 -H $*_shbin.h $*.shbin | $(AS) -o $*.shbin.o
-include $(DEPSDIR)/*.d
endif
