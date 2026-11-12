#!/usr/bin/env python3
"""Unlock all context-window gates in the agy CLI (plus optional env-var restore).

Verified end-to-end on macOS (arm64), Linux (x86-64 & arm64), and Windows
(x86-64 & arm64), agy 1.4.12 - 1.5.3. Container + Go pclntab locating supports
Mach-O, ELF, and PE (arm64 & x86-64).

Gate 1 -- model.SelectorModel.offersContextWindow
    Exposes /model -> Context: [Default] [256K] [512K] [900K] (Tab key) for all
    models instead of gating on "Gemini 4 Argon" + internal experiment state.
        arm64 : mov w0, #1 ; ret           -> 20 00 80 52 c0 03 5f d6
        x86-64: mov eax, 1 ; ret           -> b8 01 00 00 00 c3

Gate 2 -- store.(*Store).ContextWindow -> store.(*Store).ContextWindowChoice
    Bypasses the "Gemini 4 Argon" whitelist and release-folded experiment gate
    so Store.ContextWindow() returns the user's persisted ContextWindowChoice
    ("256K", "512K", "900K", or "" for Default) to Manager.syncContextWindow().
    On arm64, preserves store.AutoCompactOverride priority before tail-calling
    Store.ContextWindowChoice with a clean deallocated frame (Go PCDATA safe);
    falls back to a direct entry tail-jump on x86-64 or altered layouts.

Gate 3A -- utils.GetCheckpointConfig
    Removes the hardcoded min(threshold, 256000) (0x3e800) ceiling inside
    cortex/utils/utils.GetCheckpointConfig so defaultCp.MaxTokenLimit holds the
    model's full physical budget: contextWindow - max(maxOutputTokens, 40000)
    (983,040 tokens on 1M-context / 64K-output models).
        arm64 : 24-byte 0x3e800 cmp/csel   -> 6 x nop (1f 20 03 d5)
        x86-64: 14-byte 0x3e800 cmp/cmovb  -> mov %edx, %esi + 12 x nop

Gate 3B -- config.applyFeatureProviderCheckpointOverrideConfig
    Clears expCp.MaxTokenLimit (offset +0xc) before MergeConfigsDedup merges
    CASCADE_USE_EXPERIMENT_CHECKPOINTER (which hardcodes "max_token_limit":
    "256000" in model_profiles), preserving defaultCp.MaxTokenLimit (983,040)
    while retaining all modern checkpointer strategy and summary settings.

Gate 4A -- chatconverters.(*TrajectoryChatConverter).getMaxTokensForStep
    Bypasses the hardcoded MaxTokensPerUserInputStep = 64000 (192,000 bytes)
    per-step truncation on USER_INPUT steps by returning 0 for step types 8
    and 19, while preserving MaxTokensPerStep = 16,384 for non-user/tool steps.
        arm64 : mov x2, xzr ; b <epilogue mv x0,x2> at the USER_INPUT branch
        x86-64: xor %eax, %eax ; jmp <epilogue dealloc>

Gate 4B -- serializers.DefaultSerializer.stepToMessage (OneHarness, agy >= 1.5.0)
    Routes STEP_TYPE_USER_INPUT (0xe) to the zero-token-limit branch so the
    OneHarness / AgentScript serialiser drops its 64K (192,000-byte) clamp,
    while STEP_TYPE_CHECKPOINT (0x17) keeps limit = 0 (unlimited) and every
    other step keeps MaxTokensPerStep = 16,384.
        arm64 : b.eq <rel+16 mov xM,xzr>  (was b.eq <rel+24>, skipping it)
        x86-64: je <xor zeroing>          (rel8 displacement widened)

Bonus  -- store.AutoCompactOverride (arm64, optional; skip with --no-env-fix)
    Restores os.Getenv("AGY_CLI_AUTO_COMPACT_OVERRIDE") in place of the
    constant-folded strings.TrimSpace("") call so percentage ("80%") or integer
    token overrides ("650000") work via environment variable.

Usage:
    python3 agy-patch.py --binary ~/.local/bin/agy --in-place
    python3 agy-patch.py --binary ~/.local/bin/agy --locate
    python3 agy-patch.py --binary ~/.local/bin/agy --dry-run
"""

import argparse
import hashlib
import io
import os
import plistlib
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import zipfile
import zlib

PCLNTAB_MAGIC = 0xFFFFFFF1
ENV_NAME = b"AGY_CLI_AUTO_COMPACT_OVERRIDE"

FUNC_OFFERS = "model.SelectorModel.offersContextWindow"
FUNC_STORE_CW = "store.(*Store).ContextWindow"
FUNC_STORE_CHOICE = "store.(*Store).ContextWindowChoice"
FUNC_CHECKPOINT = "utils.GetCheckpointConfig"
FUNC_EXP_CHECKPOINT = "config.applyFeatureProviderCheckpointOverrideConfig"
FUNC_SELECT_EXP_KEY = "config.selectCheckpointExperimentKey"
FUNC_MAX_STEP_TOKENS = "chatconverters.(*TrajectoryChatConverter).getMaxTokensForStep"
FUNC_ONEHARNESS_STEP = "serializers.DefaultSerializer.stepToMessage"
FUNC_OVERRIDE = "store.AutoCompactOverride"
FUNC_HUB_WEB_BUNDLE = "agy_hub_web_bundle.GetFS"

TARGET_SYMBOLS = (
    FUNC_OFFERS,
    FUNC_STORE_CW,
    FUNC_STORE_CHOICE,
    FUNC_CHECKPOINT,
    FUNC_EXP_CHECKPOINT,
    FUNC_SELECT_EXP_KEY,
    FUNC_MAX_STEP_TOKENS,
    FUNC_ONEHARNESS_STEP,
    FUNC_OVERRIDE,
    FUNC_HUB_WEB_BUNDLE,
    "strings.TrimSpace",
    "os.Getenv",
)

NOP_ARM64 = struct.pack("<I", 0xD503201F)
RET_TRUE_ARM64 = struct.pack("<II", 0x52800020, 0xD65F03C0)      # mov w0,#1 ; ret
RET_TRUE_X64 = bytes.fromhex("b801000000c3")                     # mov eax,1 ; ret

# arm64 folded Getenv setup: nop ; mov x0,xzr ; mov x1,xzr
GETENV_STUB_ARM64 = struct.pack("<III", 0xD503201F, 0xAA1F03E0, 0xAA1F03E1)

# Gate 3 exact 0x3e800 (256,000) clamp sequences inside utils.GetCheckpointConfig
CLAMP_256K_ARM64 = bytes.fromhex(
    "04009dd2"  # mov  x4, #0xe800
    "6400a0f2"  # movk x4, #0x3, lsl #16   (0x3e800 = 256000)
    "7f00046b"  # cmp  w3, w4
    "04009dd2"  # mov  x4, #0xe800
    "6400a0f2"  # movk x4, #0x3, lsl #16
    "6330849a"  # csel x3, x3, x4, lo
)
CLAMP_256K_X64 = bytes.fromhex(
    "81fa00e80300"  # cmp  $0x3e800, %edx
    "be00e80300"    # mov  $0x3e800, %esi
    "0f42f2"        # cmovb %edx, %esi
)
PATCH_256K_X64 = bytes.fromhex("89d6") + b"\x90" * 12  # mov %edx, %esi + 12x nop

# Gate 4A step-type dispatch (USER_INPUT = 8 and 19) and return-value plumbing
USER_INPUT_STEP_TYPES = (0x8, 0x13)
MOV_X2_XZR = 0xAA1F03E2   # mov x2, xzr
MOV_X0_X2 = 0xAA0203E0    # mov x0, x2 (epilogue return-value setup)
X64_ZERO_EAX = b"\x31\xc0"  # xor %eax, %eax

# Gate 4B OneHarness stepToMessage dispatch: `cmp wR,#0xe ; b.eq ...`
ONEHARNESS_BEQ_WORD = 0x540000C0   # b.eq +0x18 (skips the zero-limit branch)
ONEHARNESS_BEQ_PATCHED = 0x54000060  # b.eq +0xc (lands on mov xM, xzr)


class PatchError(Exception):
    pass


# --------------------------------------------------------------------------
# Binary container: locate machine-code section (.text / __text)
# --------------------------------------------------------------------------

def find_text_section(data):
    """Return (text_vmaddr, text_fileoff, arch)."""
    if data[:4] == b"\xcf\xfa\xed\xfe":
        return _macho_text(data)
    if data[:4] == b"\x7fELF":
        return _elf_text(data)
    if data[:2] == b"MZ":
        return _pe_text(data)
    raise PatchError("unrecognised binary container (expected Mach-O, ELF, or PE)")


def _macho_text(data):
    ncmds = struct.unpack_from("<I", data, 16)[0]
    cputype = struct.unpack_from("<I", data, 4)[0]
    arch = {0x0100000C: "arm64", 0x01000007: "x86-64"}.get(cputype)
    if arch is None:
        raise PatchError(f"unsupported Mach-O cputype {cputype:#x}")
    off = 32
    for _ in range(ncmds):
        cmd, size = struct.unpack_from("<II", data, off)
        if cmd == 0x19:  # LC_SEGMENT_64
            nsects = struct.unpack_from("<I", data, off + 64)[0]
            sect = off + 72
            for _ in range(nsects):
                if data[sect:sect + 16].rstrip(b"\0") == b"__text":
                    addr, _sz, offset = struct.unpack_from("<QQI", data, sect + 32)
                    return addr, offset, arch
                sect += 80
        off += size
    raise PatchError("__text section not found in Mach-O")


def _elf_file_to_va(data, file_off):
    phoff = struct.unpack_from("<Q", data, 32)[0]
    phentsize, phnum = struct.unpack_from("<HH", data, 54)
    for i in range(phnum):
        ph = phoff + i * phentsize
        if ph + 40 > len(data):
            break
        if struct.unpack_from("<I", data, ph)[0] != 1:  # PT_LOAD
            continue
        p_offset, p_vaddr, _p_paddr, p_filesz = struct.unpack_from("<QQQQ", data, ph + 8)
        if p_offset <= file_off < p_offset + p_filesz:
            return p_vaddr + (file_off - p_offset)
    return None


def _elf_go_text_va(data, t_va, t_sz, relas, symtabs):
    """Locate Go `runtime.text` inside ELF `.text`.

    External C linkers (glibc / musl cgo PIE) place CRT startup stubs at the
    head of `.text` (`+0x80` on glibc x86-64, `+0xc0` on musl x86-64 and glibc
    arm64), while Go `functab` `entryoff` values are relative to `runtime.text`.
    """
    pcln_off, _ = find_pclntab(data)
    text_start, funcname_off = struct.unpack_from("<QQ", data, pcln_off + 24)
    if t_va <= text_start < t_va + t_sz:
        return text_start

    pcln_va = _elf_file_to_va(data, pcln_off)
    if pcln_va is not None:
        fn_va = pcln_va + funcname_off
        needle = struct.pack("<q", pcln_va)
        for roff, rsz, entsz in relas:
            pos = roff
            end = roff + rsz
            while True:
                hit = data.find(needle, pos, end)
                if hit < 0:
                    break
                pos = hit + 1
                if (hit - roff) % entsz != 16:
                    continue
                ep = hit - 16
                if ep + 2 * entsz > end:
                    continue
                mod_va = struct.unpack_from("<Q", data, ep)[0]
                r_off2, _, r_add2 = struct.unpack_from("<QQq", data, ep + entsz)
                if r_off2 != mod_va + 8 or r_add2 != fn_va:
                    continue
                mod_text = mod_etext = None
                for k in range(2, 24):
                    if ep + (k + 1) * entsz > end:
                        break
                    ro, _, ra = struct.unpack_from("<QQq", data, ep + k * entsz)
                    if ro == mod_va + 0xB0:
                        mod_text = ra
                    elif ro == mod_va + 0xB8:
                        mod_etext = ra
                if (mod_text is not None and mod_etext is not None
                        and t_va <= mod_text < mod_etext <= t_va + t_sz):
                    return mod_text

        pair = struct.pack("<QQ", pcln_va, fn_va)
        pos = 0
        while True:
            hit = data.find(pair, pos)
            if hit < 0 or hit + 0xC0 > len(data):
                break
            pos = hit + 1
            mod_text, mod_etext = struct.unpack_from("<QQ", data, hit + 0xB0)
            if t_va <= mod_text < mod_etext <= t_va + t_sz:
                return mod_text

    for sym_off, sym_sz, entsz, str_off in symtabs:
        for pos in range(sym_off, sym_off + sym_sz, entsz):
            if pos + 24 > len(data):
                break
            st_name, _info, _other, _shndx, st_val, _sz = struct.unpack_from("<IBBHQQ", data, pos)
            end = data.find(b"\0", str_off + st_name, str_off + st_name + 64)
            if end > 0 and data[str_off + st_name:end] == b"runtime.text":
                if t_va <= st_val < t_va + t_sz:
                    return st_val

    raise PatchError("cannot locate Go runtime.text base in ELF")


def _elf_text(data):
    if data[4] != 2 or data[5] != 1:
        raise PatchError("only 64-bit little-endian ELF is supported")
    machine = struct.unpack_from("<H", data, 18)[0]
    arch = {0xB7: "arm64", 0x3E: "x86-64"}.get(machine)
    if arch is None:
        raise PatchError(f"unsupported ELF machine {machine:#x}")
    shoff = struct.unpack_from("<Q", data, 40)[0]
    shentsize, shnum, shstrndx = struct.unpack_from("<HHH", data, 58)
    strtab_off = struct.unpack_from("<Q", data, shoff + shstrndx * shentsize + 24)[0]
    text_sec = None
    relas = []
    symtabs = []
    for i in range(shnum):
        sh = shoff + i * shentsize
        nameoff, sh_type = struct.unpack_from("<II", data, sh)
        end = data.find(b"\0", strtab_off + nameoff)
        if data[strtab_off + nameoff:end] == b".text":
            addr, offset, size = struct.unpack_from("<QQQ", data, sh + 16)
            text_sec = (addr, offset, size)
        elif sh_type == 4:  # SHT_RELA
            r_off, r_sz = struct.unpack_from("<QQ", data, sh + 24)
            r_ent = struct.unpack_from("<Q", data, sh + 56)[0] or 24
            relas.append((r_off, r_sz, r_ent))
        elif sh_type == 2:  # SHT_SYMTAB
            s_off, s_sz = struct.unpack_from("<QQ", data, sh + 24)
            s_link = struct.unpack_from("<I", data, sh + 40)[0]
            s_ent = struct.unpack_from("<Q", data, sh + 56)[0] or 24
            if s_link < shnum:
                s_str = struct.unpack_from("<Q", data, shoff + s_link * shentsize + 24)[0]
                symtabs.append((s_off, s_sz, s_ent, s_str))
    if text_sec is None:
        raise PatchError(".text section not found in ELF")
    t_va, t_off, t_sz = text_sec
    go_va = _elf_go_text_va(data, t_va, t_sz, relas, symtabs)
    return go_va, t_off + (go_va - t_va), arch


def _pe_text(data):
    pe = struct.unpack_from("<I", data, 0x3C)[0]
    if data[pe:pe + 4] != b"PE\0\0":
        raise PatchError("invalid PE header signature")
    machine = struct.unpack_from("<H", data, pe + 4)[0]
    arch = {0xAA64: "arm64", 0x8664: "x86-64"}.get(machine)
    if arch is None:
        raise PatchError(f"unsupported PE machine {machine:#x}")
    nsec = struct.unpack_from("<H", data, pe + 6)[0]
    opt_size = struct.unpack_from("<H", data, pe + 20)[0]
    opt = pe + 24
    image_base = struct.unpack_from("<Q", data, opt + 24)[0]
    sect = opt + opt_size
    for _ in range(nsec):
        if data[sect:sect + 8].rstrip(b"\0") == b".text":
            _vsize, vaddr, _rawsize, rawptr = struct.unpack_from("<IIII", data, sect + 8)
            return image_base + vaddr, rawptr, arch
        sect += 40
    raise PatchError(".text section not found in PE")


# --------------------------------------------------------------------------
# Go .gopclntab symbol resolution
# --------------------------------------------------------------------------

def _pclntab_section_hint(data):
    """Return a file-offset hint for .rdata / __gopclntab / .gopclntab to skip .text."""
    if data[:2] == b"MZ":
        pe = struct.unpack_from("<I", data, 0x3C)[0]
        if pe + 24 <= len(data) and data[pe:pe + 4] == b"PE\0\0":
            nsec = struct.unpack_from("<H", data, pe + 6)[0]
            opt_size = struct.unpack_from("<H", data, pe + 20)[0]
            sect = pe + 24 + opt_size
            for _ in range(nsec):
                if sect + 40 > len(data):
                    break
                if data[sect:sect + 8].rstrip(b"\0") == b".rdata":
                    return struct.unpack_from("<I", data, sect + 20)[0]
                sect += 40
    elif data[:4] == b"\xcf\xfa\xed\xfe":
        ncmds = struct.unpack_from("<I", data, 16)[0]
        off = 32
        for _ in range(ncmds):
            if off + 8 > len(data):
                break
            cmd, size = struct.unpack_from("<II", data, off)
            if cmd == 0x19 and off + 72 <= len(data):  # LC_SEGMENT_64
                nsects = struct.unpack_from("<I", data, off + 64)[0]
                sect = off + 72
                for _ in range(nsects):
                    if sect + 80 > len(data):
                        break
                    if data[sect:sect + 16].rstrip(b"\0") == b"__gopclntab":
                        return struct.unpack_from("<I", data, sect + 48)[0]
                    sect += 80
            off += size
    elif data[:4] == b"\x7fELF" and len(data) >= 64 and data[4] == 2 and data[5] == 1:
        shoff = struct.unpack_from("<Q", data, 40)[0]
        shentsize, shnum, shstrndx = struct.unpack_from("<HHH", data, 58)
        if shoff and shstrndx < shnum and shoff + shnum * shentsize <= len(data):
            strtab = struct.unpack_from("<Q", data, shoff + shstrndx * shentsize + 24)[0]
            for idx in range(shnum):
                sh = shoff + idx * shentsize
                noff = struct.unpack_from("<I", data, sh)[0]
                end = data.find(b"\0", strtab + noff, strtab + noff + 32)
                if end > 0 and data[strtab + noff:end] in (b".gopclntab", b".data.rel.ro.gopclntab"):
                    return struct.unpack_from("<Q", data, sh + 24)[0]
    return 0


def find_pclntab(data):
    magic = struct.pack("<I", PCLNTAB_MAGIC) + b"\x00\x00"
    hint = _pclntab_section_hint(data)
    for start in ((hint, 0) if 0 < hint < len(data) else (0,)):
        pos = start
        while True:
            i = data.find(magic, pos)
            if i < 0:
                break
            pos = i + 1
            # minLC is 1 on x86-64, 2 on s390x, 4 on arm64; ptrSize is 8 on 64-bit
            if i + 72 > len(data) or data[i + 6] not in (1, 2, 4) or data[i + 7] != 8:
                continue
            nfunc = struct.unpack_from("<Q", data, i + 8)[0]
            if not 1000 < nfunc < 5_000_000:
                continue
            fn_off = struct.unpack_from("<Q", data, i + 32)[0]
            if i + fn_off + 16 > len(data):
                continue
            head = data[i + fn_off:i + fn_off + 16]
            if not head or not all(0x20 <= c < 0x7F or c == 0 for c in head):
                continue
            return i, nfunc
    raise PatchError("Go pclntab not found")


def build_functions(data, target_names=None):
    """Map trailing Go func name -> (entryoff, size).

    When `target_names` is provided, resolves only those symbols via C-level
    `funcnametab` searches + `ftab` binary search (~15 ms instead of ~350 ms).
    Pass `target_names=None` for the full symbol map.
    """
    base, nfunc = find_pclntab(data)
    fn_off, cu_off, file_off, pc_off, pcln_off = struct.unpack_from("<5Q", data, base + 32)
    fnstart = base + fn_off
    fnend = base + cu_off if 0 < fn_off < cu_off and base + cu_off <= len(data) else len(data)
    pcln = base + pcln_off

    if not target_names:
        funcs = {}
        for i in range(nfunc):
            off = pcln + i * 8
            if off + 16 > len(data):
                break
            entryoff, funcoff, next_entryoff = struct.unpack_from("<III", data, off)
            rec = pcln + funcoff
            if rec + 8 > len(data):
                continue
            nameoff = struct.unpack_from("<i", data, rec + 4)[0]
            start = fnstart + nameoff
            end = data.find(b"\x00", start, start + 1024)
            if end < 0:
                continue
            try:
                name = data[start:end].decode("ascii")
            except UnicodeDecodeError:
                continue
            funcs.setdefault(name.rsplit("/", 1)[-1], (entryoff, max(0, next_entryoff - entryoff)))
        if not funcs:
            raise PatchError("functab produced no symbols")
        return funcs

    filestart, fileend = base + file_off, base + pc_off
    has_filetab = (
        0 < cu_off < file_off < pc_off
        and base + pc_off <= len(data)
        and fileend - filestart >= 4096
    )

    by_pkg = {}
    for sym in set(target_names):
        pkg = sym.split(".", 1)[0] if "." in sym else ""
        by_pkg.setdefault(pkg, []).append(sym)

    u32 = struct.Struct("<I").unpack_from
    i32 = struct.Struct("<i").unpack_from
    u32_pair = struct.Struct("<II").unpack_from

    wanted_pkgs_b = {pk.encode("ascii"): pk for pk in by_pkg if pk}
    pkg_centers = {pk: [] for pk in by_pkg}
    for idx in range(0, nfunc, 96):
        off = pcln + idx * 8
        if off + 8 > len(data):
            break
        foff = u32(data, off + 4)[0]
        if pcln + foff + 8 > len(data):
            continue
        noff = i32(data, pcln + foff + 4)[0]
        s = fnstart + noff
        if not fnstart <= s < fnend:
            continue
        end = data.find(b"\x00", s, min(fnend, s + 256))
        if end < 0:
            continue
        slash = data.rfind(b"/", s, end)
        p_start = (slash + 1) if slash >= s else s
        dot = data.find(b".", p_start, end)
        if dot > p_start:
            pk_b = bytes(data[p_start:dot])
            if pk_b in wanted_pkgs_b:
                pkg_centers[wanted_pkgs_b[pk_b]].append(s)

    wanted_nameoffs = {}

    def _scan_sym(sym, ranges):
        needle = sym.encode("ascii")
        nlen = len(needle)
        found = False
        for s_lo, s_hi in ranges:
            pos = s_lo
            while True:
                hit = data.find(needle, pos, s_hi)
                if hit < 0:
                    break
                end_idx = hit + nlen
                if end_idx < fnend and data[end_idx] == 0 and (hit == fnstart or data[hit - 1] in (0, 0x2F)):
                    if hit > fnstart and data[hit - 1] == 0x2F:
                        prev_nul = data.rfind(b"\x00", fnstart, hit)
                        full_start = (prev_nul + 1) if prev_nul >= fnstart else fnstart
                    else:
                        full_start = hit
                    wanted_nameoffs[full_start - fnstart] = sym
                    found = True
                pos = hit + 1
            if found:
                return True
        return False

    for pkg, syms in by_pkg.items():
        if has_filetab and pkg and pkg != "main":
            fneedle = b"/" + pkg.encode("ascii") + b"/"
            pkg_dirs = set()
            fpos = filestart
            while True:
                fhit = data.find(fneedle, fpos, fileend)
                if fhit < 0:
                    break
                fend = data.find(b"\x00", fhit, fileend)
                if fend < 0:
                    break
                fpos = fend + 1
                if data.find(b"/", fhit + len(fneedle), fend) < 0:
                    fnul = data.rfind(b"\x00", filestart, fhit)
                    f_start = (fnul + 1) if fnul >= filestart else filestart
                    pkg_dirs.add(bytes(data[f_start:fhit + len(fneedle)]))
            if not pkg_dirs:
                continue
        else:
            pkg_dirs = None

        wins = []
        for c in pkg_centers.get(pkg, ()):
            lo, hi = max(fnstart, c - 0x18000), min(fnend, c + 0x18000)
            if wins and lo <= wins[-1][1]:
                wins[-1] = (wins[-1][0], max(wins[-1][1], hi))
            else:
                wins.append((lo, hi))

        need_fallback = (pkg_dirs is None) or (len(wins) < len(pkg_dirs))
        syms.sort(key=len, reverse=True)
        anchor = None
        for sym in syms:
            if wins and _scan_sym(sym, wins):
                continue
            if not need_fallback:
                continue
            fallback = (
                ((max(fnstart, anchor - 0x20000), min(fnend, anchor + 0x20000)), (fnstart, fnend))
                if anchor is not None else ((fnstart, fnend),)
            )
            if _scan_sym(sym, fallback) and anchor is None:
                for noff, s_val in wanted_nameoffs.items():
                    if s_val == sym:
                        anchor = fnstart + noff
                        break

    if not wanted_nameoffs:
        return build_functions(data, target_names=None)

    first_foff = u32(data, pcln + 4)[0]
    last_foff = u32(data, pcln + (nfunc - 1) * 8 + 4)[0]
    rec_min = pcln + first_foff
    rec_max = min(len(data), pcln + last_foff + 64)

    matched_by_idx = {}
    for nameoff, sym in wanted_nameoffs.items():
        packed = struct.pack("<i", nameoff)
        lo, hi = 0, nfunc - 1
        while lo < hi:
            mid = (lo + hi) >> 1
            m_foff = u32(data, pcln + mid * 8 + 4)[0]
            m_noff = i32(data, pcln + m_foff + 4)[0]
            if m_noff < nameoff:
                lo = mid + 1
            else:
                hi = mid
        center_foff = u32(data, pcln + lo * 8 + 4)[0]
        w_lo = max(rec_min, pcln + center_foff - 0x40000)
        w_hi = min(rec_max, pcln + center_foff + 0x40000)

        for r_lo, r_hi in ((w_lo, w_hi), (rec_min, rec_max)):
            found_rec = False
            pos = r_lo
            while True:
                hit = data.find(packed, pos, r_hi)
                if hit < 0:
                    break
                r_off = hit - 4
                funcoff = r_off - pcln
                eoff_cand = u32(data, r_off)[0]
                flo, fhi = 0, nfunc - 1
                while flo <= fhi:
                    fmid = (flo + fhi) >> 1
                    ke, kf = u32_pair(data, pcln + fmid * 8)
                    if kf < funcoff:
                        flo = fmid + 1
                    elif kf > funcoff:
                        fhi = fmid - 1
                    else:
                        if ke == eoff_cand and pcln + (fmid + 1) * 8 + 4 <= len(data):
                            ne = u32(data, pcln + (fmid + 1) * 8)[0]
                            matched_by_idx[fmid] = (sym, ke, max(0, ne - ke))
                            found_rec = True
                        break
                pos = hit + 1
            if found_rec:
                break

    if not matched_by_idx:
        return build_functions(data, target_names=None)

    funcs = {}
    for idx in sorted(matched_by_idx):
        sym, ke, sz = matched_by_idx[idx]
        funcs.setdefault(sym, (ke, sz))
    return funcs


# --------------------------------------------------------------------------
# Instruction encoders / helpers
# --------------------------------------------------------------------------

def encode_adrp(rd, pc, target):
    delta = ((target & ~0xFFF) - (pc & ~0xFFF)) >> 12
    if not -(1 << 20) <= delta < (1 << 20):
        raise PatchError("adrp target out of range")
    imm21 = delta & 0x1FFFFF
    return 0x90000000 | ((imm21 & 3) << 29) | ((imm21 >> 2) << 5) | rd


def encode_add_imm(rd, rn, imm12):
    if not 0 <= imm12 <= 0xFFF:
        raise PatchError("add immediate out of range")
    return 0x91000000 | (imm12 << 10) | (rn << 5) | rd


def encode_movz64(rd, imm16):
    return 0xD2800000 | (imm16 << 5) | rd


def encode_b(pc, target):
    delta = target - pc
    if delta & 3 or not -(1 << 27) <= delta < (1 << 27):
        raise PatchError("b target out of range")
    return 0x14000000 | ((delta >> 2) & 0x3FFFFFF)


def encode_bl(pc, target):
    delta = target - pc
    if delta & 3 or not -(1 << 27) <= delta < (1 << 27):
        raise PatchError("bl target out of range")
    return 0x94000000 | ((delta >> 2) & 0x3FFFFFF)


def decode_bl_target(pc, word):
    if (word & 0xFC000000) != 0x94000000:
        return None
    imm26 = word & 0x3FFFFFF
    if imm26 & 0x2000000:
        imm26 -= 0x4000000
    return pc + imm26 * 4


# --------------------------------------------------------------------------
# Gate 1: model.SelectorModel.offersContextWindow -> return true
# --------------------------------------------------------------------------

def patch_gate1_ui(data, funcs, text_va, text_off, arch):
    if FUNC_OFFERS not in funcs:
        raise PatchError(f"symbol not found: {FUNC_OFFERS}")
    entryoff, _ = funcs[FUNC_OFFERS]
    blob = RET_TRUE_ARM64 if arch == "arm64" else RET_TRUE_X64
    off = text_off + entryoff
    if bytes(data[off:off + len(blob)]) == blob:
        return False, f"[Gate 1] {FUNC_OFFERS}: already patched"
    data[off:off + len(blob)] = blob
    return True, (f"[Gate 1] {FUNC_OFFERS}: entry {text_va + entryoff:#x} -> "
                  f"return true ({blob.hex(' ')})")


# --------------------------------------------------------------------------
# Gate 2: store.(*Store).ContextWindow -> store.(*Store).ContextWindowChoice
# --------------------------------------------------------------------------

def _try_arm64_rich_gate2(data, funcs, text_va, text_off):
    """Preserve AutoCompactOverride priority, then tail-call ContextWindowChoice.

    Returns (handled: bool, changed: bool, message: str).
    """
    if FUNC_OVERRIDE not in funcs:
        return False, False, ""

    cw_entry, cw_size = funcs[FUNC_STORE_CW]
    ch_entry, _ = funcs[FUNC_STORE_CHOICE]
    ov_entry, _ = funcs[FUNC_OVERRIDE]

    fn_off = text_off + cw_entry
    fn_va = text_va + cw_entry
    ch_va = text_va + ch_entry
    ov_va = text_va + ov_entry

    if cw_size < 0x100:
        return False, False, ""

    # 1. Locate `str x0, [sp, #imm]` in prologue (saves receiver s *Store)
    ldr_x0_sp = None
    for rel in range(0, min(0x40, cw_size), 4):
        w = struct.unpack_from("<I", data, fn_off + rel)[0]
        if (w & 0xFFC003FF) == 0xF90003E0:
            ldr_x0_sp = 0xF94003E0 | (w & 0x003FFC00)
            break
    if ldr_x0_sp is None:
        return False, False, ""

    # 2. Locate `mov x0, #-1` followed by `tbnz x0, #0x3f, ...` (or already nop)
    argon_br_rel = None
    for rel in range(0x40, cw_size - 8, 4):
        w0, w1 = struct.unpack_from("<II", data, fn_off + rel)
        if w0 == 0x92800000 and ((w1 & 0xFFF8001F) == 0xB7F80000 or w1 == 0xD503201F):
            argon_br_rel = rel + 4
            break
    if argon_br_rel is None:
        return False, False, ""

    # 3. Locate `bl store.AutoCompactOverride` + `cbnz x1, epilogue`
    bl_ov_rel = None
    for rel in range(argon_br_rel + 4, cw_size - 24, 4):
        w = struct.unpack_from("<I", data, fn_off + rel)[0]
        if decode_bl_target(fn_va + rel, w) == ov_va:
            bl_ov_rel = rel
            break
    if bl_ov_rel is None:
        return False, False, ""

    w_cbnz = struct.unpack_from("<I", data, fn_off + bl_ov_rel + 4)[0]
    if (w_cbnz & 0xFF00001F) != 0xB5000001:
        return False, False, ""
    imm19 = (w_cbnz >> 5) & 0x7FFFF
    if imm19 & 0x40000:
        imm19 -= 0x80000
    epi_rel = bl_ov_rel + 4 + imm19 * 4
    if not (0 < epi_rel <= cw_size - 12):
        return False, False, ""

    epi_w0, epi_w1, epi_w2 = struct.unpack_from("<III", data, fn_off + epi_rel)
    if epi_w2 != 0xD65F03C0:  # ret
        return False, False, ""

    # Skip optional compiler nop right after cbnz
    tail_rel = bl_ov_rel + 8
    if struct.unpack_from("<I", data, fn_off + tail_rel)[0] == 0xD503201F:
        tail_rel += 4
    if tail_rel + 16 > epi_rel:
        return False, False, ""

    b_choice = encode_b(fn_va + tail_rel + 12, ch_va)
    tail_blob = struct.pack("<IIII", ldr_x0_sp, epi_w0, epi_w1, b_choice)

    argon_site = fn_off + argon_br_rel
    tail_site = fn_off + tail_rel

    already_nop = bytes(data[argon_site:argon_site + 4]) == NOP_ARM64
    already_tail = bytes(data[tail_site:tail_site + 16]) == tail_blob
    if already_nop and already_tail:
        return True, False, f"[Gate 2] {FUNC_STORE_CW}: already patched"

    data[argon_site:argon_site + 4] = NOP_ARM64
    data[tail_site:tail_site + 16] = tail_blob
    return True, True, (
        f"[Gate 2] {FUNC_STORE_CW}: whitelist bypassed @ {fn_va + argon_br_rel:#x} "
        f"+ tail-call {FUNC_STORE_CHOICE} @ {fn_va + tail_rel:#x}"
    )


def patch_gate2_store(data, funcs, text_va, text_off, arch):
    for sym in (FUNC_STORE_CW, FUNC_STORE_CHOICE):
        if sym not in funcs:
            raise PatchError(f"symbol not found: {sym}")

    cw_entry, _ = funcs[FUNC_STORE_CW]
    ch_entry, _ = funcs[FUNC_STORE_CHOICE]
    fn_off = text_off + cw_entry
    fn_va = text_va + cw_entry
    ch_va = text_va + ch_entry

    if arch == "arm64":
        entry_jmp = struct.pack("<I", encode_b(fn_va, ch_va))
        if bytes(data[fn_off:fn_off + 4]) == entry_jmp:
            return False, f"[Gate 2] {FUNC_STORE_CW}: already patched (entry jump)"
        handled, changed, msg = _try_arm64_rich_gate2(data, funcs, text_va, text_off)
        if handled:
            return changed, msg
        data[fn_off:fn_off + 4] = entry_jmp
        return True, (f"[Gate 2] {FUNC_STORE_CW}: entry {fn_va:#x} -> "
                      f"b {FUNC_STORE_CHOICE} ({entry_jmp.hex(' ')})")

    # x86-64: 5-byte jmp rel32 at entry -> ContextWindowChoice (%rax receiver intact)
    rel32 = ch_va - (fn_va + 5)
    if not -(1 << 31) <= rel32 < (1 << 31):
        raise PatchError("jmp rel32 out of range for Store.ContextWindow")
    entry_jmp = b"\xe9" + struct.pack("<i", rel32)
    if bytes(data[fn_off:fn_off + 5]) == entry_jmp:
        return False, f"[Gate 2] {FUNC_STORE_CW}: already patched"
    data[fn_off:fn_off + 5] = entry_jmp
    return True, (f"[Gate 2] {FUNC_STORE_CW}: entry {fn_va:#x} -> "
                  f"jmp {FUNC_STORE_CHOICE} ({entry_jmp.hex(' ')})")


# --------------------------------------------------------------------------
# Gate 3A: utils.GetCheckpointConfig -> remove 256,000 (0x3e800) ceiling
# --------------------------------------------------------------------------

def _find_generic_arm64_256k_clamp(body):
    """Fallback matcher for mov xR,#0xe800 ; movk xR,#3,lsl#16 ... csel xD,xD,xR,lo."""
    hits = []
    for rel in range(0, len(body) - 24, 4):
        w0, w1 = struct.unpack_from("<II", body, rel)
        if (w0 & 0xFFFFFFE0) != 0xD29D0000 or (w1 & 0xFFFFFFE0) != 0xF2A00060:
            continue
        reg_lim = w0 & 0x1F
        if (w1 & 0x1F) != reg_lim:
            continue
        # Look ahead up to 5 instructions for `csel xD, xD, xR, lo`
        for end_rel in range(rel + 12, min(rel + 32, len(body)), 4):
            wc = struct.unpack_from("<I", body, end_rel)[0]
            if (wc & 0xFFE0FC00) == 0x9A803000:  # csel xD, xN, xM, lo (cond=0011)
                rd = wc & 0x1F
                rn = (wc >> 5) & 0x1F
                rm = (wc >> 16) & 0x1F
                if rd == rn and rm == reg_lim:
                    hits.append((rel, end_rel + 4 - rel))
                    break
    return hits


def patch_gate3a_checkpoint(data, funcs, text_va, text_off, arch):
    if FUNC_CHECKPOINT not in funcs:
        raise PatchError(f"symbol not found: {FUNC_CHECKPOINT}")
    entryoff, size = funcs[FUNC_CHECKPOINT]
    if size == 0:
        size = 0x600
    fn_off = text_off + entryoff
    fn_va = text_va + entryoff
    body = bytes(data[fn_off:fn_off + size])

    if arch == "arm64":
        nop6 = NOP_ARM64 * 6
        idx = body.find(CLAMP_256K_ARM64)
        if idx >= 0:
            if body.find(CLAMP_256K_ARM64, idx + 1) >= 0:
                raise PatchError(f"{FUNC_CHECKPOINT}: multiple 256K clamps found")
            site = fn_off + idx
            data[site:site + 24] = nop6
            return True, (f"[Gate 3A] {FUNC_CHECKPOINT}: removed 256K (0x3e800) clamp "
                          f"@ {fn_va + idx:#x} (6x nop)")

        generic = _find_generic_arm64_256k_clamp(body)
        if len(generic) == 1:
            rel, length = generic[0]
            site = fn_off + rel
            data[site:site + length] = NOP_ARM64 * (length // 4)
            return True, (f"[Gate 3A] {FUNC_CHECKPOINT}: removed 256K (0x3e800) clamp "
                          f"@ {fn_va + rel:#x} ({length // 4}x nop)")

        if body.find(nop6) >= 0:
            return False, f"[Gate 3A] {FUNC_CHECKPOINT}: already patched"
        raise PatchError(f"{FUNC_CHECKPOINT}: 256K (0x3e800) clamp pattern not found")

    # x86-64
    idx = body.find(CLAMP_256K_X64)
    if idx >= 0:
        if body.find(CLAMP_256K_X64, idx + 1) >= 0:
            raise PatchError(f"{FUNC_CHECKPOINT}: multiple 256K clamps found")
        site = fn_off + idx
        data[site:site + len(PATCH_256K_X64)] = PATCH_256K_X64
        return True, (f"[Gate 3A] {FUNC_CHECKPOINT}: removed 256K (0x3e800) clamp "
                      f"@ {fn_va + idx:#x} (mov %edx,%esi + 12x nop)")

    if body.find(PATCH_256K_X64) >= 0:
        return False, f"[Gate 3A] {FUNC_CHECKPOINT}: already patched"
    raise PatchError(f"{FUNC_CHECKPOINT}: x86-64 256K (0x3e800) clamp pattern not found")


# --------------------------------------------------------------------------
# Gate 3B: config.applyFeatureProviderCheckpointOverrideConfig
#          prevent CASCADE_USE_EXPERIMENT_CHECKPOINTER from resetting 256K
# --------------------------------------------------------------------------

def patch_gate3b_exp_checkpoint(data, funcs, text_va, text_off, arch):
    if FUNC_EXP_CHECKPOINT not in funcs:
        raise PatchError(f"symbol not found: {FUNC_EXP_CHECKPOINT}")
    entryoff, size = funcs[FUNC_EXP_CHECKPOINT]
    if size == 0:
        size = 0x200
    fn_off = text_off + entryoff
    fn_va = text_va + entryoff
    body = bytes(data[fn_off:fn_off + size])

    if arch == "arm64":
        # Primary: clear expCp.MaxTokenLimit ([x0, #0xc] = 0) before MergeConfigsDedup
        for rel in range(4, len(body) - 16, 4):
            w_prev, w0, w1 = struct.unpack_from("<III", body, rel - 4)
            if (w_prev & 0xFC000000) != 0x94000000:
                continue
            if (w0 & 0xFF00001F) == 0xB4000000 and w1 == 0xB9000C1F:
                return False, f"[Gate 3B] {FUNC_EXP_CHECKPOINT}: already patched"
            if (w0 & 0xFF00001F) != 0xB4000001:  # cbz x1, <merge_prep>
                continue
            imm19 = (w0 >> 5) & 0x7FFFF
            if imm19 & 0x40000:
                imm19 -= 0x80000
            span = imm19 * 4
            if not (8 <= span <= 64 and rel + span + 8 <= len(body)):
                continue
            wm0, wm1 = struct.unpack_from("<II", body, rel + span)
            if (wm0 & 0xFFC003FF) == 0xF94003E1 and wm1 == 0xAA0003E2:
                cbz_x0 = w0 & 0xFFFFFFE0          # cbz x0, <merge_prep>
                str_wzr = 0xB9000C1F              # str wzr, [x0, #0xc]
                blob = (
                    struct.pack("<II", cbz_x0, str_wzr)
                    + NOP_ARM64 * ((span - 8) // 4)
                )
                data[fn_off + rel:fn_off + rel + span] = blob
                return True, (
                    f"[Gate 3B] {FUNC_EXP_CHECKPOINT}: zeroed expCp.MaxTokenLimit "
                    f"@ {fn_va + rel:#x} (cbz x0 + str wzr,[x0,#0xc])"
                )

        # Fallback: bypass HasFlag(expKey) branch after selectCheckpointExperimentKey
        if FUNC_SELECT_EXP_KEY in funcs:
            sel_va = text_va + funcs[FUNC_SELECT_EXP_KEY][0]
            for rel in range(0, len(body) - 16, 4):
                w = struct.unpack_from("<I", body, rel)[0]
                if decode_bl_target(fn_va + rel, w) != sel_va:
                    continue
                for scan in range(rel + 4, min(rel + 72, len(body) - 4), 4):
                    wb, wn = struct.unpack_from("<II", body, scan)
                    if (wb & 0xFFFFFC1F) != 0xD63F0000:  # blr xR
                        continue
                    if wn == 0xD503201F:
                        return False, f"[Gate 3B] {FUNC_EXP_CHECKPOINT}: already patched"
                    if (wn & 0xFFFE001F) == 0x37000000 or (wn & 0x7F00001F) == 0x35000000:
                        data[fn_off + scan + 4:fn_off + scan + 8] = NOP_ARM64
                        return True, (
                            f"[Gate 3B] {FUNC_EXP_CHECKPOINT}: bypassed HasFlag "
                            f"override @ {fn_va + scan + 4:#x}"
                        )
        raise PatchError(f"{FUNC_EXP_CHECKPOINT}: arm64 override site not found")

    # x86-64: bypass HasFlag(expKey) `test %al,%al ; jne ...` after selectCheckpointExperimentKey
    if FUNC_SELECT_EXP_KEY not in funcs:
        raise PatchError(f"symbol not found: {FUNC_SELECT_EXP_KEY}")
    sel_va = text_va + funcs[FUNC_SELECT_EXP_KEY][0]
    for rel in range(0, len(body) - 12):
        if body[rel] != 0xE8:
            continue
        disp = struct.unpack_from("<i", body, rel + 1)[0]
        if fn_va + rel + 5 + disp != sel_va:
            continue
        window = body[rel + 5:min(len(body), rel + 105)]
        t_idx = window.find(b"\x84\xc0")  # test %al, %al
        if t_idx < 0:
            continue
        j_off = rel + 5 + t_idx + 2
        if body[j_off:j_off + 2] == b"\x90\x90":
            return False, f"[Gate 3B] {FUNC_EXP_CHECKPOINT}: already patched"
        if body[j_off] == 0x75:  # jne rel8
            data[fn_off + j_off:fn_off + j_off + 2] = b"\x90\x90"
            return True, (f"[Gate 3B] {FUNC_EXP_CHECKPOINT}: bypassed HasFlag "
                          f"override @ {fn_va + j_off:#x} (2x nop)")
        if body[j_off:j_off + 2] == b"\x0f\x85":  # jne rel32
            data[fn_off + j_off:fn_off + j_off + 6] = b"\x90" * 6
            return True, (f"[Gate 3B] {FUNC_EXP_CHECKPOINT}: bypassed HasFlag "
                          f"override @ {fn_va + j_off:#x} (6x nop)")
    raise PatchError(f"{FUNC_EXP_CHECKPOINT}: x86-64 override site not found")


# --------------------------------------------------------------------------
# Gate 4A: chatconverters.(*TrajectoryChatConverter).getMaxTokensForStep
#          disable the 64K per-USER_INPUT-step token clamp
# --------------------------------------------------------------------------

def _decode_imm19(word):
    imm = (word >> 5) & 0x7FFFF
    return imm - 0x80000 if imm & 0x40000 else imm


def _scan_arm64_step_dispatch(body):
    """Find `cmp w0,#8 ; b.eq ; cmp w0,#0x13 ; b.ne <non_user>`.

    Returns (bne_rel, non_user_rel) or None.
    """
    for rel in range(0, max(0, len(body) - 16), 4):
        w_cmp8, w_beq, w_cmp19, w_bne = struct.unpack_from("<IIII", body, rel)
        if w_cmp8 != 0x7100201F or w_cmp19 != 0x71004C1F:
            continue
        if (w_beq & 0xFF00001F) != 0x54000000:   # b.eq
            continue
        if (w_bne & 0xFF00001F) != 0x54000001:   # b.ne
            continue
        bne_rel = rel + 12
        return bne_rel, bne_rel + _decode_imm19(w_bne) * 4
    return None


def _x86_step_dispatch(body):
    """Find `cmp $0x8,%eax` ... `cmp $0x13,%eax` ; jne <non_user>.

    Returns (jne_end_rel, non_user_rel) or None.
    """
    for rel in range(0, max(0, len(body) - 24)):
        if body[rel:rel + 3] != b"\x83\xf8\x08":
            continue
        i19 = body.find(b"\x83\xf8\x13", rel + 3, rel + 19)
        if i19 < 0:
            continue
        j = i19 + 3
        if body[j] == 0x75 and j + 2 <= len(body):          # jne rel8
            target = j + 2 + struct.unpack_from("<b", body, j + 1)[0]
            return j + 2, target
        if body[j:j + 2] == b"\x0f\x85" and j + 6 <= len(body):  # jne rel32
            target = j + 6 + struct.unpack_from("<i", body, j + 2)[0]
            return j + 6, target
    return None


def _x86_find_epilogue(body, start):
    """Locate `add $imm,%rsp ; pop %rbp ; ret` or `leave ; ret` at/after `start`.

    Returns (jump_rel, end_rel) where jump_rel is the stack-dealloc site
    (or the `pop %rbp` / `leave` site when no `add` precedes it), or None.
    """
    window_end = min(len(body), start + 64)
    pop = body.find(b"\x5d\xc3", start, window_end)
    leave = body.find(b"\xc9\xc3", start, window_end)
    if leave >= 0 and (pop < 0 or leave < pop):
        return leave, leave
    if pop < 4:
        return None
    add = None
    for rel in range(max(0, pop - 16), pop - 2):
        if body[rel:rel + 3] == b"\x48\x83\xc4":       # add $imm8, %rsp
            add = rel
        elif body[rel:rel + 3] == b"\x48\x81\xc4":     # add $imm32, %rsp
            add = rel
    return (add if add is not None else pop), pop


def patch_gate4a_user_input_step(data, funcs, text_va, text_off, arch):
    if FUNC_MAX_STEP_TOKENS not in funcs:
        raise PatchError(f"symbol not found: {FUNC_MAX_STEP_TOKENS}")
    entryoff, size = funcs[FUNC_MAX_STEP_TOKENS]
    if size == 0:
        size = 0x100
    fn_off = text_off + entryoff
    fn_va = text_va + entryoff
    body = bytes(data[fn_off:fn_off + size])

    if arch == "arm64":
        found = _scan_arm64_step_dispatch(body)
        if found is None:
            raise PatchError(f"{FUNC_MAX_STEP_TOKENS}: arm64 step-type dispatch not found")
        bne_rel, non_user_rel = found
        if not 0 < non_user_rel <= len(body) - 8:
            raise PatchError(f"{FUNC_MAX_STEP_TOKENS}: arm64 b.ne target out of range")
        if struct.unpack_from("<I", body, non_user_rel + 4)[0] != MOV_X0_X2:
            raise PatchError(f"{FUNC_MAX_STEP_TOKENS}: arm64 non-user branch "
                             f"does not end in `mov x0, x2`")

        epi_mov_va = fn_va + non_user_rel + 4
        patch_rel = bne_rel + 4
        patch_va = fn_va + patch_rel
        blob = struct.pack("<II", MOV_X2_XZR, encode_b(patch_va + 4, epi_mov_va))

        if body[patch_rel:patch_rel + 8] == blob:
            return False, f"[Gate 4A] {FUNC_MAX_STEP_TOKENS}: already patched"

        data[fn_off + patch_rel:fn_off + patch_rel + 8] = blob
        return True, (f"[Gate 4A] {FUNC_MAX_STEP_TOKENS}: disabled 64K USER_INPUT "
                      f"step clamp @ {patch_va:#x} (mov x2,xzr ; b epilogue)")

    found = _x86_step_dispatch(body)
    if found is None:
        raise PatchError(f"{FUNC_MAX_STEP_TOKENS}: x86-64 step-type dispatch not found")
    user_rel, non_user_rel = found
    if not 0 < non_user_rel <= len(body) - 1:
        raise PatchError(f"{FUNC_MAX_STEP_TOKENS}: x86-64 jne target out of range")

    epi = _x86_find_epilogue(body, non_user_rel)
    if epi is None:
        raise PatchError(f"{FUNC_MAX_STEP_TOKENS}: x86-64 epilogue not found")
    jump_rel, _pop_rel = epi

    patch_va = fn_va + user_rel
    rel32 = (fn_va + jump_rel) - (patch_va + 7)
    if not -(1 << 31) <= rel32 < (1 << 31):
        raise PatchError(f"{FUNC_MAX_STEP_TOKENS}: x86-64 epilogue jump out of range")
    blob = X64_ZERO_EAX + b"\xe9" + struct.pack("<i", rel32)

    if body[user_rel:user_rel + len(blob)] == blob:
        return False, f"[Gate 4A] {FUNC_MAX_STEP_TOKENS}: already patched"

    data[fn_off + user_rel:fn_off + user_rel + len(blob)] = blob
    return True, (f"[Gate 4A] {FUNC_MAX_STEP_TOKENS}: disabled 64K USER_INPUT "
                  f"step clamp @ {patch_va:#x} (xor eax,eax ; jmp epilogue)")


# --------------------------------------------------------------------------
# Gate 4B: serializers.DefaultSerializer.stepToMessage (OneHarness, agy >= 1.5.0)
#          route STEP_TYPE_USER_INPUT to the zero-token-limit branch
# --------------------------------------------------------------------------

def _scan_oneharness_arm64(body):
    """Find `cmp wR,#0xe ; b.eq ; cmp wR,#0x17 ; b.ne ; mov xM,xzr`.

    Returns (beq_rel, beq_word) or None.
    """
    for rel in range(0, max(0, len(body) - 16), 4):
        w0, w1, w2, w3, w4 = struct.unpack_from("<IIIII", body, rel)
        if (w0 & 0xFFFFFC1F) != 0x7100381F:      # cmp wR, #0xe
            continue
        if (w1 & 0xFF00001F) != 0x54000000:      # b.eq
            continue
        if w1 not in (ONEHARNESS_BEQ_WORD, ONEHARNESS_BEQ_PATCHED):
            continue
        if w2 != (0x71005C1F | (w0 & 0x03E0)):   # cmp wR, #0x17 (same reg)
            continue
        if (w3 & 0xFF00001F) != 0x54000001:      # b.ne
            continue
        if (w4 & 0xFFFFFFE0) != 0xAA1F03E0:      # mov xM, xzr @ rel+16
            continue
        return rel + 4, w1
    return None


def _x86_match_cmp_imm8(body, pos, rex, modrm, imm8):
    n = len(body)
    if rex is not None:
        if pos + 4 <= n and body[pos] == rex and body[pos + 1] == 0x83 and body[pos + 2] == modrm and body[pos + 3] == imm8:
            return pos + 4
        return None
    if pos + 3 <= n and body[pos] == 0x83 and body[pos + 1] == modrm and body[pos + 2] == imm8:
        return pos + 3
    return None


def _x86_zero_insn_len(body, pos):
    """Return byte length (2 or 3) of `xor %r32,%r32` at `pos`, else 0."""
    n = len(body)
    if pos + 2 <= n and body[pos] == 0x31:
        modrm = body[pos + 1]
        if modrm >> 6 == 3 and (modrm >> 3) & 7 == modrm & 7:
            return 2
    if pos + 3 <= n and body[pos] == 0x45 and body[pos + 1] == 0x31:
        modrm = body[pos + 2]
        if modrm >> 6 == 3 and (modrm >> 3) & 7 == modrm & 7:
            return 3
    return 0


def _x86_find_zeroing_xor(body, start):
    """Locate `xor %r32,%r32` (or `45 31 ..` r8d-r15d form) at/after `start`."""
    n = len(body)
    for rel in range(start, min(n - 2, start + 48)):
        if _x86_zero_insn_len(body, rel) > 0:
            return rel
    return None


def _scan_oneharness_x86(body):
    """Find the OneHarness `0xe` / `0x17` step-type dispatch in x86-64 code.

    Supports both compiler branch layouts:
      - ("je", je_rel, zero_rel):
            cmp $0xe,%r32 ; je <64k> ; cmp $0x17,%r32 ; jne <16k> ; xor %rDst,%rDst
      - ("jne_fallthrough", fall_start, patched_fall):
            cmp $0xe,%r32 ; jne <c17> ; <load 64k> ; jmp <merge>
        c17:cmp $0x17,%r32 ; jne <16k> ; xor %rDst,%rDst ; jmp <merge>
    """
    n = len(body)
    for rel in range(0, max(0, n - 16)):
        if body[rel] == 0x83:
            rex, modrm, c0, br_rel = None, body[rel + 1], rel, rel + 3
        elif body[rel] == 0x41 and rel + 4 < n and body[rel + 1] == 0x83:
            rex, modrm, c0, br_rel = 0x41, body[rel + 2], rel + 1, rel + 4
        else:
            continue
        if not 0xF8 <= modrm <= 0xFF or body[c0 + 2] != 0x0E or br_rel + 1 >= n:
            continue

        op = body[br_rel]
        if op == 0x74:  # je rel8
            jne = _x86_match_cmp_imm8(body, br_rel + 2, rex, modrm, 0x17)
            if jne is None or jne + 1 >= n or body[jne] != 0x75:
                continue
            zero_rel = _x86_find_zeroing_xor(body, jne + 2)
            if zero_rel is not None:
                return ("je", br_rel, zero_rel)

        elif op == 0x75:  # jne rel8 (Go 1.24+ x86-64 fallthrough layout)
            disp0 = body[br_rel + 1]
            if not 4 <= disp0 <= 32:
                continue
            fall_start = br_rel + 2
            c17 = fall_start + disp0
            if c17 > n or body[c17 - 2] != 0xEB:  # fallthrough ends in jmp rel8
                continue
            merge_from_0e = c17 + struct.unpack_from("<b", body, c17 - 1)[0]
            jne = _x86_match_cmp_imm8(body, c17, rex, modrm, 0x17)
            if jne is None or jne + 1 >= n or body[jne] != 0x75:
                continue
            zero_rel = jne + 2
            zlen = _x86_zero_insn_len(body, zero_rel)
            jmp17 = zero_rel + zlen
            if zlen == 0 or jmp17 + 1 >= n or body[jmp17] != 0xEB:
                continue
            merge_from_17 = (jmp17 + 2) + struct.unpack_from("<b", body, jmp17 + 1)[0]
            slot_len = (c17 - 2) - fall_start
            if merge_from_0e != merge_from_17 or slot_len < zlen:
                continue
            patched_fall = body[zero_rel:zero_rel + zlen] + (b"\x90" * (slot_len - zlen))
            return ("jne_fallthrough", fall_start, patched_fall)
    return None


def patch_gate4b_oneharness_user_input(data, funcs, text_va, text_off, arch):
    if FUNC_ONEHARNESS_STEP not in funcs:
        return False, (f"[Gate 4B] {FUNC_ONEHARNESS_STEP}: not present "
                       f"(pre-1.5 binary), skipped")
    entryoff, size = funcs[FUNC_ONEHARNESS_STEP]
    if size == 0:
        size = 0x100
    fn_off = text_off + entryoff
    fn_va = text_va + entryoff
    body = bytes(data[fn_off:fn_off + size])

    if arch == "arm64":
        found = _scan_oneharness_arm64(body)
        if found is None:
            raise PatchError(f"{FUNC_ONEHARNESS_STEP}: arm64 OneHarness step "
                             f"dispatch not found")
        beq_rel, beq_word = found
        if beq_word == ONEHARNESS_BEQ_PATCHED:
            return False, f"[Gate 4B] {FUNC_ONEHARNESS_STEP}: already patched"
        struct.pack_into("<I", data, fn_off + beq_rel, ONEHARNESS_BEQ_PATCHED)
        return True, (f"[Gate 4B] {FUNC_ONEHARNESS_STEP}: routed "
                      f"STEP_TYPE_USER_INPUT (0xe) to zero limit "
                      f"@ {fn_va + beq_rel:#x}")

    found = _scan_oneharness_x86(body)
    if found is None:
        raise PatchError(f"{FUNC_ONEHARNESS_STEP}: x86-64 OneHarness step "
                         f"dispatch not found")
    if found[0] == "jne_fallthrough":
        _, fall_start, patched_fall = found
        end = fall_start + len(patched_fall)
        if body[fall_start:end] == patched_fall:
            return False, f"[Gate 4B] {FUNC_ONEHARNESS_STEP}: already patched"
        data[fn_off + fall_start:fn_off + end] = patched_fall
        return True, (f"[Gate 4B] {FUNC_ONEHARNESS_STEP}: routed "
                      f"STEP_TYPE_USER_INPUT (0xe) to zero limit "
                      f"@ {fn_va + fall_start:#x}")

    _, je_rel, zero_rel = found
    patched_disp = zero_rel - (je_rel + 2)
    if not 0 <= patched_disp <= 0x7F:
        raise PatchError(f"{FUNC_ONEHARNESS_STEP}: x86-64 je target out of range")
    if body[je_rel + 1] == patched_disp:
        return False, f"[Gate 4B] {FUNC_ONEHARNESS_STEP}: already patched"
    data[fn_off + je_rel + 1] = patched_disp
    return True, (f"[Gate 4B] {FUNC_ONEHARNESS_STEP}: routed "
                  f"STEP_TYPE_USER_INPUT (0xe) to zero limit "
                  f"@ {fn_va + je_rel:#x}")


# --------------------------------------------------------------------------
# Bonus Patch 4 (arm64 optional): restore AGY_CLI_AUTO_COMPACT_OVERRIDE
# --------------------------------------------------------------------------

def patch_env(data, funcs, text_va, text_off, arch):
    if arch != "arm64":
        raise PatchError("env-var restore is arm64-only")
    for needed in (FUNC_OVERRIDE, "strings.TrimSpace", "os.Getenv"):
        if needed not in funcs:
            raise PatchError(f"symbol not found: {needed}")

    entryoff, size = funcs[FUNC_OVERRIDE]
    start = text_off + entryoff
    body = bytes(data[start:start + (size or 512)])

    name_off = data.find(ENV_NAME)
    if name_off < 0:
        raise PatchError("env var string missing from rodata")
    target = text_va - text_off + name_off
    site_va = text_va + entryoff

    hits = [m.start() for m in re.finditer(re.escape(GETENV_STUB_ARM64), body)]
    if len(hits) != 1:
        # Check if `bl os.Getenv` is already wired inside store.AutoCompactOverride
        getenv_va = text_va + funcs["os.Getenv"][0]
        for rel in range(0, len(body) - 4, 4):
            w = struct.unpack_from("<I", body, rel)[0]
            if decode_bl_target(site_va + rel, w) == getenv_va:
                return False, f"[EnvFix] {FUNC_OVERRIDE}: already patched"
        raise PatchError(f"{FUNC_OVERRIDE}: expected 1 folded Getenv stub, found {len(hits)}")

    site = start + hits[0]
    bl_site = site + 12
    old_bl = struct.unpack_from("<I", data, bl_site)[0]
    bl_pc = site_va + hits[0] + 12
    trim_va = text_va + funcs["strings.TrimSpace"][0]
    if decode_bl_target(bl_pc, old_bl) != trim_va:
        raise PatchError("stub bl does not target strings.TrimSpace")

    words = (
        encode_adrp(0, site_va + hits[0], target),
        encode_add_imm(0, 0, target & 0xFFF),
        encode_movz64(1, len(ENV_NAME)),
        encode_bl(bl_pc, text_va + funcs["os.Getenv"][0]),
    )
    data[site:site + 16] = b"".join(struct.pack("<I", w) for w in words)
    return True, (f"[EnvFix] {FUNC_OVERRIDE}: restored {ENV_NAME.decode()} "
                  f"(strings.TrimSpace -> os.Getenv @ {bl_pc:#x})")


# --------------------------------------------------------------------------
# Gate 5: agy_hub_web_bundle.GetFS (Hub Desktop UI: Context Ring + Selector)
# --------------------------------------------------------------------------

HUB_UI_MARKER = "/*AGY_HUB_CTX_UI_V1*/"


def _slice_around(js, anchor, before=400, after=2200):
    idx = js.find(anchor)
    if idx < 0:
        raise PatchError(f"Hub UI: anchor {anchor!r} not found in main.js")
    return js[max(0, idx - before):min(len(js), idx + after)]


def _resolve_hub_js_idents(js):
    """Dynamically resolve minified React & UI component identifiers in ~5 ms using local windows."""
    win_er = _slice_around(js, ".contextWindow?.EffortRow", 400, 2200)
    m_ms = re.search(
        r'var \{setSelectedModel:[A-Za-z0-9_$]+,selectedModel:[A-Za-z0-9_$]+,'
        r'bestOfNConfig:[A-Za-z0-9_$]+,effortIndex:[A-Za-z0-9_$]+,'
        r'contextWindow:[A-Za-z0-9_$]+\}=([A-Za-z0-9_$]+)\(\)',
        win_er,
    )
    m_menu = re.search(
        r'([A-Za-z0-9_$]+)\.createElement\(([A-Za-z0-9_$]+),\{title:[A-Za-z0-9_$]+\.base\},\s*'
        r'\1\.createElement\(([A-Za-z0-9_$]+),\{selected:.*?'
        r'\1\.createElement\("span",\{"data-testid":"model-selector-effort-group".*?([A-Za-z0-9_$]+)\[[A-Za-z0-9_$]+\]\).*?'
        r'\1\.createElement\(([A-Za-z0-9_$]+),\{side:"right",align:"start".*?'
        r'\.contextWindowOptions\?\.length\?\1\.createElement\([A-Za-z0-9_$]+,\{key:[A-Za-z0-9_$]+,option:[A-Za-z0-9_$]+,effort:[A-Za-z0-9_$]+,selected:[A-Za-z0-9_$]+,contextWindow:[A-Za-z0-9_$]+\}\):'
        r'\1\.createElement\(([A-Za-z0-9_$]+),\{key:',
        win_er, re.S,
    )
    win_cws = _slice_around(js, '.get("contextWindowSelector")', 120, 3600)
    m_hkb = re.search(
        r'function ([A-Za-z0-9_$]+)\(a,b\)\{var c=[A-Za-z0-9_$]+\(\)\.get\("contextWindowSelector"\)\?\.enabled',
        win_cws,
    )
    m_seg = re.search(
        r'([A-Za-z0-9_$]+)="var\(--chart-blue.*?'
        r'([A-Za-z0-9_$]+)=\(\{segments:a,pct:b,heightClass:c="h-2"\}\)=>.*?'
        r'([A-Za-z0-9_$]+)=\(\{segments:a,groups:b,truncatedGroups:c',
        win_cws, re.S,
    )
    win_sb = _slice_around(js, "Removing the message from the queue", 600, 100)
    m_sb = re.search(
        r'([A-Za-z0-9_$]+)\([A-Za-z0-9_$]+,\{selector:[A-Za-z0-9_$]+,equalityFn:([A-Za-z0-9_$]+)\}\)\?\?\[\]',
        win_sb,
    )
    win_usage = _slice_around(js, '"View Usage"', 600, 600)
    m_usage = re.search(
        r'createElement\(([A-Za-z0-9_$]+),null\),[A-Za-z0-9_$]+\.createElement\("div",\{className:"flex flex-col gap-0\.5"\}.*?'
        r'"View Usage".*?createElement\([A-Za-z0-9_$]+,\{side:"right",align:"start",className:"w-80 max-h-80"\},'
        r'[A-Za-z0-9_$]+\.createElement\(([A-Za-z0-9_$]+),null\)\)',
        win_usage, re.S,
    )
    win_pop = _slice_around(js, '"model-selector-panel"', 200, 900)
    m_pop = re.search(
        r'createElement\(([A-Za-z0-9_$]+),\{open:[A-Za-z0-9_$]+,onOpenChange:[A-Za-z0-9_$]+\},'
        r'[A-Za-z0-9_$]+\.createElement\("div",\{className:"min-w-0"\},'
        r'[A-Za-z0-9_$]+\.createElement\(([A-Za-z0-9_$]+),\{asChild:!0\}',
        win_pop,
    )
    win_tip = _slice_around(js, '"Edit Conversation Title"', 200, 100)
    m_tip = re.search(
        r'createElement\(([A-Za-z0-9_$]+),\{id:[A-Za-z0-9_$]+,content:"Edit Conversation Title"',
        win_tip,
    )
    missing = [
        name for name, match in (
            ("ModelSelectorHook", m_ms),
            ("EffortMenu", m_menu),
            ("ContextWindowHook", m_hkb),
            ("AgentStateSelector", m_sb),
            ("UsagePanel", m_usage),
            ("PopoverMenu", m_pop),
            ("SegmentedBar", m_seg),
            ("Tooltip", m_tip),
        ) if match is None
    ]
    if missing:
        raise PatchError(f"Hub UI: unresolved minified identifiers: {', '.join(missing)}")

    R, sub_root, sub_trig, eff_map, menu_pop, menu_item = m_menu.groups()
    sb_hook, eq_fn = m_sb.groups()
    menu_sep, usage_panel = m_usage.groups()
    pop_root, pop_trig = m_pop.groups()
    palette, seg_bar, grp_tree = m_seg.groups()
    return {
        "R": R,
        "ms_hook": m_ms.group(1),
        "sub_root": sub_root,
        "sub_trig": sub_trig,
        "eff_map": eff_map,
        "menu_pop": menu_pop,
        "menu_item": menu_item,
        "cw_hook": m_hkb.group(1),
        "sb_hook": sb_hook,
        "eq_fn": eq_fn,
        "menu_sep": menu_sep,
        "usage_panel": usage_panel,
        "pop_root": pop_root,
        "pop_trig": pop_trig,
        "palette": palette,
        "seg_bar": seg_bar,
        "grp_tree": grp_tree,
        "tooltip": m_tip.group(1),
    }


def _build_hub_ui_inject_expr(idmap):
    R = idmap["R"]
    return (
        "(()=>{" + HUB_UI_MARKER
        + "var _tokFor=n=>n===\"256K\"?256000:n===\"512K\"?512000:n===\"900K\"?900000:0;"
        + "var _fmtK=n=>n>=1e6?(n/1e6).toFixed(2).replace(/\\.00$/,\"\")+\"M\":n>=1e3?Math.round(n/1e3)+\"K\":String(n);"
        + "var _EffortRow=({option:a,effort:b,selected:c,contextWindow:d})=>{"
        + f"var {{setIsOpen:f}}={idmap['ms_hook']}(),g=d?.selected||\"Default\",h=a.contextWindowOptions||[];"
        + f"return {R}.createElement({idmap['sub_root']},{{title:{idmap['eff_map']}[b]||b}},"
        + f"{R}.createElement({idmap['sub_trig']},{{selected:c,onClick:()=>{{d.select(a,g);f(!1)}}}},"
        + f"{R}.createElement(\"span\",{{\"data-testid\":\"model-selector-effort-option\",\"data-effort\":b,className:\"truncate flex-1 text-left\"}},{idmap['eff_map']}[b]||b),"
        + f"c&&{R}.createElement(\"span\",{{className:\"text-xs opacity-60 ml-2\"}},g)),"
        + f"{R}.createElement({idmap['menu_pop']},{{side:\"right\",align:\"start\",className:\"min-w-48 p-1\"}},"
        + f"{R}.createElement(\"div\",{{className:\"text-xs px-2 py-1 text-muted-foreground font-medium\"}},\"上下文窗口\"),"
        + f"h.map(k=>{{var l=c&&g===k.name;return {R}.createElement({idmap['menu_item']},{{key:k.name,selected:l,description:k.description,onClick:()=>{{d.select(a,k.name);f(!1)}}}},"
        + f"{R}.createElement(\"span\",{{className:\"flex items-center justify-between gap-2 w-full\"}},"
        + f"{R}.createElement(\"span\",{{className:\"font-medium\"}},k.name),"
        + f"k.expensive&&{R}.createElement(\"span\",{{className:\"text-xs text-amber-500\"}},\"高消耗\")))}})))}};"
        + "var _Indicator=({conversationId:a})=>{"
        + f"var [b,c]=(0,{R}.useState)(!1),d=(0,{R}.useId)(),"
        + f"f={idmap['sb_hook']}(a,{{selector:w=>w?.agentState?.contextWindowUsage,equalityFn:{idmap['eq_fn']}}}),"
        + f"g={idmap['cw_hook']}(a),h=_tokFor(g),k=f?.estimatedTokensUsed??0,"
        + "l=h>0?h:f?.maxContextTokens||983040,"
        + "m=l>0?Math.min(k/l,1):0,n=Math.round(m*100),"
        + "p=2*Math.PI*11,q=p-m*p,"
        + "t=m>=.85?\"#ef4444\":m>=.65?\"#f59e0b\":\"#22c55e\","
        + "u=(f?.tokenGroups||[]).filter(w=>w.numTokens>0),"
        + f"v=u.map((w,x)=>({{label:w.name,tokens:w.numTokens,color:{idmap['palette']}[x%{idmap['palette']}.length],pctValue:l>0?w.numTokens/l*100:0}}));"
        + f"return {R}.createElement({idmap['pop_root']},{{open:b,onOpenChange:c}},"
        + f"{R}.createElement({idmap['pop_trig']},{{asChild:!0}},"
        + f"{R}.createElement(\"button\",{{type:\"button\",\"data-tooltip-id\":d,\"aria-label\":\"上下文与额度\",className:\"flex items-center gap-1 h-7 px-1.5 rounded-full hover:bg-secondary text-xs text-muted-foreground hover:text-foreground transition-colors cursor-pointer select-none\"}},"
        + f"{R}.createElement(\"svg\",{{viewBox:\"0 0 28 28\",className:\"w-4 h-4 -rotate-90 shrink-0\"}},"
        + f"{R}.createElement(\"circle\",{{cx:\"14\",cy:\"14\",r:\"11\",fill:\"none\",stroke:\"currentColor\",strokeOpacity:\"0.22\",strokeWidth:\"2.8\"}}),"
        + f"{R}.createElement(\"circle\",{{cx:\"14\",cy:\"14\",r:\"11\",fill:\"none\",stroke:t,strokeWidth:\"3\",strokeDasharray:p,strokeDashoffset:q,strokeLinecap:\"round\"}})),"
        + f"{R}.createElement(\"span\",{{className:\"tabular-nums\"}},k>0?_fmtK(k)+\"/\"+_fmtK(l):g===\"Default\"?\"1M\":g))),"
        + f"!b&&{R}.createElement({idmap['tooltip']},{{id:d,place:\"top\"}},\"上下文: \"+k.toLocaleString()+\" / \"+l.toLocaleString()+\" tokens (\"+n+\"%) · 档位: \"+(g===\"Default\"?\"默认 1M\":g)),"
        + f"{R}.createElement({idmap['menu_pop']},{{side:\"top\",align:\"end\",offset:6,className:\"w-80 p-3 flex flex-col gap-3\"}},"
        + f"{R}.createElement(\"div\",{{className:\"flex flex-col gap-1.5\"}},"
        + f"{R}.createElement(\"div\",{{className:\"flex items-center justify-between text-sm\"}},"
        + f"{R}.createElement(\"span\",{{className:\"font-medium text-foreground\"}},\"上下文窗口 (\"+(g===\"Default\"?\"已解锁 1M\":g)+\")\"),"
        + f"{R}.createElement(\"span\",{{className:\"tabular-nums text-muted-foreground\"}},k.toLocaleString()+\" / \"+l.toLocaleString()+\" (\"+n+\"%)\")),"
        + f"v.length>0&&{R}.createElement({idmap['seg_bar']},{{segments:v,pct:w=>l>0?w/l*100:0}}),"
        + f"v.length>0&&{R}.createElement({idmap['grp_tree']},{{segments:v,groups:u,showPct:!0,hideBreakdownForGroups:[]}})),"
        + f"{R}.createElement({idmap['menu_sep']},null),"
        + f"{R}.createElement({idmap['usage_panel']},null)))}};"
        + "return{contextUsageIndicator:{Indicator:_Indicator},contextWindow:{EffortRow:_EffortRow}}})()"
    )


def _va_to_file_off(data, va):
    if data[:2] == b"MZ":
        pe = struct.unpack_from("<I", data, 0x3C)[0]
        nsec = struct.unpack_from("<H", data, pe + 6)[0]
        opt_size = struct.unpack_from("<H", data, pe + 20)[0]
        opt = pe + 24
        image_base = struct.unpack_from("<Q", data, opt + 24)[0]
        rva = va - image_base
        sect = opt + opt_size
        for _ in range(nsec):
            vsz, sva, raw_sz, raw_off = struct.unpack_from("<IIII", data, sect + 8)
            if sva <= rva < sva + raw_sz:
                return raw_off + (rva - sva)
            sect += 40
        return None
    if data[:4] == b"\xcf\xfa\xed\xfe":
        ncmds = struct.unpack_from("<I", data, 16)[0]
        off = 32
        for _ in range(ncmds):
            cmd, size = struct.unpack_from("<II", data, off)
            if cmd == 0x19:
                vmaddr, vmsize, fileoff, filesize = struct.unpack_from("<QQQQ", data, off + 24)
                if vmaddr <= va < vmaddr + filesize:
                    return fileoff + (va - vmaddr)
            off += size
        return None
    if data[:4] == b"\x7fELF":
        phoff = struct.unpack_from("<Q", data, 32)[0]
        phentsize, phnum = struct.unpack_from("<HH", data, 54)
        for i in range(phnum):
            ph = phoff + i * phentsize
            if struct.unpack_from("<I", data, ph)[0] != 1:
                continue
            p_offset, p_vaddr, _p_paddr, p_filesz = struct.unpack_from("<QQQQ", data, ph + 8)
            if p_vaddr <= va < p_vaddr + p_filesz:
                return p_offset + (va - p_vaddr)
    return None


def _locate_embedded_hub_zip(data, funcs, text_va, text_off, arch):
    """Return (zip_file_off, zip_orig_cap, len_global_offs) from agy_hub_web_bundle.GetFS."""
    if FUNC_HUB_WEB_BUNDLE not in funcs:
        return None
    entryoff, size = funcs[FUNC_HUB_WEB_BUNDLE]
    fn_off = text_off + entryoff
    fn_va = text_va + entryoff
    body = bytes(data[fn_off:fn_off + (size or 0xE0)])

    global_vas = set()
    if arch == "x86-64":
        for i in range(len(body) - 7):
            if body[i] in (0x48, 0x4C) and body[i + 1] == 0x8B and (body[i + 2] & 0xC7) == 0x05:
                disp = struct.unpack_from("<i", body, i + 3)[0]
                global_vas.add(fn_va + i + 7 + disp)
    elif arch == "arm64":
        for i in range(0, len(body) - 8, 4):
            w0, w1 = struct.unpack_from("<II", body, i)
            if (w0 & 0x9F000000) == 0x90000000 and (w1 & 0xFFC00000) == 0xF9400000:
                rd = w0 & 0x1F
                rn = (w1 >> 5) & 0x1F
                if rd == rn:
                    immlo = (w0 >> 29) & 3
                    immhi = (w0 >> 5) & 0x7FFFF
                    imm = (immhi << 2) | immlo
                    if imm & (1 << 20):
                        imm -= 1 << 21
                    page = ((fn_va + i) & ~0xFFF) + (imm << 12)
                    off12 = ((w1 >> 10) & 0xFFF) * 8
                    global_vas.add(page + off12)

    for gva in sorted(global_vas):
        # A Go slice header is {Data uintptr, Len int, Cap int} (24 bytes)
        hdr_fo = _va_to_file_off(data, gva)
        if hdr_fo is None or hdr_fo + 24 > len(data):
            continue
        ptr_va, sl_len, sl_cap = struct.unpack_from("<QQQ", data, hdr_fo)
        if not (0x10000 < sl_len <= sl_cap < 0x10000000):
            continue
        zip_fo = _va_to_file_off(data, ptr_va)
        if zip_fo is None or zip_fo + sl_cap > len(data):
            continue
        if data[zip_fo:zip_fo + 4] == b"PK\x03\x04":
            return zip_fo, sl_len, sl_cap, hdr_fo + 8
    return None


def _transform_hub_main_js(js):
    """Unlock Context Usage Indicator ring & Context Window tier selector in Hub UI main.js."""
    if HUB_UI_MARKER in js:
        return None

    # 1. Enable contextWindowSelector feature for external cohort too (no customFlag needed)
    old_cws = (
        'contextWindowSelector:{feature:{enabled:!1},applyAll:[{runtimeGating:'
        '{customFlag:"context-window-selector",userCohort:["google"]},feature:{enabled:!0}}]}'
    )
    new_cws = 'contextWindowSelector:{feature:{enabled:!0},applyAll:[]}'
    if old_cws not in js:
        raise PatchError("Hub UI: contextWindowSelector gate pattern not found")
    js = js.replace(old_cws, new_cws, 1)

    # 2. Enable contextUsageIndicator by default unless in cli-hosted-session
    w2_idx = js.find("contextUsageIndicator:{feature:!1")
    if w2_idx < 0:
        raise PatchError("Hub UI: contextUsageIndicator gate anchor not found")
    w2_end = min(len(js), w2_idx + 240)
    seg2, n1 = re.subn(
        r'contextUsageIndicator:\{feature:!1,applyAll:\[\{runtimeGating:\{customFlag:"enable-context-usage-indicator"\},\s*feature:!0\},',
        'contextUsageIndicator:{feature:!0,applyAll:[',
        js[w2_idx:w2_end],
        count=1,
    )
    if n1 != 1:
        raise PatchError("Hub UI: contextUsageIndicator gate pattern not found")
    js = js[:w2_idx] + seg2 + js[w2_end:]

    # 3. Make Context and Quota Indicator toggle visible in Appearance Settings
    w3_idx = js.find("contextUsageIndicatorEnabled:{feature:{visible:!1}")
    if w3_idx < 0:
        raise PatchError("Hub UI: contextUsageIndicatorEnabled settings anchor not found")
    w3_end = min(len(js), w3_idx + 240)
    seg3, n2 = re.subn(
        r'contextUsageIndicatorEnabled:\{feature:\{visible:!1\},applyAll:\[\{runtimeGating:\{customFlag:"enable-context-usage-indicator"\},\s*feature:\{visible:!0\}\}\]',
        'contextUsageIndicatorEnabled:{feature:{visible:!0},applyAll:[]',
        js[w3_idx:w3_end],
        count=1,
    )
    if n2 != 1:
        raise PatchError("Hub UI: contextUsageIndicatorEnabled settings gate not found")
    js = js[:w3_idx] + seg3 + js[w3_end:]

    # 4. Populate Default / 256K / 512K / 900K options when server omits contextWindowOptions
    default_opts = (
        '[{name:"Default",description:"已解锁完整物理上下文窗口 (~1M tokens)"},'
        '{name:"256K",description:"256,000 tokens (标准 1.0x 消耗)"},'
        '{name:"512K",description:"512,000 tokens (推荐甜点档位)"},'
        '{name:"900K",description:"900,000 tokens (近极限大窗口)",expensive:!0}]'
    )
    w4_idx = js.find("contextWindowOptions:a.contextWindowOptions.length?")
    if w4_idx < 0:
        raise PatchError("Hub UI: Kpa contextWindowOptions anchor not found")
    w4_end = min(len(js), w4_idx + 260)
    seg4, n3 = re.subn(
        r'contextWindowOptions:a\.contextWindowOptions\.length\?\s*a\.contextWindowOptions\.map\(\(\{name:b,description:c,expensive:d\}\)=>\(\{name:b,description:c\|\|void 0,expensive:d\|\|void 0\}\)\):void 0',
        f'contextWindowOptions:a.contextWindowOptions.length?a.contextWindowOptions.map(({{name:b,description:c,expensive:d}})=>({{name:b,description:c||void 0,expensive:d||void 0}})):{default_opts}',
        js[w4_idx:w4_end],
        count=1,
    )
    if n3 != 1:
        raise PatchError("Hub UI: Kpa contextWindowOptions mapper not found")
    js = js[:w4_idx] + seg4 + js[w4_end:]

    # 4b. Also render Indicator on empty new-chat composer before first message
    w4b_idx = js.find("contextUsageIndicator:")
    while w4b_idx >= 0:
        chunk4b = js[w4b_idx:w4b_idx + 140]
        m4b = re.match(
            r'contextUsageIndicator:([A-Za-z0-9_$]+)&&([A-Za-z0-9_$]+)&&([A-Za-z0-9_$]+)\?([A-Za-z0-9_$]+)\.createElement\(\1,\{conversationId:\3\}\):void 0',
            chunk4b,
        )
        if m4b:
            s_comp, s_en, s_cid, s_R = m4b.groups()
            repl4b = f'contextUsageIndicator:{s_comp}&&{s_en}?{s_R}.createElement({s_comp},{{conversationId:{s_cid}}}):void 0'
            js = js[:w4b_idx] + repl4b + js[w4b_idx + len(m4b.group(0)):]
            break
        w4b_idx = js.find("contextUsageIndicator:", w4b_idx + 22)

    # 5. Map selected tier to CheckpointConfig.maxTokenLimit so LS enforces 256K/512K/900K/1M cleanly
    idx_cp = js.find("CheckpointConfigSchema,{contextWindow:")
    if idx_cp < 0:
        raise PatchError("Hub UI: JS() checkpointConfig builder not found")
    s_lo, s_hi = max(0, idx_cp - 60), min(len(js), idx_cp + 120)
    seg_cp, n4 = re.subn(
        r'checkpointConfig:([A-Za-z0-9_$]+)\?([A-Za-z0-9_$]+)\(([A-Za-z0-9_$]+\.CheckpointConfigSchema),\{contextWindow:\1\}\):void 0',
        r'checkpointConfig:\1==="256K"?\2(\3,{maxTokenLimit:256000}):\1==="512K"?\2(\3,{maxTokenLimit:512000}):\1==="900K"?\2(\3,{maxTokenLimit:900000}):void 0',
        js[s_lo:s_hi],
        count=1,
    )
    if n4 != 1:
        raise PatchError("Hub UI: JS() checkpointConfig builder pattern mismatch")
    js = js[:s_lo] + seg_cp + js[s_hi:]

    idx_cp2 = js.find("CheckpointConfigSchema,{contextWindow:", idx_cp + 100)
    if idx_cp2 >= 0:
        s2_lo, s2_hi = max(0, idx_cp2 - 80), min(len(js), idx_cp2 + 120)
        seg_cp2 = re.sub(
            r'b\.checkpointConfig=([A-Za-z0-9_$]+)!==void 0\?([A-Za-z0-9_$]+)\(([A-Za-z0-9_$]+\.CheckpointConfigSchema),\{contextWindow:\1\}\):void 0',
            r'b.checkpointConfig=\1==="256K"?\2(\3,{maxTokenLimit:256000}):\1==="512K"?\2(\3,{maxTokenLimit:512000}):\1==="900K"?\2(\3,{maxTokenLimit:900000}):void 0',
            js[s2_lo:s2_hi],
            count=1,
        )
        js = js[:s2_lo] + seg_cp2 + js[s2_hi:]

    # 6. Inject Indicator & EffortRow components into buildBindings bootstrap
    idmap = _resolve_hub_js_idents(js)
    inject_expr = _build_hub_ui_inject_expr(idmap)
    old_bind = 'contextUsageIndicator:void 0,contextWindow:void 0,'
    new_bind = f'...{inject_expr},'
    if old_bind not in js:
        raise PatchError("Hub UI: buildBindings contextUsageIndicator/contextWindow slot not found")
    js = js.replace(old_bind, new_bind, 1)

    return js


def _fast_replace_zip_member(zip_bytes, target_name, new_raw_bytes, compresslevel=9):
    """Rebuild ZIP copying raw compressed streams for untouched entries and recompressing only target_name."""
    zin = zipfile.ZipFile(io.BytesIO(zip_bytes), "r")
    raw_view = memoryview(zip_bytes)
    out = io.BytesIO()
    central_dir = []

    for info in zin.infolist():
        lhdr_off = info.header_offset
        sig, ver, flags, comp_method, mtime, mdate, crc, csize, usize, fn_len, ex_len = struct.unpack_from(
            "<4sHHHHHIIIHH", raw_view, lhdr_off
        )
        if sig != b"PK\x03\x04":
            raise PatchError(f"Hub UI ZIP: bad local header signature at {lhdr_off:#x}")
        fn_bytes = bytes(raw_view[lhdr_off + 30:lhdr_off + 30 + fn_len])
        extra_bytes = bytes(raw_view[lhdr_off + 30 + fn_len:lhdr_off + 30 + fn_len + ex_len])
        data_start = lhdr_off + 30 + fn_len + ex_len

        if info.filename == target_name:
            usize = len(new_raw_bytes)
            crc = zlib.crc32(new_raw_bytes) & 0xFFFFFFFF
            cobj = zlib.compressobj(compresslevel, zlib.DEFLATED, -zlib.MAX_WBITS)
            comp_payload = cobj.compress(new_raw_bytes) + cobj.flush()
            csize = len(comp_payload)
            comp_method = zipfile.ZIP_DEFLATED
            flags &= ~0x08
        else:
            comp_payload = raw_view[data_start:data_start + info.compress_size]
            crc = info.CRC
            csize = info.compress_size
            usize = info.file_size
            flags &= ~0x08

        new_lhdr_off = out.tell()
        out.write(struct.pack(
            "<4sHHHHHIIIHH",
            b"PK\x03\x04", ver, flags, comp_method, mtime, mdate,
            crc, csize, usize, len(fn_bytes), len(extra_bytes),
        ))
        out.write(fn_bytes)
        out.write(extra_bytes)
        out.write(comp_payload)

        cd_hdr = struct.pack(
            "<4sHHHHHHIIIHHHHHII",
            b"PK\x01\x02", info.create_version, info.extract_version, flags,
            comp_method, mtime, mdate, crc, csize, usize,
            len(fn_bytes), len(info.extra), len(info.comment),
            0, info.internal_attr, info.external_attr, new_lhdr_off,
        )
        central_dir.append(cd_hdr + fn_bytes + info.extra + info.comment)

    cd_start = out.tell()
    for cd in central_dir:
        out.write(cd)
    cd_size = out.tell() - cd_start
    out.write(struct.pack(
        "<4sHHHHIIH",
        b"PK\x05\x06", 0, 0, len(central_dir), len(central_dir), cd_size, cd_start, 0,
    ))
    return out.getvalue()


def patch_gate5_hub_ui(data, funcs, text_va, text_off, arch):
    loc = _locate_embedded_hub_zip(data, funcs, text_va, text_off, arch)
    if loc is None:
        return False, f"[Gate 5] {FUNC_HUB_WEB_BUNDLE}: embedded UI archive not present, skipped"
    zip_fo, zip_len, zip_cap, len_field_fo = loc
    zip_bytes = bytes(data[zip_fo:zip_fo + zip_len])
    try:
        zin = zipfile.ZipFile(io.BytesIO(zip_bytes), "r")
        if "main.js" not in zin.namelist():
            return False, f"[Gate 5] {FUNC_HUB_WEB_BUNDLE}: main.js not in bundle, skipped"
        orig_js = zin.read("main.js").decode("utf-8")
    except Exception as exc:
        raise PatchError(f"Hub UI ZIP read failed: {exc}") from exc

    new_js = _transform_hub_main_js(orig_js)
    if new_js is None:
        return False, f"[Gate 5] {FUNC_HUB_WEB_BUNDLE}: already patched"

    new_zip = _fast_replace_zip_member(zip_bytes, "main.js", new_js.encode("utf-8"), compresslevel=9)
    if len(new_zip) > zip_cap:
        raise PatchError(f"Hub UI ZIP repack ({len(new_zip)} B) exceeds slice cap ({zip_cap} B)")

    data[zip_fo:zip_fo + len(new_zip)] = new_zip
    if len(new_zip) < zip_cap:
        data[zip_fo + len(new_zip):zip_fo + zip_cap] = b"\x00" * (zip_cap - len(new_zip))
    struct.pack_into("<Q", data, len_field_fo, len(new_zip))
    return True, (
        f"[Gate 5] {FUNC_HUB_WEB_BUNDLE}: unlocked Hub UI Context Ring + "
        f"[Default/256K/512K/900K] Selector (zip {zip_len} -> {len(new_zip)} B)"
    )


# --------------------------------------------------------------------------
# Signing & atomic write
# --------------------------------------------------------------------------

def _extract_adhoc_safe_entitlements(src_path):
    """Read existing Mach-O entitlements (e.g. allow-jit) safe for ad-hoc signing."""
    if sys.platform != "darwin" or not os.path.exists(src_path):
        return None
    res = subprocess.run(
        ["codesign", "-d", "--entitlements", ":-", src_path],
        capture_output=True,
    )
    if res.returncode != 0 or not res.stdout.strip().startswith(b"<?xml"):
        return None
    try:
        ent = plistlib.loads(res.stdout)
    except Exception:
        return None
    if not isinstance(ent, dict):
        return None
    filtered = {
        k: v for k, v in ent.items()
        if not k.startswith("com.apple.developer.")
        and k not in ("keychain-access-groups", "com.apple.application-identifier")
    }
    return plistlib.dumps(filtered) if filtered else None


def _extract_identifier(src_path):
    """Read the existing code-signing identifier (defaults to file basename).

    Signing the temp copy would otherwise bake in "<name>.tmp.<pid>" as the
    identifier, which changes the code identity that TCC and the keychain
    bind to (the desktop backend is signed as "language_server", not
    "language_server.tmp.35559").
    """
    if sys.platform != "darwin" or not os.path.exists(src_path):
        return None
    res = subprocess.run(["codesign", "-d", "--verbose=2", src_path],
                         capture_output=True, text=True)
    # codesign writes its description to stderr.
    m = re.search(r"^Identifier=(.+)$", res.stderr or res.stdout, re.M)
    return m.group(1).strip() if m else None


def resign(path, entitlements_xml=None, identifier=None):
    if sys.platform != "darwin":
        print("[i] non-macOS host: skipping codesign")
        return
    ent_file = None
    try:
        cmd = ["codesign", "--force", "--sign", "-", "--options", "runtime"]
        if identifier:
            cmd.extend(["--identifier", identifier])
        if entitlements_xml:
            fd, ent_file = tempfile.mkstemp(suffix=".entitlements.plist")
            os.write(fd, entitlements_xml)
            os.close(fd)
            cmd.extend(["--entitlements", ent_file])
        cmd.append(path)
        subprocess.run(cmd, check=True, capture_output=True)
        subprocess.run(
            ["codesign", "--verify", "--verbose=1", path],
            check=True,
            capture_output=True,
        )
    finally:
        if ent_file and os.path.exists(ent_file):
            os.unlink(ent_file)


def _strip_pe_authenticode(data):
    """Clear PE CheckSum and trailing Authenticode Certificate Table after patching."""
    if data[:2] != b"MZ":
        return
    pe = struct.unpack_from("<I", data, 0x3C)[0]
    if pe + 24 > len(data) or data[pe:pe + 4] != b"PE\0\0":
        return
    opt = pe + 24
    magic = struct.unpack_from("<H", data, opt)[0]
    dd_sec = opt + (144 if magic == 0x20B else 128 if magic == 0x10B else 0)
    if dd_sec == 0 or dd_sec + 8 > len(data):
        return
    struct.pack_into("<I", data, opt + 64, 0)
    cert_off, cert_size = struct.unpack_from("<II", data, dd_sec)
    if cert_off > 0 and cert_size > 0:
        struct.pack_into("<II", data, dd_sec, 0, 0)
        if cert_off + cert_size == len(data):
            del data[cert_off:]


def _default_binary_path():
    if sys.platform == "win32":
        local_app = os.environ.get("LOCALAPPDATA") or os.path.join(
            os.path.expanduser("~"), "AppData", "Local"
        )
        return os.path.join(local_app, "agy", "bin", "agy.exe")
    return os.path.expanduser("~/.local/bin/agy")


def _running_pids_for_binary(src):
    if sys.platform == "win32":
        exe_name = os.path.basename(src)
        res = subprocess.run(
            ["tasklist", "/FI", f"IMAGENAME eq {exe_name}", "/NH", "/FO", "CSV"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        pids = []
        for line in res.stdout.splitlines():
            parts = [p.strip('"') for p in line.split('","')]
            if len(parts) >= 2 and parts[0].lower() == exe_name.lower() and parts[1].isdigit():
                pids.append(parts[1])
        return pids
    running = subprocess.run(["pgrep", "-f", src], capture_output=True, text=True)
    return [p for p in running.stdout.split() if p != str(os.getpid())]


def _cleanup_stale_running_old(dst):
    """Best-effort removal of unlocked <dst>.running.old* files from prior Windows swaps."""
    dst_dir = os.path.dirname(dst) or "."
    prefix = os.path.basename(dst) + ".running.old"
    try:
        entries = os.listdir(dst_dir)
    except OSError:
        return
    for name in entries:
        if name == prefix or name.startswith(prefix + "."):
            try:
                os.unlink(os.path.join(dst_dir, name))
            except OSError:
                pass


def _replace_atomic_or_hotswap(tmp, dst):
    """Atomically place `tmp` at `dst`, parking a running Windows PE as `.running.old`."""
    if sys.platform == "win32":
        _cleanup_stale_running_old(dst)

    try:
        os.replace(tmp, dst)
        return None
    except PermissionError as orig_exc:
        if sys.platform != "win32" or not os.path.exists(dst):
            raise

        pid = os.getpid()
        candidates = [f"{dst}.running.old", f"{dst}.running.old.{pid}"]
        candidates.extend(f"{dst}.running.old.{pid}.{n}" for n in range(1, 8))

        parked = None
        for cand in candidates:
            if os.path.exists(cand):
                try:
                    os.unlink(cand)
                except OSError:
                    continue
            try:
                os.replace(dst, cand)
                parked = cand
                break
            except OSError:
                continue

        if parked is None:
            raise orig_exc

        try:
            os.replace(tmp, dst)
        except Exception:
            if not os.path.exists(dst) and os.path.exists(parked):
                try:
                    os.replace(parked, dst)
                except OSError:
                    pass
            raise
        return parked


def write_atomic_signed(dst, data, src_for_entitlements=None, make_backup=False):
    """Write to a fresh inode, sign, then os.replace() (with Windows hot-swap fallback)."""
    _strip_pe_authenticode(data)
    ent_xml = _extract_adhoc_safe_entitlements(src_for_entitlements or dst)
    ident = _extract_identifier(src_for_entitlements or dst)
    if make_backup and os.path.exists(dst):
        bak = dst + ".bak"
        if not os.path.exists(bak):
            shutil.copy2(dst, bak)
            print(f"[i] saved backup: {bak}")

    tmp = f"{dst}.tmp.{os.getpid()}"
    parked_old = None
    try:
        with open(tmp, "wb") as fh:
            fh.write(data)
        os.chmod(tmp, 0o755)
        resign(tmp, entitlements_xml=ent_xml, identifier=ident)
        try:
            parked_old = _replace_atomic_or_hotswap(tmp, dst)
        except PermissionError as exc:
            raise PatchError(
                f"cannot overwrite {dst} ({exc.strerror or 'access denied'}); "
                f"quit any running {os.path.basename(dst)} processes and retry"
            ) from exc
    except Exception:
        if os.path.exists(tmp):
            try:
                os.unlink(tmp)
            except OSError:
                pass
        raise

    if parked_old:
        print(f"[i] hot-swap: parked running binary at {parked_old} "
              f"(restart {os.path.basename(dst)} to load patched image)")
    if ident:
        print(f"[i] signature: identifier preserved as {ident!r}")
    if sys.platform == "darwin" and ".app/Contents/" in dst:
        print("[!] target lives inside an .app bundle: the bundle's sealed "
              "resources no longer verify\n"
              "    (codesign --verify --deep <app> now fails; the app still "
              "launches, at your own risk).")
        if os.path.exists(dst + ".bak"):
            print(f"    restore: cp -p '{dst}.bak' '{dst}'")


def main():
    for stream in (sys.stdout, sys.stderr):
        reconf = getattr(stream, "reconfigure", None)
        if callable(reconf):
            reconf(encoding="utf-8", errors="replace")

    ap = argparse.ArgumentParser(
        description="Unlock all context-window gates (1-5) in agy CLI & Antigravity Hub."
    )
    ap.add_argument("--binary", default=_default_binary_path())
    ap.add_argument("--no-env-fix", action="store_true",
                    help="skip optional AGY_CLI_AUTO_COMPACT_OVERRIDE restore")
    ap.add_argument("--no-hub-ui", action="store_true",
                    help="skip Gate 5 Hub Desktop UI Context Ring & Selector unlock")
    ap.add_argument("--no-backup", action="store_true",
                    help="skip creating <binary>.bak on --in-place")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--in-place", action="store_true")
    ap.add_argument("--locate", action="store_true", help="print target symbols only")
    args = ap.parse_args()

    src = os.path.abspath(os.path.expanduser(args.binary))
    if not os.path.exists(src):
        sys.exit(f"error: no such file: {src}")

    with open(src, "rb") as fh:
        data = bytearray(fh.read())
    before = hashlib.sha256(data).hexdigest()

    pids = _running_pids_for_binary(src)
    if pids:
        if sys.platform == "win32":
            print(f"[i] {len(pids)} process(es) are running this binary "
                  f"(pid {', '.join(pids[:5])}); Windows hot-swap will park the active image")
        else:
            print(f"[!] {len(pids)} process(es) are running this binary "
                  f"(pid {', '.join(pids[:5])}); quit them first or replacing / "
                  f"reloading the binary may fail")

    text_va, text_off, arch = find_text_section(data)
    funcs = build_functions(data, target_names=TARGET_SYMBOLS)
    print(f"[i] binary : {src}")
    print(f"[i] target : arch={arch} text_va={text_va:#x} text_off={text_off:#x} "
          f"matched_symbols={len(funcs)}")

    inspect_syms = (
        FUNC_OFFERS,
        FUNC_STORE_CW,
        FUNC_STORE_CHOICE,
        FUNC_CHECKPOINT,
        FUNC_EXP_CHECKPOINT,
        FUNC_MAX_STEP_TOKENS,
        FUNC_ONEHARNESS_STEP,
        FUNC_HUB_WEB_BUNDLE,
        FUNC_OVERRIDE,
        "strings.TrimSpace",
        "os.Getenv",
    )
    for key in inspect_syms:
        if key in funcs:
            eoff, sz = funcs[key]
            print(f"[i] symbol : {key} -> {text_va + eoff:#x} (size={sz:#x})")
        else:
            print(f"[i] symbol : {key} -> MISSING")

    if args.locate:
        return

    is_server_only = (FUNC_OFFERS not in funcs and FUNC_STORE_CW not in funcs)
    base_lower = os.path.basename(src).lower()
    if is_server_only and (base_lower == "agy" or base_lower.startswith("agy.")):
        raise PatchError(
            f"{src} is missing CLI context-window symbols (outdated agy < 1.4); "
            f"upgrade first with `agy update` or reinstall from "
            f"https://antigravity.google/docs/install?tab=cli"
        )
    if is_server_only:
        print("[i] mode   : standalone language_server (Gate 3A/3B + Gate 4A/4B + Gate 5 Hub UI)")
        patch_pipeline = [
            patch_gate3a_checkpoint,
            patch_gate3b_exp_checkpoint,
            patch_gate4a_user_input_step,
            patch_gate4b_oneharness_user_input,
        ]
    else:
        patch_pipeline = [
            patch_gate1_ui,
            patch_gate2_store,
            patch_gate3a_checkpoint,
            patch_gate3b_exp_checkpoint,
            patch_gate4a_user_input_step,
            patch_gate4b_oneharness_user_input,
        ]
    if not args.no_hub_ui:
        patch_pipeline.append(patch_gate5_hub_ui)

    for patch_fn in patch_pipeline:
        changed, msg = patch_fn(data, funcs, text_va, text_off, arch)
        print(("[*] " if changed else "[=] ") + msg)

    if not is_server_only and not args.no_env_fix:
        try:
            changed, msg = patch_env(data, funcs, text_va, text_off, arch)
            print(("[*] " if changed else "[=] ") + msg)
        except PatchError as exc:
            print(f"[!] skipped optional env-var patch: {exc}")

    after = hashlib.sha256(data).hexdigest()
    if before == after:
        print("[=] all gates already unlocked; nothing to write")
        return
    if args.dry_run:
        print(f"[*] dry run complete: sha256 {before[:12]} -> {after[:12]}")
        return

    dst = src if args.in_place else src + ".patched"
    write_atomic_signed(
        dst,
        data,
        src_for_entitlements=src,
        make_backup=(args.in_place and not args.no_backup),
    )
    with open(dst, "rb") as fh:
        final_sha = hashlib.sha256(fh.read()).hexdigest()
    print(f"[*] wrote  : {dst}\n    sha256 : {before[:12]} -> {final_sha[:12]}")


if __name__ == "__main__":
    try:
        main()
    except PatchError as exc:
        sys.exit(f"error: {exc}")
