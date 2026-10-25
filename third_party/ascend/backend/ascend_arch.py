"""Centralized Ascend target-architecture classification for the compiler backend.

Single source of truth for generation membership, UB / RF capacities, and
backend feature flags (register-based lowering, FFTS disable).  A6 devices
reuse the A5 register-based pipeline, so they are treated as a sibling
generation rather than hardcoded inline everywhere.
"""
from enum import IntEnum, auto
from typing import Optional

__all__ = [
    "AscendArchGen",
    "is_register_based_arch",
    "arch_generation",
    "ub_size_in_kbytes_for_arch",
    "rf_size_in_kbytes_for_arch",
    "is_ffts_disabled_arch",
    "is_ffts_supported",
]


class AscendArchGen(IntEnum):
    """Compiler backend generations that need distinct lowering or policy."""

    LEGACY = auto()  # A2 / A3: 910A/B/D, 910_93, 310B
    GEN5 = auto()    # A5: Ascend 950 / 910_95
    GEN6 = auto()    # A6: Ascend 960 / 910_96


# Ordered table -- longer/more-specific prefixes must come first so that
# e.g. "Ascend910_96" does not shadow "Ascend910A".
# Tuple shape: (prefixes, gen, ub_kb, rf_kb, flags)
_ARCH_TABLE = (
    (("Ascend910_96", "Ascend960"), AscendArchGen.GEN6, 320, 128, {"ffts_disabled": True}),
    (("Ascend910_95", "Ascend950"), AscendArchGen.GEN5, 256, 128, {"ffts_disabled": True}),
    (("Ascend910A", "Ascend310B4"), AscendArchGen.LEGACY, 192, None,
     {"ffts_disabled": True}),
    (("Ascend910B", "Ascend910D", "Ascend910_93", "Ascend310B"),
     AscendArchGen.LEGACY, 192, None, {"ffts_disabled": False}),
)


def _match_entry(arch: Optional[str]):
    """Return the (gen, ub_kb, rf_kb, flags) tuple for *arch* or None.

    Matching is case-insensitive so this function works with both the
    uppercase arch prefixes used in compiler options (e.g. "Ascend910_96")
    and the lowercase SOC names returned by acl.get_soc_name().
    """
    if not isinstance(arch, str) or not arch:
        return None
    arch_lower = arch.lower()
    for prefixes, gen, ub_kb, rf_kb, flags in _ARCH_TABLE:
        for p in prefixes:
            if arch_lower.startswith(p.lower()):
                return gen, ub_kb, rf_kb, flags
    return None


def arch_generation(arch: Optional[str]) -> AscendArchGen:
    """Resolve the target generation for a (possibly partial) arch string.

    Unknown / empty / non-string targets resolve to AscendArchGen.LEGACY.
    """
    entry = _match_entry(arch)
    return entry[0] if entry is not None else AscendArchGen.LEGACY


def is_register_based_arch(arch: Optional[str]) -> bool:
    """Return True when *arch* uses the register-based (A5/A6) compile pipeline."""
    entry = _match_entry(arch)
    return entry is not None and entry[0] in (AscendArchGen.GEN5, AscendArchGen.GEN6)


def ub_size_in_kbytes_for_arch(arch: Optional[str]) -> int:
    """Return the raw UB capacity (KiB) for *arch*, or 0 for unknown targets."""
    entry = _match_entry(arch)
    return entry[1] if entry is not None else 0


def rf_size_in_kbytes_for_arch(arch: Optional[str]) -> Optional[int]:
    """Return the raw RF capacity (KiB) for *arch*, or None when not applicable."""
    entry = _match_entry(arch)
    return entry[2] if entry is not None else None


def is_ffts_disabled_arch(arch: Optional[str]) -> bool:
    """Return True when *arch* requires FFTS to be disabled at compile time."""
    entry = _match_entry(arch)
    return entry[3]["ffts_disabled"] if entry is not None else False


def is_ffts_supported(arch: Optional[str]) -> bool:
    """Return whether the specified compilation target supports FFTS."""
    if not isinstance(arch, str) or not arch:
        return False
    return not is_ffts_disabled_arch(arch)
