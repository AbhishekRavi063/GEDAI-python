"""gedai_core – Python GEDAI implementation with ENOVA-based rejection."""

from .gedai import GEDAICore, GEDAIResult
from .enova import compute_enova_per_epoch, compute_enova_per_channel, enova_summary
from .reject import (
    reject_epochs_by_enova,
    identify_bad_channels,
    two_pass_channel_rejection,
    DEFAULT_ENOVA_THRESHOLD,
)
from .sliding_window import SlidingWindowGEDAI, SlidingWindowResult, compare_global_vs_sliding
from .leadfield import get_reference_cov

__all__ = [
    "GEDAICore", "GEDAIResult",
    "compute_enova_per_epoch", "compute_enova_per_channel", "enova_summary",
    "reject_epochs_by_enova", "identify_bad_channels", "two_pass_channel_rejection",
    "DEFAULT_ENOVA_THRESHOLD",
    "SlidingWindowGEDAI", "SlidingWindowResult", "compare_global_vs_sliding",
    "get_reference_cov",
]
