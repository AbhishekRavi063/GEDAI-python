"""Controlled artifact injection for GEDAI benchmarking."""

from .inject_eog import inject_blink, inject_horizontal_eye_movement, ArtifactMeta
from .inject_emg import inject_emg
from .inject_line_noise import inject_line_noise
from .inject_drift import inject_drift
from .inject_channel_noise import inject_channel_noise
from .inject_electrode_pop import inject_electrode_pop

__all__ = [
    "ArtifactMeta",
    "inject_blink", "inject_horizontal_eye_movement",
    "inject_emg", "inject_line_noise", "inject_drift",
    "inject_channel_noise", "inject_electrode_pop",
]
