"""Reference covariance (leadfield gram matrix) loading and matching.

Three strategies (matching MATLAB GEDAI ref_matrix_type):
  'precomputed' – load the BEM gram matrix bundled with GEDAI-master/auxiliaries
  'channel_positions' – Gaussian RBF kernel on channel XYZ positions (fallback)
  custom np.ndarray – pass directly

The gram matrix G = L @ L.T where L is the leadfield (channels × sources).
For EEG, L should already be average-referenced.
"""

from __future__ import annotations

import os
from pathlib import Path
import logging

import numpy as np

logger = logging.getLogger(__name__)

# Path to MATLAB leadfield .mat bundled with the cloned GEDAI-master repo
_THIS_DIR = Path(__file__).parent
_MATLAB_REPO_DEFAULT = _THIS_DIR.parent.parent / "GEDAI-master" / "auxiliaries"
_LEADFIELD_MAT = _MATLAB_REPO_DEFAULT / "fsavLEADFIELD_4_GEDAI.mat"


def load_precomputed_leadfield(
    ch_names: list[str],
    mat_path: str | Path | None = None,
) -> np.ndarray:
    """Load BEM precomputed gram matrix and extract rows/cols for ch_names.

    Parameters
    ----------
    ch_names : list of str – EEG channel names (must match template labels)
    mat_path : path to fsavLEADFIELD_4_GEDAI.mat; defaults to GEDAI-master clone

    Returns
    -------
    gram : (n_ch, n_ch) float64 reference covariance
    """
    mat_path = Path(mat_path) if mat_path else _LEADFIELD_MAT
    if not mat_path.exists():
        raise FileNotFoundError(
            f"Leadfield file not found: {mat_path}\n"
            "Either clone GEDAI-master to the expected location or pass mat_path."
        )

    try:
        import h5py
    except ImportError:
        raise ImportError("h5py required to read MATLAB v7.3 files: pip install h5py")

    with h5py.File(str(mat_path), "r") as hf:
        lf = hf["leadfield4GEDAI"]
        gram_full: np.ndarray = lf["gram_matrix_avref"][:]  # (343, 343)
        name_refs = lf["electrodes"]["Name"]  # (343, 1) of HDF5 refs
        template_names: list[str] = [
            "".join(chr(c) for c in hf[name_refs[i, 0]][:].flatten())
            for i in range(name_refs.shape[0])
        ]

    lower_template = [n.lower() for n in template_names]
    indices: list[int] = []
    missing: list[str] = []

    for ch in ch_names:
        ch_l = ch.lower()
        if ch_l in lower_template:
            indices.append(lower_template.index(ch_l))
        else:
            # substring match
            found = False
            for j, tl in enumerate(lower_template):
                if tl in ch_l or ch_l in tl:
                    indices.append(j)
                    found = True
                    break
            if not found:
                missing.append(ch)

    if missing:
        raise ValueError(
            f"Could not match channels in leadfield: {missing}\n"
            f"Use ref_type='channel_positions' for non-standard montages."
        )

    idx = np.array(indices)
    return gram_full[np.ix_(idx, idx)].astype(np.float64)


def load_interpolated_leadfield(
    ch_names: list[str],
    ch_positions: np.ndarray,
    mat_path: str | Path | None = None,
) -> np.ndarray:
    """Spherically interpolate the BEM leadfield onto the actual EEG positions.

    Equivalent of MATLAB GEDAI's ref_matrix_type='interpolated' mode
    (interp_mont_GEDAI.m). Use this for non-standard montages where
    'precomputed' lookup-by-name is insufficient — the leadfield Gain
    matrix values are interpolated from the 343 template positions onto
    the recording montage positions using spherical-spline interpolation
    (Perrin et al. 1989), then the gram matrix is recomputed.

    Parameters
    ----------
    ch_names : list of EEG channel names (only used for size/order)
    ch_positions : (n_ch, 3) XYZ positions in metres (MNE convention)
    mat_path : optional override for leadfield .mat path

    Returns
    -------
    ref_cov : (n_ch, n_ch) float64 — interpolated reference covariance
    """
    mat_path = Path(mat_path) if mat_path else _LEADFIELD_MAT
    if not mat_path.exists():
        raise FileNotFoundError(f"Leadfield file not found: {mat_path}")

    if ch_positions is None:
        raise ValueError("ch_positions required for interpolated leadfield")
    if not np.all(np.isfinite(ch_positions)) or np.allclose(ch_positions, 0):
        raise RuntimeError("Channel positions invalid (zeros or NaN)")

    import h5py

    with h5py.File(str(mat_path), "r") as hf:
        lf = hf["leadfield4GEDAI"]
        # Gain matrix: HDF5 stores as (n_sources, n_channels) → transpose to (n_ch, n_src)
        gain_raw = lf["Gain"][:].astype(np.float64)
        if gain_raw.shape[1] == 343:
            gain = gain_raw.T  # → (343, n_src)
        elif gain_raw.shape[0] == 343:
            gain = gain_raw
        else:
            raise RuntimeError(f"Unexpected Gain shape: {gain_raw.shape}")
        # Template electrode positions — stored as HDF5 references in v7.3 .mat
        loc_refs = lf["electrodes"]["Loc"]
        template_pos = np.array([
            np.array(hf[loc_refs[i, 0]][:]).flatten()[:3]
            for i in range(loc_refs.shape[0])
        ], dtype=np.float64)
        # template_pos: (343, 3) XYZ in metres

    n_ch_template = gain.shape[0]
    if template_pos.shape[0] != n_ch_template:
        raise RuntimeError(
            f"Template positions ({template_pos.shape[0]}) and Gain rows "
            f"({n_ch_template}) disagree"
        )

    # Average-reference the Gain matrix across channels (matches MATLAB:
    # non-rank-deficient average reference, line 463 of GEDAI.m)
    n = n_ch_template
    gain_avref = gain - gain.sum(axis=0, keepdims=True) / (n + 1)

    # Build spherical-spline interpolation matrix mapping
    # (343 template positions) → (n_ch recording positions)
    from mne.channels.interpolation import _make_interpolation_matrix

    # MNE expects positions normalized to unit sphere (it does this internally)
    # _make_interpolation_matrix(pos_from, pos_to, alpha=1e-5)
    # Returns matrix of shape (n_to, n_from)
    interp_matrix = _make_interpolation_matrix(template_pos, ch_positions, alpha=1e-5)

    # Interpolate each source's spatial projection from template → EEG positions
    # interp_matrix: (n_ch, 343), gain_avref: (343, n_src)
    # → interpolated_gain: (n_ch, n_src)
    interpolated_gain = interp_matrix @ gain_avref

    # Reference covariance = Gain @ Gain.T  (gram matrix in channel space)
    ref_cov = interpolated_gain @ interpolated_gain.T
    return ref_cov.astype(np.float64)


def compute_channel_position_cov(
    ch_positions: np.ndarray,
    eps: float = 1e-6,
) -> np.ndarray:
    """Gaussian RBF kernel from 3-D channel positions – fallback reference.

    Parameters
    ----------
    ch_positions : (n_ch, 3) XYZ positions in metres

    Returns
    -------
    cov : (n_ch, n_ch) float64
    """
    from sklearn.metrics import pairwise_distances

    D = pairwise_distances(ch_positions, metric="euclidean")
    nonzero = D[D > 0]
    ell = float(np.median(nonzero)) if nonzero.size > 0 else 1.0
    cov = np.exp(-(D**2) / (2 * ell**2))
    cov += eps * np.eye(len(ch_positions))
    return cov.astype(np.float64)


def compute_warped_bem_leadfield(
    ch_names: list[str],
    ch_positions: np.ndarray,
) -> np.ndarray:
    """Compute BEM forward solution at actual electrode positions using MNE.

    Equivalent to MATLAB GEDAI ref_matrix_type='warped':
      - MATLAB: EEGLAB coregister → FieldTrip ft_prepare_leadfield (Colin27 BEM)
      - Python:  MNE fsaverage BEM → make_forward_solution

    The fsaverage BEM (5120-5120-5120 ico-4 surfaces) is downloaded
    automatically via mne.datasets.fetch_fsaverage() and cached locally.

    Parameters
    ----------
    ch_names    : list of EEG channel names
    ch_positions: (n_ch, 3) XYZ positions in metres (MNE head coordinates)

    Returns
    -------
    ref_cov : (n_ch, n_ch) float64 — Gram matrix (leadfield @ leadfield.T)
    """
    import mne
    from pathlib import Path

    # 1. Fetch fsaverage (downloads once, ~50 MB, then cached)
    subjects_dir = str(Path(mne.datasets.fetch_fsaverage(verbose=False)).parent)
    subject = "fsaverage"

    # 2. Load precomputed BEM solution (5120-5120-5120 = inner skull + outer skull + scalp)
    bem_path = (Path(subjects_dir) / subject / "bem"
                / "fsaverage-5120-5120-5120-bem.fif")
    bem = mne.read_bem_surfaces(str(bem_path))
    bem_sol = mne.make_bem_solution(bem, verbose=False)

    # 3. Source space (precomputed ico-5 surface source space)
    src_path = (Path(subjects_dir) / subject / "bem"
                / "fsaverage-ico-5-src.fif")
    src = mne.read_source_spaces(str(src_path), verbose=False)

    # 4. Build MNE Info with actual electrode positions
    info = mne.create_info(ch_names=ch_names, sfreq=256.0, ch_types="eeg",
                           verbose=False)
    # ch_positions in metres → build DigMontage
    ch_pos_dict = {name: pos for name, pos in zip(ch_names, ch_positions)}
    montage = mne.channels.make_dig_montage(ch_pos=ch_pos_dict)
    info.set_montage(montage, verbose=False)

    # 5. Co-registration transform — use fsaverage identity trans
    trans = str(Path(subjects_dir) / subject / "bem" / "fsaverage-trans.fif")

    # 6. Compute forward solution (EEG leadfield)
    fwd = mne.make_forward_solution(
        info, trans=trans, src=src, bem=bem_sol,
        eeg=True, meg=False, verbose=False,
    )

    # 7. Extract leadfield: (n_ch × n_sources*3); average-reference across channels
    L = fwd["sol"]["data"].astype(np.float64)          # (n_ch, n_src*3)
    n_ch = L.shape[0]
    L_avref = L - L.sum(axis=0, keepdims=True) / (n_ch + 1)  # non-rank-deficient avg ref

    # 8. Gram matrix = L @ L.T (channel-space reference covariance)
    ref_cov = L_avref @ L_avref.T
    logger.info(f"  Warped BEM leadfield computed: {n_ch} channels, "
                f"{L.shape[1]} sources → Gram ({n_ch}×{n_ch})")
    return ref_cov.astype(np.float64)


def get_reference_cov(
    ref_type: str | np.ndarray,
    ch_names: list[str],
    ch_positions: np.ndarray | None = None,
    mat_path: str | Path | None = None,
) -> np.ndarray:
    """Unified interface for reference covariance.

    Parameters
    ----------
    ref_type : 'precomputed' | 'channel_positions' | np.ndarray
    ch_names : list[str]
    ch_positions : (n_ch, 3) required when ref_type='channel_positions'
    mat_path : optional override for leadfield mat path

    Returns
    -------
    ref_cov : (n_ch, n_ch) float64
    """
    if isinstance(ref_type, np.ndarray):
        assert ref_type.shape == (len(ch_names), len(ch_names)), (
            f"Custom ref_cov must be ({len(ch_names)}, {len(ch_names)})"
        )
        return ref_type.astype(np.float64)

    if ref_type == "precomputed":
        try:
            return load_precomputed_leadfield(ch_names, mat_path)
        except FileNotFoundError as exc:
            logger.warning(
                f"{exc}\nFalling back to channel_positions covariance."
            )
            if ch_positions is None:
                raise ValueError(
                    "ch_positions required when leadfield mat is unavailable."
                ) from exc
            return compute_channel_position_cov(ch_positions)

    if ref_type == "interpolated":
        if ch_positions is None:
            raise ValueError("ch_positions required for ref_type='interpolated'")
        try:
            return load_interpolated_leadfield(ch_names, ch_positions, mat_path)
        except FileNotFoundError as exc:
            logger.warning(f"{exc}\nFalling back to channel_positions covariance.")
            return compute_channel_position_cov(ch_positions)

    if ref_type == "warped":
        # Full BEM forward solve at actual electrode positions.
        # Matches MATLAB GEDAI ref_matrix_type='warped' (EEGLAB coregister +
        # FieldTrip ft_prepare_leadfield on Colin27 BEM).
        # Python uses MNE fsaverage BEM (equivalent standard adult head model).
        if ch_positions is None:
            raise ValueError("ch_positions required for ref_type='warped'")
        try:
            return compute_warped_bem_leadfield(ch_names, ch_positions)
        except Exception as exc:
            logger.warning(
                f"  Warped BEM failed ({exc}); falling back to interpolated leadfield."
            )
            try:
                return load_interpolated_leadfield(ch_names, ch_positions, mat_path)
            except Exception as exc2:
                logger.warning(f"  Interpolated also failed ({exc2}); using channel_positions.")
                return compute_channel_position_cov(ch_positions)

    if ref_type == "channel_positions":
        if ch_positions is None:
            raise ValueError("ch_positions must be provided for ref_type='channel_positions'")
        return compute_channel_position_cov(ch_positions)

    raise ValueError(
        f"Unknown ref_type '{ref_type}'. "
        f"Use 'precomputed', 'interpolated', 'warped', 'channel_positions', or ndarray."
    )
