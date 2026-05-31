#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Analysis script for photon-induced tau-pair production with a tau-dipole SMEFT benchmark.

Process:
    gamma gamma -> tau+ tau-

Samples:
    1. SM sample:
       generate a a > ta+ ta- / h h1 NP=0 SMHLOOP=0

    2. EFT/BSM benchmark sample:
       generate a a > ta+ ta- / h h1 NP<=2 SMHLOOP=0
       benchmark: Delta a_tau = 5.0e-4

Default local sample directories:
    SM:
      /home/hamzeh-khanpour/MG5_aMC_v3_6_6/aa_tautau_SM_NP_0_SMEFTsim_top_alphaScheme_UFO_LHeC/Events

    EFT:
      /home/hamzeh-khanpour/MG5_aMC_v3_6_6/aa_tautau_SM_NP_2_SMEFTsim_top_alphaScheme_UFO_LHeC_delta_atau_0_0005/Events

Main outputs:
    plots_aa_tautau_SM_EFT_delta_atau_0_0005/tautau_mass_SM_vs_EFT_ratio.pdf
    plots_aa_tautau_SM_EFT_delta_atau_0_0005/tautau_mass_SM_vs_EFT_ratio.png
    plots_aa_tautau_SM_EFT_delta_atau_0_0005/Ratio_EFT_over_SM_MtauTau_custom_bins_fit.pdf
    plots_aa_tautau_SM_EFT_delta_atau_0_0005/Ratio_EFT_over_SM_MtauTau_custom_bins_fit.png
    plots_aa_tautau_SM_EFT_delta_atau_0_0005/Ratio_EFT_over_SM_MtauTau_uniform_fit.pdf
    plots_aa_tautau_SM_EFT_delta_atau_0_0005/Ratio_EFT_over_SM_MtauTau_uniform_fit.png
    plots_aa_tautau_SM_EFT_delta_atau_0_0005/tautau_rapidity_SM_vs_EFT.pdf
    plots_aa_tautau_SM_EFT_delta_atau_0_0005/tautau_rapidity_SM_vs_EFT.png

Important updates in this version:
    - the main mass plot no longer uses the simple step EFT/SM ratio;
    - the lower panel of the main mass plot now contains the custom-bin fitted ratio
      with statistical uncertainties and a linear fit y = a x + b;
    - rapidity is shown over -10 < Y_{tau tau} < +10;
    - fonts are enlarged for paper/slide readability;
    - independent run_XX samples are combined as improved MC statistics, not added luminosity.

Important convention:
    Ratio = EFT / SM = BSM benchmark prediction divided by SM prediction.
"""

from __future__ import annotations

import argparse
import gzip
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt

try:
    import mplhep as hep
    hep.style.use("ATLAS")
except Exception:
    print("WARNING: mplhep is not available. Using default Matplotlib style.")

# Bigger fonts, applied after mplhep style so they override it.
plt.rcParams.update(
    {
        "font.size": 18,
        "axes.labelsize": 26,
        "axes.titlesize": 26,
        "legend.fontsize": 17,
        "xtick.labelsize": 22,
        "ytick.labelsize": 22,
        "axes.linewidth": 1.5,
        "xtick.major.size": 8,
        "ytick.major.size": 8,
        "xtick.minor.size": 4,
        "ytick.minor.size": 4,
        "xtick.major.width": 1.4,
        "ytick.major.width": 1.4,
        "xtick.minor.width": 1.1,
        "ytick.minor.width": 1.1,
        "savefig.bbox": "tight",
    }
)


# ============================================================
# Default user configuration
# ============================================================

BASE = Path("/home/hamzeh-khanpour/MG5_aMC_v3_6_6")

DEFAULT_SM_EVENTS_DIR = (
    BASE / "aa_tautau_SM_NP_0_SMEFTsim_top_alphaScheme_UFO_LHeC" / "Events"
)

DEFAULT_EFT_EVENTS_DIR = (
    BASE
    / "aa_tautau_SM_NP_2_SMEFTsim_top_alphaScheme_UFO_LHeC_delta_atau_0_0005"
    / "Events"
)

# Fallback values used only if per-run cross sections cannot be parsed
# from the LHE header or run banner.
DEFAULT_SM_FALLBACK_SIGMA_PB = 50.696
DEFAULT_EFT_FALLBACK_SIGMA_PB = 51.153

DEFAULT_OUTDIR = Path("plots_aa_tautau_SM_EFT_delta_atau_0_0005")

# Luminosity used for expected statistical uncertainties in the ratio plots.
DEFAULT_LUMINOSITY_FB = 111.1

DELTA_A_TAU_LABEL = r"5.0\times10^{-4}"
COLLIDER_LABEL = r"LHeC"
PROCESS_NOTE = r"Elastic ($ep\to e(\gamma\gamma\to\tau^+\tau^-)p^*$)"

SM_LABEL = r"$\tau^+\tau^-$ (SM)"
EFT_LABEL = r"$\tau^+\tau^-$ ($\Delta a_\tau=5.0\times10^{-4}$)"

SM_COLOR = "red"
EFT_COLOR = "blue"
FIT_COLOR = "magenta"
UNITY_COLOR = "green"

# Mass plot range and binning.
MASS_MIN = 10.0
MASS_MAX = 500.0
UNIFORM_NBINS = 50

# Custom bins used for the fitted ratio in the lower panel of the main mass plot.
# These are intentionally wider in the high-mass tail to reduce statistical noise.
CUSTOM_BINS = np.array(
    [10, 20, 35, 50, 70, 100, 130, 170, 210, 260, 310, 360, 420, 500],
    dtype=float,
)

# Rapidity plot range. Updated as requested.
Y_MIN = -10.0
Y_MAX = 10.0
Y_NBINS = 80

# Ratio plot controls.
RATIO_YLIM = (0.0, 2.0)
MAIN_RATIO_YLIM = (0.80, 1.45)
MIN_EXPECTED_SM_EVENTS_FOR_FIT = 1.0

# Statistical uncertainty mode for R = EFT/SM.
#   "poisson_on_eft" : sigma_R = sqrt(N_EFT) / N_SM, Obs/Exp-style.
#   "both"           : sigma_R = R sqrt(1/N_EFT + 1/N_SM), includes SM and EFT stats.
#   "sm_only"        : sigma_R = R / sqrt(N_SM), simple expected SM-stat precision.
RATIO_STAT_MODE = "poisson_on_eft"

SAVE_DPI = 600


# ============================================================
# Four-vector utilities
# ============================================================

@dataclass
class FourVector:
    px: float
    py: float
    pz: float
    e: float

    def __add__(self, other: "FourVector") -> "FourVector":
        return FourVector(
            px=self.px + other.px,
            py=self.py + other.py,
            pz=self.pz + other.pz,
            e=self.e + other.e,
        )

    @property
    def mass(self) -> float:
        m2 = self.e**2 - self.px**2 - self.py**2 - self.pz**2
        return math.sqrt(max(m2, 0.0))

    @property
    def rapidity(self) -> float:
        numerator = self.e + self.pz
        denominator = self.e - self.pz
        if numerator <= 0.0 or denominator <= 0.0:
            return float("nan")
        return 0.5 * math.log(numerator / denominator)


@dataclass
class RunData:
    path: Path
    sigma_pb: float
    masses: np.ndarray
    rapidities: np.ndarray


@dataclass
class SampleData:
    name: str
    runs: List[RunData]

    @property
    def n_events(self) -> int:
        return int(sum(len(run.masses) for run in self.runs))

    @property
    def combined_sigma_pb(self) -> float:
        """
        Combine independent MC runs as improved Monte Carlo statistics,
        not as doubled/tripled luminosity.

        The combined cross section is the event-count-weighted average
        of the per-run integrated weights.
        """
        n_total = self.n_events
        if n_total == 0:
            return 0.0
        return sum(len(run.masses) * run.sigma_pb for run in self.runs) / n_total

    def values_and_weights(self, observable: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return concatenated observable values and cross-section weights.

        Each event in run i gets weight:
            sigma_i / N_total

        where N_total is the total number of accepted events in all runs.
        This makes histogram integrals equal to the combined effective
        physical cross section, not the sum of the run cross sections.
        """
        n_total = self.n_events
        if n_total == 0:
            raise RuntimeError(f"No accepted events found for sample {self.name}")

        values_list = []
        weights_list = []

        for run in self.runs:
            if observable == "mass":
                values = run.masses
            elif observable == "rapidity":
                values = run.rapidities
            else:
                raise ValueError(f"Unknown observable: {observable}")

            weights = np.full(len(values), run.sigma_pb / n_total, dtype=float)
            values_list.append(values)
            weights_list.append(weights)

        return np.concatenate(values_list), np.concatenate(weights_list)


@dataclass
class FitResult:
    slope: float
    intercept: float
    slope_err: float
    intercept_err: float
    chi2_ndf: float
    n_fit: int


@dataclass
class RatioFitData:
    bins: np.ndarray
    centers: np.ndarray
    sm_bin_pb: np.ndarray
    eft_bin_pb: np.ndarray
    ratio: np.ndarray
    ratio_err: np.ndarray
    n_sm: np.ndarray
    fit_mask: np.ndarray
    fit: FitResult


# ============================================================
# LHE path discovery and cross-section parsing
# ============================================================

def open_text_maybe_gzip(path: Path):
    """Open plain-text or gzip-compressed file."""
    if str(path).endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return open(path, "rt", encoding="utf-8", errors="replace")


def resolve_lhe_path(run_dir: Path) -> Path:
    """Return unweighted_events.lhe or unweighted_events.lhe.gz inside one run directory."""
    candidates = [
        run_dir / "unweighted_events.lhe",
        run_dir / "unweighted_events.lhe.gz",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        f"Could not find unweighted_events.lhe(.gz) in run directory:\n  {run_dir}"
    )


def discover_run_dirs(events_dir: Path) -> List[Path]:
    """Discover run_XX directories under Events."""
    events_dir = Path(events_dir)
    if not events_dir.exists():
        raise FileNotFoundError(f"Events directory does not exist:\n  {events_dir}")

    run_dirs = sorted(
        [p for p in events_dir.iterdir() if p.is_dir() and re.match(r"run_\d+", p.name)]
    )

    if not run_dirs:
        raise FileNotFoundError(f"No run_XX directories found under:\n  {events_dir}")

    return run_dirs


def parse_integrated_weight_from_text(path: Path, max_lines: int = 200000) -> Optional[float]:
    """
    Parse '#  Integrated weight (pb)  :  51.153' from an LHE header or banner.
    Returns None if not found.
    """
    pattern = re.compile(r"Integrated\s+weight\s*\(pb\)\s*:\s*([-+0-9.eE]+)")

    try:
        with open_text_maybe_gzip(path) as handle:
            for i, line in enumerate(handle):
                match = pattern.search(line)
                if match:
                    return float(match.group(1))
                if i >= max_lines:
                    break
    except Exception:
        return None

    return None


def find_run_sigma_pb(run_dir: Path, lhe_path: Path, fallback_sigma_pb: float) -> float:
    """
    Find the integrated cross section for a run.
    Priority:
      1. LHE file header;
      2. run banner file, e.g. run_01_tag_1_banner.txt;
      3. fallback value supplied by the user.
    """
    sigma = parse_integrated_weight_from_text(lhe_path)
    if sigma is not None:
        return sigma

    banner_candidates = sorted(run_dir.glob("*_banner.txt"))
    for banner in banner_candidates:
        sigma = parse_integrated_weight_from_text(banner)
        if sigma is not None:
            return sigma

    print(
        f"WARNING: Could not parse integrated weight for {run_dir.name}. "
        f"Using fallback sigma = {fallback_sigma_pb:.9f} pb"
    )
    return float(fallback_sigma_pb)


# ============================================================
# LHE reader for gamma gamma -> tau+ tau-
# ============================================================

def read_tautau_observables_from_lhe(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Read one LHE file and reconstruct M_{tau tau} and Y_{tau tau}.

    PDG IDs:
        tau- : +15
        tau+ : -15

    We accept status 1 or 2 because in some LHE samples the taus are stable
    final-state particles, while in decay-chain samples they may appear as
    intermediate particles.
    """
    masses: List[float] = []
    rapidities: List[float] = []

    in_event = False
    first_line_after_event_tag = False

    tau_minus: Optional[FourVector] = None
    tau_plus: Optional[FourVector] = None

    with open_text_maybe_gzip(path) as lhe:
        for raw_line in lhe:
            line = raw_line.strip()
            if not line:
                continue

            if line.startswith("<event"):
                in_event = True
                first_line_after_event_tag = True
                tau_minus = None
                tau_plus = None
                continue

            if line.startswith("</event"):
                if tau_minus is not None and tau_plus is not None:
                    tautau = tau_minus + tau_plus
                    y_tautau = tautau.rapidity
                    if math.isfinite(y_tautau):
                        masses.append(tautau.mass)
                        rapidities.append(y_tautau)
                in_event = False
                first_line_after_event_tag = False
                continue

            if not in_event:
                continue

            # First non-empty line after <event> is the event header:
            # NUP IDPRUP XWGTUP SCALUP AQEDUP AQCDUP
            if first_line_after_event_tag:
                first_line_after_event_tag = False
                continue

            if line.startswith("#") or line.startswith("<"):
                continue

            parts = line.split()
            if len(parts) < 10:
                continue

            try:
                pid = int(parts[0])
                status = int(parts[1])
            except ValueError:
                continue

            if abs(pid) != 15 or status not in (1, 2):
                continue

            try:
                px = float(parts[6])
                py = float(parts[7])
                pz = float(parts[8])
                energy = float(parts[9])
            except ValueError:
                continue

            vec = FourVector(px=px, py=py, pz=pz, e=energy)

            if pid == 15:
                tau_minus = vec
            elif pid == -15:
                tau_plus = vec

    return np.asarray(masses, dtype=float), np.asarray(rapidities, dtype=float)


# ============================================================
# Sample loading
# ============================================================

def load_sample_from_events_dir(
    name: str,
    events_dir: Path,
    fallback_sigma_pb: float,
) -> SampleData:
    run_dirs = discover_run_dirs(events_dir)
    runs: List[RunData] = []

    print("\n" + "=" * 88)
    print(f"Loading sample: {name}")
    print(f"Events directory: {events_dir}")
    print("=" * 88)

    for run_dir in run_dirs:
        lhe_path = resolve_lhe_path(run_dir)
        sigma_pb = find_run_sigma_pb(run_dir, lhe_path, fallback_sigma_pb)

        print(f"Reading {run_dir.name}: {lhe_path}")
        masses, rapidities = read_tautau_observables_from_lhe(lhe_path)

        if len(masses) == 0:
            raise RuntimeError(f"No tau+tau- events reconstructed from:\n  {lhe_path}")

        print(f"  accepted tau+tau- events : {len(masses):,}")
        print(f"  integrated weight used   : {sigma_pb:.9f} pb")

        runs.append(RunData(path=lhe_path, sigma_pb=sigma_pb, masses=masses, rapidities=rapidities))

    sample = SampleData(name=name, runs=runs)

    print("-" * 88)
    print(f"Total accepted events       : {sample.n_events:,}")
    print(f"Combined effective sigma    : {sample.combined_sigma_pb:.9f} pb")
    print("=" * 88)

    return sample


# ============================================================
# Histogram and uncertainty utilities
# ============================================================

def make_histogram(
    values: np.ndarray,
    weights: np.ndarray,
    bins: np.ndarray,
    differential: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Weighted histogram.

    If differential=True, returns dσ/dx.
    If differential=False, returns σ_bin.
    """
    hist, _ = np.histogram(values, bins=bins, weights=weights)
    hist_w2, _ = np.histogram(values, bins=bins, weights=weights**2)
    err = np.sqrt(hist_w2)

    if differential:
        widths = np.diff(bins)
        hist = hist / widths
        err = err / widths

    return hist, err


def safe_ratio(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    ratio = np.full_like(num, np.nan, dtype=float)
    mask = den > 0.0
    ratio[mask] = num[mask] / den[mask]
    return ratio


def expected_ratio_uncertainty(
    sm_bin_pb: np.ndarray,
    eft_bin_pb: np.ndarray,
    luminosity_fb: float,
    mode: str = RATIO_STAT_MODE,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Expected statistical uncertainty for R = EFT/SM.

    Inputs are bin-integrated cross sections in pb.
    Luminosity is given in fb^-1.

    Returns:
        ratio, ratio_err, n_sm
    """
    lumi_pb = luminosity_fb * 1000.0
    n_sm = sm_bin_pb * lumi_pb
    n_eft = eft_bin_pb * lumi_pb

    ratio = safe_ratio(n_eft, n_sm)
    err = np.full_like(ratio, np.nan, dtype=float)

    valid = (n_sm > 0.0) & (n_eft > 0.0) & np.isfinite(ratio)

    if mode == "poisson_on_eft":
        err[valid] = np.sqrt(n_eft[valid]) / n_sm[valid]
    elif mode == "both":
        err[valid] = ratio[valid] * np.sqrt(1.0 / n_eft[valid] + 1.0 / n_sm[valid])
    elif mode == "sm_only":
        err[valid] = ratio[valid] / np.sqrt(n_sm[valid])
    else:
        raise ValueError(f"Unknown ratio uncertainty mode: {mode}")

    return ratio, err, n_sm


def weighted_linear_fit(x: np.ndarray, y: np.ndarray, yerr: np.ndarray) -> FitResult:
    """
    Weighted linear fit y = a x + b.
    """
    valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(yerr) & (yerr > 0.0)
    x = x[valid]
    y = y[valid]
    yerr = yerr[valid]

    if len(x) < 2:
        raise RuntimeError("Need at least two valid points for a linear fit.")

    w = 1.0 / yerr**2
    X = np.column_stack([x, np.ones_like(x)])
    XT_W = X.T * w
    cov = np.linalg.inv(XT_W @ X)
    beta = cov @ (XT_W @ y)

    slope = float(beta[0])
    intercept = float(beta[1])

    residual = y - (slope * x + intercept)
    chi2 = float(np.sum((residual / yerr) ** 2))
    ndf = max(len(x) - 2, 1)
    chi2_ndf = chi2 / ndf

    # Scale parameter errors when scatter exceeds the assigned statistical uncertainties.
    scale = math.sqrt(max(chi2_ndf, 1.0))
    slope_err = float(math.sqrt(cov[0, 0]) * scale)
    intercept_err = float(math.sqrt(cov[1, 1]) * scale)

    return FitResult(
        slope=slope,
        intercept=intercept,
        slope_err=slope_err,
        intercept_err=intercept_err,
        chi2_ndf=chi2_ndf,
        n_fit=len(x),
    )


def compute_mass_ratio_fit_data(
    sm: SampleData,
    eft: SampleData,
    bins: np.ndarray,
    luminosity_fb: float,
) -> RatioFitData:
    """Compute custom/uniform-bin EFT/SM ratio and linear fit."""
    sm_values, sm_weights = sm.values_and_weights("mass")
    eft_values, eft_weights = eft.values_and_weights("mass")

    sm_mask = (sm_values >= bins[0]) & (sm_values <= bins[-1])
    eft_mask = (eft_values >= bins[0]) & (eft_values <= bins[-1])

    sm_bin_pb, _ = make_histogram(sm_values[sm_mask], sm_weights[sm_mask], bins, differential=False)
    eft_bin_pb, _ = make_histogram(eft_values[eft_mask], eft_weights[eft_mask], bins, differential=False)

    ratio, ratio_err, n_sm = expected_ratio_uncertainty(
        sm_bin_pb, eft_bin_pb, luminosity_fb=luminosity_fb, mode=RATIO_STAT_MODE
    )

    centers = 0.5 * (bins[:-1] + bins[1:])
    fit_mask = (
        np.isfinite(ratio)
        & np.isfinite(ratio_err)
        & (ratio_err > 0.0)
        & (n_sm >= MIN_EXPECTED_SM_EVENTS_FOR_FIT)
        & (sm_bin_pb > 0.0)
        & (eft_bin_pb > 0.0)
    )

    fit = weighted_linear_fit(centers[fit_mask], ratio[fit_mask], ratio_err[fit_mask])

    return RatioFitData(
        bins=bins,
        centers=centers,
        sm_bin_pb=sm_bin_pb,
        eft_bin_pb=eft_bin_pb,
        ratio=ratio,
        ratio_err=ratio_err,
        n_sm=n_sm,
        fit_mask=fit_mask,
        fit=fit,
    )


def save_ratio_csv(outpath: Path, data: RatioFitData) -> None:
    widths = np.diff(data.bins)
    table = np.column_stack(
        [
            data.bins[:-1],
            data.bins[1:],
            data.centers,
            widths,
            data.sm_bin_pb,
            data.eft_bin_pb,
            data.ratio,
            data.ratio_err,
            data.n_sm,
            data.fit_mask.astype(int),
        ]
    )
    header = (
        "bin_low_GeV,bin_high_GeV,bin_center_GeV,bin_width_GeV,"
        "SM_sigma_bin_pb,EFT_sigma_bin_pb,EFT_over_SM,ratio_stat_err,"
        "expected_SM_events,used_in_fit"
    )
    np.savetxt(outpath, table, delimiter=",", header=header, comments="")
    print(f"Saved CSV: {outpath}")


# ============================================================
# Plotting helpers
# ============================================================

def style_axis(ax) -> None:
    ax.grid(True, which="major", linestyle="--", alpha=0.55)
    ax.grid(True, which="minor", linestyle=":", alpha=0.25)
    ax.minorticks_on()
    ax.tick_params(axis="both", which="both", direction="in", top=True, right=True)


def draw_step(ax, bins: np.ndarray, values: np.ndarray, label: str, color: str, linestyle: str) -> None:
    ax.stairs(values, bins, label=label, color=color, linestyle=linestyle, linewidth=2.6)


def set_safe_log_y(ax, arrays: List[np.ndarray]) -> None:
    positive = []
    for arr in arrays:
        mask = np.isfinite(arr) & (arr > 0.0)
        positive.extend(arr[mask])
    if not positive:
        return
    positive = np.asarray(positive)
    ax.set_yscale("log")
    ax.set_ylim(max(np.min(positive) * 0.3, 1e-12), np.max(positive) * 4.5)


def add_process_note(ax, loc: str = "upper right", fontsize: int = 18) -> None:
    if loc == "upper right":
        xy = (0.97, 0.78)
        ha = "right"
    else:
        xy = (0.03, 0.80)
        ha = "left"

    ax.text(
        xy[0],
        xy[1],
        PROCESS_NOTE,
        transform=ax.transAxes,
        fontsize=fontsize,
        ha=ha,
        va="top",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="none", alpha=0.76),
    )


def draw_fitted_ratio_axis(
    ax,
    data: RatioFitData,
    *,
    xlim: Tuple[float, float],
    ylim: Tuple[float, float],
    show_xlabel: bool,
    show_legend: bool,
    legend_loc: str = "upper left",
) -> None:
    """Draw ratio points, y=1, and linear fit on an axis."""
    xfit = np.linspace(xlim[0], xlim[1], 500)
    yfit = data.fit.slope * xfit + data.fit.intercept

    ax.axhline(1.0, color=UNITY_COLOR, linestyle="--", linewidth=2.0, label=r"$y=1$")
    ax.plot(
        xfit,
        yfit,
        color=FIT_COLOR,
        linestyle="--",
        linewidth=2.8,
        label=(
            rf"Fit: $y=({data.fit.slope:.4f}\pm{data.fit.slope_err:.4f})x"
            rf"+({data.fit.intercept:.4f}\pm{data.fit.intercept_err:.4f})$"
        ),
    )

    ax.errorbar(
        data.centers[data.fit_mask],
        data.ratio[data.fit_mask],
        yerr=data.ratio_err[data.fit_mask],
        fmt="o",
        color="red",
        ecolor="red",
        elinewidth=2.0,
        capsize=0,
        markersize=7.0,
        label="Stat. uncertainty",
        zorder=5,
    )

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_ylabel(r"Ratio (EFT/SM)", fontsize=24)
    if show_xlabel:
        ax.set_xlabel(r"$M_{\tau^+\tau^-}$ [GeV]", fontsize=26)
    style_axis(ax)

    if show_legend:
        ax.legend(loc=legend_loc, fontsize=16, frameon=False)


def plot_mass_distribution_with_fitted_ratio(
    sm: SampleData,
    eft: SampleData,
    outdir: Path,
    luminosity_fb: float,
    main_ratio_ylim: Tuple[float, float] = MAIN_RATIO_YLIM,
) -> FitResult:
    """
    Main paper-style plot:
      - top panel: dσ/dM for SM and EFT;
      - bottom panel: custom-bin EFT/SM ratio with statistical uncertainties and linear fit.

    The old simple step EFT/SM ratio has intentionally been removed.
    """
    dist_bins = np.linspace(MASS_MIN, MASS_MAX, UNIFORM_NBINS + 1)
    ratio_data = compute_mass_ratio_fit_data(sm, eft, CUSTOM_BINS, luminosity_fb)

    sm_values, sm_weights = sm.values_and_weights("mass")
    eft_values, eft_weights = eft.values_and_weights("mass")

    sm_mask = (sm_values >= MASS_MIN) & (sm_values <= MASS_MAX)
    eft_mask = (eft_values >= MASS_MIN) & (eft_values <= MASS_MAX)

    sm_hist, _ = make_histogram(sm_values[sm_mask], sm_weights[sm_mask], dist_bins, differential=True)
    eft_hist, _ = make_histogram(eft_values[eft_mask], eft_weights[eft_mask], dist_bins, differential=True)

    fig, (ax, rax) = plt.subplots(
        2,
        1,
        figsize=(10.0, 10.2),
        sharex=True,
        gridspec_kw={"height_ratios": [3.35, 1.25], "hspace": 0.055},
    )
    plt.subplots_adjust(left=0.13, right=0.97, bottom=0.115, top=0.965)

    draw_step(ax, dist_bins, sm_hist, SM_LABEL, SM_COLOR, "-")
    draw_step(ax, dist_bins, eft_hist, EFT_LABEL, EFT_COLOR, "--")
    set_safe_log_y(ax, [sm_hist, eft_hist])
    ax.set_ylabel(r"$d\sigma/dM_{\tau^+\tau^-}$ [pb/GeV]", fontsize=26)
    ax.legend(loc="upper right", fontsize=18, frameon=False)
    style_axis(ax)
    add_process_note(ax, loc="upper right", fontsize=18)

    draw_fitted_ratio_axis(
        rax,
        ratio_data,
        xlim=(MASS_MIN, MASS_MAX),
        ylim=main_ratio_ylim,
        show_xlabel=True,
        show_legend=True,
        legend_loc="upper left",
    )

    # Compact fit-quality note in the lower panel.
    note = (
        rf"$\sigma_{{\rm SM}}={sm.combined_sigma_pb:.3f}$ pb, "
        rf"$\sigma_{{\rm EFT}}={eft.combined_sigma_pb:.3f}$ pb, "
        rf"$\chi^2/\mathrm{{ndf}}={ratio_data.fit.chi2_ndf:.2f}$, "
        rf"$N_{{\rm fit}}={ratio_data.fit.n_fit}$"
    )
    rax.text(
        0.03,
        0.055,
        note,
        transform=rax.transAxes,
        fontsize=13,
        ha="left",
        va="bottom",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="none", alpha=0.76),
    )

    fig.align_ylabels([ax, rax])

    for ext in ("pdf", "png"):
        out = outdir / f"tautau_mass_SM_vs_EFT_ratio.{ext}"
        fig.savefig(out, dpi=SAVE_DPI if ext == "png" else None)
        print(f"Saved plot: {out}")
    plt.close(fig)

    # Save the exact custom-bin ratio points used in the lower panel.
    save_ratio_csv(outdir / "tautau_mass_fitted_ratio_lower_panel_custom_bins.csv", ratio_data)

    return ratio_data.fit


def make_ratio_fit_plot(
    sm: SampleData,
    eft: SampleData,
    bins: np.ndarray,
    luminosity_fb: float,
    outdir: Path,
    tag: str,
    ratio_ylim: Tuple[float, float] = RATIO_YLIM,
) -> FitResult:
    """Create standalone ratio-only plot with statistical error bars and linear fit."""
    data = compute_mass_ratio_fit_data(sm, eft, bins, luminosity_fb)

    fig, ax = plt.subplots(figsize=(14.0, 6.0))
    plt.subplots_adjust(left=0.095, right=0.98, bottom=0.16, top=0.875)

    draw_fitted_ratio_axis(
        ax,
        data,
        xlim=(bins[0], bins[-1]),
        ylim=ratio_ylim,
        show_xlabel=True,
        show_legend=True,
        legend_loc="upper left",
    )

    title = (
        rf"{COLLIDER_LABEL} $\gamma\gamma\to\tau^+\tau^-$ "
        rf"($\Delta a_\tau={DELTA_A_TAU_LABEL}$, "
        rf"$\mathcal{{L}}={luminosity_fb:.1f}\,\mathrm{{fb}}^{{-1}}$)"
    )
    ax.set_title(title, fontsize=27, pad=8)

    note = (
        rf"$\sigma_{{\rm SM}}={sm.combined_sigma_pb:.3f}$ pb, "
        rf"$\sigma_{{\rm EFT}}={eft.combined_sigma_pb:.3f}$ pb, "
        rf"$\chi^2/\mathrm{{ndf}}={data.fit.chi2_ndf:.2f}$, "
        rf"$N_{{\rm fit}}={data.fit.n_fit}$"
    )
    ax.text(
        0.03,
        0.05,
        note,
        transform=ax.transAxes,
        fontsize=14,
        ha="left",
        va="bottom",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="none", alpha=0.76),
    )

    for ext in ("pdf", "png"):
        out = outdir / f"Ratio_EFT_over_SM_MtauTau_{tag}_fit.{ext}"
        fig.savefig(out, dpi=SAVE_DPI if ext == "png" else None)
        print(f"Saved plot: {out}")
    plt.close(fig)

    save_ratio_csv(outdir / f"Ratio_EFT_over_SM_MtauTau_{tag}_fit.csv", data)

    print("-" * 88)
    print(f"Fit summary for {tag} binning")
    print(f"  slope       = {data.fit.slope:.8e} +/- {data.fit.slope_err:.8e} 1/GeV")
    print(f"  intercept   = {data.fit.intercept:.8e} +/- {data.fit.intercept_err:.8e}")
    print(f"  chi2/ndf    = {data.fit.chi2_ndf:.4f}")
    print(f"  fit points  = {data.fit.n_fit}")
    print("-" * 88)

    return data.fit


def plot_rapidity_distribution(sm: SampleData, eft: SampleData, outdir: Path) -> None:
    """Plot dσ/dY over -10 < Y_{tau tau} < 10."""
    bins = np.linspace(Y_MIN, Y_MAX, Y_NBINS + 1)

    sm_values, sm_weights = sm.values_and_weights("rapidity")
    eft_values, eft_weights = eft.values_and_weights("rapidity")

    sm_mask = (sm_values >= Y_MIN) & (sm_values <= Y_MAX)
    eft_mask = (eft_values >= Y_MIN) & (eft_values <= Y_MAX)

    sm_hist, _ = make_histogram(sm_values[sm_mask], sm_weights[sm_mask], bins, differential=True)
    eft_hist, _ = make_histogram(eft_values[eft_mask], eft_weights[eft_mask], bins, differential=True)

    fig, ax = plt.subplots(figsize=(10.5, 7.6))
    plt.subplots_adjust(left=0.13, right=0.97, bottom=0.14, top=0.965)

    draw_step(ax, bins, sm_hist, SM_LABEL, SM_COLOR, "-")
    draw_step(ax, bins, eft_hist, EFT_LABEL, EFT_COLOR, "--")
    ax.set_xlim(Y_MIN, Y_MAX)
    ax.set_ylim(0.0, max(np.nanmax(sm_hist), np.nanmax(eft_hist)) * 1.25)
    ax.set_xlabel(r"$Y_{\tau^+\tau^-}$", fontsize=26)
    ax.set_ylabel(r"$d\sigma/dY_{\tau^+\tau^-}$ [pb]", fontsize=26)
    ax.legend(loc="upper left", fontsize=18, frameon=False)
    style_axis(ax)
    add_process_note(ax, loc="upper left", fontsize=18)

    for ext in ("pdf", "png"):
        out = outdir / f"tautau_rapidity_SM_vs_EFT.{ext}"
        fig.savefig(out, dpi=SAVE_DPI if ext == "png" else None)
        print(f"Saved plot: {out}")
    plt.close(fig)


# ============================================================
# Summary output
# ============================================================

def format_fit_result(name: str, fit: FitResult) -> str:
    return (
        f"{name}:\n"
        f"  slope [1/GeV]     = {fit.slope:.10e} +/- {fit.slope_err:.10e}\n"
        f"  intercept         = {fit.intercept:.10e} +/- {fit.intercept_err:.10e}\n"
        f"  chi2/ndf          = {fit.chi2_ndf:.6f}\n"
        f"  N_fit             = {fit.n_fit}\n"
    )


def write_summary(
    outdir: Path,
    sm: SampleData,
    eft: SampleData,
    luminosity_fb: float,
    main_fit: FitResult,
    uniform_fit: FitResult,
    custom_fit: FitResult,
) -> None:
    summary = outdir / "summary.txt"
    with open(summary, "w", encoding="utf-8") as f:
        f.write("gamma gamma -> tau+ tau- SM/EFT summary\n")
        f.write("=" * 70 + "\n")
        f.write(f"SM runs: {len(sm.runs)}\n")
        f.write(f"EFT runs: {len(eft.runs)}\n")
        f.write(f"SM accepted events: {sm.n_events}\n")
        f.write(f"EFT accepted events: {eft.n_events}\n")
        f.write(f"SM combined sigma [pb]: {sm.combined_sigma_pb:.9f}\n")
        f.write(f"EFT combined sigma [pb]: {eft.combined_sigma_pb:.9f}\n")
        f.write(f"Inclusive EFT/SM ratio: {eft.combined_sigma_pb / sm.combined_sigma_pb:.9f}\n")
        f.write(f"Luminosity [fb^-1]: {luminosity_fb:.6f}\n")
        f.write(f"Ratio uncertainty mode: {RATIO_STAT_MODE}\n")
        f.write(f"Rapidity range: [{Y_MIN}, {Y_MAX}]\n")
        f.write("\nFit results\n")
        f.write("-" * 70 + "\n")
        f.write(format_fit_result("Main lower panel custom-bin fit", main_fit))
        f.write("\n")
        f.write(format_fit_result("Standalone uniform-bin fit", uniform_fit))
        f.write("\n")
        f.write(format_fit_result("Standalone custom-bin fit", custom_fit))
        f.write("\nPer-run cross sections used:\n")
        for sample in (sm, eft):
            f.write(f"\n{sample.name}:\n")
            for run in sample.runs:
                f.write(f"  {run.path}: sigma = {run.sigma_pb:.9f} pb, events = {len(run.masses)}\n")
    print(f"Saved summary: {summary}")


# ============================================================
# Command line and main
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot gamma gamma -> tau+ tau- SM/EFT distributions and fitted EFT/SM ratios."
    )
    parser.add_argument("--sm-events-dir", type=Path, default=DEFAULT_SM_EVENTS_DIR)
    parser.add_argument("--eft-events-dir", type=Path, default=DEFAULT_EFT_EVENTS_DIR)
    parser.add_argument("--sm-fallback-sigma-pb", type=float, default=DEFAULT_SM_FALLBACK_SIGMA_PB)
    parser.add_argument("--eft-fallback-sigma-pb", type=float, default=DEFAULT_EFT_FALLBACK_SIGMA_PB)
    parser.add_argument("--luminosity-fb", type=float, default=DEFAULT_LUMINOSITY_FB)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--ratio-ymin", type=float, default=RATIO_YLIM[0])
    parser.add_argument("--ratio-ymax", type=float, default=RATIO_YLIM[1])
    parser.add_argument("--main-ratio-ymin", type=float, default=MAIN_RATIO_YLIM[0])
    parser.add_argument("--main-ratio-ymax", type=float, default=MAIN_RATIO_YLIM[1])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("\nStarting gamma gamma -> tau+ tau- SM/EFT ratio analysis")
    print("=" * 88)
    print(f"SM Events dir     : {args.sm_events_dir}")
    print(f"EFT Events dir    : {args.eft_events_dir}")
    print(f"Luminosity        : {args.luminosity_fb:.6f} fb^-1")
    print(f"Output dir        : {outdir.resolve()}")
    print(f"Ratio stat mode   : {RATIO_STAT_MODE}")
    print(f"Rapidity range    : {Y_MIN} < Y_tautau < {Y_MAX}")
    print("=" * 88)

    sm_sample = load_sample_from_events_dir(
        "SM", args.sm_events_dir, fallback_sigma_pb=args.sm_fallback_sigma_pb
    )
    eft_sample = load_sample_from_events_dir(
        "EFT", args.eft_events_dir, fallback_sigma_pb=args.eft_fallback_sigma_pb
    )

    print("\nCross-section summary")
    print("-" * 88)
    print(f"SM combined effective sigma  = {sm_sample.combined_sigma_pb:.9f} pb")
    print(f"EFT combined effective sigma = {eft_sample.combined_sigma_pb:.9f} pb")
    print(f"Inclusive EFT/SM ratio       = {eft_sample.combined_sigma_pb / sm_sample.combined_sigma_pb:.9f}")
    print("-" * 88)

    for sample in (sm_sample, eft_sample):
        masses, _ = sample.values_and_weights("mass")
        rapidities, _ = sample.values_and_weights("rapidity")
        print(
            f"{sample.name}: M_tautau range = "
            f"[{np.nanmin(masses):.3f}, {np.nanmax(masses):.3f}] GeV, "
            f"mean = {np.nanmean(masses):.3f} GeV"
        )
        print(
            f"{sample.name}: Y_tautau range = "
            f"[{np.nanmin(rapidities):.3f}, {np.nanmax(rapidities):.3f}], "
            f"mean = {np.nanmean(rapidities):.3f}"
        )
    print("-" * 88)

    standalone_ratio_ylim = (args.ratio_ymin, args.ratio_ymax)
    main_ratio_ylim = (args.main_ratio_ymin, args.main_ratio_ymax)

    # Main distribution with fitted custom-bin ratio in the lower panel.
    main_fit = plot_mass_distribution_with_fitted_ratio(
        sm_sample,
        eft_sample,
        outdir,
        luminosity_fb=args.luminosity_fb,
        main_ratio_ylim=main_ratio_ylim,
    )

    # Standalone ratio-only fit plots.
    uniform_bins = np.linspace(MASS_MIN, MASS_MAX, UNIFORM_NBINS + 1)
    uniform_fit = make_ratio_fit_plot(
        sm_sample,
        eft_sample,
        bins=uniform_bins,
        luminosity_fb=args.luminosity_fb,
        outdir=outdir,
        tag="uniform",
        ratio_ylim=standalone_ratio_ylim,
    )

    custom_fit = make_ratio_fit_plot(
        sm_sample,
        eft_sample,
        bins=CUSTOM_BINS,
        luminosity_fb=args.luminosity_fb,
        outdir=outdir,
        tag="custom_bins",
        ratio_ylim=standalone_ratio_ylim,
    )

    # Rapidity comparison over -10 < Y < 10.
    plot_rapidity_distribution(sm_sample, eft_sample, outdir)

    write_summary(
        outdir,
        sm_sample,
        eft_sample,
        luminosity_fb=args.luminosity_fb,
        main_fit=main_fit,
        uniform_fit=uniform_fit,
        custom_fit=custom_fit,
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
