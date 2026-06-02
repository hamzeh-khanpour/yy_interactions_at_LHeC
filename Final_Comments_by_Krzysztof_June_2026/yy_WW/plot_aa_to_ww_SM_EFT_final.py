#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Analysis script for photon-induced W+W- production at the LHeC.

Process:
    gamma gamma -> W+ W-

Samples:
    1. SM:
       generate a a > w+ w- NP=0

    2. EFT/aQGC benchmark:
       generate a a > w+ w- NP=1
       with f_M2/Lambda^4 = 1 TeV^{-4}
       corresponding to FM2 = 1.0e-12 GeV^{-4}

LHeC setup:
    E_p = 7 TeV
    E_e = 50 GeV
    sqrt(s_ep) ~ 1.2 TeV

Photon-flux setup:
    lpp1 = 2
    lpp2 = 3
    pdlabel1 = iww
    pdlabel2 = iww
    Elastic photon emission
    Q_p^2 < 1e5 GeV^2
    Q_e^2 < 1e5 GeV^2

Outputs:
    plots_aa_ww_SM_EFT/ww_mass_SM_vs_EFT_ratio.pdf
    plots_aa_ww_SM_EFT/ww_mass_SM_vs_EFT_ratio.png
    plots_aa_ww_SM_EFT/ww_rapidity_SM_vs_EFT_ratio.pdf
    plots_aa_ww_SM_EFT/ww_rapidity_SM_vs_EFT_ratio.png
    plots_aa_ww_SM_EFT/ww_mass_histograms.csv
    plots_aa_ww_SM_EFT/ww_rapidity_histograms.csv

Main updates in this version:
    - mass plot is restricted to M_WW <= 800 GeV;
    - rapidity plot is restricted to -5 < Y_WW < +5;
    - rapidity plot uses 10 bins and no ratio panel;
    - the large plot title is removed from both figures;
    - both plots include the annotation:
      Elastic (Q_e^2 < 10^5 GeV^2; Q_p^2 < 10^5 GeV^2);
    - mass-ratio panel includes an optional MC statistical uncertainty band.
"""

from __future__ import annotations

import gzip
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt

try:
    import mplhep as hep
    hep.style.use("ATLAS")
except Exception:
    print("WARNING: mplhep is not available. Using default Matplotlib style.")


# ============================================================
# User configuration
# ============================================================

BASE = Path("/home/hamzeh-khanpour/MG5_aMC_v3_6_6")

SM_FILES = [
    BASE / "LHeC_aa_ww_SM_NP0/Events/run_01/unweighted_events.lhe",
    BASE / "LHeC_aa_ww_SM_NP0/Events/run_02/unweighted_events.lhe",
]

SM_SIGMA_PB = [
    0.039381,  # run_01
    0.039383,  # run_02
]

EFT_FILES = [
    BASE / "LHeC_aa_ww_EFT_NP1/Events/run_01/unweighted_events.lhe",
    BASE / "LHeC_aa_ww_EFT_NP1/Events/run_02/unweighted_events.lhe",
]

EFT_SIGMA_PB = [
    0.039647,  # run_01  0.039347
    0.039646,  # run_02  0.039346
]

OUTDIR = Path("plots_aa_ww_SM_EFT")
OUTDIR.mkdir(parents=True, exist_ok=True)

# The large plot title is intentionally disabled for the final version.
PLOT_TITLE = None
#PROCESS_NOTE = r"Elastic ($Q_e^2<10^5~\mathrm{GeV}^2$; $Q_p^2<10^5~\mathrm{GeV}^2$)"
PROCESS_NOTE = r"Elastic ($ep \to e(\gamma \gamma \to W^+W^-)p^*$)"

SM_LABEL = r"$W^+W^-$ (SM)"
EFT_LABEL = r"$W^+W^-$ ($f_{M2}/\Lambda^4=1~\mathrm{TeV}^{-4}$)"

SM_COLOR = "red"
EFT_COLOR = "blue"
RATIO_BAND_COLOR = "tab:blue"

# Invariant-mass plot: updated to stop at 800 GeV
MASS_MIN = 160.0
MASS_MAX = 800.0
MASS_NBINS = 40
MASS_BINNING = "linear"  # choose "linear" or "log"

# Rapidity plot: updated to -5 < Y_WW < +5 and no ratio panel
Y_MIN = -5.001
Y_MAX = 5.001
Y_NBINS = 40

# Ratio panel settings for mass plot only
RATIO_LABEL = r"EFT/SM"
MASS_RATIO_YLIM = (0.85, 1.15)
SHOW_RATIO_BAND = True

SAVE_DPI = 600

LEGEND_FONTSIZE = 18
PROCESS_NOTE_FONTSIZE = 18


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
        not as doubled luminosity.

        We use an event-count-weighted average of the run cross sections.
        For equal-statistics runs, this is the arithmetic mean.
        """
        n_total = self.n_events
        if n_total == 0:
            return 0.0
        return sum(len(run.masses) * run.sigma_pb for run in self.runs) / n_total

    def values_and_weights(self, observable: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return concatenated observable values and event weights.

        Each event in run i gets weight:
            sigma_i / N_total

        Therefore, the total histogram integral is the combined effective
        cross section, not the sum of the two run cross sections.
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

            weights = np.full(len(values), run.sigma_pb / n_total)
            values_list.append(values)
            weights_list.append(weights)

        return np.concatenate(values_list), np.concatenate(weights_list)


# ============================================================
# LHE reader
# ============================================================

def resolve_lhe_path(path: Path) -> Path:
    """
    Return an existing LHE path. If the plain .lhe path does not exist,
    try the compressed .lhe.gz version.
    """
    path = Path(path)

    if path.exists():
        return path

    gz_path = Path(str(path) + ".gz")
    if gz_path.exists():
        return gz_path

    raise FileNotFoundError(
        f"Could not find LHE file:\n"
        f"  {path}\n"
        f"or compressed file:\n"
        f"  {gz_path}"
    )


def open_lhe(path: Path):
    """Open .lhe or .lhe.gz as text."""
    path = resolve_lhe_path(path)

    if str(path).endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")

    return open(path, "rt", encoding="utf-8", errors="replace")


def read_ww_observables_from_lhe(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Read one LHE file and reconstruct M_WW and Y_WW.

    The function searches for W+ and W- using PDG IDs:
        W+ : +24
        W- : -24

    It accepts status 1 or 2:
        status 1: final-state W bosons in undecayed samples;
        status 2: intermediate W bosons if decays are later included.
    """
    masses: List[float] = []
    rapidities: List[float] = []

    in_event = False
    first_line_after_event_tag = False

    wplus: Optional[FourVector] = None
    wminus: Optional[FourVector] = None

    with open_lhe(path) as lhe:
        for raw_line in lhe:
            line = raw_line.strip()

            if not line:
                continue

            if line.startswith("<event"):
                in_event = True
                first_line_after_event_tag = True
                wplus = None
                wminus = None
                continue

            if line.startswith("</event"):
                if wplus is not None and wminus is not None:
                    ww = wplus + wminus
                    y_ww = ww.rapidity
                    if math.isfinite(y_ww):
                        masses.append(ww.mass)
                        rapidities.append(y_ww)

                in_event = False
                first_line_after_event_tag = False
                continue

            if not in_event:
                continue

            # The first non-empty line after <event> is the event header:
            # NUP IDPRUP XWGTUP SCALUP AQEDUP AQCDUP
            if first_line_after_event_tag:
                first_line_after_event_tag = False
                continue

            if line.startswith("#") or line.startswith("<"):
                continue

            parts = line.split()

            # LHE particle line has at least:
            # IDUP ISTUP MOTH1 MOTH2 ICOL1 ICOL2 PX PY PZ E M VTIM SPIN
            if len(parts) < 10:
                continue

            try:
                pid = int(parts[0])
                status = int(parts[1])
            except ValueError:
                continue

            if abs(pid) != 24:
                continue

            if status not in (1, 2):
                continue

            try:
                px = float(parts[6])
                py = float(parts[7])
                pz = float(parts[8])
                energy = float(parts[9])
            except ValueError:
                continue

            vec = FourVector(px=px, py=py, pz=pz, e=energy)

            if pid == 24:
                wplus = vec
            elif pid == -24:
                wminus = vec

    return np.asarray(masses, dtype=float), np.asarray(rapidities, dtype=float)


def load_sample(
    name: str,
    files: Iterable[Path],
    cross_sections_pb: Iterable[float],
) -> SampleData:
    files = list(files)
    cross_sections_pb = list(cross_sections_pb)

    if len(files) != len(cross_sections_pb):
        raise ValueError(f"Sample {name}: number of files and cross sections differ.")

    runs: List[RunData] = []

    print("\n" + "=" * 80)
    print(f"Loading sample: {name}")
    print("=" * 80)

    for path, sigma_pb in zip(files, cross_sections_pb):
        path = resolve_lhe_path(path)

        print(f"Reading: {path}")
        masses, rapidities = read_ww_observables_from_lhe(path)

        if len(masses) == 0:
            raise RuntimeError(f"No W+W- events were reconstructed from file:\n{path}")

        print(f"  accepted W+W- events : {len(masses):,}")
        print(f"  cross section used   : {sigma_pb:.9f} pb")

        runs.append(
            RunData(
                path=path,
                sigma_pb=float(sigma_pb),
                masses=masses,
                rapidities=rapidities,
            )
        )

    sample = SampleData(name=name, runs=runs)

    print("-" * 80)
    print(f"Total accepted events       : {sample.n_events:,}")
    print(f"Combined effective sigma    : {sample.combined_sigma_pb:.9f} pb")
    print("=" * 80)

    return sample


# ============================================================
# Histogram utilities
# ============================================================

def make_histogram(
    values: np.ndarray,
    weights: np.ndarray,
    bins: np.ndarray,
    differential: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create a weighted histogram and MC statistical uncertainty.

    If differential=True, returns dσ/dx with uncertainties.
    If differential=False, returns bin-integrated σ_bin with uncertainties.
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


def ratio_uncertainty(
    numerator: np.ndarray,
    numerator_err: np.ndarray,
    denominator: np.ndarray,
    denominator_err: np.ndarray,
) -> np.ndarray:
    """
    Propagate independent MC statistical uncertainties for R = A/B:
        δR = R sqrt[(δA/A)^2 + (δB/B)^2]

    This is a Monte Carlo statistical uncertainty band, not an
    experimental systematic uncertainty.
    """
    ratio = safe_ratio(numerator, denominator)
    err = np.full_like(ratio, np.nan, dtype=float)

    mask = (numerator > 0.0) & (denominator > 0.0) & np.isfinite(ratio)
    err[mask] = ratio[mask] * np.sqrt(
        (numerator_err[mask] / numerator[mask]) ** 2
        + (denominator_err[mask] / denominator[mask]) ** 2
    )

    return err


def save_histogram_csv(
    outpath: Path,
    bins: np.ndarray,
    sm_hist: np.ndarray,
    sm_err: np.ndarray,
    eft_hist: np.ndarray,
    eft_err: np.ndarray,
    ratio: np.ndarray,
    ratio_err: Optional[np.ndarray] = None,
) -> None:
    centers = 0.5 * (bins[:-1] + bins[1:])

    if ratio_err is None:
        ratio_err = np.full_like(ratio, np.nan, dtype=float)

    table = np.column_stack(
        [
            bins[:-1],
            bins[1:],
            centers,
            sm_hist,
            sm_err,
            eft_hist,
            eft_err,
            ratio,
            ratio_err,
        ]
    )

    header = (
        "bin_low,bin_high,bin_center,"
        "SM,SM_MCerr,EFT,EFT_MCerr,SM_over_EFT,SM_over_EFT_MCerr"
    )

    np.savetxt(outpath, table, delimiter=",", header=header, comments="")
    print(f"Saved CSV: {outpath}")


# ============================================================
# Plotting utilities
# ============================================================

def setup_ratio_axes(figsize=(8.0, 9.0)):
    fig, (ax, rax) = plt.subplots(
        2,
        1,
        figsize=figsize,
        sharex=True,
        gridspec_kw={"height_ratios": [3.4, 1.0], "hspace": 0.06},
    )

    plt.subplots_adjust(left=0.15, right=0.95, bottom=0.12, top=0.95)

    return fig, ax, rax


def setup_single_axis(figsize=(8.0, 9.0)):
    fig, ax = plt.subplots(figsize=figsize)
    plt.subplots_adjust(left=0.15, right=0.95, bottom=0.12, top=0.95)
    return fig, ax


def draw_step(
    ax,
    bins: np.ndarray,
    values: np.ndarray,
    label: str,
    color: str,
    linestyle: str,
    linewidth: float = 2.2,
) -> None:
    ax.stairs(values, bins, label=label, color=color, linestyle=linestyle, linewidth=linewidth)


def set_safe_log_y(ax, arrays: List[np.ndarray]) -> None:
    positive_values = []

    for array in arrays:
        mask = np.isfinite(array) & (array > 0.0)
        positive_values.extend(array[mask])

    if len(positive_values) == 0:
        return

    positive_values = np.asarray(positive_values)
    ymin = max(np.min(positive_values) * 0.3, 1e-12)
    ymax = np.max(positive_values) * 3.0

    ax.set_yscale("log")
    ax.set_ylim(ymin, ymax)


def style_axis(ax) -> None:
    ax.grid(True, which="major", linestyle="--", alpha=0.55)
    ax.grid(True, which="minor", linestyle=":", alpha=0.25)
    ax.tick_params(axis="both", which="both", direction="in", top=True, right=True)


def style_ratio_axis(rax) -> None:
    rax.axhline(1.0, color="black", linestyle="-", linewidth=1.2)
    rax.grid(True, which="major", linestyle="--", alpha=0.55)
    rax.grid(True, which="minor", linestyle=":", alpha=0.25)
    rax.tick_params(axis="both", which="both", direction="in", top=True, right=True)


def add_process_note(ax, *, loc: str = "upper right") -> None:
    """Add the elastic/Q^2 annotation inside the main axes."""
    if loc == "upper right":
        # Slightly below the larger line legend.
        xy = (0.97, 0.80)
        ha = "right"
        va = "top"
    elif loc == "upper left":
        # On the left side, below the SM/EFT legend.
        xy = (0.03, 0.82)
        ha = "left"
        va = "top"
    elif loc == "middle left":
        xy = (0.03, 0.58)
        ha = "left"
        va = "center"
    else:
        xy = (0.5, 0.76)
        ha = "center"
        va = "top"

    ax.text(
        xy[0],
        xy[1],
        PROCESS_NOTE,
        transform=ax.transAxes,
        fontsize=PROCESS_NOTE_FONTSIZE,
        ha=ha,
        va=va,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="none", alpha=0.75),
    )


# ============================================================
# Plot: invariant mass with ratio panel
# ============================================================

def plot_mass_distribution(sm: SampleData, eft: SampleData) -> None:
    if MASS_BINNING.lower() == "linear":
        bins = np.linspace(MASS_MIN, MASS_MAX, MASS_NBINS + 1)
    elif MASS_BINNING.lower() == "log":
        bins = np.geomspace(MASS_MIN, MASS_MAX, MASS_NBINS + 1)
    else:
        raise ValueError("MASS_BINNING must be either 'linear' or 'log'.")

    sm_values, sm_weights = sm.values_and_weights("mass")
    eft_values, eft_weights = eft.values_and_weights("mass")

    sm_mask = (sm_values >= MASS_MIN) & (sm_values <= MASS_MAX)
    eft_mask = (eft_values >= MASS_MIN) & (eft_values <= MASS_MAX)

    sm_hist, sm_err = make_histogram(sm_values[sm_mask], sm_weights[sm_mask], bins, differential=True)
    eft_hist, eft_err = make_histogram(eft_values[eft_mask], eft_weights[eft_mask], bins, differential=True)

    ratio = safe_ratio(eft_hist, sm_hist)
    ratio_err = ratio_uncertainty(eft_hist, eft_err, sm_hist, sm_err)

    fig, ax, rax = setup_ratio_axes(figsize=(10.0, 10.2))

    draw_step(ax, bins, sm_hist, label=SM_LABEL, color=SM_COLOR, linestyle="-")
    draw_step(ax, bins, eft_hist, label=EFT_LABEL, color=EFT_COLOR, linestyle="--")

    set_safe_log_y(ax, [sm_hist, eft_hist])

    ax.set_ylabel(r"$d\sigma/dM_{W^+W^-}$ [pb/GeV]", fontsize=24)
    if PLOT_TITLE:
        ax.set_title(PLOT_TITLE, fontsize=32, pad=10)
    ax.legend(loc="upper right", fontsize=LEGEND_FONTSIZE, frameon=False)
    style_axis(ax)
    add_process_note(ax, loc="upper right")

    if MASS_BINNING.lower() == "log":
        ax.set_xscale("log")

    centers = 0.5 * (bins[:-1] + bins[1:])

    if SHOW_RATIO_BAND:
        valid_band = np.isfinite(ratio) & np.isfinite(ratio_err)
        rax.fill_between(
            centers[valid_band],
            ratio[valid_band] - ratio_err[valid_band],
            ratio[valid_band] + ratio_err[valid_band],
            step="mid",
            color=RATIO_BAND_COLOR,
            alpha=0.22,
            linewidth=0,
            label="MC stat.",
        )

    rax.stairs(ratio, bins, color="black", linestyle="-", linewidth=1.8)

    rax.set_ylabel(RATIO_LABEL, fontsize=20)
    rax.set_xlabel(r"$M_{W^+W^-}$ [GeV]", fontsize=24)
    rax.set_xlim(MASS_MIN, MASS_MAX)
    rax.set_ylim(*MASS_RATIO_YLIM)
    style_ratio_axis(rax)

    fig.align_ylabels([ax, rax])

    out_pdf = OUTDIR / "ww_mass_SM_vs_EFT_ratio.pdf"
    out_png = OUTDIR / "ww_mass_SM_vs_EFT_ratio.png"

    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=SAVE_DPI)
    plt.close(fig)

    print(f"Saved plot: {out_pdf}")
    print(f"Saved plot: {out_png}")

    save_histogram_csv(
        OUTDIR / "ww_mass_histograms.csv",
        bins,
        sm_hist,
        sm_err,
        eft_hist,
        eft_err,
        ratio,
        ratio_err,
    )


# ============================================================
# Plot: rapidity without ratio panel, Y in [-5, 5]
# ============================================================

def plot_rapidity_distribution(sm: SampleData, eft: SampleData) -> None:
    bins = np.linspace(Y_MIN, Y_MAX, Y_NBINS + 1)

    sm_values, sm_weights = sm.values_and_weights("rapidity")
    eft_values, eft_weights = eft.values_and_weights("rapidity")

    sm_mask = (sm_values >= Y_MIN) & (sm_values <= Y_MAX)
    eft_mask = (eft_values >= Y_MIN) & (eft_values <= Y_MAX)

    sm_hist, sm_err = make_histogram(sm_values[sm_mask], sm_weights[sm_mask], bins, differential=True)
    eft_hist, eft_err = make_histogram(eft_values[eft_mask], eft_weights[eft_mask], bins, differential=True)

    ratio = safe_ratio(sm_hist, eft_hist)
    ratio_err = ratio_uncertainty(sm_hist, sm_err, eft_hist, eft_err)

    fig, ax = setup_single_axis(figsize=(8.0, 9.0))

    draw_step(ax, bins, sm_hist, label=SM_LABEL, color=SM_COLOR, linestyle="-")
    draw_step(ax, bins, eft_hist, label=EFT_LABEL, color=EFT_COLOR, linestyle="--")

    ymax = max(np.nanmax(sm_hist), np.nanmax(eft_hist))
#    ax.set_ylim(0.0, 1.25 * ymax if ymax > 0.0 else 1.0)
    ax.set_ylim(0.0, 3.01e-2)
    ax.set_xlim(Y_MIN, Y_MAX)

    ax.set_ylabel(r"$d\sigma/dY_{W^+W^-}$ [pb]", fontsize=24)
    ax.set_xlabel(r"$Y_{W^+W^-}$", fontsize=24)
    if PLOT_TITLE:
        ax.set_title(PLOT_TITLE, fontsize=32, pad=10)
    ax.legend(loc="upper left", fontsize=LEGEND_FONTSIZE, frameon=False)
    style_axis(ax)
    add_process_note(ax, loc="upper left")

    out_pdf = OUTDIR / "ww_rapidity_SM_vs_EFT_ratio.pdf"
    out_png = OUTDIR / "ww_rapidity_SM_vs_EFT_ratio.png"

    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=SAVE_DPI)
    plt.close(fig)

    print(f"Saved plot: {out_pdf}")
    print(f"Saved plot: {out_png}")

    save_histogram_csv(
        OUTDIR / "ww_rapidity_histograms.csv",
        bins,
        sm_hist,
        sm_err,
        eft_hist,
        eft_err,
        ratio,
        ratio_err,
    )


# ============================================================
# Main
# ============================================================

def main() -> None:
    print("\nStarting LHeC gamma gamma -> W+W- SM/EFT analysis")
    print("=" * 80)
    print(r"Observable 1: dσ/dM_{W+W-}, up to 800 GeV")
    print(r"Observable 2: dσ/dY_{W+W-}, with -5 < Y < +5 and 10 bins")
    print(r"Mass ratio panel: SM/EFT")
    print(r"Rapidity plot: no ratio panel")
    print(f"Output dir: {OUTDIR.resolve()}")
    print("=" * 80)

    sm_sample = load_sample("SM", SM_FILES, SM_SIGMA_PB)
    eft_sample = load_sample("EFT", EFT_FILES, EFT_SIGMA_PB)

    print("\nCross-section summary")
    print("-" * 80)
    print(f"SM combined effective sigma  = {sm_sample.combined_sigma_pb:.9f} pb")
    print(f"EFT combined effective sigma = {eft_sample.combined_sigma_pb:.9f} pb")
    print(f"Inclusive SM/EFT ratio       = {sm_sample.combined_sigma_pb / eft_sample.combined_sigma_pb:.9f}")
    print("-" * 80)

    # Diagnostic for the WW-system rapidity.  In the MadGraph/LHE convention,
    # beam 1 travels along +z.  Here beam 1 is the 7 TeV proton and beam 2 is
    # the 50 GeV electron, so the gamma-gamma system is naturally boosted
    # toward +z.  Therefore a mostly positive Y_WW distribution is expected.
    for sample in (sm_sample, eft_sample):
        y_all, _ = sample.values_and_weights("rapidity")
        print(
            f"{sample.name} rapidity range: "
            f"min={np.nanmin(y_all):.4f}, "
            f"mean={np.nanmean(y_all):.4f}, "
            f"max={np.nanmax(y_all):.4f}"
        )
    print("-" * 80)

    plot_mass_distribution(sm_sample, eft_sample)
    plot_rapidity_distribution(sm_sample, eft_sample)

    print("\nDone.")


if __name__ == "__main__":
    main()
