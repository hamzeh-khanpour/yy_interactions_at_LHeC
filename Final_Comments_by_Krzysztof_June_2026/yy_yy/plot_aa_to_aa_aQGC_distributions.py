#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Plot differential distributions for pure dimension-eight
gamma gamma -> gamma gamma EFT samples.

Input:
  MadGraph Events directory containing run_XX folders with:
    - run_XX_tag_1_banner.txt
    - unweighted_events.lhe.gz

Default input directory:
  /home/hamzeh-khanpour/MG5_aMC_v3_6_6/aa_to_aa_FT8_FT9_NP1/Events

Usage:
  python3 plot_aa_to_aa_aQGC_distributions.py \
    --events-dir /home/hamzeh-khanpour/MG5_aMC_v3_6_6/aa_to_aa_FT8_FT9_NP1/Events \
    --targets 1,5,10

Outputs:
  plots_aa_aa_aQGC_distributions/*.pdf
  plots_aa_aa_aQGC_distributions/*.png
  plots_aa_aa_aQGC_distributions/*.csv

Physics note:
  The generated process is a a > a a NP=1, so these distributions are the
  pure tree-level dimension-eight EFT contribution. There is no SM loop-induced
  light-by-light contribution and no SM--EFT interference included here.

Histogram convention:
  - M_aa and Y_aa are filled once per event.
  - eta_gamma and pT_gamma are inclusive photon distributions; both final-state
    photons are filled into the same histogram. Therefore, the integral of
    d sigma / d eta_gamma is 2 * sigma after the mass cut.


python3 plot_aa_to_aa_aQGC_distributions.py \
  --events-dir /home/hamzeh-khanpour/MG5_aMC_v3_6_6/aa_to_aa_FT8_FT9_NP1/Events \
  --targets 1,5,10



"""

from __future__ import annotations

import argparse
import gzip
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt

try:
    import mplhep as hep
    hep.style.use("CMS")
except Exception:
    print("WARNING: mplhep is not available. Using default Matplotlib style.")


# Same style settings as the uploaded ZZ scripts.
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

DEFAULT_EVENTS_DIR = Path(
    "/home/hamzeh-khanpour/MG5_aMC_v3_6_6/aa_to_aa_FT8_FT9_NP1/Events"
)
DEFAULT_OUTDIR = Path("plots_aa_aa_aQGC_distributions")
SAVE_DPI = 600

DEFAULT_TARGETS = [1.0, 5.0, 10.0]

DEFAULT_M_BINS = np.linspace(0.0, 1200.0, 49, dtype=float)
DEFAULT_ETA_BINS = np.linspace(-10.0, 10.0, 81, dtype=float)
DEFAULT_Y_BINS = np.linspace(-6.0, 8.0, 57, dtype=float)
DEFAULT_PT_BINS = np.linspace(0.0, 600.0, 49, dtype=float)


@dataclass
class RunPoint:
    run: str
    ft8_gev: float
    ft8_tev: float
    sigma_pb: float
    nevents_banner: int
    banner: Path
    lhe: Path


@dataclass
class EventKinematics:
    maa: float
    yaa: float
    eta1: float
    eta2: float
    pt1: float
    pt2: float


def style_axis(ax) -> None:
    ax.grid(True, which="major", linestyle="--", alpha=0.55)
    ax.grid(True, which="minor", linestyle=":", alpha=0.25)
    ax.minorticks_on()
    ax.tick_params(axis="both", which="both", direction="in", top=True, right=True)


def parse_banner(banner: Path, lhe_path: Path) -> RunPoint:
    text = banner.read_text(encoding="utf-8", errors="replace")

    ft8_match = re.search(r"\n\s*19\s+([-+0-9.eE]+)\s*#\s*FT8", text)
    xs_match = re.search(r"Integrated\s+weight\s*\(pb\)\s*:\s*([-+0-9.eE]+)", text)
    nev_match = re.search(r"Number\s+of\s+Events\s*:\s*([0-9]+)", text)

    if ft8_match is None:
        raise RuntimeError(f"Could not find FT8 in banner: {banner}")
    if xs_match is None:
        raise RuntimeError(f"Could not find Integrated weight in banner: {banner}")
    if nev_match is None:
        raise RuntimeError(f"Could not find Number of Events in banner: {banner}")

    ft8_gev = float(ft8_match.group(1))
    sigma_pb = float(xs_match.group(1))
    nevents = int(nev_match.group(1))
    ft8_tev = ft8_gev * 1.0e12

    return RunPoint(
        run=banner.parent.name,
        ft8_gev=ft8_gev,
        ft8_tev=ft8_tev,
        sigma_pb=sigma_pb,
        nevents_banner=nevents,
        banner=banner,
        lhe=lhe_path,
    )


def resolve_lhe_path(run_dir: Path) -> Path:
    candidates = [run_dir / "unweighted_events.lhe.gz", run_dir / "unweighted_events.lhe"]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"No unweighted_events.lhe(.gz) found in {run_dir}")


def discover_points(events_dir: Path) -> List[RunPoint]:
    run_dirs = sorted([p for p in events_dir.iterdir() if p.is_dir() and p.name.startswith("run_")])
    if not run_dirs:
        raise FileNotFoundError(f"No run_XX directories found under: {events_dir}")

    points: List[RunPoint] = []
    for run_dir in run_dirs:
        banners = sorted(run_dir.glob("*_banner.txt"))
        if not banners:
            print(f"WARNING: no banner found in {run_dir}; skipping")
            continue
        lhe_path = resolve_lhe_path(run_dir)
        points.append(parse_banner(banners[0], lhe_path))

    if not points:
        raise RuntimeError(f"No valid run points found under: {events_dir}")

    return sorted(points, key=lambda p: p.ft8_tev)


def find_target_runs(points: Sequence[RunPoint], targets: Sequence[float], tolerance: float = 1.0e-8) -> List[RunPoint]:
    selected: List[RunPoint] = []
    for target in targets:
        matches = [p for p in points if abs(p.ft8_tev - target) < tolerance]
        if not matches:
            available = ", ".join(f"{p.ft8_tev:g}" for p in points)
            raise RuntimeError(
                f"Could not find target fT8/Lambda^4 = {target:g} TeV^-4.\n"
                f"Available values are: {available}"
            )
        selected.append(matches[0])
    return selected


def open_lhe(path: Path):
    if str(path).endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return open(path, "rt", encoding="utf-8", errors="replace")


def invariant_mass(vectors: Sequence[Tuple[float, float, float, float]]) -> float:
    """Input tuple order: (E, px, py, pz)."""
    e = sum(v[0] for v in vectors)
    px = sum(v[1] for v in vectors)
    py = sum(v[2] for v in vectors)
    pz = sum(v[3] for v in vectors)
    m2 = e * e - px * px - py * py - pz * pz
    return math.sqrt(max(m2, 0.0))


def rapidity(vectors: Sequence[Tuple[float, float, float, float]]) -> float:
    """Rapidity of the sum of vectors. Input tuple order: (E, px, py, pz)."""
    e = sum(v[0] for v in vectors)
    pz = sum(v[3] for v in vectors)
    denom = e - pz
    numer = e + pz
    if denom <= 0.0 or numer <= 0.0:
        return float("nan")
    return 0.5 * math.log(numer / denom)


def pt(v: Tuple[float, float, float, float]) -> float:
    _, px, py, _ = v
    return math.sqrt(px * px + py * py)


def eta(v: Tuple[float, float, float, float]) -> float:
    """Pseudorapidity. Input tuple order: (E, px, py, pz)."""
    _, px, py, pz = v
    p = math.sqrt(px * px + py * py + pz * pz)
    denom = p - pz
    numer = p + pz
    if denom <= 0.0 or numer <= 0.0:
        return float("nan")
    return 0.5 * math.log(numer / denom)


def read_aa_kinematics_from_lhe(path: Path) -> List[EventKinematics]:
    events: List[EventKinematics] = []

    in_event = False
    first_line_after_event = False
    photon_vectors: List[Tuple[float, float, float, float]] = []

    with open_lhe(path) as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue

            if line.startswith("<event"):
                in_event = True
                first_line_after_event = True
                photon_vectors = []
                continue

            if line.startswith("</event"):
                if len(photon_vectors) == 2:
                    maa = invariant_mass(photon_vectors)
                    yaa = rapidity(photon_vectors)
                    eta1 = eta(photon_vectors[0])
                    eta2 = eta(photon_vectors[1])
                    pt1 = pt(photon_vectors[0])
                    pt2 = pt(photon_vectors[1])

                    if all(math.isfinite(x) for x in (maa, yaa, eta1, eta2, pt1, pt2)):
                        events.append(
                            EventKinematics(
                                maa=maa,
                                yaa=yaa,
                                eta1=eta1,
                                eta2=eta2,
                                pt1=pt1,
                                pt2=pt2,
                            )
                        )

                in_event = False
                first_line_after_event = False
                continue

            if not in_event:
                continue

            # First non-empty line after <event> is the event-level header.
            if first_line_after_event:
                first_line_after_event = False
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

            # Keep only final-state photons.
            # Initial-state photons have status -1 and are ignored.
            if pid != 22 or status != 1:
                continue

            try:
                px = float(parts[6])
                py = float(parts[7])
                pz = float(parts[8])
                energy = float(parts[9])
            except ValueError:
                continue

            photon_vectors.append((energy, px, py, pz))

    return events


def make_event_hist(values: np.ndarray, sigma_pb: float, total_events: int, bins: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    counts, edges = np.histogram(values, bins=bins)
    widths = np.diff(edges)
    dsigma = sigma_pb * counts / total_events / widths
    return dsigma, counts


def make_inclusive_photon_hist(values: np.ndarray, sigma_pb: float, total_events: int, bins: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Inclusive photon histogram. Both photons are filled.
    Normalization uses the number of generated events, so the integral is
    2 * sigma after the applied event selection.
    """
    counts, edges = np.histogram(values, bins=bins)
    widths = np.diff(edges)
    dsigma = sigma_pb * counts / total_events / widths
    return dsigma, counts


def save_distribution_csv(
    outpath: Path,
    observable_name: str,
    bins: np.ndarray,
    rows_by_label: Dict[str, Tuple[np.ndarray, np.ndarray, float, int, int]],
) -> None:
    with open(outpath, "w", encoding="utf-8") as f:
        f.write(
            "observable,run_label,sigma_total_pb,generated_events,selected_events,"
            "bin_low,bin_high,bin_center,counts,dsigma_pb_per_unit\n"
        )
        for label, (dsigma, counts, sigma_total, n_total, n_selected) in rows_by_label.items():
            centers = 0.5 * (bins[:-1] + bins[1:])
            for low, high, center, count, value in zip(bins[:-1], bins[1:], centers, counts, dsigma):
                f.write(
                    f"{observable_name},{label},{sigma_total:.12e},{n_total},{n_selected},"
                    f"{low:.8e},{high:.8e},{center:.8e},{int(count)},{value:.12e}\n"
                )
    print(f"Saved CSV: {outpath}")


def plot_distribution(
    selected: Sequence[RunPoint],
    events_by_run: Dict[str, List[EventKinematics]],
    bins: np.ndarray,
    observable: str,
    xlabel: str,
    ylabel: str,
    outbase: Path,
    w0: float | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(10.0, 10.2))
    plt.subplots_adjust(left=0.15, right=0.95, bottom=0.12, top=0.95)

    linestyles = {
        0.5: "-",
        1.0: "--",
        5.0: ":",
        10.0: "-.",
        -0.5: "-",
        -1.0: "--",
        -5.0: ":",
        -10.0: "-.",
    }

    csv_rows: Dict[str, Tuple[np.ndarray, np.ndarray, float, int, int]] = {}

    for point in selected:
        events = events_by_run[point.run]
        total_events = len(events)
        if total_events == 0:
            raise RuntimeError(f"No reconstructed aa events for {point.run}")

        if w0 is None:
            selected_events = events
        else:
            selected_events = [ev for ev in events if ev.maa > w0]

        if observable == "Maa":
            values = np.array([ev.maa for ev in selected_events], dtype=float)
            dsigma, counts = make_event_hist(values, point.sigma_pb, total_events, bins)
        elif observable == "Yaa":
            values = np.array([ev.yaa for ev in selected_events], dtype=float)
            dsigma, counts = make_event_hist(values, point.sigma_pb, total_events, bins)
        elif observable == "eta_gamma":
            values = np.array(
                [x for ev in selected_events for x in (ev.eta1, ev.eta2)],
                dtype=float,
            )
            dsigma, counts = make_inclusive_photon_hist(values, point.sigma_pb, total_events, bins)
        elif observable == "pt_gamma":
            values = np.array(
                [x for ev in selected_events for x in (ev.pt1, ev.pt2)],
                dtype=float,
            )
            dsigma, counts = make_inclusive_photon_hist(values, point.sigma_pb, total_events, bins)
        else:
            raise ValueError(f"Unknown observable: {observable}")

        label = rf"$f_{{T8}}/\Lambda^4={point.ft8_tev:g}$ TeV$^{{-4}}$"
        linestyle = linestyles.get(float(point.ft8_tev), "-")

        ax.stairs(dsigma, bins, linewidth=2.7, linestyle=linestyle, label=label)

        csv_label = f"{point.run}_fT8_{point.ft8_tev:g}_TeV^-4"
        csv_rows[csv_label] = (dsigma, counts, point.sigma_pb, total_events, len(selected_events))

        cut_text = "inclusive" if w0 is None else rf"$M_{{\gamma\gamma}}>{w0:g}$ GeV"
        print(
            f"{point.run}: fT8={point.ft8_tev:g} TeV^-4, "
            f"{cut_text}, selected {len(selected_events):,}/{total_events:,} events, "
            f"sigma_after_cut = {point.sigma_pb * len(selected_events) / total_events:.6e} pb"
        )

    ax.set_yscale("log")

    # Let the lower bound remain readable even for sparse tails.
    positive_values = []
    for dsigma, _, _, _, _ in csv_rows.values():
        positive_values.extend([x for x in dsigma if x > 0.0])
    if positive_values:
        ymin = 1.0e-8   #   max(min(positive_values) * 0.3, 1.0e-14)
        ymax = 1.0e0   #   max(positive_values) * 5.0
        ax.set_ylim(ymin, ymax)

    ax.set_xlim(float(bins[0]), float(bins[-1]))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(loc="upper right", frameon=False)
    style_axis(ax)

    if w0 is None:
        annotation_cut = r"no $M_{\gamma\gamma}$ cut"
    else:
        annotation_cut = rf"$M_{{\gamma\gamma}}>{w0:g}$ GeV"

    ax.text(
        0.04,
        0.96,
        r"$ep\to e(\gamma\gamma\to\gamma\gamma)p^{(*)}$" "\n"
        r"tree-level dimension-eight contribution" "\n"
        + annotation_cut,
        transform=ax.transAxes,
        fontsize=18,
        ha="left",
        va="top",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="none", alpha=0.78),
    )

    for ext in ("pdf", "png"):
        out = outbase.with_suffix(f".{ext}")
        fig.savefig(out, dpi=SAVE_DPI if ext == "png" else None)
        print(f"Saved plot: {out}")
    plt.close(fig)

    save_distribution_csv(outbase.with_suffix(".csv"), observable, bins, csv_rows)


def parse_targets(text: str) -> List[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def parse_bins(text: str) -> np.ndarray:
    return np.array([float(x.strip()) for x in text.split(",") if x.strip()], dtype=float)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot aa -> aa EFT distributions from LHE files.")
    parser.add_argument("--events-dir", type=Path, default=DEFAULT_EVENTS_DIR)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument(
        "--targets",
        type=str,
        default=",".join(str(x) for x in DEFAULT_TARGETS),
        help="Comma-separated target fT8/Lambda^4 values in TeV^-4, e.g. '1,5,10'",
    )
    parser.add_argument(
        "--m-bins",
        type=str,
        default=",".join(str(float(x)) for x in DEFAULT_M_BINS),
        help="Comma-separated M_aa bin edges in GeV.",
    )
    parser.add_argument(
        "--eta-bins",
        type=str,
        default=",".join(str(float(x)) for x in DEFAULT_ETA_BINS),
        help="Comma-separated eta_gamma bin edges.",
    )
    parser.add_argument(
        "--y-bins",
        type=str,
        default=",".join(str(float(x)) for x in DEFAULT_Y_BINS),
        help="Comma-separated Y_aa bin edges.",
    )
    parser.add_argument(
        "--pt-bins",
        type=str,
        default=",".join(str(float(x)) for x in DEFAULT_PT_BINS),
        help="Comma-separated pT_gamma bin edges in GeV.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    targets = parse_targets(args.targets)
    m_bins = parse_bins(args.m_bins)
    eta_bins = parse_bins(args.eta_bins)
    y_bins = parse_bins(args.y_bins)
    pt_bins = parse_bins(args.pt_bins)

    points = discover_points(args.events_dir)
    selected = find_target_runs(points, targets)

    print("\nSelected runs")
    print("-" * 100)
    print(
        f"{'run':8s} {'FT8 [GeV^-4]':>18s} "
        f"{'fT8/Lambda^4 [TeV^-4]':>28s} {'sigma [pb]':>16s} {'LHE':>25s}"
    )
    for p in selected:
        print(
            f"{p.run:8s} {p.ft8_gev:18.6e} {p.ft8_tev:28.6g} "
            f"{p.sigma_pb:16.6e} {p.lhe.name:>25s}"
        )
    print("-" * 100)

    events_by_run: Dict[str, List[EventKinematics]] = {}

    for point in selected:
        print(f"\nReading LHE file for {point.run}: {point.lhe}")
        events = read_aa_kinematics_from_lhe(point.lhe)
        if not events:
            raise RuntimeError(f"No aa events reconstructed from {point.lhe}")
        events_by_run[point.run] = events

        maa_values = np.array([ev.maa for ev in events], dtype=float)
        print(
            f"  reconstructed events: {len(events):,}; "
            f"Maa range = ({maa_values.min():.4f}, {maa_values.max():.4f}) GeV"
        )

    # 1) Mass distribution without additional mass cut.
    plot_distribution(
        selected=selected,
        events_by_run=events_by_run,
        bins=m_bins,
        observable="Maa",
        xlabel=r"$M_{\gamma\gamma}$ [GeV]",
        ylabel=r"$d\sigma_{\rm EFT}/dM_{\gamma\gamma}$ [pb/GeV]",
        outbase=args.outdir / "aa_to_aa_aQGC_dsigma_dMaa_selected_FT8",
        w0=None,
    )

    # 2) Inclusive photon eta distributions for W0 = 10 and 100 GeV.
    for w0 in (10.0, 100.0):
        plot_distribution(
            selected=selected,
            events_by_run=events_by_run,
            bins=eta_bins,
            observable="eta_gamma",
            xlabel=r"$\eta_{\gamma}$",
            ylabel=r"$d\sigma_{\rm EFT}/d\eta_{\gamma}$ [pb]",
            outbase=args.outdir / f"aa_to_aa_aQGC_dsigma_deta_gamma_W{int(w0)}_selected_FT8",
            w0=w0,
        )

    # 3) Diphoton-system rapidity distributions for W0 = 10 and 100 GeV.
    for w0 in (10.0, 100.0):
        plot_distribution(
            selected=selected,
            events_by_run=events_by_run,
            bins=y_bins,
            observable="Yaa",
            xlabel=r"$Y_{\gamma\gamma}$",
            ylabel=r"$d\sigma_{\rm EFT}/dY_{\gamma\gamma}$ [pb]",
            outbase=args.outdir / f"aa_to_aa_aQGC_dsigma_dYaa_W{int(w0)}_selected_FT8",
            w0=w0,
        )

    # 4) Inclusive photon pT distributions for W0 = 10 and 100 GeV.
    for w0 in (10.0, 100.0):
        plot_distribution(
            selected=selected,
            events_by_run=events_by_run,
            bins=pt_bins,
            observable="pt_gamma",
            xlabel=r"$p_{T}^{\gamma}$ [GeV]",
            ylabel=r"$d\sigma_{\rm EFT}/dp_{T}^{\gamma}$ [pb/GeV]",
            outbase=args.outdir / f"aa_to_aa_aQGC_dsigma_dpT_gamma_W{int(w0)}_selected_FT8",
            w0=w0,
        )


if __name__ == "__main__":
    main()
