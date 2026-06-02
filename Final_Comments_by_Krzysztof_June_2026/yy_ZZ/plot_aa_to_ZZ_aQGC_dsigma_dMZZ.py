#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Plot d sigma / d M_ZZ for gamma gamma -> ZZ AQGC samples.

Default input directory:
  /home/hamzeh-khanpour/MG5_aMC_v3_6_6/LHeC_aa_zz_EFT_NP1/Events

Default target coefficients:
  f_M2/Lambda^4 = -5, -1, +1, +5 TeV^{-4}

Usage:
  python3 plot_aa_to_ZZ_aQGC_dsigma_dMZZ.py \
    --events-dir /home/hamzeh-khanpour/MG5_aMC_v3_6_6/LHeC_aa_zz_EFT_NP1/Events

Outputs:
  plots_aa_ZZ_aQGC_dsigma_dMZZ/aa_to_ZZ_aQGC_dsigma_dMZZ_selected_FM2.pdf
  plots_aa_ZZ_aQGC_dsigma_dMZZ/aa_to_ZZ_aQGC_dsigma_dMZZ_selected_FM2.png
  plots_aa_ZZ_aQGC_dsigma_dMZZ/aa_to_ZZ_aQGC_dsigma_dMZZ_selected_FM2.csv

Physics note:
  These samples are generated with a a > z z NP=1, so they correspond to the
  pure AQGC-squared contribution. Therefore, +f_M2 and -f_M2 are expected to
  have the same total rate and very similar M_ZZ shapes, up to MC fluctuations.

python3 plot_aa_to_ZZ_aQGC_dsigma_dMZZ.py \
  --events-dir /home/hamzeh-khanpour/MG5_aMC_v3_6_6/LHeC_aa_zz_EFT_NP1/Events


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

# Style chosen to be close to the tau-pair plotting script.
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
    "/home/hamzeh-khanpour/MG5_aMC_v3_6_6/LHeC_aa_zz_EFT_NP1/Events"
)
DEFAULT_OUTDIR = Path("plots_aa_ZZ_aQGC_dsigma_dMZZ")
SAVE_DPI = 600

DEFAULT_TARGETS = [1.0, 5.0, 10.0]
DEFAULT_BINS = np.linspace(180.0, 1200.0, 41, dtype=float)


@dataclass
class RunPoint:
    run: str
    fm2_gev: float
    fm2_tev: float
    sigma_pb: float
    nevents_banner: int
    banner: Path
    lhe: Path


def style_axis(ax) -> None:
    ax.grid(True, which="major", linestyle="--", alpha=0.55)
    ax.grid(True, which="minor", linestyle=":", alpha=0.25)
    ax.minorticks_on()
    ax.tick_params(axis="both", which="both", direction="in", top=True, right=True)


def parse_banner(banner: Path, lhe_path: Path) -> RunPoint:
    text = banner.read_text(encoding="utf-8", errors="replace")

    fm2_match = re.search(r"\n\s*5\s+([-+0-9.eE]+)\s*#\s*FM2", text)
    xs_match = re.search(r"Integrated\s+weight\s*\(pb\)\s*:\s*([-+0-9.eE]+)", text)
    nev_match = re.search(r"Number\s+of\s+Events\s*:\s*([0-9]+)", text)

    if fm2_match is None:
        raise RuntimeError(f"Could not find FM2 in banner: {banner}")
    if xs_match is None:
        raise RuntimeError(f"Could not find Integrated weight in banner: {banner}")
    if nev_match is None:
        raise RuntimeError(f"Could not find Number of Events in banner: {banner}")

    fm2_gev = float(fm2_match.group(1))
    sigma_pb = float(xs_match.group(1))
    nevents = int(nev_match.group(1))
    fm2_tev = fm2_gev * 1.0e12

    return RunPoint(
        run=banner.parent.name,
        fm2_gev=fm2_gev,
        fm2_tev=fm2_tev,
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

    return sorted(points, key=lambda p: p.fm2_tev)


def find_target_runs(points: Sequence[RunPoint], targets: Sequence[float], tolerance: float = 1.0e-8) -> List[RunPoint]:
    selected: List[RunPoint] = []
    for target in targets:
        matches = [p for p in points if abs(p.fm2_tev - target) < tolerance]
        if not matches:
            available = ", ".join(f"{p.fm2_tev:g}" for p in points)
            raise RuntimeError(
                f"Could not find target fM2/Lambda^4 = {target:g} TeV^-4.\n"
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


def read_mzz_from_lhe(path: Path) -> np.ndarray:
    masses: List[float] = []
    in_event = False
    first_line_after_event = False
    z_vectors: List[Tuple[float, float, float, float]] = []

    with open_lhe(path) as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue

            if line.startswith("<event"):
                in_event = True
                first_line_after_event = True
                z_vectors = []
                continue

            if line.startswith("</event"):
                if len(z_vectors) == 2:
                    masses.append(invariant_mass(z_vectors))
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

            # In this 2 -> 2 process the Z bosons are normally status 1.
            # Status 2 is accepted as a fallback for alternative LHE conventions.
            if pid != 23 or status not in (1, 2):
                continue

            try:
                px = float(parts[6])
                py = float(parts[7])
                pz = float(parts[8])
                energy = float(parts[9])
            except ValueError:
                continue

            z_vectors.append((energy, px, py, pz))

    return np.asarray(masses, dtype=float)


def make_dsigma(mzz: np.ndarray, sigma_pb: float, bins: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    counts, edges = np.histogram(mzz, bins=bins)
    widths = np.diff(edges)
    centers = 0.5 * (edges[:-1] + edges[1:])
    if len(mzz) == 0:
        raise RuntimeError("No M_ZZ values found; cannot normalize histogram.")
    dsigma = sigma_pb * counts / len(mzz) / widths
    return centers, dsigma, counts


def save_csv(outpath: Path, rows: List[Dict[str, float]]) -> None:
    with open(outpath, "w", encoding="utf-8") as f:
        f.write("run,fM2_over_Lambda4_TeV^-4,sigma_total_pb,bin_low_GeV,bin_high_GeV,bin_center_GeV,counts,dsigma_dMZZ_pb_per_GeV\n")
        for row in rows:
            f.write(
                f"{row['run']},{row['fm2']:.8g},{row['sigma']:.12e},"
                f"{row['low']:.6f},{row['high']:.6f},{row['center']:.6f},"
                f"{int(row['counts'])},{row['dsigma']:.12e}\n"
            )
    print(f"Saved CSV: {outpath}")


def plot_dsigma(selected: Sequence[RunPoint], bins: np.ndarray, outdir: Path) -> None:

    fig, ax = plt.subplots(figsize=(10.0, 10.2))

    plt.subplots_adjust(left=0.15, right=0.95, bottom=0.12, top=0.95)

    rows: List[Dict[str, float]] = []

    # Distinct line styles help because +f and -f can overlap in a pure NP^2 calculation.
    linestyles = {
        -5.0: "-",
        -1.0: "--",
        1.0: ":",
        5.0: "-.",
    }

    for point in selected:
        print(f"Reading {point.run}: fM2/Lambda^4={point.fm2_tev:g} TeV^-4, sigma={point.sigma_pb:.6e} pb")
        mzz = read_mzz_from_lhe(point.lhe)
        if len(mzz) == 0:
            raise RuntimeError(f"No ZZ events reconstructed from {point.lhe}")

        print(
            f"  reconstructed events: {len(mzz):,}; "
            f"MZZ range = ({mzz.min():.2f}, {mzz.max():.2f}) GeV"
        )

        centers, dsigma, counts = make_dsigma(mzz, point.sigma_pb, bins)
        label = rf"$f_{{M2}}/\Lambda^4={point.fm2_tev:g}$ TeV$^{{-4}}$"
        linestyle = linestyles.get(float(point.fm2_tev), "-")

        ax.stairs(dsigma, bins, linewidth=2.7, linestyle=linestyle, label=label)

        for low, high, center, count, value in zip(bins[:-1], bins[1:], centers, counts, dsigma):
            rows.append(
                {
                    "run": point.run,
                    "fm2": point.fm2_tev,
                    "sigma": point.sigma_pb,
                    "low": low,
                    "high": high,
                    "center": center,
                    "counts": float(count),
                    "dsigma": value,
                }
            )

    ax.set_yscale("log")

    ax.set_ylim(1.0e-12, 1.0e-5)

    ax.set_xlim(float(bins[0]), float(bins[-1]))

    ax.set_xlabel(r"$M_{ZZ}$ [GeV]")
    ax.set_ylabel(r"$d\sigma_{\rm aQGC}/dM_{ZZ}$ [pb/GeV]")
    ax.legend(loc="upper right", frameon=False)
    style_axis(ax)

    ax.text(
        0.04,
        0.96,
        r"Elastic, $ep\to e(\gamma\gamma\to ZZ)p^{(*)}$" "\n" "\n"
        r"tree-level aQGC contribution",
        transform=ax.transAxes,
        fontsize=18,
        ha="left",
        va="top",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="none", alpha=0.78),
    )

    for ext in ("pdf", "png"):
        out = outdir / f"aa_to_ZZ_aQGC_dsigma_dMZZ_selected_FM2.{ext}"
        fig.savefig(out, dpi=SAVE_DPI if ext == "png" else None)
        print(f"Saved plot: {out}")
    plt.close(fig)

    save_csv(outdir / "aa_to_ZZ_aQGC_dsigma_dMZZ_selected_FM2.csv", rows)


def parse_targets(text: str) -> List[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def parse_bins(text: str) -> np.ndarray:
    return np.array([float(x.strip()) for x in text.split(",") if x.strip()], dtype=float)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot d sigma / d M_ZZ for selected AQGC fM2 values.")
    parser.add_argument("--events-dir", type=Path, default=DEFAULT_EVENTS_DIR)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument(
        "--targets",
        type=str,
        default=",".join(str(x) for x in DEFAULT_TARGETS),
        help="Comma-separated target fM2/Lambda^4 values in TeV^-4, e.g. '-5,-1,1,5'",
    )
    parser.add_argument(
        "--bins",
        type=str,
        default=",".join(str(int(x)) for x in DEFAULT_BINS),
        help="Comma-separated MZZ bin edges in GeV.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    targets = parse_targets(args.targets)
    bins = parse_bins(args.bins)

    points = discover_points(args.events_dir)
    selected = find_target_runs(points, targets)

    print("\nSelected runs")
    print("-" * 94)
    print(f"{'run':8s} {'FM2 [GeV^-4]':>18s} {'fM2/Lambda^4 [TeV^-4]':>28s} {'sigma [pb]':>16s} {'LHE':>10s}")
    for p in selected:
        print(f"{p.run:8s} {p.fm2_gev:18.6e} {p.fm2_tev:28.6g} {p.sigma_pb:16.6e} {p.lhe.name:>10s}")
    print("-" * 94)

    plot_dsigma(selected, bins, args.outdir)


if __name__ == "__main__":
    main()
