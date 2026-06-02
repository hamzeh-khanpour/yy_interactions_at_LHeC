#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Plot the total AQGC cross section for gamma gamma -> ZZ as a function of
f_M2/Lambda^4 using MadGraph run banners.

Default input directory:
  /home/hamzeh-khanpour/MG5_aMC_v3_6_6/LHeC_aa_zz_EFT_NP1/Events

Usage:
  python3 plot_aa_to_ZZ_aQGC_sigma_vs_fM2.py \
    --events-dir /home/hamzeh-khanpour/MG5_aMC_v3_6_6/LHeC_aa_zz_EFT_NP1/Events

Outputs:
  plots_aa_ZZ_aQGC_sigma_vs_fM2/aa_to_ZZ_aQGC_sigma_vs_abs_fM2.pdf
  plots_aa_ZZ_aQGC_sigma_vs_fM2/aa_to_ZZ_aQGC_sigma_vs_abs_fM2.png
  plots_aa_ZZ_aQGC_sigma_vs_fM2/aa_to_ZZ_aQGC_sigma_vs_signed_fM2.pdf
  plots_aa_ZZ_aQGC_sigma_vs_fM2/aa_to_ZZ_aQGC_sigma_vs_signed_fM2.png
  plots_aa_ZZ_aQGC_sigma_vs_fM2/aa_to_ZZ_aQGC_sigma_vs_fM2.csv

Important convention:
  The UFO parameter FM2 is in GeV^{-4}.
  Therefore: f_M2/Lambda^4 [TeV^{-4}] = FM2 [GeV^{-4}] * 1e12.

Physics note:
  The generated process is a a > z z NP=1, so these cross sections are the
  pure tree-level AQGC contribution. In this setup the rate should scale as
  sigma_AQGC = A * (f_M2/Lambda^4)^2.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List

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
DEFAULT_OUTDIR = Path("plots_aa_ZZ_aQGC_sigma_vs_fM2")
SAVE_DPI = 600


@dataclass
class RunPoint:
    run: str
    fm2_gev: float
    fm2_tev: float
    sigma_pb: float
    nevents: int
    banner: Path


def style_axis(ax) -> None:
    ax.grid(True, which="major", linestyle="--", alpha=0.55)
    ax.grid(True, which="minor", linestyle=":", alpha=0.25)
    ax.minorticks_on()
    ax.tick_params(axis="both", which="both", direction="in", top=True, right=True)


def parse_banner(banner: Path) -> RunPoint:
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
        nevents=nevents,
        banner=banner,
    )


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
        points.append(parse_banner(banners[0]))

    if not points:
        raise RuntimeError(f"No valid banner files found under: {events_dir}")

    return sorted(points, key=lambda p: p.fm2_tev)


def fit_quadratic_coefficient(points: List[RunPoint]) -> float:
    """Fit sigma = A * f^2, where f is in TeV^{-4}."""
    f = np.array([p.fm2_tev for p in points], dtype=float)
    sig = np.array([p.sigma_pb for p in points], dtype=float)
    x = f**2
    mask = x > 0.0
    return float(np.sum(x[mask] * sig[mask]) / np.sum(x[mask] ** 2))


def save_csv(points: List[RunPoint], outpath: Path, A_fit: float) -> None:
    with open(outpath, "w", encoding="utf-8") as f:
        f.write("run,FM2_GeV^-4,fM2_over_Lambda4_TeV^-4,sigma_AQGC_pb,nevents,sigma_fit_pb,ratio_to_fit\n")
        for p in points:
            sigma_fit = A_fit * p.fm2_tev**2
            ratio = p.sigma_pb / sigma_fit if sigma_fit > 0 else float("nan")
            f.write(
                f"{p.run},{p.fm2_gev:.12e},{p.fm2_tev:.8g},{p.sigma_pb:.12e},"
                f"{p.nevents},{sigma_fit:.12e},{ratio:.8f}\n"
            )
    print(f"Saved CSV: {outpath}")


def plot_abs(points: List[RunPoint], A_fit: float, outdir: Path) -> None:
    pos = [p for p in points if p.fm2_tev > 0]
    neg = [p for p in points if p.fm2_tev < 0]

    fig, ax = plt.subplots(figsize=(8.0, 7.0))
    plt.subplots_adjust(left=0.16, right=0.96, bottom=0.14, top=0.94)

    if pos:
        ax.plot(
            [abs(p.fm2_tev) for p in pos],
            [p.sigma_pb for p in pos],
            "o",
            markersize=8,
            label=r"$f_{M2}/\Lambda^4>0$",
        )
    if neg:
        ax.plot(
            [abs(p.fm2_tev) for p in neg],
            [p.sigma_pb for p in neg],
            "s",
            markersize=8,
            fillstyle="none",
            label=r"$f_{M2}/\Lambda^4<0$",
        )

    xfit = np.logspace(-1.2, 1.2, 400)
    yfit = A_fit * xfit**2
    ax.plot(
        xfit,
        yfit,
        "--",
        linewidth=2.6,
        label=rf"$\sigma={A_fit:.3e}\,|f_{{M2}}/\Lambda^4|^2$ pb",
    )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$|f_{M2}/\Lambda^4|$ [TeV$^{-4}$]")
    ax.set_ylabel(r"$\sigma_{\rm aQGC}$ [pb]")
    ax.legend(loc="upper left", frameon=False)
    style_axis(ax)

    for ext in ("pdf", "png"):
        out = outdir / f"aa_to_ZZ_aQGC_sigma_vs_abs_fM2.{ext}"
        fig.savefig(out, dpi=SAVE_DPI if ext == "png" else None)
        print(f"Saved plot: {out}")
    plt.close(fig)


def plot_signed(points: List[RunPoint], A_fit: float, outdir: Path) -> None:
    f = np.array([p.fm2_tev for p in points], dtype=float)
    sig = np.array([p.sigma_pb for p in points], dtype=float)

    fig, ax = plt.subplots(figsize=(8.0, 7.0))
    plt.subplots_adjust(left=0.16, right=0.96, bottom=0.14, top=0.94)

    ax.plot(f, sig, "o", markersize=8, label="")  # MadGraph samples

    # Avoid x=0 on a log-y plot because sigma_fit would be zero.
    xneg = np.linspace(min(f), -0.05, 300)
    xpos = np.linspace(0.05, max(f), 300)
    ax.plot(xneg, A_fit * xneg**2, "--", linewidth=2.4, label=r"")  # quadratic fit
    ax.plot(xpos, A_fit * xpos**2, "--", linewidth=2.4)

    ax.set_yscale("log")
    ax.set_xlabel(r"$f_{M2}/\Lambda^4$ [TeV$^{-4}$]")
    ax.set_ylabel(r"$\sigma_{\rm aQGC}$ [pb]")
    ax.legend(loc="upper center", frameon=False)
    style_axis(ax)

    for ext in ("pdf", "png"):
        out = outdir / f"aa_to_ZZ_aQGC_sigma_vs_signed_fM2.{ext}"
        fig.savefig(out, dpi=SAVE_DPI if ext == "png" else None)
        print(f"Saved plot: {out}")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot sigma(gamma gamma -> ZZ) AQGC vs fM2/Lambda^4.")
    parser.add_argument("--events-dir", type=Path, default=DEFAULT_EVENTS_DIR)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    points = discover_points(args.events_dir)
    A_fit = fit_quadratic_coefficient(points)

    print("\nRun mapping from banners")
    print("-" * 92)
    print(f"{'run':8s} {'FM2 [GeV^-4]':>18s} {'fM2/Lambda^4 [TeV^-4]':>28s} {'sigma [pb]':>16s} {'events':>10s}")
    for p in points:
        print(f"{p.run:8s} {p.fm2_gev:18.6e} {p.fm2_tev:28.6g} {p.sigma_pb:16.6e} {p.nevents:10d}")
    print("-" * 92)
    print(rf"Fitted coefficient: sigma_AQGC = {A_fit:.8e} * (fM2/Lambda^4 [TeV^-4])^2 pb")

    save_csv(points, args.outdir / "aa_to_ZZ_aQGC_sigma_vs_fM2.csv", A_fit)
    plot_abs(points, A_fit, args.outdir)
    plot_signed(points, A_fit, args.outdir)


if __name__ == "__main__":
    main()
