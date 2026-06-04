#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Plot the pure dimension-eight AQGC cross section for
gamma gamma -> gamma gamma as a function of f_T8/Lambda^4
using MadGraph run banners.

Default input directory:
  /home/hamzeh-khanpour/MG5_aMC_v3_6_6/aa_to_aa_FT8_FT9_NP1/Events

Usage:
  python3 plot_aa_to_aa_aQGC_sigma_vs_fT8.py \
    --events-dir /home/hamzeh-khanpour/MG5_aMC_v3_6_6/aa_to_aa_FT8_FT9_NP1/Events

Outputs:
  plots_aa_aa_aQGC_sigma_vs_fT8/aa_to_aa_aQGC_sigma_vs_abs_fT8.pdf
  plots_aa_aa_aQGC_sigma_vs_fT8/aa_to_aa_aQGC_sigma_vs_abs_fT8.png
  plots_aa_aa_aQGC_sigma_vs_fT8/aa_to_aa_aQGC_sigma_vs_signed_fT8.pdf
  plots_aa_aa_aQGC_sigma_vs_fT8/aa_to_aa_aQGC_sigma_vs_signed_fT8.png
  plots_aa_aa_aQGC_sigma_vs_fT8/aa_to_aa_aQGC_sigma_vs_fT8.csv

Important convention:
  The UFO parameter FT8 is in GeV^{-4}.
  Therefore:
      f_T8/Lambda^4 [TeV^{-4}] = FT8 [GeV^{-4}] * 1e12.

Physics note:
  The generated process is a a > a a NP=1, so these cross sections are the
  pure tree-level dimension-eight EFT contribution. In this setup the rate
  should scale as sigma_EFT = A * (f_T8/Lambda^4)^2.


python3 plot_aa_to_aa_aQGC_sigma_vs_fT8.py \
  --events-dir /home/hamzeh-khanpour/MG5_aMC_v3_6_6/aa_to_aa_FT8_FT9_NP1/Events


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
    hep.style.use("ATLAS")
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
DEFAULT_OUTDIR = Path("plots_aa_aa_aQGC_sigma_vs_fT8")
SAVE_DPI = 600


@dataclass
class RunPoint:
    run: str
    ft8_gev: float
    ft8_tev: float
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

    return sorted(points, key=lambda p: p.ft8_tev)


def fit_quadratic_coefficient(points: List[RunPoint]) -> float:
    """Fit sigma = A * f^2, where f is in TeV^{-4}."""
    f = np.array([p.ft8_tev for p in points], dtype=float)
    sig = np.array([p.sigma_pb for p in points], dtype=float)
    x = f**2
    mask = x > 0.0
    return float(np.sum(x[mask] * sig[mask]) / np.sum(x[mask] ** 2))


def save_csv(points: List[RunPoint], outpath: Path, A_fit: float) -> None:
    with open(outpath, "w", encoding="utf-8") as f:
        f.write(
            "run,FT8_GeV^-4,fT8_over_Lambda4_TeV^-4,"
            "sigma_EFT_pb,raw_events_at_1ab^-1,nevents,"
            "sigma_fit_pb,ratio_to_fit\n"
        )
        for p in points:
            sigma_fit = A_fit * p.ft8_tev**2
            ratio = p.sigma_pb / sigma_fit if sigma_fit > 0 else float("nan")
            raw_events = p.sigma_pb * 1.0e6
            f.write(
                f"{p.run},{p.ft8_gev:.12e},{p.ft8_tev:.8g},"
                f"{p.sigma_pb:.12e},{raw_events:.8e},{p.nevents},"
                f"{sigma_fit:.12e},{ratio:.8f}\n"
            )
    print(f"Saved CSV: {outpath}")


def plot_abs(points: List[RunPoint], A_fit: float, outdir: Path) -> None:
    pos = [p for p in points if p.ft8_tev > 0]
    neg = [p for p in points if p.ft8_tev < 0]

    fig, ax = plt.subplots(figsize=(8.0, 9.0))
    plt.subplots_adjust(left=0.15, right=0.95, bottom=0.12, top=0.95)

    if pos:
        ax.plot(
            [abs(p.ft8_tev) for p in pos],
            [p.sigma_pb for p in pos],
            "o",
            markersize=8,
            label=r"$f_{T8}/\Lambda^4>0$",
        )
    if neg:
        ax.plot(
            [abs(p.ft8_tev) for p in neg],
            [p.sigma_pb for p in neg],
            "s",
            markersize=8,
            fillstyle="none",
            label=r"$f_{T8}/\Lambda^4<0$",
        )

    x_min = min(abs(p.ft8_tev) for p in points if p.ft8_tev != 0) * 0.7
    x_max = max(abs(p.ft8_tev) for p in points if p.ft8_tev != 0) * 1.3
    xfit = np.logspace(np.log10(x_min), np.log10(x_max), 400)
    yfit = A_fit * xfit**2

    ax.plot(
        xfit,
        yfit,
        "--",
        linewidth=2.6,
        label=rf"$\sigma={A_fit:.3e}\,|f_{{T8}}/\Lambda^4|^2$ pb",
    )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$|f_{T8}/\Lambda^4|$ [TeV$^{-4}$]")
    ax.set_ylabel(r"$\sigma_{\rm EFT}$ [pb]")
    ax.legend(loc="upper left", frameon=False)
    style_axis(ax)

    ax.text(
        0.04,
        0.96,
        r"$ep\to e(\gamma\gamma\to\gamma\gamma)p^{(*)}$" "\n"
        r"pure tree-level dimension-eight EFT contribution",
        transform=ax.transAxes,
        fontsize=16,
        ha="left",
        va="top",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="none", alpha=0.78),
    )

    for ext in ("pdf", "png"):
        out = outdir / f"aa_to_aa_aQGC_sigma_vs_abs_fT8.{ext}"
        fig.savefig(out, dpi=SAVE_DPI if ext == "png" else None)
        print(f"Saved plot: {out}")
    plt.close(fig)


def plot_signed(points: List[RunPoint], A_fit: float, outdir: Path) -> None:
    f = np.array([p.ft8_tev for p in points], dtype=float)
    sig = np.array([p.sigma_pb for p in points], dtype=float)

    fig, ax = plt.subplots(figsize=(9.0, 9.0))
    plt.subplots_adjust(left=0.15, right=0.95, bottom=0.12, top=0.95)

    ax.plot(f, sig, "o", markersize=8, label="") #MadGraph samples

    x_min = min(f)
    x_max = max(f)
    xneg = np.linspace(x_min, -0.01, 300)
    xpos = np.linspace(0.01, x_max, 300)
    ax.plot(xneg, A_fit * xneg**2, "--", linewidth=2.4, label="") #quadratic fit
    ax.plot(xpos, A_fit * xpos**2, "--", linewidth=2.4)

    ax.set_yscale("log")
    ax.set_ylim(1.0e-6, 1.0e-1)

    ax.set_xlabel(r"$f_{T8}/\Lambda^4$ [TeV$^{-4}$]")
    ax.set_ylabel(r"$\sigma_{\rm EFT}$ [pb]")
    ax.legend(loc="upper center", frameon=False)
    style_axis(ax)

    ax.text(
        0.04,
        0.96,
        r"$ep\to e(\gamma\gamma\to\gamma\gamma)p^{(*)}$" "\n"
        r"pure tree-level dimension-eight EFT contribution",
        transform=ax.transAxes,
        fontsize=16,
        ha="left",
        va="top",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="none", alpha=0.78),
    )

    for ext in ("pdf", "png"):
        out = outdir / f"aa_to_aa_aQGC_sigma_vs_signed_fT8.{ext}"
        fig.savefig(out, dpi=SAVE_DPI if ext == "png" else None)
        print(f"Saved plot: {out}")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot sigma(gamma gamma -> gamma gamma) EFT vs fT8/Lambda^4."
    )
    parser.add_argument("--events-dir", type=Path, default=DEFAULT_EVENTS_DIR)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    points = discover_points(args.events_dir)
    A_fit = fit_quadratic_coefficient(points)

    print("\nRun mapping from banners")
    print("-" * 96)
    print(
        f"{'run':8s} {'FT8 [GeV^-4]':>18s} "
        f"{'fT8/Lambda^4 [TeV^-4]':>28s} {'sigma [pb]':>16s} {'events@1ab^-1':>16s}"
    )
    for p in points:
        print(
            f"{p.run:8s} {p.ft8_gev:18.6e} {p.ft8_tev:28.6g} "
            f"{p.sigma_pb:16.6e} {p.sigma_pb * 1.0e6:16.6e}"
        )
    print("-" * 96)
    print(
        rf"Fitted coefficient: sigma_EFT = {A_fit:.8e} "
        r"* (fT8/Lambda^4 [TeV^-4])^2 pb"
    )

    save_csv(points, args.outdir / "aa_to_aa_aQGC_sigma_vs_fT8.csv", A_fit)
    plot_abs(points, A_fit, args.outdir)
    plot_signed(points, A_fit, args.outdir)


if __name__ == "__main__":
    main()
