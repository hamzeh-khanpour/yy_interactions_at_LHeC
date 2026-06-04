#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SM light-by-light eta and Y distributions for:
    ep -> e (gamma gamma -> gamma gamma) p^(*) at the LHeC

This script uses the loop-induced Standard Model gamma gamma -> gamma gamma
matrix element from the local ggMatrixElements Python module.

It produces:
  plots_SM_LbL_eta_Y/SM_LbL_dsigma_deta_gamma_W10.pdf
  plots_SM_LbL_eta_Y/SM_LbL_dsigma_deta_gamma_W100.pdf
  plots_SM_LbL_eta_Y/SM_LbL_dsigma_dY_gammagamma_W10_W100.pdf
  plots_SM_LbL_eta_Y/SM_LbL_eta_Y_summary.csv
  plus PNG copies and histogram CSV files.

Important physics note
----------------------
The existing yy_to_yy package contains a one-dimensional photon-photon
luminosity grid dL/dW. That is enough for total cross sections vs W0, but not
for a lab-frame eta distribution, because eta_lab also needs the boost
Y_{gamma gamma}, hence the two photon momentum fractions y_e and y_p.

Therefore this script uses an explicit Monte Carlo integration over:
    W_{gamma gamma}, Y_{gamma gamma}, cos(theta*)
with analytic EPA photon fluxes for the electron and elastic proton emission.
The histogram shapes are obtained from this 2D EPA sampling, and by default the
overall normalization is rescaled to the existing integrated elastic cross
section in output_values_ep_epgg.txt. This keeps the normalization consistent
with the already-used yy_to_yy package.

Default assumptions:
  E_e = 50 GeV
  E_p = 7000 GeV
  Q_e^2 max = 1e5 GeV^2
  tagged elastic proton photon flux: Drees-Zeppenfeld elastic EPA approximation

Usage:
  cd /home/hamzeh-khanpour/Documents/GitHub/yy_interactions_at_LHeC/JHEP/yy_to_yy

  python3 plot_SM_LbL_eta_Y_LHeC.py

For better statistics:
  python3 plot_SM_LbL_eta_Y_LHeC.py --n-samples 1000000

If output_values_ep_epgg.txt is not available:
  python3 plot_SM_LbL_eta_Y_LHeC.py --no-normalize-to-existing-table

Author:
  prepared for the gamma gamma at LHeC photon-pair eta-distribution study.
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt

try:
    import mplhep as hep
    hep.style.use("CMS")
except Exception:
    print("WARNING: mplhep is not available. Using default Matplotlib style.")

try:
    import ggMatrixElements
except Exception as exc:
    print("\nERROR: Could not import ggMatrixElements.")
    print("Make sure ggMatrixElements.so is in the current directory or PYTHONPATH.")
    print("For example:")
    print("  export PYTHONPATH=$PWD:$PYTHONPATH")
    print(f"\nOriginal import error:\n  {exc}")
    raise


# ---------------------------------------------------------------------------
# Plot style: same numerical settings as the previous tau/WW/ZZ scripts.
# ---------------------------------------------------------------------------
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

SAVE_DPI = 600

# Constants
ALPHA_EM = 1.0 / 137.035999084
ME_GEV = 0.000510998950
MP_GEV = 0.93827208816
HBARC2_PB = 0.389379338e9  # 1 GeV^{-2} in pb


@dataclass
class HistResult:
    w0: float
    sigma_mc_pb: float
    sigma_norm_pb: float
    scale_factor: float
    eta_edges: np.ndarray
    eta_dsigma: np.ndarray
    y_edges: np.ndarray
    y_dsigma: np.ndarray
    n_samples: int
    n_finite: int


def style_axis(ax) -> None:
    ax.grid(True, which="major", linestyle="--", alpha=0.55)
    ax.grid(True, which="minor", linestyle=":", alpha=0.25)
    ax.minorticks_on()
    ax.tick_params(axis="both", which="both", direction="in", top=True, right=True)


def sqme_sm_cached(s: float, t: float) -> float:
    """Loop-induced SM gamma gamma -> gamma gamma squared matrix element."""
    return float(ggMatrixElements.sqme_sm(float(s), float(t), False))


def dsigma_dcostheta_sm_pb(W: float, costheta: float) -> float:
    """
    d sigma(gamma gamma -> gamma gamma) / d cos(theta*) in pb.

    Existing yy_to_yy code uses:
        d sigma / dt = |M|^2 / (16 pi s^2)
        t = -s/2 * (1 - cos(theta*))
    Therefore:
        d sigma / d cos(theta*) = |M|^2 / (32 pi s)
    """
    s = W * W
    t = -0.5 * s * (1.0 - costheta)
    sqme = sqme_sm_cached(s, t)
    return sqme / (32.0 * math.pi * s) * HBARC2_PB


def photon_flux_electron_iww(y: float, q2max: float = 1.0e5) -> float:
    """
    Improved Weizsaecker-Williams photon flux from the electron.

    f_{gamma/e}(y) =
      alpha/(2 pi) [
        (1 + (1-y)^2)/y * log(Qmax^2/Qmin^2)
        - 2 m_e^2 y * (1/Qmin^2 - 1/Qmax^2)
      ]

    with Qmin^2 = m_e^2 y^2/(1-y).
    """
    if y <= 0.0 or y >= 1.0:
        return 0.0

    q2min = ME_GEV * ME_GEV * y * y / (1.0 - y)
    if q2min <= 0.0 or q2min >= q2max:
        return 0.0

    term1 = (1.0 + (1.0 - y) ** 2) / y * math.log(q2max / q2min)
    term2 = -2.0 * ME_GEV * ME_GEV * y * (1.0 / q2min - 1.0 / q2max)
    flux = ALPHA_EM / (2.0 * math.pi) * (term1 + term2)
    return max(flux, 0.0)


def photon_flux_proton_elastic_dz(y: float) -> float:
    """
    Drees-Zeppenfeld elastic photon flux from the proton.

    This is a compact analytic approximation for the elastic EPA proton flux.
    It is used here to obtain the lab-frame rapidity shape. The final
    normalization is rescaled to the existing yy_to_yy elastic cross section
    unless --no-normalize-to-existing-table is used.
    """
    if y <= 0.0 or y >= 1.0:
        return 0.0

    q2min = MP_GEV * MP_GEV * y * y / (1.0 - y)
    if q2min <= 0.0:
        return 0.0

    A = 1.0 + 0.71 / q2min
    if A <= 1.0:
        return 0.0

    phi = math.log(A) - 11.0 / 6.0 + 3.0 / A - 3.0 / (2.0 * A * A) + 1.0 / (3.0 * A**3)
    flux = ALPHA_EM / (2.0 * math.pi) * (1.0 + (1.0 - y) ** 2) / y * phi
    return max(flux, 0.0)


def eta_star_from_costheta(costheta: float) -> float:
    c = max(min(costheta, 1.0 - 1.0e-15), -1.0 + 1.0e-15)
    return 0.5 * math.log((1.0 + c) / (1.0 - c))


def read_existing_elastic_cross_section(table_path: Path, w0: float) -> float | None:
    """
    Read the existing output_values_ep_epgg.txt table and interpolate the
    elastic integrated cross section at the requested W0.

    The table columns are:
      W_Value  Elastic  Inelastic
    """
    if not table_path.exists():
        return None

    data = np.loadtxt(table_path, comments="#")
    if data.ndim != 2 or data.shape[1] < 2:
        return None

    W = data[:, 0]
    sigma_el = data[:, 1]

    positive = (W > 0.0) & (sigma_el > 0.0)
    W = W[positive]
    sigma_el = sigma_el[positive]

    if len(W) < 2:
        return None

    # If w0 is exactly present, use it directly.
    exact = np.where(np.isclose(W, w0, rtol=1.0e-10, atol=1.0e-12))[0]
    if len(exact):
        return float(sigma_el[exact[0]])

    # Log-log interpolation is appropriate for the smooth falling curve.
    if w0 < W.min() or w0 > W.max():
        return None
    return float(np.exp(np.interp(np.log(w0), np.log(W), np.log(sigma_el))))


def monte_carlo_histograms(
    *,
    w0: float,
    wmax: float,
    ee: float,
    ep: float,
    qe2max: float,
    n_samples: int,
    seed: int,
    eta_edges: np.ndarray,
    y_edges: np.ndarray,
) -> Tuple[float, np.ndarray, np.ndarray, int]:
    """
    Compute eta_gamma and Y_gammagamma histograms for tagged-elastic SM LbL.

    Integration variables:
      u = log(W)
      Y = Y_{gamma gamma}
      c = cos(theta*)

    y_p = W exp(Y)/(2 E_p)
    y_e = W exp(-Y)/(2 E_e)

    dy_p dy_e = W/(2 E_p E_e) dW dY
              = W^2/(2 E_p E_e) dlogW dY
    """
    rng = np.random.default_rng(seed)

    logw_min = math.log(w0)
    logw_max = math.log(wmax)
    logw_range = logw_max - logw_min

    eta_hist = np.zeros(len(eta_edges) - 1, dtype=float)
    y_hist = np.zeros(len(y_edges) - 1, dtype=float)

    sigma_mc = 0.0
    n_finite = 0

    # Generate in chunks to avoid too much memory use.
    chunk_size = min(50000, n_samples)
    processed = 0

    while processed < n_samples:
        n = min(chunk_size, n_samples - processed)
        processed += n

        u_rand = rng.random(n)
        c_rand = rng.random(n)
        y_rand = rng.random(n)

        W_values = np.exp(logw_min + logw_range * u_rand)

        for W, rc, ry in zip(W_values, c_rand, y_rand):
            # Kinematic rapidity range from 0 < y_e,y_p < 1:
            # y_p = W e^Y/(2Ep) < 1  => Y < ln(2Ep/W)
            # y_e = W e^-Y/(2Ee) < 1 => Y > ln(W/(2Ee))
            ymin = math.log(W / (2.0 * ee))
            ymax = math.log(2.0 * ep / W)
            ywidth = ymax - ymin
            if ywidth <= 0.0:
                continue

            Ygg = ymin + ywidth * ry
            yp = W * math.exp(Ygg) / (2.0 * ep)
            ye = W * math.exp(-Ygg) / (2.0 * ee)

            fe = photon_flux_electron_iww(ye, qe2max)
            fp = photon_flux_proton_elastic_dz(yp)
            if fe <= 0.0 or fp <= 0.0:
                continue

            costheta = -1.0 + 2.0 * rc
            dsdc = dsigma_dcostheta_sm_pb(W, costheta)
            if dsdc <= 0.0 or not math.isfinite(dsdc):
                continue

            # MC integration weight.
            # Integrand in (logW, Y, cos(theta)):
            # W^2/(2EeEp) * fe * fp * dsigma/dcos(theta)
            jac = W * W / (2.0 * ee * ep)
            weight = logw_range * ywidth * 2.0 / float(n_samples) * jac * fe * fp * dsdc

            if not math.isfinite(weight) or weight <= 0.0:
                continue

            sigma_mc += weight
            n_finite += 1

            eta_star = eta_star_from_costheta(costheta)
            eta1 = Ygg + eta_star
            eta2 = Ygg - eta_star

            # Histogram Y once per event.
            y_bin = np.searchsorted(y_edges, Ygg, side="right") - 1
            if 0 <= y_bin < len(y_hist):
                y_hist[y_bin] += weight

            # Inclusive photon eta: fill both photons.
            eta_bin_1 = np.searchsorted(eta_edges, eta1, side="right") - 1
            if 0 <= eta_bin_1 < len(eta_hist):
                eta_hist[eta_bin_1] += weight

            eta_bin_2 = np.searchsorted(eta_edges, eta2, side="right") - 1
            if 0 <= eta_bin_2 < len(eta_hist):
                eta_hist[eta_bin_2] += weight

        print(f"  processed {processed:,}/{n_samples:,} samples for W0={w0:g} GeV", flush=True)

    eta_widths = np.diff(eta_edges)
    y_widths = np.diff(y_edges)

    eta_dsigma = eta_hist / eta_widths
    y_dsigma = y_hist / y_widths

    return sigma_mc, eta_dsigma, y_dsigma, n_finite


def save_hist_csv(outpath: Path, edges: np.ndarray, values: np.ndarray, column_name: str) -> None:
    centers = 0.5 * (edges[:-1] + edges[1:])
    with open(outpath, "w", encoding="utf-8") as f:
        f.write(f"bin_low,bin_high,bin_center,{column_name}\n")
        for low, high, center, val in zip(edges[:-1], edges[1:], centers, values):
            f.write(f"{low:.8e},{high:.8e},{center:.8e},{val:.12e}\n")
    print(f"Saved CSV: {outpath}")


def plot_eta(result: HistResult, outdir: Path) -> None:

    fig, ax = plt.subplots(figsize=(8.0, 9.0))
    plt.subplots_adjust(left=0.15, right=0.95, bottom=0.12, top=0.95)

    ax.stairs(result.eta_dsigma, result.eta_edges, linewidth=2.8, label="tagged elastic SM LbL")

    ax.set_yscale("log")
    positive = result.eta_dsigma[result.eta_dsigma > 0.0]
    if len(positive):
        ax.set_ylim(1.0e-8, 1.0e0)   #max(positive.min() * 0.3, 1.0e-12), positive.max() * 5.0

    ax.set_xlim(float(result.eta_edges[0]), float(result.eta_edges[-1]))
    ax.set_xlabel(r"$\eta_{\gamma}$")
    ax.set_ylabel(r"$d\sigma_{\rm SM}/d\eta_{\gamma}$ [pb]")
    ax.legend(loc="upper right", frameon=False)
    style_axis(ax)

    ax.text(
        0.04,
        0.96,
        r"$ep\to e(\gamma\gamma\to\gamma\gamma)p$" "\n"
        r"SM loop-induced LbL, tagged elastic" "\n"
        rf"$W_{{\gamma\gamma}}>{result.w0:g}$ GeV",
        transform=ax.transAxes,
        fontsize=18,
        ha="left",
        va="top",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="none", alpha=0.78),
    )

    base = outdir / f"SM_LbL_dsigma_deta_gamma_W{int(result.w0)}"
    for ext in ("pdf", "png"):
        out = base.with_suffix(f".{ext}")
        fig.savefig(out, dpi=SAVE_DPI if ext == "png" else None)
        print(f"Saved plot: {out}")
    plt.close(fig)

    save_hist_csv(base.with_suffix(".csv"), result.eta_edges, result.eta_dsigma, "dsigma_deta_gamma_pb")


def plot_y(results: List[HistResult], outdir: Path) -> None:

    fig, ax = plt.subplots(figsize=(8.0, 9.0))
    plt.subplots_adjust(left=0.15, right=0.95, bottom=0.12, top=0.95)

    linestyles = {10.0: "-", 100.0: "--"}

    for result in results:
        ax.stairs(
            result.y_dsigma,
            result.y_edges,
            linewidth=2.8,
            linestyle=linestyles.get(float(result.w0), "-"),
            label=rf"$W_{{\gamma\gamma}}>{result.w0:g}$ GeV",
        )

    ax.set_yscale("log")
    positive_values = np.concatenate([r.y_dsigma[r.y_dsigma > 0.0] for r in results if np.any(r.y_dsigma > 0.0)])
    if len(positive_values):
        ax.set_ylim(1.0e-8, 1.0e0)    #max(positive_values.min() * 0.3, 1.0e-12), positive_values.max() * 5.0

    ax.set_xlim(float(results[0].y_edges[0]), float(results[0].y_edges[-1]))
    ax.set_xlabel(r"$Y_{\gamma\gamma}$")
    ax.set_ylabel(r"$d\sigma_{\rm SM}/dY_{\gamma\gamma}$ [pb]")
    ax.legend(loc="upper right", frameon=False)
    style_axis(ax)

    ax.text(
        0.04,
        0.96,
        r"$ep\to e(\gamma\gamma\to\gamma\gamma)p$" "\n"
        r"SM loop-induced LbL, tagged elastic",
        transform=ax.transAxes,
        fontsize=18,
        ha="left",
        va="top",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="none", alpha=0.78),
    )

    base = outdir / "SM_LbL_dsigma_dY_gammagamma_W10_W100"
    for ext in ("pdf", "png"):
        out = base.with_suffix(f".{ext}")
        fig.savefig(out, dpi=SAVE_DPI if ext == "png" else None)
        print(f"Saved plot: {out}")
    plt.close(fig)

    # Save both Y histograms to one CSV.
    centers = 0.5 * (results[0].y_edges[:-1] + results[0].y_edges[1:])
    with open(base.with_suffix(".csv"), "w", encoding="utf-8") as f:
        f.write("bin_low,bin_high,bin_center")
        for r in results:
            f.write(f",dsigma_dY_W{int(r.w0)}_pb")
        f.write("\n")

        for i, center in enumerate(centers):
            f.write(f"{results[0].y_edges[i]:.8e},{results[0].y_edges[i+1]:.8e},{center:.8e}")
            for r in results:
                f.write(f",{r.y_dsigma[i]:.12e}")
            f.write("\n")
    print(f"Saved CSV: {base.with_suffix('.csv')}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot SM LbL eta and Y distributions at the LHeC.")
    parser.add_argument("--outdir", type=Path, default=Path("plots_SM_LbL_eta_Y"))
    parser.add_argument("--normalization-table", type=Path, default=Path("output_values_ep_epgg.txt"))
    parser.add_argument("--no-normalize-to-existing-table", action="store_true")
    parser.add_argument("--n-samples", type=int, default=300000)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--ee", type=float, default=50.0, help="Electron beam energy in GeV.")
    parser.add_argument("--ep", type=float, default=7000.0, help="Proton beam energy in GeV.")
    parser.add_argument("--wmax", type=float, default=1000.0, help="Maximum W_gammagamma in GeV.")
    parser.add_argument("--qe2max", type=float, default=1.0e5, help="Maximum electron photon virtuality in GeV^2.")
    parser.add_argument("--eta-min", type=float, default=-10.0)
    parser.add_argument("--eta-max", type=float, default=10.0)
    parser.add_argument("--eta-bins", type=int, default=80)
    parser.add_argument("--y-min", type=float, default=-6.0)
    parser.add_argument("--y-max", type=float, default=8.0)
    parser.add_argument("--y-bins", type=int, default=80)
    parser.add_argument(
        "--w0-values",
        type=str,
        default="10,100",
        help="Comma-separated W0 values in GeV. The standard choice is '10,100'.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    w0_values = [float(x.strip()) for x in args.w0_values.split(",") if x.strip()]
    if w0_values != [10.0, 100.0]:
        print(f"WARNING: standard filenames are intended for W0=10 and 100 GeV. Got: {w0_values}")

    eta_edges = np.linspace(args.eta_min, args.eta_max, args.eta_bins + 1)
    y_edges = np.linspace(args.y_min, args.y_max, args.y_bins + 1)

    sqrt_s_ep = math.sqrt(4.0 * args.ee * args.ep)
    if args.wmax > sqrt_s_ep:
        print(
            f"WARNING: requested wmax={args.wmax:g} GeV is above sqrt(s_ep)={sqrt_s_ep:g} GeV. "
            f"Using sqrt(s_ep)."
        )
        args.wmax = sqrt_s_ep

    print("\nSM light-by-light eta/Y distribution setup")
    print("-" * 78)
    print(f"Ee = {args.ee:g} GeV, Ep = {args.ep:g} GeV, sqrt(s_ep) = {sqrt_s_ep:.3f} GeV")
    print(f"Wmax = {args.wmax:g} GeV")
    print(f"n_samples per W0 = {args.n_samples:,}")
    print(f"normalization table = {args.normalization_table}")
    print("-" * 78)

    results: List[HistResult] = []

    for index, w0 in enumerate(w0_values):
        if w0 >= args.wmax:
            raise RuntimeError(f"W0={w0:g} GeV must be smaller than Wmax={args.wmax:g} GeV.")

        print(f"\nRunning Monte Carlo for W0 = {w0:g} GeV")
        sigma_mc, eta_dsigma, y_dsigma, n_finite = monte_carlo_histograms(
            w0=w0,
            wmax=args.wmax,
            ee=args.ee,
            ep=args.ep,
            qe2max=args.qe2max,
            n_samples=args.n_samples,
            seed=args.seed + index,
            eta_edges=eta_edges,
            y_edges=y_edges,
        )

        sigma_norm = sigma_mc
        if not args.no_normalize_to_existing_table:
            existing = read_existing_elastic_cross_section(args.normalization_table, w0)
            if existing is not None and existing > 0.0:
                sigma_norm = existing
                print(
                    f"  Using existing elastic normalization from {args.normalization_table}: "
                    f"sigma(W>{w0:g}) = {sigma_norm:.8e} pb"
                )
            else:
                print("  WARNING: could not read existing normalization; using MC normalization.")

        scale = sigma_norm / sigma_mc if sigma_mc > 0.0 else 1.0
        eta_dsigma_scaled = eta_dsigma * scale
        y_dsigma_scaled = y_dsigma * scale

        print(f"  raw MC sigma      = {sigma_mc:.8e} pb")
        print(f"  normalized sigma  = {sigma_norm:.8e} pb")
        print(f"  scale factor      = {scale:.6e}")
        print(f"  finite samples    = {n_finite:,}/{args.n_samples:,}")

        result = HistResult(
            w0=w0,
            sigma_mc_pb=sigma_mc,
            sigma_norm_pb=sigma_norm,
            scale_factor=scale,
            eta_edges=eta_edges,
            eta_dsigma=eta_dsigma_scaled,
            y_edges=y_edges,
            y_dsigma=y_dsigma_scaled,
            n_samples=args.n_samples,
            n_finite=n_finite,
        )
        results.append(result)
        plot_eta(result, args.outdir)

    if len(results) >= 2:
        plot_y(results, args.outdir)

    summary = args.outdir / "SM_LbL_eta_Y_summary.csv"
    with open(summary, "w", encoding="utf-8") as f:
        f.write("W0_GeV,sigma_mc_pb,sigma_used_for_normalization_pb,scale_factor,n_samples,n_finite,raw_events_at_1ab^-1\n")
        for r in results:
            f.write(
                f"{r.w0:.8g},{r.sigma_mc_pb:.12e},{r.sigma_norm_pb:.12e},"
                f"{r.scale_factor:.12e},{r.n_samples},{r.n_finite},{r.sigma_norm_pb * 1.0e6:.8e}\n"
            )
    print(f"\nSaved summary: {summary}")

    print("\nDone. Main outputs:")
    print(f"  {args.outdir / 'SM_LbL_dsigma_deta_gamma_W10.pdf'}")
    print(f"  {args.outdir / 'SM_LbL_dsigma_deta_gamma_W100.pdf'}")
    print(f"  {args.outdir / 'SM_LbL_dsigma_dY_gammagamma_W10_W100.pdf'}")


if __name__ == "__main__":
    main()
