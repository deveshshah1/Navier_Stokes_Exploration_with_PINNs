import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from scipy.interpolate import griddata


FIELDS = [
    ("Ux", "Ux_pred"),
    ("Uy", "Uy_pred"),
    ("p",  "p_pred"),
]

FIELD_COLORS = {"Ux": "#1f77b4", "Uy": "#ff7f0e", "p": "#2ca02c"}


def l2_relative_error(true: np.ndarray, pred: np.ndarray) -> float:
    return float(np.linalg.norm(pred - true) / np.linalg.norm(true))


def compute_l2_error_table(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
        summary_df  — one row per field with overall L2 relative error
        per_time_df — one row per (t, field) with per-snapshot L2 relative error
    """
    summary_rows = []
    for true_col, pred_col in FIELDS:
        err = l2_relative_error(df[true_col].values, df[pred_col].values)
        summary_rows.append({"field": true_col, "l2_relative_error": err})
    summary_df = pd.DataFrame(summary_rows).set_index("field")

    per_time_rows = []
    for t, group in df.groupby("t", sort=True):
        for true_col, pred_col in FIELDS:
            err = l2_relative_error(group[true_col].values, group[pred_col].values)
            per_time_rows.append({"t": t, "field": true_col, "l2_relative_error": err})
    per_time_df = pd.DataFrame(per_time_rows)

    return summary_df, per_time_df


def save_l2_error_results(
    summary_df: pd.DataFrame,
    per_time_df: pd.DataFrame,
    benchmarks_dir: str,
) -> None:
    # CSVs
    summary_df.to_csv(os.path.join(benchmarks_dir, "l2_error_summary.csv"))
    per_time_df.to_csv(os.path.join(benchmarks_dir, "l2_error_per_timestep.csv"), index=False)

    # Plot: top = overall bar chart, bottom = per-timestep line chart
    fig, (ax_bar, ax_line) = plt.subplots(2, 1, figsize=(9, 8))
    fig.suptitle("L2 Relative Error", fontsize=13, fontweight="bold")

    # --- Panel 1: horizontal bar chart (overall) ---
    fields = summary_df.index.tolist()
    values = summary_df["l2_relative_error"].values
    colors = [FIELD_COLORS[f] for f in fields]
    bars = ax_bar.barh(fields, values, color=colors, height=0.5)
    ax_bar.bar_label(bars, fmt="{:.4f}", padding=4, fontsize=9)
    ax_bar.set_xlabel("L2 relative error")
    ax_bar.set_title("Overall")
    ax_bar.set_xlim(0, values.max() * 1.2)
    ax_bar.invert_yaxis()
    ax_bar.grid(axis="x", alpha=0.3)

    # --- Panel 2: line chart (per timestep) ---
    for field, color in FIELD_COLORS.items():
        subset = per_time_df[per_time_df["field"] == field]
        ax_line.plot(subset["t"], subset["l2_relative_error"], label=field, color=color, linewidth=1.5)
    ax_line.set_xlabel("t")
    ax_line.set_ylabel("L2 relative error")
    ax_line.set_title("Per Timestep")
    ax_line.legend()
    ax_line.grid(alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(benchmarks_dir, "l2_error.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()

    print("\nL2 Relative Error (overall)")
    print(summary_df.to_string(float_format="{:.6f}".format))
    print(f"\nSaved to {benchmarks_dir}")


def _draw_cylinder(ax, cx: float, cy: float, r: float):
    theta = np.linspace(0, 2 * np.pi, 100)
    ax.fill(cx + r * np.cos(theta), cy + r * np.sin(theta), color="gray", zorder=5)


def plot_field_comparison(
    df: pd.DataFrame,
    benchmarks_dir: str,
    cylinder_geometry: dict,
    t: float | None = None,
) -> None:
    """
    3-row × 3-col figure: one row per field (Ux, Uy, p),
    columns = True | Predicted | Absolute Error.
    Defaults to the last (largest) time snapshot.
    """
    times = np.sort(df["t"].unique())
    t = times[-1] if t is None else times[np.argmin(np.abs(times - t))]
    snap = df[df["t"] == t]

    x, y = snap["x"].values, snap["y"].values
    triang = tri.Triangulation(x, y)

    field_labels = ["Ux", "Uy", "p"]
    pred_cols    = ["Ux_pred", "Uy_pred", "p_pred"]

    fig, axes = plt.subplots(3, 3, figsize=(16, 7))
    fig.suptitle(f"Field Comparison  (t = {t:.2f})", fontsize=13, fontweight="bold")

    col_titles = ["True", "Predicted", "Absolute Error"]
    for ax, title in zip(axes[0], col_titles):
        ax.set_title(title, fontsize=11)

    for row, (field, pred_col) in enumerate(zip(field_labels, pred_cols)):
        true_vals  = snap[field].values
        pred_vals  = snap[pred_col].values
        error_vals = np.abs(pred_vals - true_vals)

        # Shared colorscale for True/Predicted based on the true field
        abs_max = np.abs(true_vals).max()
        vmin, vmax = -abs_max, abs_max

        for col, (vals, cmap, v0, v1) in enumerate([
            (true_vals,  "RdBu_r", vmin, vmax),
            (pred_vals,  "RdBu_r", vmin, vmax),
            (error_vals, "viridis", 0,   error_vals.max()),
        ]):
            ax = axes[row, col]
            tcf = ax.tricontourf(triang, vals, levels=50, cmap=cmap, vmin=v0, vmax=v1)
            plt.colorbar(tcf, ax=ax, fraction=0.046, pad=0.04)
            _draw_cylinder(ax, **cylinder_geometry)
            ax.set_xlim(0, 2.2)
            ax.set_ylim(0, 0.41)
            ax.set_aspect("equal")
            # Show tick labels only on left column (y) and bottom row (x)
            ax.tick_params(labelleft=(col == 0), labelbottom=(row == 2), labelsize=7)
            if col == 0:
                ax.set_ylabel(f"{field}\ny (m)", fontsize=8)
                ax.set_yticks([0.0, 0.1, 0.2, 0.3, 0.4])
            if row == 2:
                ax.set_xlabel("x (m)", fontsize=8)
                ax.set_xticks([0.0, 0.5, 1.0, 1.5, 2.0])

    plt.tight_layout()
    save_path = os.path.join(benchmarks_dir, f"field_comparison_t{t:.2f}.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved field comparison → {save_path}")


def plot_wake_profiles(
    df: pd.DataFrame,
    benchmarks_dir: str,
    x_locations: list[float] = [0.5, 1.0, 1.5],
    band: float = 0.025,
    n_bins: int = 50,
    t: float | None = None,
) -> None:
    """
    Ux(y) velocity profiles at cross-sections downstream of the cylinder.
    Points within ±band of each x location are y-binned and averaged to
    produce clean profiles on the unstructured mesh.
    """
    times = np.sort(df["t"].unique())
    t = times[-1] if t is None else times[np.argmin(np.abs(times - t))]
    snap = df[df["t"] == t]

    y_min, y_max = snap["y"].min(), snap["y"].max()
    bin_edges = np.linspace(y_min, y_max, n_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    fig, axes = plt.subplots(1, len(x_locations), figsize=(5 * len(x_locations), 5), sharey=True)
    fig.suptitle(f"Wake Velocity Profiles — Ux(y)  (t = {t:.2f})", fontsize=13, fontweight="bold")

    for ax, x_target in zip(axes, x_locations):
        slice_df = snap[np.abs(snap["x"] - x_target) < band]

        labels = slice_df["y"].values
        bin_idx = np.digitize(labels, bin_edges) - 1
        bin_idx = np.clip(bin_idx, 0, n_bins - 1)

        true_means = np.full(n_bins, np.nan)
        pred_means = np.full(n_bins, np.nan)
        for b in range(n_bins):
            mask = bin_idx == b
            if mask.sum() > 0:
                true_means[b] = slice_df["Ux"].values[mask].mean()
                pred_means[b] = slice_df["Ux_pred"].values[mask].mean()

        valid = ~(np.isnan(true_means) | np.isnan(pred_means))
        ax.plot(true_means[valid], bin_centers[valid], color="#1f77b4", linewidth=1.8, label="CFD (true)")
        ax.plot(pred_means[valid], bin_centers[valid], color="#d62728", linewidth=1.8, linestyle="--", label="PINN")

        ax.set_title(f"x = {x_target} m")
        ax.set_xlabel("Ux (m/s)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

    axes[0].set_ylabel("y (m)")
    plt.tight_layout()
    save_path = os.path.join(benchmarks_dir, f"wake_profiles_t{t:.2f}.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved wake profiles → {save_path}")


# Schäfer-Turek (1996) reference probe locations
_PROBE_UPSTREAM   = (0.15, 0.2)
_PROBE_DOWNSTREAM = (0.25, 0.2)


def _nearest_point_pressure(snap: pd.DataFrame, x_target: float, y_target: float) -> tuple[float, float]:
    """Return (p_true, p_pred) at the mesh point closest to (x_target, y_target)."""
    dist = (snap["x"] - x_target) ** 2 + (snap["y"] - y_target) ** 2
    row = snap.loc[dist.idxmin()]
    return float(row["p"]), float(row["p_pred"])


def compute_delta_p(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Δp = p_upstream - p_downstream at the Schäfer-Turek probe points
    for every time snapshot. Returns a DataFrame with columns:
        t, delta_p_true, delta_p_pred, abs_error
    """
    rows = []
    for t, snap in df.groupby("t", sort=True):
        p_up_true,  p_up_pred  = _nearest_point_pressure(snap, *_PROBE_UPSTREAM)
        p_dn_true,  p_dn_pred  = _nearest_point_pressure(snap, *_PROBE_DOWNSTREAM)
        dp_true = p_up_true - p_dn_true
        dp_pred = p_up_pred - p_dn_pred
        rows.append({
            "t": t,
            "delta_p_true": dp_true,
            "delta_p_pred": dp_pred,
            "abs_error": abs(dp_pred - dp_true),
        })
    return pd.DataFrame(rows)


def save_delta_p_results(
    delta_p_df: pd.DataFrame,
    benchmarks_dir: str,
    ref_lo: float | None = None,
    ref_hi: float | None = None,
) -> None:
    delta_p_df.to_csv(os.path.join(benchmarks_dir, "delta_p.csv"), index=False)

    last = delta_p_df.iloc[-1]
    print("\nΔp probe comparison (Schäfer-Turek)")
    print(f"  CFD true  : {last['delta_p_true']:.6f}")
    print(f"  PINN pred : {last['delta_p_pred']:.6f}")
    print(f"  Abs error : {last['abs_error']:.6f}")
    if ref_lo is not None:
        print(f"  Reference : {ref_lo} – {ref_hi}")

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(delta_p_df["t"], delta_p_df["delta_p_true"], color="#1f77b4", linewidth=1.8, label="CFD (true)")
    ax.plot(delta_p_df["t"], delta_p_df["delta_p_pred"], color="#d62728", linewidth=1.8, linestyle="--", label="PINN")
    if ref_lo is not None:
        ax.axhspan(ref_lo, ref_hi, alpha=0.15, color="green", label=f"Reference [{ref_lo}, {ref_hi}]")
    ax.set_xlabel("t")
    ax.set_ylabel("Δp")
    ax.set_title("Pressure Drop — Schäfer-Turek Probe Points")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    save_path = os.path.join(benchmarks_dir, "delta_p.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved Δp plot → {save_path}")


def plot_pde_residuals(
    df: pd.DataFrame,
    benchmarks_dir: str,
    nu: float,
    cylinder_geometry: dict,
    domain_bounds: dict,
    t: float | None = None,
    steady: bool = True,
    grid_resolution: int = 300,
) -> None:
    """
    Interpolates PINN predictions onto a regular grid, computes spatial
    derivatives via np.gradient, and plots the absolute residuals of the
    incompressible Navier-Stokes equations.

    steady=True  (Re=20): drops ∂/∂t — valid at steady state.
    steady=False (Re=100): estimates ∂/∂t from adjacent snapshots via
                           central finite difference.
    """
    times = np.sort(df["t"].unique())
    t = times[-1] if t is None else times[np.argmin(np.abs(times - t))]
    t_idx = np.argmin(np.abs(times - t))
    snap = df[df["t"] == t]

    x_min, x_max = domain_bounds["x_min"], domain_bounds["x_max"]
    y_min, y_max = domain_bounds["y_min"], domain_bounds["y_max"]
    cx, cy, r = cylinder_geometry["cx"], cylinder_geometry["cy"], cylinder_geometry["r"]

    # Regular grid — aspect-correct spacing
    nx = grid_resolution
    ny = max(int(grid_resolution * (y_max - y_min) / (x_max - x_min)), 20)
    xi = np.linspace(x_min, x_max, nx)
    yi = np.linspace(y_min, y_max, ny)
    Xi, Yi = np.meshgrid(xi, yi)
    dx = xi[1] - xi[0]
    dy = yi[1] - yi[0]

    cyl_mask = (Xi - cx) ** 2 + (Yi - cy) ** 2 < r ** 2

    def snap_to_grid(snapshot, col):
        g = griddata(snapshot[["x", "y"]].values, snapshot[col].values, (Xi, Yi), method="linear")
        g[cyl_mask] = np.nan
        return g

    Ux = snap_to_grid(snap, "Ux_pred")
    Uy = snap_to_grid(snap, "Uy_pred")
    p  = snap_to_grid(snap, "p_pred")

    # First-order spatial derivatives
    dUx_dx = np.gradient(Ux, dx, axis=1)
    dUx_dy = np.gradient(Ux, dy, axis=0)
    dUy_dx = np.gradient(Uy, dx, axis=1)
    dUy_dy = np.gradient(Uy, dy, axis=0)
    dp_dx  = np.gradient(p,  dx, axis=1)
    dp_dy  = np.gradient(p,  dy, axis=0)

    # Second-order spatial derivatives (for viscous terms)
    d2Ux_dx2 = np.gradient(dUx_dx, dx, axis=1)
    d2Ux_dy2 = np.gradient(dUx_dy, dy, axis=0)
    d2Uy_dx2 = np.gradient(dUy_dx, dx, axis=1)
    d2Uy_dy2 = np.gradient(dUy_dy, dy, axis=0)

    # Temporal derivatives — central difference using adjacent snapshots
    if steady:
        dUx_dt = np.zeros_like(Ux)
        dUy_dt = np.zeros_like(Uy)
    else:
        i_prev = max(t_idx - 1, 0)
        i_next = min(t_idx + 1, len(times) - 1)
        dt = times[i_next] - times[i_prev]
        snap_prev = df[df["t"] == times[i_prev]]
        snap_next = df[df["t"] == times[i_next]]
        Ux_prev = snap_to_grid(snap_prev, "Ux_pred")
        Ux_next = snap_to_grid(snap_next, "Ux_pred")
        Uy_prev = snap_to_grid(snap_prev, "Uy_pred")
        Uy_next = snap_to_grid(snap_next, "Uy_pred")
        dUx_dt = (Ux_next - Ux_prev) / dt
        dUy_dt = (Uy_next - Uy_prev) / dt

    continuity = np.abs(dUx_dx + dUy_dy)
    momentum_x = np.abs(dUx_dt + Ux * dUx_dx + Uy * dUx_dy + dp_dx - nu * (d2Ux_dx2 + d2Ux_dy2))
    momentum_y = np.abs(dUy_dt + Ux * dUy_dx + Uy * dUy_dy + dp_dy - nu * (d2Uy_dx2 + d2Uy_dy2))

    residuals = {
        "Continuity  |∂Ux/∂x + ∂Uy/∂y|": continuity,
        "Momentum x": momentum_x,
        "Momentum y": momentum_y,
    }

    fig, axes = plt.subplots(3, 1, figsize=(14, 9))
    fig.suptitle(f"PDE Residuals — PINN  (t = {t:.2f})", fontsize=13, fontweight="bold")

    theta = np.linspace(0, 2 * np.pi, 100)
    cyl_x = cx + r * np.cos(theta)
    cyl_y = cy + r * np.sin(theta)

    for ax, (label, res) in zip(axes, residuals.items()):
        # Clip upper tail so the cylinder-boundary artifacts don't crush the colorscale
        vmax = float(np.nanpercentile(res, 98))
        pcm = ax.pcolormesh(Xi, Yi, res, cmap="Reds", vmin=0, vmax=vmax, shading="auto")
        plt.colorbar(pcm, ax=ax, fraction=0.02, pad=0.01)
        ax.fill(cyl_x, cyl_y, color="gray", zorder=5)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_aspect("equal")
        ax.set_title(label, fontsize=9, pad=3)
        ax.set_ylabel("y (m)", fontsize=8)
        ax.set_yticks([0.0, 0.1, 0.2, 0.3, 0.4])
        ax.tick_params(labelbottom=False, labelsize=7)

    axes[-1].tick_params(labelbottom=True)
    axes[-1].set_xlabel("x (m)")
    axes[-1].set_xticks([0.0, 0.5, 1.0, 1.5, 2.0])
    plt.tight_layout()
    save_path = os.path.join(benchmarks_dir, f"pde_residuals_t{t:.2f}.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved PDE residuals → {save_path}")


# Schäfer-Turek (1996) reference Strouhal number range for Re=100
_ST_REF_LO = 0.295
_ST_REF_HI = 0.305


def compute_strouhal_number(
    df: pd.DataFrame,
    D: float,
    U_mean: float,
    probe_x: float = 0.5,
    probe_y: float = 0.2,
) -> tuple[float, float]:
    """
    Estimate Strouhal number from FFT of the predicted Uy time series at a
    downstream probe point. Returns (St, dominant_frequency).
    Reference for Re=100: St = 0.295 – 0.305.
    """
    times = np.sort(df["t"].unique())
    dt = float(times[1] - times[0])

    # Find the fixed mesh point closest to the probe location
    snap0 = df[df["t"] == times[0]]
    dist = (snap0["x"] - probe_x) ** 2 + (snap0["y"] - probe_y) ** 2
    nearest = snap0.loc[dist.idxmin(), ["x", "y"]]
    px, py = float(nearest["x"]), float(nearest["y"])

    # Extract Uy_pred time series at that point
    probe_df = df[
        (np.abs(df["x"] - px) < 1e-8) & (np.abs(df["y"] - py) < 1e-8)
    ].sort_values("t")
    Uy_series = probe_df["Uy_pred"].values

    # Remove mean and apply FFT
    Uy_series = Uy_series - Uy_series.mean()
    n = len(Uy_series)
    freqs = np.fft.rfftfreq(n, d=dt)
    fft_mag = np.abs(np.fft.rfft(Uy_series))

    # Dominant frequency (skip DC bin at index 0)
    peak_idx = int(np.argmax(fft_mag[1:])) + 1
    f_peak = float(freqs[peak_idx])
    St = f_peak * D / U_mean

    print(f"\nStrouhal number estimate  (probe at x={px:.3f}, y={py:.3f})")
    print(f"  Dominant frequency : {f_peak:.4f} Hz")
    print(f"  St = f·D/U_mean    : {St:.4f}")
    print(f"  Reference          : {_ST_REF_LO} – {_ST_REF_HI}")

    return St, f_peak
