import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.tri as tri


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
            ax.set_ylabel(field if col == 0 else "")
            ax.set_yticks([])
            ax.set_xticks([])

    plt.tight_layout()
    save_path = os.path.join(benchmarks_dir, f"field_comparison_t{t:.2f}.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved field comparison → {save_path}")
