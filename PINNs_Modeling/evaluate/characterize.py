import yaml
import os
import pandas as pd
from characterize_helper import (
    compute_l2_error_table,
    save_l2_error_results,
    plot_field_comparison,
    plot_wake_profiles,
    compute_delta_p,
    save_delta_p_results,
    plot_pde_residuals,
    compute_strouhal_number,
)


# Global config
with open("./configs/config_training.yaml", "r") as file:
    config_training = yaml.safe_load(file)
    config_training = {k: v["value"] for k, v in config_training.items()}


def characterize(ckpt_to_use):
    model_dir = config_training["experiment_details"]["model_dir"]
    exp_dir   = config_training["experiment_details"]["experiment_name"]
    results_dir    = os.path.join(model_dir, exp_dir, f"predictions{ckpt_to_use}")
    benchmarks_dir = os.path.join(model_dir, exp_dir, f"benchmarks{ckpt_to_use}")
    os.makedirs(benchmarks_dir, exist_ok=True)

    df = pd.read_parquet(os.path.join(results_dir, "predictions.parquet"))

    dataset_cfg       = config_training["dataset_configs"]
    cylinder_geometry = dataset_cfg["cylinder_geometry"]
    domain_bounds     = dataset_cfg["domain_bounds"]
    nu                = dataset_cfg["nu"]
    U_mean            = dataset_cfg["U_mean"]
    Re                = dataset_cfg["reynolds_number"]
    D                 = 2 * cylinder_geometry["r"]  # cylinder diameter

    is_steady = Re == 20

    # Snapshots to visualize for field comparison, wake profiles, and PDE residuals.
    # Re=20: last snapshot only (steady state).
    # Re=100: 4 snapshots spaced ~quarter-period apart in the developed flow region.
    if is_steady:
        viz_snapshots = [None]
    else:
        viz_snapshots = [5.0, 5.4, 5.8, 6.2]

    # --- L2 relative error table ---
    summary_df, per_time_df = compute_l2_error_table(df)
    save_l2_error_results(summary_df, per_time_df, benchmarks_dir)

    # --- Field comparison plots ---
    for t in viz_snapshots:
        plot_field_comparison(df, benchmarks_dir, cylinder_geometry, t=t)

    # --- Wake velocity profiles ---
    for t in viz_snapshots:
        plot_wake_profiles(df, benchmarks_dir, t=t)

    # --- Δp probe comparison ---
    # Re=20 has a published steady-state reference; Re=100 Δp oscillates so no band.
    delta_p_df = compute_delta_p(df)
    ref_lo, ref_hi = (0.1172, 0.1176) if is_steady else (None, None)
    save_delta_p_results(delta_p_df, benchmarks_dir, ref_lo=ref_lo, ref_hi=ref_hi)

    # --- PDE residual maps ---
    for t in viz_snapshots:
        plot_pde_residuals(df, benchmarks_dir, nu, cylinder_geometry, domain_bounds, t=t, steady=is_steady)

    # --- Strouhal number (Re=100 only) ---
    if not is_steady:
        compute_strouhal_number(df, D=D, U_mean=U_mean)


if __name__ == "__main__":
    all_ckpts = ["_best_train_loss"]

    for ckpt in all_ckpts:
        characterize(ckpt_to_use=ckpt)

    print("Done characterizing the predictions!")
