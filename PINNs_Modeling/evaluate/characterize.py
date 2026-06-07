import yaml
import os
import pandas as pd
from characterize_helper import compute_l2_error_table, save_l2_error_results, plot_field_comparison


# Global config
with open("./configs/config_training.yaml", "r") as file:
    config_training = yaml.safe_load(file)
    config_training = {k: v["value"] for k, v in config_training.items()}


def characterize(ckpt_to_use):
    # Find correct ckpt for given model run
    model_dir = config_training["experiment_details"]["model_dir"]
    exp_dir = config_training["experiment_details"]["experiment_name"]
    results_dir = os.path.join(model_dir, exp_dir, f"predictions{ckpt_to_use}")

    # Load the dataset
    df = pd.read_parquet(
        os.path.join(results_dir, "predictions.parquet")
    )

    # Create the benchmarks directory
    benchmarks_dir = os.path.join(
        model_dir, exp_dir, f"benchmarks{ckpt_to_use}"
    )
    os.makedirs(benchmarks_dir, exist_ok=True)

    # --- L2 relative error table ---
    summary_df, per_time_df = compute_l2_error_table(df)
    save_l2_error_results(summary_df, per_time_df, benchmarks_dir)

    # --- Field comparison plots ---
    cylinder_geometry = config_training["dataset_configs"]["cylinder_geometry"]
    plot_field_comparison(df, benchmarks_dir, cylinder_geometry)


if __name__ == "__main__":
    all_ckpts = ["_best_train_loss"]

    for ckpt in all_ckpts:
        characterize(
            ckpt_to_use=ckpt,
        )

    print("Done characterizing the predictions!")