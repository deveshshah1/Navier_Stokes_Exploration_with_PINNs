import os
import yaml
import pandas as pd
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from custom_dataset import Cylinder2DDataset
from pyL_modules import PyLModel

# Global config
with open("./configs/config_training.yaml", "r") as file:
    config_training = yaml.safe_load(file)
    config_training = {k: v["value"] for k, v in config_training.items()}


def predict(
    ckpt_to_use,
    ground_truth_dataset_path,
):
    # Find correct ckpt for given model run
    model_dir = config_training["experiment_details"]["model_dir"]
    exp_dir = config_training["experiment_details"]["experiment_name"]
    ckpt_dir = os.path.join(model_dir, exp_dir, "checkpoints")
    all_ckpts = os.listdir(ckpt_dir)
    run_name = all_ckpts[0].split("_")[0]
    ckpt_path = f"{run_name}{ckpt_to_use}.ckpt"

    print(
        f"------------------Generating predictions for {os.path.join(ckpt_dir, ckpt_path)}------------------"
    )

    # Create results directory
    results_dir = os.path.join(model_dir, exp_dir, f"predictions{ckpt_to_use}")
    os.makedirs(results_dir, exist_ok=True)

    # Load the dataset
    df = pd.read_parquet(ground_truth_dataset_path)

    # Define dataset and data loader
    print("Loading dataset")
    test_data = Cylinder2DDataset(ground_truth_dataset_path=ground_truth_dataset_path, use_ground_truth_dataset=True)
    test_loader = DataLoader(test_data, batch_size=256, shuffle=False, num_workers=8)
    assert len(df) == len(test_data), "Dataset length mismatch"

    # Load the model
    model_path = os.path.join(ckpt_dir, ckpt_path)
    print(f"Loading model from {model_path}")
    model = PyLModel.load_from_checkpoint(
        model_path,
        map_location="cpu",
    )

    trainer = pl.Trainer(accelerator="auto", devices="auto")

    # Make predictions
    print("Making predictions...")
    out_batches = trainer.predict(model, test_loader)

    # Accumulate predictions
    print("Accumulating predictions...")
    id, u_pred, v_pred, p_pred = [], [], [], []
    for batch in out_batches:
        id.append(np.array(batch["test_point_id"]))
        u_pred.append(batch["u_pred"].cpu().numpy().astype(np.float32))
        v_pred.append(batch["v_pred"].cpu().numpy().astype(np.float32))
        p_pred.append(batch["p_pred"].cpu().numpy().astype(np.float32))

    id = np.concatenate(id, axis=0)
    u_pred = np.concatenate(u_pred, axis=0)
    v_pred = np.concatenate(v_pred, axis=0)
    p_pred = np.concatenate(p_pred, axis=0)

    # Save the predictions
    df_to_add = pd.DataFrame(
        {
            "test_point_id": id,
            "Ux_pred": u_pred,
            "Uy_pred": v_pred,
            "p_pred": p_pred,
        }
    )

    df = pd.merge(df, df_to_add, on="test_point_id", how="left")

    # Save to parquet
    save_path = os.path.join(results_dir, "predictions.parquet")
    df.to_parquet(save_path, index=False)

    return


if __name__ == "__main__":
    ground_truth_dataset_path = config_training["dataset_configs"]["ground_truth_dataset_path"]
    for ckpt in ["_best_train_loss"]:
        predict(
            ckpt_to_use=ckpt,
            ground_truth_dataset_path=ground_truth_dataset_path,
        )
    print("Predictions saved successfully")
