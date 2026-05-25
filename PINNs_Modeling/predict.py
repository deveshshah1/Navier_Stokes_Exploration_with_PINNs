import os
import yaml
import pandas as pd
import numpy as np
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from custom_dataset import PlantPathologyDataset
from pyL_modules import PyLModel
from utils.misc import get_device_params

# Global config
with open("./configs/config_training.yaml", "r") as file:
    config_training = yaml.safe_load(file)
    config_training = {k: v["value"] for k, v in config_training.items()}


def predict(
    stage,
    device,
    base_dataset_path,
    model_dataset_name,
    model_dataset_path,
    ckpt_to_use="_best_val_loss",
):
    # Find correct ckpt for given model run
    model_dir = config_training["experiment_details"]["model_dir"]
    exp_dir = config_training["experiment_details"]["experiment_name"]
    ckpt_dir = os.path.join(model_dir, exp_dir, "checkpoints")
    all_ckpts = os.listdir(ckpt_dir)
    run_name = all_ckpts[0].split("_")[0]
    ckpt_path = f"{run_name}{ckpt_to_use}.ckpt"

    print(
        f"------------------Generating {stage} predictions for {os.path.join(ckpt_dir, ckpt_path)}------------------"
    )

    # Create results directory
    results_dir = os.path.join(model_dir, exp_dir, f"predictions{ckpt_to_use}")
    os.makedirs(results_dir, exist_ok=True)

    # Load the dataset
    df = pd.read_csv(os.path.join(base_dataset_path, model_dataset_path, "dataset.csv"))

    # Define dataset and data loader
    print("Loading dataset")
    test_data = PlantPathologyDataset(stage=stage, base_dataset_path=base_dataset_path, dataset_name=model_dataset_path)
    test_loader = DataLoader(test_data, batch_size=32, shuffle=False, num_workers=8)
    assert len(df) == len(test_data), "Dataset length mismatch"

    # Load the model
    print(f"Device: {device}")
    model_path = os.path.join(ckpt_dir, ckpt_path)
    print(f"Loading model from {model_path}")
    model = PyLModel.load_from_checkpoint(
        model_path,
        map_location=device,
    ).to(device)

    if device.type == "cpu":
        trainer = pl.Trainer(accelerator="cpu", devices="auto")
    elif device.type == "cuda":
        trainer = pl.Trainer(accelerator="gpu", devices=[0])
    elif device.type == "mps":
        trainer = pl.Trainer(accelerator="mps", devices=1)
    else:
        raise ValueError("Unsupported device type")

    # Make predictions
    print("Making predictions...")
    out_batches = trainer.predict(model, test_loader)

    # Accumulate predictions
    print("Accumulating predictions...")
    id, embedding, predicted_label, true_label, outputs, logits = [], [], [], [], [], []
    for batch in out_batches:
        id.append(np.array(batch["id"]))
        embedding.append(batch["embedding"].cpu().numpy().astype(np.float32))
        predicted_label.append(np.array(batch["predicted_label"]))
        true_label.append(np.array(batch["true_label"]))
        outputs.append(batch["outputs"].cpu().numpy().astype(np.float32))
        logits.append(batch["logits"].cpu().numpy().astype(np.float32))

    id = np.concatenate(id, axis=0)
    embedding = np.concatenate(embedding, axis=0)
    predicted_label = np.concatenate(predicted_label, axis=0)
    true_label = np.concatenate(true_label, axis=0)
    outputs = np.concatenate(outputs, axis=0)
    logits = np.concatenate(logits, axis=0)

    embedding_list = []
    for idx in range(len(embedding)):
        embedding_list.append(embedding[idx])

    outputs_list = []
    for idx in range(len(outputs)):
        outputs_list.append(outputs[idx])

    logits_list = []
    for idx in range(len(logits)):
        logits_list.append(logits[idx])

    # Save the predictions
    df_to_add = pd.DataFrame(
        {
            "id": id,
            "true_label": true_label,
            "predicted_label": predicted_label,
            "embedding": embedding_list,
            "outputs": outputs_list,
            "logits": logits_list,
        }
    )

    df = pd.merge(df, df_to_add, on="id", how="left")

    # Save to parquet
    save_path = os.path.join(results_dir, f"{model_dataset_name}_predictions.parquet")
    df.to_parquet(save_path, index=False)

    return


if __name__ == "__main__":
    device_params = get_device_params()
    device = torch.device(device_params["accelerator"])
    base_dataset_path = config_training["dataset_configs"]["base_dataset_path"]
    datasets_of_interest = {
        "plantpathology": config_training["dataset_configs"]["plantpathology"],
        "stanfordcars": config_training["dataset_configs"]["stanfordcars"],
        "flowers102": config_training["dataset_configs"]["flowers102"],
        "dtd": config_training["dataset_configs"]["dtd"],
    }
    for ckpt in ["_best_val_loss", "_best_train_loss", "_best_val_balanced_accuracy"]:
        for model_dataset_name, model_dataset_path in datasets_of_interest.items():
            predict(
                stage="ALL",
                device=device,
                ckpt_to_use=ckpt,
                base_dataset_path=base_dataset_path,
                model_dataset_name=model_dataset_name,
                model_dataset_path=model_dataset_path,
            )
    print("Predictions saved successfully")
