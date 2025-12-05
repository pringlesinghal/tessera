# train_ssl.py

import os

os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ["NUMEXPR_MAX_THREADS"] = "24"
import time
import math
import subprocess
import argparse
import logging
from datetime import datetime
from contextlib import nullcontext

import numpy as np
import pandas as pd
from collections import Counter
import torch
import torch.nn as nn
import torch.cuda.amp as amp
import wandb

# Modified imports for local structure
# Use absolute imports to allow running as a script
import sys
from pathlib import Path

# Add parent directory to path to enable imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from tessera_ml.dataset import TreeDataset
from tessera_ml.models.modules import (
    TransformerEncoder,
    ProjectionHead,
    SpectralTemporalTransformer,
    SimpleMLP,
)
from tessera_ml.models.ssl_model import (
    MultimodalBTModel,
    BarlowTwinsLoss,
    compute_cross_correlation,
)
from tessera_ml.utils.lr_scheduler import adjust_learning_rate
from tessera_ml.utils.metrics import linear_probe_evaluate, rankme
from tessera_ml.utils.misc import remove_dir, save_checkpoint, plot_cross_corr
import matplotlib.pyplot as plt

# Default Configuration
DEFAULT_CONFIG = {
    "data_root": "data/ready_to_use_40",  # Placeholder, will be overridden by args
    "index_dir": "data/index",  # Placeholder
    "batch_size": 256,
    "epochs": 100,
    "learning_rate": 0.002,
    "barlow_lambda": 5e-3,
    "fusion_method": "concat",
    "latent_dim": 128,
    "projector_hidden_dim": 4096,
    "projector_out_dim": 4096,
    "min_valid_timesteps": 0,
    "sample_size_s2": 40,
    "sample_size_s1": 40,
    "num_workers": 4,
    "shuffle_tiles": True,
    "warmup_ratio": 0,
    "plateau_ratio": 0.0,
    "apply_amp": True,
    "apply_mixup": True,
    "mixup_lambda": 1.0,
    "beta_alpha": 1.0,
    "beta_beta": 1.0,
    "log_interval_steps": 10,
    "val_interval_steps": 0,  # Disable validation by default for now
    "total_samples": 100000,  # Approximate, will be updated by dataset
}


def parse_args():
    parser = argparse.ArgumentParser(description="SSL Training")
    parser.add_argument(
        "--config", type=str, default=None, help="Path to config file (optional)"
    )
    parser.add_argument(
        "--index_dir", type=str, required=True, help="Path to index directory"
    )
    parser.add_argument(
        "--data_dir", type=str, required=True, help="Path to data directory"
    )
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument(
        "--dry_run", action="store_true", help="Run in dry-run mode (no wandb)"
    )
    parser.add_argument(
        "--resume_from",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from",
    )
    return parser.parse_args()


def main():
    args_cli = parse_args()

    # Load config
    config = DEFAULT_CONFIG.copy()
    if args_cli.config:
        config_module = {}
        with open(args_cli.config, "r") as f:
            exec(f.read(), config_module)
        if "config" in config_module:
            config.update(config_module["config"])

    # Override with CLI args
    config["index_dir"] = args_cli.index_dir
    config["data_root"] = args_cli.data_dir  # Using data_root as data_dir key
    config["batch_size"] = args_cli.batch_size
    config["epochs"] = args_cli.epochs

    # Logging setup
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    run_name = f"BT_Iter_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    if args_cli.dry_run:
        wandb_mode = "disabled"
    else:
        wandb_mode = "online"

    wandb_run = wandb.init(
        project="btfm-iterable-temp", name=run_name, config=config, mode=wandb_mode
    )

    # Log source code artifacts
    if not args_cli.dry_run:
        artifact = wandb.Artifact("source-code", type="source")
        script_dir = Path(__file__).parent
        artifact.add_file(str(script_dir / "train_ssl.py"))
        artifact.add_file(str(script_dir / "dataset.py"))
        artifact.add_file(str(script_dir / "models" / "modules.py"))
        artifact.add_file(str(script_dir / "models" / "ssl_model.py"))
        artifact.add_file(str(script_dir / "utils" / "lr_scheduler.py"))
        artifact.add_file(str(script_dir / "utils" / "metrics.py"))
        artifact.add_file(str(script_dir / "utils" / "misc.py"))
        wandb.log_artifact(artifact)

    # Initialize Dataset
    logging.info("Initializing TreeDataset...")
    dataset_train = TreeDataset(
        index_dir=config["index_dir"],
        data_dir=config["data_root"],
        year=None,  # Random year from years list
        years=list(range(2017, 2025)),  # Use years 2017-2024
        sample_size_s2=config["sample_size_s2"],
        sample_size_s1=config["sample_size_s1"],
        normalize=True,
    )

    # Update total samples based on dataset length
    config["total_samples"] = len(dataset_train)
    total_steps = config["epochs"] * config["total_samples"] // config["batch_size"]
    logging.info(f"Dataset length: {len(dataset_train)}")
    logging.info(f"Total steps = {total_steps}")

    # Model Setup
    s2_num_heads = 4
    s2_num_layers = 4
    s2_dim_feedforward = 1024
    s1_num_heads = 4
    s1_num_layers = 4
    s1_dim_feedforward = 1024

    wandb.config.update(
        {
            "s2_num_heads": s2_num_heads,
            "s2_num_layers": s2_num_layers,
            "s2_dim_feedforward": s2_dim_feedforward,
            "s1_num_heads": s1_num_heads,
            "s1_num_layers": s1_num_layers,
            "s1_dim_feedforward": s1_dim_feedforward,
        }
    )

    s2_enc = TransformerEncoder(
        band_num=10,
        latent_dim=config["latent_dim"],
        nhead=s2_num_heads,
        num_encoder_layers=s2_num_layers,
        dim_feedforward=s2_dim_feedforward,
        dropout=0.1,
        max_seq_len=config["sample_size_s2"],  # Now 40
    ).to(device)

    s1_enc = TransformerEncoder(
        band_num=2,
        latent_dim=config["latent_dim"],
        nhead=s1_num_heads,
        num_encoder_layers=s1_num_layers,
        dim_feedforward=s1_dim_feedforward,
        dropout=0.1,
        max_seq_len=config["sample_size_s1"],  # Now 40
    ).to(device)

    if config["fusion_method"] == "concat":
        proj_in_dim = config["latent_dim"]
    else:
        proj_in_dim = config["latent_dim"]

    projector = ProjectionHead(
        proj_in_dim, config["projector_hidden_dim"], config["projector_out_dim"]
    ).to(device)

    if config["fusion_method"] == "transformer":
        model = MultimodalBTModel(
            s2_enc,
            s1_enc,
            projector,
            fusion_method=config["fusion_method"],
            return_repr=True,
            latent_dim=config["latent_dim"],
        ).to(device)
    else:
        model = MultimodalBTModel(
            s2_enc,
            s1_enc,
            projector,
            fusion_method=config["fusion_method"],
            return_repr=True,
        ).to(device)

    criterion = BarlowTwinsLoss(lambda_coeff=config["barlow_lambda"])

    logging.info(
        f"Model has {sum(p.numel() for p in model.parameters() if p.requires_grad)} trainable parameters."
    )

    weight_params = [p for n, p in model.named_parameters() if p.ndim > 1]
    bias_params = [p for n, p in model.named_parameters() if p.ndim == 1]
    optimizer = torch.optim.AdamW(
        [{"params": weight_params}, {"params": bias_params}],
        lr=config["learning_rate"],
        weight_decay=1e-6,
    )

    if config.get("apply_amp", False):
        scaler = amp.GradScaler()
    else:
        scaler = None

    step = 0
    examples = 0
    start_epoch = 0
    last_time = time.time()
    last_examples = 0
    rolling_loss = []
    rolling_size = 40
    best_val_acc = 0.0
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    best_ckpt_path = os.path.join("checkpoints", "ssl", f"best_model_{timestamp}.pt")

    # Load checkpoint if resuming
    if args_cli.resume_from and os.path.exists(args_cli.resume_from):
        logging.info(f"Resuming training from checkpoint: {args_cli.resume_from}")
        checkpoint = torch.load(args_cli.resume_from, map_location=device)

        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if scaler is not None and "scaler_state_dict" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler_state_dict"])

        start_epoch = checkpoint.get("epoch", 0) + 1  # Start from next epoch
        step = checkpoint.get("step", 0)
        examples = checkpoint.get("examples", 0)
        best_val_acc = checkpoint.get("best_val_acc", 0.0)

        logging.info(
            f"Resumed from epoch {start_epoch}, step {step}, examples {examples}"
        )
        logging.info(f"Best validation accuracy so far: {best_val_acc:.4f}")
    elif args_cli.resume_from:
        logging.warning(
            f"Checkpoint not found at {args_cli.resume_from}, starting from scratch"
        )

    # Training Loop
    for epoch in range(start_epoch, start_epoch + config["epochs"]):
        train_loader = torch.utils.data.DataLoader(
            dataset_train,
            batch_size=config["batch_size"],
            num_workers=config["num_workers"],
            drop_last=True,
            shuffle=False,  # since index is already globally shuffled
        )
        model.train()
        for batch_data in train_loader:
            s2_aug1 = batch_data["s2_aug1"].to(device, non_blocking=True)
            s2_aug2 = batch_data["s2_aug2"].to(device, non_blocking=True)
            s1_aug1 = batch_data["s1_aug1"].to(device, non_blocking=True)
            s1_aug2 = batch_data["s1_aug2"].to(device, non_blocking=True)

            adjust_learning_rate(
                optimizer,
                step,
                total_steps,
                config["learning_rate"],
                config["warmup_ratio"],
                config["plateau_ratio"],
            )
            optimizer.zero_grad()

            with amp.autocast() if config.get("apply_amp", False) else nullcontext():
                z1, repr1 = model(s2_aug1, s1_aug1)
                z2, repr2 = model(s2_aug2, s1_aug2)
                loss_main, bar_main, off_main = criterion(z1, z2)
                loss_mix = 0.0
                if config["apply_mixup"]:
                    B = s2_aug1.size(0)
                    idxs = torch.randperm(B, device=device)
                    alpha = (
                        torch.distributions.Beta(
                            config["beta_alpha"], config["beta_beta"]
                        )
                        .sample()
                        .to(device)
                    )
                    y_m_s2 = alpha * s2_aug1 + (1 - alpha) * s2_aug2[idxs, :]
                    y_m_s1 = alpha * s1_aug1 + (1 - alpha) * s1_aug2[idxs, :]
                    z_m, _ = model(y_m_s2, y_m_s1)
                    cc_m_a = compute_cross_correlation(z_m, z1)
                    cc_m_b = compute_cross_correlation(z_m, z2)
                    cc_z1_z1 = compute_cross_correlation(z1, z1)
                    cc_z2idx_z1 = compute_cross_correlation(z2[idxs], z1)
                    cc_z1_z2 = compute_cross_correlation(z1, z2)
                    cc_z2idx_z2 = compute_cross_correlation(z2[idxs], z2)
                    cc_m_a_gt = alpha * cc_z1_z1 + (1 - alpha) * cc_z2idx_z1
                    cc_m_b_gt = alpha * cc_z1_z2 + (1 - alpha) * cc_z2idx_z2
                    diff_a = (cc_m_a - cc_m_a_gt).pow(2).sum()
                    diff_b = (cc_m_b - cc_m_b_gt).pow(2).sum()
                    loss_mix = (
                        config["mixup_lambda"]
                        * config["barlow_lambda"]
                        * (diff_a + diff_b)
                    )
                total_loss = loss_main + loss_mix

            if config.get("apply_amp", False):
                scaler.scale(total_loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=2.0, norm_type=2
                )
                scaler.step(optimizer)
                scaler.update()
            else:
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=2.0, norm_type=2
                )
                optimizer.step()

            examples += s2_aug1.size(0)
            if step % config["log_interval_steps"] == 0:
                current_time = time.time()
                exps = (examples - last_examples) / (current_time - last_time)
                last_time = current_time
                last_examples = examples
                rolling_loss.append(loss_main.item())
                if len(rolling_loss) > rolling_size:
                    rolling_loss = rolling_loss[-rolling_size:]
                avg_loss = sum(rolling_loss) / len(rolling_loss)
                current_lr = optimizer.param_groups[0]["lr"]
                print(f"{current_lr=}")
                erank_z = rankme(z1)
                erank_repr = rankme(repr1)
                mix_loss_val = (
                    loss_mix if isinstance(loss_mix, float) else loss_mix.item()
                )
                logging.info(
                    f"[Epoch={epoch}, Step={step}] Loss={loss_main.item():.2f}, MixLoss={mix_loss_val:.2f}, AvgLoss={avg_loss:.2f}, LR={current_lr:.4f}, batchsize={s2_aug1.size(0)}, Examples/sec={exps:.2f}, Rank(z)={erank_z:.4f}, Rank(repr)={erank_repr:.4f}"
                )
                wandb_dict = {
                    "epoch": epoch,
                    "loss_main": loss_main.item(),
                    "mix_loss": mix_loss_val,
                    "avg_loss": avg_loss,
                    "lr": current_lr,
                    "examples/sec": exps,
                    "total_loss": total_loss.item(),
                    "rank_z": erank_z,
                    "rank_repr": erank_repr,
                }
                wandb.log(wandb_dict, step=step)

            step += 1

        # Save checkpoint at the end of each epoch
        epoch_ckpt_path = os.path.join(
            "checkpoints", "ssl", f"checkpoint_epoch_{epoch}_step_{step}.pt"
        )
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            step=step,
            val_acc=best_val_acc,
            ckpt_path=epoch_ckpt_path,
            scaler=scaler,
            examples=examples,
            best_val_acc=best_val_acc,
        )

        logging.info(f"Epoch {epoch} finished, current step = {step}")

    # Save final checkpoint
    final_ckpt_path = os.path.join("checkpoints", "ssl", f"final_model_{timestamp}.pt")
    save_checkpoint(
        model=model,
        optimizer=optimizer,
        epoch=config["epochs"] - 1,
        step=step,
        val_acc=best_val_acc,
        ckpt_path=final_ckpt_path,
        scaler=scaler,
        examples=examples,
        best_val_acc=best_val_acc,
    )

    logging.info("Training completed.")
    wandb_run.finish()


if __name__ == "__main__":
    main()
