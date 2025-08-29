#!/usr/bin/env python3
"""
SNN UniAD Inference Script (with built-in validate_snn)

Usage:
    python tools/snn_inference.py \
        --config tools/config.yaml \
        --checkpoint tools/checkpoints/Bottle/ckpt.pth.tar \
        --class_name bottle --timesteps 4 --validate
"""
import sys, os
import yaml
import torch
import argparse
import numpy as np
from easydict import EasyDict
import time
import shutil
import logging
import torch.distributed as dist
from pathlib import Path

# ==== Add project root to sys.path ====
current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, current_dir)

# ==== Utils, datasets, models ====
from utils.misc_helper import (
    AverageMeter, load_state, set_random_seed, update_config
)
from utils.criterion_helper import build_criterion
from utils.eval_helper import dump, merge_together, performances, log_metrics
from datasets.data_builder import build_dataloader
from models.model_helper import ModelHelper
from models.SNN_model import convert_uniad_to_ecmt


# ============================================================
#   Model Wrapper (thêm filename vào output)
# ============================================================
class ModelWithFilename(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    def reset(self):
        if hasattr(self.model, "reset"):
            self.model.reset()
    
    def forward(self, batch, timesteps=None):
        out = self.model(batch, timesteps=timesteps)
        if isinstance(out, dict):
            out = dict(out)
            if "filename" not in out and "filename" in batch:
                out["filename"] = batch["filename"]
        return out


# ============================================================
#   Utility: đảm bảo batch lên GPU
# ============================================================
def to_device_loader(loader, device):
    for batch in loader:
        if torch.is_tensor(batch):
            yield batch.to(device)
        elif isinstance(batch, (list, tuple)):
            yield [b.to(device) if torch.is_tensor(b) else b for b in batch]
        elif isinstance(batch, dict):
            yield {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
        else:
            yield batch


# ============================================================
#   Validate cho SNN (AUROC + các metrics khác)
# ============================================================
def validate_snn(val_loader, model, device, config, timesteps, single_gpu_mode=True, wandb_run=None, epoch=None):
    batch_time = AverageMeter(0)
    losses = AverageMeter(0)

    model.eval()
    rank = 0 if single_gpu_mode else dist.get_rank()
    logger = logging.getLogger("global_logger")
    criterion = build_criterion(config.criterion)
    end = time.time()

    # ==== eval_dir fallback nếu không có trong config ====
    eval_dir = getattr(config.get("evaluator", {}), "eval_dir", "./eval_tmp")
    Path(eval_dir).mkdir(parents=True, exist_ok=True)
    if not single_gpu_mode:
        dist.barrier()

    with torch.no_grad():
        for i, input in enumerate(val_loader):
            input = input.to(device) if torch.is_tensor(input) else input
            outputs = model(input, timesteps=timesteps)

            # === build lại dict có cả mask & filename ===
            if isinstance(input, dict):
                batch_size = outputs["pred"].shape[0]
                height = outputs["pred"].shape[2]
                width = outputs["pred"].shape[3]

                batch_outputs = {
                    "pred": outputs["pred"].detach().cpu(),
                    "mask": input["mask"].detach().cpu() if "mask" in input else None,
                    "filename": input.get("filename", []),
                    "clsname": input.get("clsname", []),
                    "height": torch.tensor([height] * batch_size),  # ✅ tensor 1-D
                    "width": torch.tensor([width] * batch_size)     # ✅ tensor 1-D
                }
            else:
                raise ValueError("Expect dataloader to return dict with keys: image, mask, filename, clsname")
            dump(eval_dir, batch_outputs)

            # record loss
            loss = 0
            for _, criterion_loss in criterion.items():
                weight = criterion_loss.weight
                loss += weight * criterion_loss(outputs)
            num = len(outputs["filename"])
            losses.update(loss.item(), num)

            # time
            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % config.trainer.print_freq_step == 0 and rank == 0 and logger:
                logger.info(
                    f"Test: [{i+1}/{len(val_loader)}]\t"
                    f"Time {batch_time.val:.3f} ({batch_time.avg:.3f})"
                )

    # ==== gather final results ====
    if single_gpu_mode:
        final_loss = losses.avg
        total_num = losses.count
    else:
        dist.barrier()
        total_num = torch.Tensor([losses.count]).cuda()
        loss_sum = torch.Tensor([losses.avg * losses.count]).cuda()
        dist.all_reduce(total_num, async_op=True)
        dist.all_reduce(loss_sum, async_op=True)
        final_loss = loss_sum.item() / total_num.item()
        total_num = total_num.item()

    ret_metrics = {}
    if rank == 0:
        if logger:
            logger.info("Gathering final results ...")
            logger.info(f" * Loss {final_loss:.5f}\ttotal_num={total_num}")

        # merge outputs
        fileinfos, preds, masks = merge_together(eval_dir)
        shutil.rmtree(eval_dir)

        # evaluate (AUROC, PRO, F1, ...)
        metrics = getattr(config.evaluator, "metrics", ["AUROC"])
        ret_metrics = performances(fileinfos, preds, masks, metrics)
        log_metrics(ret_metrics, metrics)

        # wandb logging
        if wandb_run and epoch is not None:
            wandb_metrics = {"val/loss": final_loss, "epoch": epoch + 1}
            for metric_name, metric_value in ret_metrics.items():
                wandb_metrics[f"val/{metric_name}"] = metric_value
            wandb_run.log(wandb_metrics)

    model.train()
    return ret_metrics


# ============================================================
#   Parse args
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser(description='SNN UniAD Inference/Validation')
    parser.add_argument('--config', type=str, default='./tools/config.yaml', help='Config file path')
    parser.add_argument('--checkpoint', type=str, required=True, help='ANN checkpoint path')
    parser.add_argument('--class_name', type=str, default=None, help='Single class to test (e.g., bottle)')
    parser.add_argument('--timesteps', type=int, default=4, help='Number of timesteps for SNN')
    parser.add_argument('--tau', type=float, default=2.0, help='LIF time constant')
    parser.add_argument('--num_thresholds', type=int, default=4, help='Multi-threshold neuron thresholds')
    parser.add_argument('--compensation_window', type=int, default=4, help='ECM history window')
    parser.add_argument('--output_dir', type=str, default='./snn_results', help='Output directory')
    parser.add_argument('--validate', action='store_true', help='Run validate_snn() instead of raw inference')
    return parser.parse_args()


# ============================================================
#   Main
# ============================================================
def main():
    args = parse_args()

    # load config
    with open(args.config) as f:
        config = EasyDict(yaml.load(f, Loader=yaml.FullLoader))
        config = update_config(config)

    set_random_seed(config.get('random_seed', 133), reproduce=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("=" * 60)
    print("🚀 SNN UniAD Inference with ECMT")
    print("=" * 60)
    print(f"Config: {args.config}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Class: {args.class_name or 'All classes'}")
    print(f"Timesteps: {args.timesteps}")
    print(f"Device: {device}")
    print("=" * 60)

    # 1. Load ANN model
    ann_model = ModelHelper(config.net)
    ann_model.to(device)
    ann_model.device = torch.device(device)
    load_state(args.checkpoint, ann_model)
    ann_model.eval()

    # 2. Convert to SNN
    ecmt_model = convert_uniad_to_ecmt(
        ann_model,
        tau=args.tau,
        num_thresholds=args.num_thresholds,
        compensation_window=args.compensation_window
    )
    ecmt_model = ModelWithFilename(ecmt_model).to(device)
    ecmt_model.eval()

    # 3. Build dataloader
    _, test_loader = build_dataloader(
        config.dataset,
        distributed=False,
        class_name=args.class_name
    )

    # 4. Run validate or raw inference
    if args.validate:
        print("✅ Running validate_snn() on SNN model...")
        test_loader_gpu = to_device_loader(test_loader, device)
        results = validate_snn(test_loader_gpu, ecmt_model, device, config, timesteps=args.timesteps)
        print("📊 Validation Results:")
        for k, v in results.items():
            if isinstance(v, np.ndarray):
                print(f"{k}: shape={v.shape}")
            elif isinstance(v, list):
                print(f"{k}: len={len(v)}")
            else:
                print(f"{k}: {v}")
    else:
        print("🔥 Running raw SNN inference (no validate)...")
        all_preds, all_files, all_classes = [], [], []
        with torch.no_grad():
            for batch in test_loader:
                batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
                ecmt_model.reset()
                output = ecmt_model(batch, timesteps=args.timesteps)
                pred = output['pred'].cpu()
                all_preds.extend(pred)
                all_files.extend(batch.get('filename', ['unknown']*len(pred)))
                all_classes.extend(batch.get('clsname', ['unknown']*len(pred)))

        os.makedirs(args.output_dir, exist_ok=True)
        np.savez(
            os.path.join(args.output_dir, f'snn_results_{args.class_name or "all"}.npz'),
            predictions=np.array([p.numpy() for p in all_preds]),
            filenames=np.array(all_files),
            classnames=np.array(all_classes),
            timesteps=args.timesteps
        )
        print(f"💾 Saved results to {args.output_dir}")

    print("=" * 60)
    print("🎉 Done!")


if __name__ == '__main__':
    main()
