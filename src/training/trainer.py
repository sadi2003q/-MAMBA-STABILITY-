"""Training loop for one (arm, seed) run: tqdm progress, mid-run checkpoints, resume, divergence handling,
then the full evaluation. Everything for the run goes into its own folder."""

from __future__ import annotations

import copy
import math
import time
from pathlib import Path

import torch
from tqdm.auto import tqdm

from src.evaluation.diagnostics import certificate_status, error_amplification, limit_cycle
from src.evaluation.metrics import rollout_errors, teacher_forced_one_step
from src.losses.regimes import regime_loss
from src.models.registry import build_model, count_parameters
from src.utils.config import save_yaml
from src.utils.io import save_json, write_train_log
from src.utils.seed import set_seed


def make_optimizer(model, cfg: dict):
    decay, no_decay = [], []
    for p in model.parameters():
        if not p.requires_grad:
            continue
        (no_decay if getattr(p, "_no_weight_decay", False) or p.ndim < 2 else decay).append(p)
    t = cfg["training"]
    opt = torch.optim.Adam([{"params": decay, "weight_decay": t["weight_decay"]},
                            {"params": no_decay, "weight_decay": 0.0}], lr=t["lr"])
    sched = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=t["lr_decay_gamma"])
    return opt, sched


def _batches(n: int, batch_size: int, generator: torch.Generator):
    order = torch.randperm(n, generator=generator)
    for i in range(0, n, batch_size):
        yield order[i:i + batch_size]


@torch.no_grad()
def validation_loss(model, data: dict, regime: str, device, batch_size: int = 512) -> float:
    model.eval()
    split = data["splits"]["val"]
    total, n = 0.0, 0
    for i in range(0, split.xw.shape[0], batch_size):
        x = split.xw[i:i + batch_size].to(device)
        u = split.uw[i:i + batch_size].to(device)
        loss, _ = regime_loss(model, x, u, regime, data["context_len"], data["horizon"])
        if not math.isfinite(float(loss)):
            return float("inf")
        total += float(loss) * x.shape[0]
        n += x.shape[0]
    return total / n


def time_batches(cfg: dict, data: dict, device, n_batches: int = 3) -> float:
    """Seconds per training batch (after one warm-up batch). Used for the run-time estimate."""
    set_seed(0)
    model = build_model(cfg).to(device)
    opt, _ = make_optimizer(model, cfg)
    split = data["splits"]["train"]
    bs = cfg["training"]["batch_size"]
    times = []
    for b in range(n_batches + 1):
        idx = torch.arange(b * bs, (b + 1) * bs) % split.xw.shape[0]
        x, u = split.xw[idx].to(device), split.uw[idx].to(device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        model.train()
        opt.zero_grad(set_to_none=True)
        loss, _ = regime_loss(model, x, u, cfg["training"]["regime"], data["context_len"], data["horizon"])
        if torch.isfinite(loss):
            loss.backward()
            opt.step()
        if device.type == "cuda":
            torch.cuda.synchronize()
        if b > 0:
            times.append(time.time() - t0)
    return sum(times) / len(times)


def train_run(cfg: dict, data: dict, seed: int, run_dir: Path, device, fresh: bool = False,
              position_label: str = "") -> dict:
    run_dir.mkdir(parents=True, exist_ok=True)
    save_yaml({k: v for k, v in cfg.items() if not k.startswith("_")}, run_dir / "config.yaml")
    t_cfg = cfg["training"]
    regime = t_cfg["regime"]
    C, H = data["context_len"], data["horizon"]

    set_seed(seed)
    model = build_model(cfg).to(device)
    opt, sched = make_optimizer(model, cfg)
    gen = torch.Generator().manual_seed(seed)

    ckpt_path = run_dir / "checkpoint_last.pt"
    start_epoch, log_rows = 0, []
    best_val, best_state = float("inf"), None
    nonfinite_total, diverged = 0, False
    train_seconds = 0.0
    if ckpt_path.exists() and not fresh:
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        sched.load_state_dict(ck["scheduler"])
        gen.set_state(ck["generator"])
        start_epoch, log_rows = ck["epoch"], ck["log_rows"]
        best_val, best_state = ck["best_val"], ck["best_state"]
        nonfinite_total, diverged = ck["nonfinite_total"], ck["diverged"]
        train_seconds = ck.get("train_seconds", 0.0)
        tqdm.write(f"  resuming {position_label} from epoch {start_epoch}")

    split = data["splits"]["train"]
    n_train = split.xw.shape[0]
    epochs = int(t_cfg["epochs"])
    bar = tqdm(range(start_epoch, epochs), desc=f"  {position_label}", unit="epoch", leave=False,
               initial=start_epoch, total=epochs)
    for epoch in bar:
        if diverged:
            break
        t0 = time.time()
        model.train()
        loss_sum, steps, nonfinite = 0.0, 0, 0
        for idx in _batches(n_train, t_cfg["batch_size"], gen):
            x, u = split.xw[idx].to(device), split.uw[idx].to(device)
            opt.zero_grad(set_to_none=True)
            loss, _ = regime_loss(model, x, u, regime, C, H)
            steps += 1
            if not torch.isfinite(loss):
                nonfinite += 1
                continue
            loss.backward()
            grads_ok = all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
            if not grads_ok:
                nonfinite += 1
                opt.zero_grad(set_to_none=True)
                continue
            torch.nn.utils.clip_grad_norm_(model.parameters(), t_cfg["grad_clip"])
            opt.step()
            loss_sum += float(loss)
        sched.step()
        nonfinite_total += nonfinite
        if steps == 0 or nonfinite / steps > t_cfg["max_nonfinite_fraction"]:
            diverged = True
        val = validation_loss(model, data, regime, device) if not diverged else float("inf")
        if math.isfinite(val) and val < best_val:
            best_val, best_state = val, copy.deepcopy(model.state_dict())
        train_loss = loss_sum / max(steps - nonfinite, 1)
        train_seconds += time.time() - t0
        log_rows.append({"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val,
                         "lr": opt.param_groups[0]["lr"], "nonfinite_steps": nonfinite,
                         "seconds": round(time.time() - t0, 2)})
        bar.set_postfix(loss=f"{train_loss:.3e}", val=f"{val:.3e}", bad=nonfinite)
        if (epoch + 1) % int(t_cfg["checkpoint_every"]) == 0 or epoch + 1 == epochs or diverged:
            torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(),
                        "scheduler": sched.state_dict(), "generator": gen.get_state(), "epoch": epoch + 1,
                        "log_rows": log_rows, "best_val": best_val, "best_state": best_state,
                        "nonfinite_total": nonfinite_total, "diverged": diverged,
                        "train_seconds": train_seconds}, ckpt_path)
            write_train_log(run_dir / "train_log.csv", log_rows)
    bar.close()
    write_train_log(run_dir / "train_log.csv", log_rows)

    if t_cfg.get("select_best_on_validation", True) and best_state is not None:
        model.load_state_dict(best_state)
    torch.save({"model": model.state_dict(), "config": {k: v for k, v in cfg.items() if not k.startswith("_")}},
               run_dir / "checkpoint.pt")

    metrics = evaluate_model(model, cfg, data, device)
    metrics["training"] = {"diverged_during_training": diverged, "nonfinite_steps": nonfinite_total,
                           "epochs_run": len(log_rows), "best_val_loss": best_val,
                           "train_minutes": train_seconds / 60.0}
    metrics["parameters"] = count_parameters(model)
    metrics["diverged"] = bool(diverged or metrics["primary"]["diverged"])
    save_json(metrics, run_dir / "metrics.json")
    return metrics


def evaluate_model(model, cfg: dict, data: dict, device) -> dict:
    regime = cfg["training"]["regime"]
    primary = "direct" if regime == "direct" else "chained"
    other = "chained" if primary == "direct" else "direct"
    ev = cfg["evaluation"]
    C, H = data["context_len"], data["horizon"]
    test = data["splits"]["test"]
    out = {"primary_regime": primary,
           "primary": rollout_errors(model, primary, test, data["norm"], C, H, device, ev["report_steps"]),
           "other_regime": rollout_errors(model, other, test, data["norm"], C, H, device, ev["report_steps"]),
           "teacher_forced_one_step_rmse": teacher_forced_one_step(model, test, data["norm"], device),
           "certificate": certificate_status(model)}
    try:
        out["amplification"] = error_amplification(model, primary, test, data["norm"], data["system"], C, H,
                                                    ev["jacobian_windows"], device)
    except Exception as exc:   # a diagnostic must never lose a finished run
        out["amplification"] = {"error": repr(exc)}
    try:
        out["limit_cycle"] = limit_cycle(model, primary, data["norm"], data["system"], C,
                                         ev["limit_cycle_steps"], ev["limit_cycle_start"], device)
    except Exception as exc:
        out["limit_cycle"] = {"status": "error", "error": repr(exc)}
    return out


def result_row(arm_cfg: dict, seed: int, metrics: dict) -> dict:
    arm = arm_cfg["arm"]
    p = metrics["primary"]
    amp = metrics.get("amplification", {})
    lc = metrics.get("limit_cycle", {})
    cert = metrics.get("certificate", {})
    return {
        "arm": arm["name"], "seed": seed, "group": arm.get("group", ""), "role": arm.get("role", ""),
        "training_regime": arm_cfg["training"]["regime"], "model": arm_cfg["model"]["name"],
        "decay_mode": arm_cfg["model"].get("decay_mode", ""), "parameters": metrics["parameters"],
        "diverged": metrics["diverged"], "one_step_rmse": p["one_step_rmse"], "rollout_rmse": p["rollout_rmse"],
        "error_at_10": p.get("error_at_10"), "error_at_50": p.get("error_at_50"),
        "growth_1_to_50": p.get("growth_1_to_end"),
        "other_regime_rollout_rmse": metrics["other_regime"]["rollout_rmse"],
        "certificate_holds_at_end": cert.get("certificate_holds_at_end"),
        "fraction_decay_positive": cert.get("fraction_decay_positive"),
        "model_amplification_median": amp.get("model_amplification_median"),
        "true_amplification_median": amp.get("true_amplification_median"),
        "amplification_ratio_median": amp.get("amplification_ratio_median"),
        "limit_cycle_amplitude_error": lc.get("amplitude_error"),
        "limit_cycle_period_error": lc.get("period_error"), "limit_cycle_status": lc.get("status"),
        "nonfinite_steps": metrics["training"]["nonfinite_steps"],
        "epochs_run": metrics["training"]["epochs_run"],
        "train_minutes": metrics["training"]["train_minutes"],
    }
