import torch, gc, os, csv

from torch.amp import autocast
from tqdm import tqdm
from contextlib import contextmanager


@contextmanager
def gpu_safe_context():
    try:
        yield
    except Exception:
        gc.collect()
        torch.cuda.empty_cache()
        raise


def _detach_states(st):
    if torch.is_tensor(st):
        return st.detach()
    if isinstance(st, (list, tuple)):
        return tuple(_detach_states(s) for s in st)
    return st


def process_train_batch(
    batch,
    model,
    optimizer,
    scheduler,
    scaler,
    device,
    tbptt_len: int = 4096,
    clip_grad: float = 5.0,
):
    model.train()
    inputs = batch["inputs"].to(device, non_blocking=True)
    cls = batch["cls"].to(device, non_blocking=True)
    p90 = batch["p90"].to(device, non_blocking=True)
    p10 = batch["p10"].to(device, non_blocking=True)
    sigma = batch["sigma"].to(device, non_blocking=True)

    B, L, _ = inputs.shape
    states = model.allocate_inference_cache(B)
    optimizer.zero_grad(set_to_none=True)
    h_sum = torch.zeros(B, model.d_model, device=device, dtype=inputs.dtype)
    steps_since_reset = 0

    with autocast(device.type):
        for t in range(L):
            y_t, states = model.step(inputs[:, t : t + 1], states)
            h_sum += y_t.squeeze(1)
            steps_since_reset += 1
            reached_boundary = (steps_since_reset == tbptt_len) or (t == L - 1)
            if not reached_boundary:
                continue
            h_avg = h_sum / steps_since_reset
            dir_logits, reg_out = model.heads(h_avg)
            loss, cls_loss, reg_loss = model.multi_task_loss(
                dir_logits, reg_out, cls, p90, p10, sigma
            )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            states = _detach_states(states)
            h_sum = torch.zeros_like(h_sum)
            steps_since_reset = 0
            torch.cuda.empty_cache()
    return loss.item(), cls_loss.item(), reg_loss.item()


@torch.inference_mode()
def process_val_batch(
    batch,
    model,
    device,
    tbptt_len: int = 4096,
):
    model.eval()
    inputs = batch["inputs"].to(device, non_blocking=True)
    cls = batch["cls"].to(device, non_blocking=True)
    p90 = batch["p90"].to(device, non_blocking=True)
    p10 = batch["p10"].to(device, non_blocking=True)
    sigma = batch["sigma"].to(device, non_blocking=True)

    B, L, _ = inputs.shape
    states = model.allocate_inference_cache(B)
    h_sum = torch.zeros(B, model.d_model, device=device, dtype=inputs.dtype)
    steps_since_reset = 0

    with autocast(device.type):
        for t in range(L):
            y_t, states = model.step(inputs[:, t : t + 1], states)
            h_sum += y_t.squeeze(1)
            steps_since_reset += 1
            reached_boundary = (steps_since_reset == tbptt_len) or (t == L - 1)
            if not reached_boundary:
                continue
            h_avg = h_sum / steps_since_reset
            dir_logits, reg_out = model.heads(h_avg)
            loss, cls_loss, reg_loss = model.multi_task_loss(
                dir_logits, reg_out, cls, p90, p10, sigma
            )
            states = _detach_states(states)
            h_sum = torch.zeros_like(h_sum)
            steps_since_reset = 0
            torch.cuda.empty_cache()
    return loss.item(), cls_loss.item(), reg_loss.item()


def train_loop(
    model,
    train_loader,
    val_loader,
    optimizer,
    scheduler,
    scaler,
    device,
    epochs,
    ckpt_dir,
    tbptt_len: int = 1024,
):
    os.makedirs(ckpt_dir, exist_ok=True)
    log_path = os.path.join(ckpt_dir, "loss_log.csv")
    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "epoch",
                "train_loss",
                "train_cls",
                "train_reg",
                "val_loss",
                "val_cls",
                "val_reg",
                "lr",
            ]
        )
    best_val_loss = float("inf")
    for epoch in range(1, epochs + 1):
        # ——— Training ———
        t_loss = t_cls = t_reg = 0.0
        steps = 0
        with gpu_safe_context():
            pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]")
            for batch in pbar:
                loss, c_loss, r_loss = process_train_batch(
                    batch, model, optimizer, scheduler, scaler, device, tbptt_len
                )
                steps += 1
                t_loss += loss
                t_cls += c_loss
                t_reg += r_loss
                t_avg = t_loss / steps
                pbar.set_postfix(
                    {
                        "loss": f"{loss:.4f}",
                        "cls": f"{t_cls/steps:.4f}",
                        "reg": f"{t_reg/steps:.4f}",
                        "avg": f"{t_avg:.4f}",
                        "lr": f"{scheduler.get_last_lr()[0]:.2e}",
                    }
                )
        train_loss = t_loss / steps
        train_cls = t_cls / steps
        train_reg = t_reg / steps
        print(f"Epoch {epoch} — Train Avg Loss: {train_loss:.4f}")

        # ——— Validation ———
        v_loss = v_cls = v_reg = 0.0
        v_steps = 0
        with gpu_safe_context():
            pbar = tqdm(val_loader, desc=f"Epoch {epoch} [Val]")
            for batch in pbar:
                loss, c_loss, r_loss = process_val_batch(
                    batch, model, device, tbptt_len
                )
                v_steps += 1
                v_loss += loss
                v_cls += c_loss
                v_reg += r_loss
                v_avg = v_loss / v_steps
                pbar.set_postfix(
                    {
                        "loss": f"{loss:.4f}",
                        "cls": f"{v_cls/v_steps:.4f}",
                        "reg": f"{v_reg/v_steps:.4f}",
                        "v_avg": f"{v_avg:.4f}",
                    }
                )
        val_loss = v_loss / v_steps
        val_cls = v_cls / v_steps
        val_reg = v_reg / v_steps
        print(f"Epoch {epoch} — Val Avg Loss: {val_loss:.4f}")

        lr = scheduler.get_last_lr()[0]
        with open(log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    epoch,
                    f"{train_loss:.6f}",
                    f"{train_cls:.6f}",
                    f"{train_reg:.6f}",
                    f"{val_loss:.6f}",
                    f"{val_cls:.6f}",
                    f"{val_reg:.6f}",
                    f"{lr:.6e}",
                ]
            )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "scaler": scaler.state_dict(),
                    "best_val_loss": best_val_loss,
                },
                f"{ckpt_dir}/best.pth",
            )
            print(f">>> New best model at epoch {epoch}: {best_val_loss:.4f}")
        torch.save(
            {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "best_val_loss": best_val_loss,
            },
            f"{ckpt_dir}/epoch{epoch}.pth",
        )
    print("Training complete.")
