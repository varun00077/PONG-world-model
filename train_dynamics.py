# Step 3: learn how the latent state changes over time given the action
#
# The autoencoder is frozen. Every frame is encoded once, then a GRU learns
#   (z_t, a_t) -> z_{t+1}
#
# Training has two phases:
#   1. one-step ("teacher forcing"): the model always gets the REAL z_t as input.
#      Small noise is added to the inputs so it learns to correct slightly wrong codes.
#   2. closed-loop: from a random point in an episode, the model must run K steps
#      on its OWN predictions (K = 5, then 15). Each predicted code is also decoded
#      and compared with the real frame, so the imagined ball stays sharp.
#      Phase 2 is what keeps the dream from drifting.
#
# Usage:
#   python train_dynamics.py
import argparse
import os

import torch
import torch.nn.functional as F

from models import AutoEncoder, Dynamics, LATENT_DIM
from utils import get_device, load_episodes, seed_everything, weighted_bce, weighted_mse


@torch.no_grad()
def encode_episodes(encoder, frames, device, batch_size=1024):
    """(N, T+1, 3, 32, 32) -> (N, T+1, D)"""
    N, T1 = frames.shape[:2]
    flat = frames.flatten(0, 1)
    zs = [encoder(flat[i:i + batch_size].to(device)).cpu() for i in range(0, len(flat), batch_size)]
    return torch.cat(zs).view(N, T1, -1)


def train_one_step(dynamics, z, actions, args, device):
    optimizer = torch.optim.Adam(dynamics.parameters(), lr=args.lr)
    N = len(z)
    for epoch in range(1, args.one_step_epochs + 1):
        total = 0.0
        for idx in torch.randperm(N).split(args.batch_size):
            z_ep, a_ep = z[idx].to(device), actions[idx].to(device)
            z_in = z_ep[:, :-1] + args.input_noise * torch.randn_like(z_ep[:, :-1])
            z_pred, _ = dynamics(z_in, a_ep)
            loss = F.mse_loss(z_pred, z_ep[:, 1:])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item() * len(idx)
        if epoch % 5 == 0 or epoch == 1:
            print(f"[one-step]  epoch {epoch:3d} | latent MSE {total / N:.4f}")
    return optimizer


def train_closed_loop(dynamics, decoder, z, actions, frames, z_mean, z_std, optimizer, args, device):
    N, T1 = z.shape[:2]
    stages = [(5, args.lr, args.closed_epochs_short), (15, args.lr * 0.3, args.closed_epochs_long)]
    for K, lr, n_epochs in stages:
        for group in optimizer.param_groups:
            group["lr"] = lr
        for epoch in range(1, n_epochs + 1):
            total = 0.0
            for idx in torch.randperm(N).split(args.batch_size):
                z_ep, a_ep = z[idx].to(device), actions[idx].to(device)
                start = int(torch.randint(4, T1 - K - 1, (1,)))

                # warm up the memory on real codes 0 .. start-1
                _, h = dynamics(z_ep[:, :start], a_ep[:, :start])

                # then run K steps on its own predictions, starting from the real code at `start`
                z_cur = z_ep[:, start]
                loss = 0.0
                for k in range(K):
                    t = start + k
                    z_cur, h = dynamics.step(z_cur, a_ep[:, t], h)
                    loss = loss + F.mse_loss(z_cur, z_ep[:, t + 1])

                    frame_pred = decoder(z_cur * z_std + z_mean)
                    frame_real = frames[idx, t + 1].to(device)
                    loss = loss + args.pixel_weight * weighted_bce(frame_pred, frame_real)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(dynamics.parameters(), 1.0)
                optimizer.step()
                total += loss.item() * len(idx)
            if epoch % 5 == 0 or epoch == 1:
                print(f"[closed K={K:2d}] epoch {epoch:3d} | loss {total / N:.3f}")


@torch.no_grad()
def open_loop_error(dynamics, decoder, z, actions, frames, z_mean, z_std, device, context=10, horizon=30):
    """Pixel error at each step of a dream on the test set."""
    z, a = z.to(device), actions.to(device)
    _, h = dynamics(z[:, :context], a[:, :context])
    z_cur = z[:, context]
    errors = []
    for k in range(horizon):
        t = context + k
        z_cur, h = dynamics.step(z_cur, a[:, t], h)
        pred = decoder(z_cur * z_std + z_mean)
        errors.append(weighted_mse(pred, frames[:, t + 1].to(device)).item())
    return errors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--one-step-epochs", type=int, default=25)
    parser.add_argument("--closed-epochs-short", type=int, default=25)
    parser.add_argument("--closed-epochs-long", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--input-noise", type=float, default=0.05)
    parser.add_argument("--pixel-weight", type=float, default=0.002)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    seed_everything(args.seed)
    device = get_device()

    # frozen autoencoder from step 2
    ae = AutoEncoder(LATENT_DIM).to(device)
    ae.load_state_dict(torch.load("checkpoints/autoencoder.pt", map_location=device, weights_only=True))
    ae.eval()
    for p in ae.parameters():
        p.requires_grad_(False)

    train_frames, train_actions = load_episodes(os.path.join(args.data_dir, "pong_train.npz"))
    test_frames, test_actions = load_episodes(os.path.join(args.data_dir, "pong_test.npz"))
    train_z = encode_episodes(ae.encoder, train_frames, device)
    test_z = encode_episodes(ae.encoder, test_frames, device)

    # Normalise each latent dimension. Otherwise the dimensions describing the big
    # paddle dominate the loss and the ones describing the tiny ball get ignored.
    z_mean = train_z.reshape(-1, LATENT_DIM).mean(0)
    z_std = train_z.reshape(-1, LATENT_DIM).std(0).clamp_min(1e-4)
    train_z = (train_z - z_mean) / z_std
    test_z = (test_z - z_mean) / z_std
    z_mean_d, z_std_d = z_mean.to(device), z_std.to(device)
    print(f"device: {device} | train latents {tuple(train_z.shape)} | test latents {tuple(test_z.shape)}")

    dynamics = Dynamics(LATENT_DIM).to(device)

    optimizer = train_one_step(dynamics, train_z, train_actions, args, device)
    before = open_loop_error(dynamics, ae.decoder, test_z, test_actions, test_frames, z_mean_d, z_std_d, device)

    train_closed_loop(dynamics, ae.decoder, train_z, train_actions, train_frames,
                      z_mean_d, z_std_d, optimizer, args, device)
    after = open_loop_error(dynamics, ae.decoder, test_z, test_actions, test_frames, z_mean_d, z_std_d, device)

    print("\ndream error on test episodes (weighted pixel MSE):")
    print("  step:                 " + "  ".join(f"{s:>6d}" for s in (1, 5, 10, 20, 30)))
    for name, err in [("one-step training", before), ("+ closed-loop", after)]:
        print(f"  {name:<20s}" + "  ".join(f"{err[s - 1]:.4f}" for s in (1, 5, 10, 20, 30)))

    os.makedirs("checkpoints", exist_ok=True)
    torch.save({"dynamics": dynamics.state_dict(), "z_mean": z_mean, "z_std": z_std},
               "checkpoints/dynamics.pt")
    save_plot(before, after, "outputs")
    print("saved checkpoints/dynamics.pt")


def save_plot(before, after, out_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed, skipping plot")
        return
    os.makedirs(out_dir, exist_ok=True)
    steps = range(1, len(before) + 1)
    plt.figure(figsize=(5.5, 3.5))
    plt.plot(steps, before, label="one-step training only")
    plt.plot(steps, after, label="+ closed-loop training")
    plt.xlabel("steps into the dream"), plt.ylabel("weighted pixel MSE")
    plt.legend(), plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "dream_error_vs_training.png"), dpi=120)
    plt.close()


if __name__ == "__main__":
    main()
