# Step 2: learn to compress a 32x32 frame into a small latent vector z
#
# Three things in the loss besides plain reconstruction:
#   * object weighting  - the ball is a few pixels, so its pixels count more
#   * smoothness        - consecutive frames should get nearby codes, so the
#                         dynamics model has an easy path to follow in latent space
#   * code noise        - the decoder also learns to draw frames from codes
#                         slightly off the real ones (where predictions will land)
#
# Usage:
#   python train_autoencoder.py --epochs 20
import argparse
import os

import torch

from models import AutoEncoder, LATENT_DIM
from utils import get_device, load_episodes, seed_everything, weighted_bce, weighted_mse


def evaluate(model, frames, device, batch_size=512):
    model.eval()
    flat = frames.flatten(0, 1)
    total = 0.0
    with torch.no_grad():
        for i in range(0, len(flat), batch_size):
            x = flat[i:i + batch_size].to(device)
            recon, _ = model(x)
            total += weighted_mse(recon, x).item() * len(x)
    return total / len(flat)


def save_plots(history, model, test_frames, device, out_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed, skipping plots")
        return

    # loss curves
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    axes[0].plot(history["train"]), axes[0].set_title("train loss"), axes[0].set_xlabel("epoch")
    axes[1].plot(history["test"]), axes[1].set_title("test reconstruction (weighted MSE)")
    axes[1].set_xlabel("epoch"), axes[1].set_yscale("log")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "ae_loss.png"), dpi=120)
    plt.close()

    # real (top row) vs reconstruction (bottom row)
    model.eval()
    x = test_frames[0, ::15][:8].to(device)
    with torch.no_grad():
        recon, _ = model(x)
    fig, axes = plt.subplots(2, 8, figsize=(12, 3.2))
    for i in range(len(x)):
        axes[0, i].imshow(x[i].permute(1, 2, 0).cpu())
        axes[1, i].imshow(recon[i].permute(1, 2, 0).cpu())
        axes[0, i].axis("off"), axes[1, i].axis("off")
    axes[0, 0].set_title("real", loc="left")
    axes[1, 0].set_title(f"reconstruction from {LATENT_DIM} numbers", loc="left")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "ae_reconstructions.png"), dpi=120)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--smooth-weight", type=float, default=1.0)
    parser.add_argument("--size-weight", type=float, default=0.01)
    parser.add_argument("--code-noise", type=float, default=0.1)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    seed_everything(args.seed)
    device = get_device()
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    train_frames, _ = load_episodes(os.path.join(args.data_dir, "pong_train.npz"))
    test_frames, _ = load_episodes(os.path.join(args.data_dir, "pong_test.npz"))
    N, T1 = train_frames.shape[:2]
    print(f"device: {device} | train frames: {N * T1} | test frames: {test_frames.shape[0] * T1}")

    # every (episode, t) with a next frame t+1 is one training pair
    pairs = torch.stack(torch.meshgrid(torch.arange(N), torch.arange(T1 - 1), indexing="ij"), -1).view(-1, 2)

    model = AutoEncoder(LATENT_DIM).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    history = {"train": [], "test": []}
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for batch in pairs[torch.randperm(len(pairs))].split(args.batch_size):
            ep, t = batch[:, 0], batch[:, 1]
            x = train_frames[ep, t].to(device)            # frame t
            x_next = train_frames[ep, t + 1].to(device)   # frame t+1 of the same episode

            recon, z = model(x, code_noise=args.code_noise)
            z_next = model.encoder(x_next)

            recon_loss = weighted_bce(recon, x)
            smooth_loss = ((z - z_next) ** 2).sum(1).mean()   # consecutive frames -> nearby codes
            size_loss = (z ** 2).sum(1).mean()                 # keeps the code scale bounded
            loss = recon_loss + args.smooth_weight * smooth_loss + args.size_weight * size_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item() * len(x)

        history["train"].append(total / len(pairs))
        history["test"].append(evaluate(model, test_frames, device))
        print(f"epoch {epoch:3d} | train loss {history['train'][-1]:8.2f} "
              f"| test weighted MSE {history['test'][-1]:.5f}")

    torch.save(model.state_dict(), "checkpoints/autoencoder.pt")
    save_plots(history, model, test_frames, device, "outputs")
    print("saved checkpoints/autoencoder.pt")


if __name__ == "__main__":
    main()
