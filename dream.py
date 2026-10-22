# Step 4: let the world model "dream"
#
# 1. Show the model the first few real frames of a test episode
#    (so the GRU memory can work out where the ball is going).
# 2. Then stop showing real frames. The model feeds its own predictions
#    back in, using the same actions that were played in the real game.
# 3. Decode the imagined latents and compare them with what really happened.
#    After each step the decoded frame is encoded again (re-encoding), which
#    keeps the imagined ball from fading. Use --no-reencode to see the difference.
#
# Usage:
#   python dream.py --episode 0 --context 10 --horizon 40
import argparse
import os

import numpy as np
import torch

from models import AutoEncoder, Dynamics, LATENT_DIM
from utils import get_device, load_episodes


@torch.no_grad()
def dream(ae, dynamics, frames, actions, z_mean, z_std, context, horizon, device, reencode=True):
    """frames: (T+1, 3, 32, 32), actions: (T,) for one episode.
    Returns the real and the dreamed frames context+1 .. context+horizon."""
    z_real = (ae.encoder(frames.to(device)) - z_mean) / z_std     # normalised codes (T+1, D)
    a = actions.to(device)

    # warm up the GRU memory on real codes 0 .. context-1
    _, h = dynamics(z_real[None, :context], a[None, :context])

    # start from the real code at `context` (one the memory has not seen yet),
    # then the model only sees its own predictions
    z = z_real[context:context + 1]
    imagined = []
    for t in range(context, context + horizon):
        z, h = dynamics.step(z, a[t:t + 1], h)
        frame = ae.decoder(z * z_std + z_mean)
        imagined.append(frame)
        if reencode:
            # snap the code back to one the encoder could produce for this frame
            z = (ae.encoder(frame) - z_mean) / z_std

    dreamed = torch.cat(imagined)
    real = frames[context + 1:context + horizon + 1]
    return real.cpu(), dreamed.cpu()


def to_image(frame, scale=8):
    """(3, 32, 32) tensor in [0, 1] -> upscaled uint8 HxWx3 array"""
    img = (frame.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return img.repeat(scale, axis=0).repeat(scale, axis=1)


def save_gif(real, dreamed, path, scale=8):
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        print("pillow not installed, skipping GIF")
        return False

    gap = np.full((32 * scale, 8, 3), 255, dtype=np.uint8)
    images = []
    for t, (r, d) in enumerate(zip(real, dreamed)):
        side_by_side = np.concatenate([to_image(r, scale), gap, to_image(d, scale)], axis=1)
        im = Image.fromarray(side_by_side)
        draw = ImageDraw.Draw(im)
        draw.text((6, 4), f"real  t+{t + 1}", fill=(255, 255, 255))
        draw.text((32 * scale + 14, 4), "dreamed", fill=(255, 255, 255))
        images.append(im)
    images[0].save(path, save_all=True, append_images=images[1:], duration=120, loop=0)
    return True


def save_strip_and_error(real, dreamed, out_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed, skipping plots")
        return

    # every 5th frame: real on top, dreamed below
    steps = list(range(0, len(real), 5))
    fig, axes = plt.subplots(2, len(steps), figsize=(1.6 * len(steps), 3.4))
    for col, t in enumerate(steps):
        axes[0, col].imshow(real[t].permute(1, 2, 0))
        axes[1, col].imshow(dreamed[t].permute(1, 2, 0))
        axes[0, col].set_title(f"t+{t + 1}", fontsize=8)
        axes[0, col].axis("off"), axes[1, col].axis("off")
    axes[0, 0].text(-4, 16, "real", rotation=90, va="center", ha="right")
    axes[1, 0].text(-4, 16, "dream", rotation=90, va="center", ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "dream_strip.png"), dpi=120)
    plt.close()

    # how fast the dream drifts away from reality
    error = ((real - dreamed) ** 2).mean(dim=(1, 2, 3)).numpy()
    plt.figure(figsize=(5, 3.5))
    plt.plot(np.arange(1, len(error) + 1), error)
    plt.xlabel("steps into the dream"), plt.ylabel("pixel MSE")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "dream_error.png"), dpi=120)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--context", type=int, default=10)
    parser.add_argument("--horizon", type=int, default=40)
    parser.add_argument("--no-reencode", dest="reencode", action="store_false",
                        help="feed raw predicted codes back in (shows the ball fading)")
    parser.add_argument("--out", default="outputs/dream.gif")
    parser.add_argument("--data-dir", default="data")
    args = parser.parse_args()

    device = get_device()
    os.makedirs("outputs", exist_ok=True)

    ae = AutoEncoder(LATENT_DIM).to(device)
    ae.load_state_dict(torch.load("checkpoints/autoencoder.pt", map_location=device, weights_only=True))
    ckpt = torch.load("checkpoints/dynamics.pt", map_location=device, weights_only=True)
    dynamics = Dynamics(LATENT_DIM).to(device)
    dynamics.load_state_dict(ckpt["dynamics"])
    z_mean, z_std = ckpt["z_mean"].to(device), ckpt["z_std"].to(device)
    ae.eval(), dynamics.eval()

    frames, actions = load_episodes(os.path.join(args.data_dir, "pong_test.npz"))
    real, dreamed = dream(ae, dynamics, frames[args.episode], actions[args.episode],
                          z_mean, z_std, args.context, args.horizon, device, args.reencode)

    error = ((real - dreamed) ** 2).mean(dim=(1, 2, 3))
    print(f"pixel MSE  step 1: {error[0]:.5f} | step 10: {error[min(9, len(error) - 1)]:.5f} "
          f"| step {len(error)}: {error[-1]:.5f}")

    if save_gif(real, dreamed, args.out):
        print(f"saved {args.out}")
    save_strip_and_error(real, dreamed, "outputs")


if __name__ == "__main__":
    main()
