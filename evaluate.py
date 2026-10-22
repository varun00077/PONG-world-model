# Measure how well the dream tracks the ball on the test episodes
#
# Two numbers per step into the dream:
#   * ball position error - distance (|dx| + |dy|, in pixels) between the brightest
#                           ball pixel in the dreamed frame and in the real frame
#   * ball brightness     - how bright the dreamed ball is compared with the real
#                           one (1.0 = fully visible, ~0.3 = faded into the background)
#
# Usage:
#   python evaluate.py                # with re-encoding (default)
#   python evaluate.py --no-reencode  # raw predicted codes fed back in
import argparse
import os

import numpy as np
import torch

from dream import dream
from models import AutoEncoder, Dynamics, LATENT_DIM
from utils import get_device, load_episodes


def ball(frame):
    """(3, 32, 32) -> (row, col) of the ball and its brightness score.
    The ball is the only yellow object, so red + green - blue picks it out."""
    score = frame[0] + frame[1] - frame[2]
    idx = int(torch.argmax(score))
    return np.array(divmod(idx, frame.shape[-1])), float(score.max())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--context", type=int, default=10)
    parser.add_argument("--horizon", type=int, default=40)
    parser.add_argument("--no-reencode", dest="reencode", action="store_false")
    parser.add_argument("--data-dir", default="data")
    args = parser.parse_args()

    device = get_device()
    ae = AutoEncoder(LATENT_DIM).to(device)
    ae.load_state_dict(torch.load("checkpoints/autoencoder.pt", map_location=device, weights_only=True))
    ckpt = torch.load("checkpoints/dynamics.pt", map_location=device, weights_only=True)
    dynamics = Dynamics(LATENT_DIM).to(device)
    dynamics.load_state_dict(ckpt["dynamics"])
    z_mean, z_std = ckpt["z_mean"].to(device), ckpt["z_std"].to(device)
    ae.eval(), dynamics.eval()

    frames, actions = load_episodes(os.path.join(args.data_dir, "pong_test.npz"))

    errors, brightness = [], []
    with torch.no_grad():
        for ep in range(len(frames)):
            real, dreamed = dream(ae, dynamics, frames[ep], actions[ep], z_mean, z_std,
                                  args.context, args.horizon, device, args.reencode)
            e, b = [], []
            for r, d in zip(real, dreamed):
                (r_pos, r_val), (d_pos, d_val) = ball(r), ball(d)
                e.append(np.abs(r_pos - d_pos).sum())
                b.append(d_val / r_val)
            errors.append(e)
            brightness.append(b)
    errors, brightness = np.array(errors), np.array(brightness)

    steps = [s for s in (1, 5, 10, 20, 30, 40) if s <= args.horizon]
    print(f"{len(frames)} test episodes | re-encode: {args.reencode}")
    print("step:                    " + "".join(f"{s:>7d}" for s in steps))
    print("ball position error (px) " + "".join(f"{errors[:, s - 1].mean():7.1f}" for s in steps))
    print("ball brightness vs real  " + "".join(f"{brightness[:, s - 1].mean():7.0%}" for s in steps))


if __name__ == "__main__":
    main()
