# Step 1: play the game with a random policy and save (frames, actions)
#
# Usage:
#   python collect_data.py                      # 200 train + 20 test episodes
#   python collect_data.py --episodes 50        # quicker
import argparse
import os

import numpy as np

from pong_game_env import My_pong


def to_uint8(obs):
    # store frames as uint8 to keep the file small
    return (obs * 255).round().astype(np.uint8)


def collect(num_episodes, seed, repeat_prob=0.8):
    """Random play with "sticky" actions.

    A purely random action every step makes the paddle jitter in place.
    Repeating the previous action most of the time gives longer moves,
    so the data covers more of the screen.
    """
    env = My_pong(seed=seed)
    rng = np.random.default_rng(seed)

    T = env.MAX_STEPS
    frames = np.zeros((num_episodes, T + 1, env.SIZE, env.SIZE, 3), dtype=np.uint8)
    actions = np.zeros((num_episodes, T), dtype=np.int64)

    for episode in range(num_episodes):
        obs = env.reset()
        frames[episode, 0] = to_uint8(obs)

        action = int(rng.integers(env.NUM_ACTIONS))

        for t in range(T):
            if rng.random() > repeat_prob:
                action = int(rng.integers(env.NUM_ACTIONS))

            obs, done = env.step(action)

            frames[episode, t + 1] = to_uint8(obs)
            actions[episode, t] = action

    return frames, actions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--test-episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", default="data")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # different seeds so the test episodes are never seen in training
    for name, n, seed in [("train", args.episodes, args.seed),
                          ("test", args.test_episodes, args.seed + 1000)]:
        frames, actions = collect(n, seed)
        path = os.path.join(args.out_dir, f"pong_{name}.npz")
        np.savez_compressed(path, frames=frames, actions=actions)
        print(f"{name}: frames {frames.shape}, actions {actions.shape} -> {path}")


if __name__ == "__main__":
    main()
