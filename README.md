<div align="center">

# 🏓 Pong World Model

**A neural network that learns how a Pong game works by watching it, then plays the game inside its own imagination.**

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-env-013243?logo=numpy&logoColor=white)
![CPU](https://img.shields.io/badge/runs%20on-CPU-2E8F8F)
![Train time](https://img.shields.io/badge/full%20pipeline-~7%20min-DD9F3E)

<img src="outputs/dream.gif" width="520" alt="Real game (left) vs the model's dream (right)">

*Left: the real game. Right: the world model's imagined rollout. After 10 real frames, the game engine is switched off and the model predicts every frame from its own previous predictions, given only the actions.*

</div>

---

## ✨ At a glance

| | |
|---|---|
| 🎮 **World** | A 32 × 32 RGB Pong game: paddle, glowing ball, walls, 3 actions |
| 🗜️ **Compression** | Conv autoencoder: 3,072 pixels → **16 numbers** → 3,072 pixels |
| 🧠 **Memory + prediction** | GRU dynamics model: `(z_t, action_t, memory) → z_{t+1}` |
| 🔁 **Key tricks** | Closed-loop training + **re-encoding** each imagined frame, which keeps the ball from fading |
| 📏 **Accuracy** | Dreamed ball within **~2 px for 10 steps, ~5 px at 20**, and still **~90% visible after 20 steps** (was 39%) |
| ⏱️ **Cost** | Whole pipeline in **~7 minutes on a laptop CPU**, no GPU |

---

## 🧩 How it works

```mermaid
flowchart LR
    G["Pong env"] -->|"random play"| D[("frames + actions")]
    D --> E["Encoder"]
    E --> Z(("z_t: 16 numbers"))
    Z --> M["GRU memory"]
    A["action_t"] --> M
    M -->|"predicts the change"| Z2(("z_t+1"))
    Z2 -.->|"fed back in while dreaming"| M
    Z2 --> Dec["Decoder"]
    Dec --> F["imagined frame"]
```

1. **Collect:** play the game randomly with "sticky" actions (keys are held, as a human would) and record 200 episodes × 120 steps.
2. **Compress:** an autoencoder squeezes each frame into 16 numbers. The loss weights the tiny ball 20× more, keeps consecutive frames close in code space, and trains the decoder on slightly noisy codes.
3. **Predict:** one frame shows *where* the ball is, never *where it's going*, so the dynamics model is a **GRU** whose memory recovers the velocity. It predicts the *change* in the code.
4. **Dream:** warm up the memory on 10 real frames, switch the game off, and let the model run on its own predictions. After every step the imagined frame is **decoded and encoded again**, which snaps the code back to one a real frame could produce, so the ball cannot dissolve into blur.

> 📖 **Why each of these details is needed, and what breaks without it:** see **[LESSONS.md](LESSONS.md)**.

---

## 📊 Results

**Dream vs. reality,** every 5th frame after the engine is switched off:

<img src="outputs/dream_strip.png" width="760">

<details>
<summary><b>Same episode without re-encoding: the ball fades away</b></summary>
<br>
<img src="outputs/dream_strip_no_reencode.png" width="760">
</details>

**Ball tracking on 20 unseen test episodes:**

<img src="outputs/ball_tracking.png" width="760">

| Steps into the dream | 1 | 5 | 10 | 20 | 30 | 40 |
|---|:-:|:-:|:-:|:-:|:-:|:-:|
| Ball error, closed-loop only (px) | 1.6 | 2.0 | 3.0 | 8.6 | 16.8 | 25.1 |
| **Ball error, final model (px)** | **1.2** | **1.9** | **2.2** | **4.5** | **7.7** | **11.4** |
| Ball brightness, closed-loop only | 96% | 93% | 78% | 39% | 32% | 35% |
| **Ball brightness, final model** | **95%** | **95%** | **94%** | **90%** | **87%** | **73%** |

<sub>Reproduce with <code>python evaluate.py</code> (and <code>--no-reencode</code> for the comparison).</sub>

<table>
<tr>
<td align="center"><b>Autoencoder: real vs. reconstruction from 16 numbers</b><br><img src="outputs/ae_reconstructions.png" width="520"></td>
</tr>
</table>

**Honest reading:**
- ✅ The **paddle obeys the actions** for the whole rollout.
- ✅ The **ball stays visible and on track** for roughly 20–30 steps, including wall bounces.
- ⚠️ Position error still **grows over time** (~11 px after 40 steps), and the ball can still fade after **fast bounces near the corners and the paddle**. This compounding-error problem is the central open problem of world models; re-encoding and closed-loop training reduce it but do not remove it.

---

## 🚀 Quickstart

```bash
git clone https://github.com/varun00077/PONG-world-model.git
cd PONG-world-model
pip install -r requirements.txt

python collect_data.py        # ~3 s   · 200 train + 20 test episodes
python train_autoencoder.py   # ~4 min · frame -> 16 numbers -> frame
python train_dynamics.py      # ~2 min · one-step, then closed-loop training
python dream.py               # GIF + plots in outputs/
python evaluate.py            # ball position error and brightness per dream step
```

Useful options:

```bash
python dream.py --episode 3 --context 10 --horizon 60      # dream longer on another episode
python dream.py --no-reencode                              # see the ball fade without re-encoding
python train_dynamics.py --closed-epochs-short 0 --closed-epochs-long 0 --closed-epochs-longest 0   # ablation: no closed-loop
```

---

## 🗂️ Project structure

```text
PONG-world-model/
├── pong_game_env.py       🎮 the Pong environment
├── collect_data.py        1️⃣ random play -> data/pong_{train,test}.npz
├── models.py              🧱 Encoder, Decoder, AutoEncoder, GRU Dynamics
├── train_autoencoder.py   2️⃣ compression with object weighting, smoothness, code noise
├── train_dynamics.py      3️⃣ one-step + closed-loop training (5 → 15 → 30 steps), with re-encoding
├── dream.py               4️⃣ imagined rollouts vs. reality (GIF + plots)
├── evaluate.py            📏 ball position error and visibility per dream step
├── utils.py               🔧 seeding, data loading, weighted losses
├── outputs/               🖼️ figures and GIF shown in this README
├── LESSONS.md             📖 what the basic approach misses, and why
└── requirements.txt
```

---

## 🎮 The environment

| Property | Value |
|---|---|
| Observation | 32 × 32 × 3 RGB, `float32` in [0, 1] |
| Actions | `0` left · `1` stay · `2` right |
| Episode length | 120 steps |
| Physics | Deterministic after reset; continuous ball position rendered as a soft glow |

```python
from pong_game_env import My_pong

env = My_pong(seed=42)
obs = env.reset()                    # (32, 32, 3)
obs, done = env.step(env.RIGHT)
print(env.last_hit)                  # did the paddle hit the ball this step?
```

---

## 🛣️ Roadmap

- [x] Pong environment with RGB rendering
- [x] Data collection with sticky random actions
- [x] Autoencoder with object-weighted, smooth latent space
- [x] GRU dynamics model predicting latent changes
- [x] Closed-loop training to reduce compounding error
- [x] Re-encoding imagined frames to keep the ball from fading
- [x] Dream rollouts with quantitative evaluation
- [ ] Train encoder and dynamics **jointly**, so codes are shaped for prediction
- [ ] **Stochastic** dynamics for a world with randomness
- [ ] Train a **controller inside the dream** and test it in the real game
- [ ] **Multi-agent** Pong (two paddles)

---

<div align="center">

**📖 Read next: [LESSONS.md](LESSONS.md)**, eight things the basic approach misses and how to think about them.

</div>
