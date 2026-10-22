# The two parts of the world model
#
#   frame  --Encoder-->  z (latent state)  --Decoder-->  frame
#   (z_t, action_t, memory h)  --Dynamics-->  z_{t+1}
#
# A single frame does not show which way the ball is moving, so the
# dynamics model is a GRU: its hidden state h remembers the past frames
# and can work out the ball's velocity.
import torch
import torch.nn as nn
import torch.nn.functional as F

LATENT_DIM = 16
NUM_ACTIONS = 3


class Encoder(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2, padding=1), nn.ReLU(),     # 32 -> 16
            nn.Conv2d(32, 64, 4, stride=2, padding=1), nn.ReLU(),    # 16 -> 8
            nn.Conv2d(64, 128, 4, stride=2, padding=1), nn.ReLU(),   # 8  -> 4
        )
        self.fc = nn.Linear(128 * 4 * 4, latent_dim)

    def forward(self, x):                      # x: (B, 3, 32, 32)
        h = self.conv(x)
        return self.fc(h.flatten(1))           # (B, latent_dim)


class Decoder(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 128 * 4 * 4)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1), nn.ReLU(),   # 4  -> 8
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1), nn.ReLU(),    # 8  -> 16
            nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1),                # 16 -> 32
        )

    def forward(self, z):                      # z: (B, latent_dim)
        h = self.fc(z).view(-1, 128, 4, 4)
        return torch.sigmoid(self.deconv(h))   # (B, 3, 32, 32) in [0, 1]


class AutoEncoder(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM):
        super().__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)

    def forward(self, x, code_noise=0.0):
        z = self.encoder(x)
        # Noise only goes into the decoder's input. The dynamics model's predictions
        # will land NEAR real codes, so the decoder should draw a sensible frame there too.
        z_noisy = z + code_noise * torch.randn_like(z) if code_noise > 0 else z
        return self.decoder(z_noisy), z


class Dynamics(nn.Module):
    """Predicts the next latent state from the current one and the action."""

    def __init__(self, latent_dim=LATENT_DIM, num_actions=NUM_ACTIONS, hidden_dim=256):
        super().__init__()
        self.num_actions = num_actions
        self.gru = nn.GRU(latent_dim + num_actions, hidden_dim, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def _inputs(self, z, a):
        a = F.one_hot(a, self.num_actions).float()
        return torch.cat([z, a], dim=-1)

    def forward(self, z_seq, a_seq, h=None):
        """Teacher forcing over a whole sequence.

        z_seq: (B, T, D)   real latents z_0 .. z_{T-1}
        a_seq: (B, T)      actions a_0 .. a_{T-1}
        returns predicted z_1 .. z_T, shape (B, T, D)
        """
        out, h = self.gru(self._inputs(z_seq, a_seq), h)
        # predict the change in z, not z itself (most of the frame stays the same)
        return z_seq + self.head(out), h

    def step(self, z, a, h=None):
        """One imagined step: z (B, D), a (B,) -> next z (B, D)."""
        out, h = self.gru(self._inputs(z, a).unsqueeze(1), h)
        return z + self.head(out[:, 0]), h
