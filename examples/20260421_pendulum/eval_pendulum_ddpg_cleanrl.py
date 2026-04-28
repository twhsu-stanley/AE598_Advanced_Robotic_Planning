"""
Evaluate a trained DDPG model (from train_pendulum_ddpg_cleanrl.py).

Produces:
  - Value function heatmap
  - Policy heatmap
  - Video of a rollout (optionally from a fixed initial state)

Usage:
    python eval_pendulum_ddpg.py results_ddpg_cleanrl_sparse
    python eval_pendulum_ddpg.py results_ddpg_cleanrl_sparse --x0 3.14 0
"""

import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pendulum
from gymnasium.wrappers import RecordVideo


class Actor(nn.Module):
    """Must match the architecture used during training."""
    def __init__(self, obs_dim, act_dim, max_tau):
        super().__init__()
        self.max_tau = max_tau
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, act_dim), nn.Tanh(),
        ).double()

    def forward(self, x):
        return self.max_tau * self.net(x)


class Critic(nn.Module):
    """Must match the architecture used during training."""
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + act_dim, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, 1),
        ).double()

    def forward(self, obs, action):
        return self.net(torch.cat([obs, action], dim=-1)).squeeze(-1)


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate a trained DDPG model.',
    )
    parser.add_argument('results_dir', type=str,
                        help='directory containing ddpg.pt')
    parser.add_argument('--x0', type=float, nargs=2, default=None,
                        metavar=('THETA', 'THETADOT'),
                        help='initial state for video (default: random)')
    args = parser.parse_args()

    # Load model
    checkpoint = torch.load(os.path.join(args.results_dir, 'ddpg.pt'),
                            weights_only=False)
    train_args = checkpoint['args']
    obs_dim = checkpoint['obs_dim']
    act_dim = checkpoint['act_dim']
    max_tau = checkpoint['max_tau']

    actor = Actor(obs_dim, act_dim, max_tau)
    actor.load_state_dict(checkpoint['actor'])
    actor.eval()

    critic = Critic(obs_dim, act_dim)
    critic.load_state_dict(checkpoint['critic'])
    critic.eval()

    sparse_reward = train_args['sparse']

    print(f"Loaded model from {args.results_dir}")
    print(f"  obs_dim={obs_dim}, act_dim={act_dim}, sparse={sparse_reward}")

    # --- Value function, policy, and Q-function heatmaps ---
    n = 101
    theta = np.linspace(-np.pi, np.pi, n)
    thetadot = np.linspace(-15.0, 15.0, n)
    V = np.empty((n, n))
    u = np.empty((n, n))
    for i in range(n):
        for j in range(n):
            s = torch.from_numpy(np.array([theta[i], thetadot[j]]))
            with torch.no_grad():
                a = actor(s)
                V[j, i] = critic(s, a).item()
                u[j, i] = a.item()

    plt.figure()
    plt.pcolor(theta, thetadot, V, shading='nearest',
               vmin=np.percentile(V, 2), vmax=np.percentile(V, 98))
    plt.xlabel('theta')
    plt.ylabel('thetadot')
    plt.xlim(-np.pi, np.pi)
    plt.ylim(-15, 15)
    plt.colorbar(label='Q(s, actor(s))')
    plt.title('Value function')
    plt.savefig(os.path.join(args.results_dir, 'valuefunction.png'))
    plt.close()
    print(f"Saved value function plot")

    plt.figure()
    plt.pcolor(theta, thetadot, u, shading='nearest', vmin=-max_tau, vmax=max_tau)
    plt.xlabel('theta')
    plt.ylabel('thetadot')
    plt.xlim(-np.pi, np.pi)
    plt.ylim(-15, 15)
    plt.colorbar(label='tau')
    plt.title('Policy')
    plt.savefig(os.path.join(args.results_dir, 'policy.png'))
    plt.close()
    print(f"Saved policy plot")

    # --- Record video ---
    reset_options = {}
    if args.x0 is not None:
        reset_options['x0'] = args.x0
        video_name = f'eval_theta{args.x0[0]:.2f}_thetadot{args.x0[1]:.2f}'
        print(f"Recording video from x0 = {args.x0}")
    else:
        video_name = 'eval_random'
        print(f"Recording video from random initial state")

    video_dir = os.path.join(args.results_dir, 'eval_video')
    rec_env = RecordVideo(
        pendulum.PendulumEnv(render_mode='rgb_array', sparse_reward=sparse_reward),
        video_folder=video_dir,
        episode_trigger=lambda ep: True,
        name_prefix=video_name,
        disable_logger=True,
    )
    obs, _ = rec_env.reset(options=reset_options if reset_options else None)
    done = False
    total_reward = 0.0
    while not done:
        with torch.no_grad():
            action = actor(torch.from_numpy(obs)).numpy()
        obs, r, terminated, truncated, _ = rec_env.step(action)
        done = terminated or truncated
        total_reward += r
    rec_env.close()

    print(f"Episode return: {total_reward:.1f}")
    print(f"Video saved to: {video_dir}")


if __name__ == '__main__':
    main()
