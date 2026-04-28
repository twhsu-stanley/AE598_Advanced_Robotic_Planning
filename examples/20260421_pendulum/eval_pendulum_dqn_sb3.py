"""
Evaluate a trained SB3 DQN model (from train_pendulum_dqn_sb3.py).

Produces:
  - Value function heatmap
  - Policy heatmap
  - Video of a rollout (optionally from a fixed initial state)

Usage:
    python eval_pendulum_dqn_sb3.py results_dqn_sb3_sparse_31
    python eval_pendulum_dqn_sb3.py results_dqn_sb3_sparse_31 --x0 3.14 0
"""

import argparse
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import pendulum
from stable_baselines3 import DQN
from gymnasium.wrappers import RecordVideo


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate a trained SB3 DQN model.',
    )
    parser.add_argument('results_dir', type=str,
                        help='directory containing final_model.zip '
                             '(e.g., results_dqn_sb3_sparse_31)')
    parser.add_argument('--x0', type=float, nargs=2, default=None,
                        metavar=('THETA', 'THETADOT'),
                        help='initial state for video (default: random)')
    args = parser.parse_args()

    # Parse n_tau and reward type from directory name
    # Expected format: results_dqn_sb3_{dense|sparse}_{n_tau}
    parts = os.path.basename(args.results_dir).split('_')
    sparse_reward = 'sparse' in parts
    n_tau = int(parts[-1])

    # Load model
    model = DQN.load(os.path.join(args.results_dir, 'final_model'))
    print(f"Loaded model from {args.results_dir}")

    tau_centers = np.linspace(-5.0, 5.0, n_tau)

    # --- Value function and policy heatmaps ---
    n = 101
    theta = np.linspace(-np.pi, np.pi, n)
    thetadot = np.linspace(-15.0, 15.0, n)
    V = np.empty((n, n))
    u = np.empty((n, n))
    for i in range(n):
        for j in range(n):
            s = np.array([theta[i], thetadot[j]])
            q = model.policy.q_net(torch.from_numpy(s).float().unsqueeze(0))
            V[j, i] = q.max().item()
            u[j, i] = tau_centers[q.argmax().item()]

    plt.figure()
    plt.pcolor(theta, thetadot, V, shading='nearest',
               vmin=np.percentile(V, 2), vmax=np.percentile(V, 98))
    plt.xlabel('theta')
    plt.ylabel('thetadot')
    plt.xlim(-np.pi, np.pi)
    plt.ylim(-15, 15)
    plt.colorbar(label='V(s)')
    plt.title('Value function')
    plt.savefig(os.path.join(args.results_dir, 'valuefunction.png'))
    plt.close()
    print(f"Saved value function plot")

    plt.figure()
    plt.pcolor(theta, thetadot, u, shading='nearest', vmin=-5, vmax=5)
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
        pendulum.DiscretePendulumWrapper(
            pendulum.PendulumEnv(render_mode='rgb_array', sparse_reward=sparse_reward),
            n_tau=n_tau,
        ),
        video_folder=video_dir,
        episode_trigger=lambda ep: True,
        name_prefix=video_name,
        disable_logger=True,
    )
    obs, _ = rec_env.reset(options=reset_options if reset_options else None)
    done = False
    total_reward = 0.0
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, r, terminated, truncated, _ = rec_env.step(int(action))
        done = terminated or truncated
        total_reward += r
    rec_env.close()

    print(f"Episode return: {total_reward:.1f}")
    print(f"Video saved to: {video_dir}")


if __name__ == '__main__':
    main()
