"""
DQN in CleanRL style for the simple pendulum.

This is a self-contained, single-file DQN implementation following the
conventions of CleanRL (https://github.com/vwxyzjn/cleanrl). It is
intentionally minimal — no agent class, no helper modules, just a flat
training loop with inline logging.

Usage:
    python train_pendulum_dqn_cleanrl.py
    python train_pendulum_dqn_cleanrl.py --num_steps 500000 --sparse
"""

import argparse
import os
import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pendulum
from gymnasium.wrappers import RecordVideo


def parse_args():
    parser = argparse.ArgumentParser(description='DQN (CleanRL-style) for the simple pendulum.')
    parser.add_argument('--n_tau', type=int, default=31)
    parser.add_argument('--sparse', action='store_true')
    parser.add_argument('--gamma', type=float, default=0.95)
    parser.add_argument('--epsilon', type=float, default=0.1)
    parser.add_argument('--num_steps', type=int, default=500_000)
    parser.add_argument('--buffer_size', type=int, default=100_000)
    parser.add_argument('--learning_starts', type=int, default=5_000)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--target_update_interval', type=int, default=1_000)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--seed', type=int, default=None)
    return parser.parse_args()


class QNetwork(nn.Module):
    def __init__(self, obs_dim, n_actions):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, n_actions),
        ).double()

    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity, obs_dim):
        self.capacity = capacity
        self.pos = 0
        self.size = 0
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float64)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float64)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float64)

    def add(self, obs, action, reward, next_obs):
        self.obs[self.pos] = obs
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.next_obs[self.pos] = next_obs
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size, rng):
        idx = rng.choice(self.size, size=batch_size, replace=False)
        return (
            torch.from_numpy(self.obs[idx]),
            torch.from_numpy(self.actions[idx]),
            torch.from_numpy(self.rewards[idx]),
            torch.from_numpy(self.next_obs[idx]),
        )


if __name__ == '__main__':
    args = parse_args()

    # Seeding
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    # Environment
    env = pendulum.DiscretePendulumWrapper(
        pendulum.PendulumEnv(sparse_reward=args.sparse),
        n_tau=args.n_tau,
    )
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    # Networks
    q_net = QNetwork(obs_dim, n_actions)
    q_target = QNetwork(obs_dim, n_actions)
    q_target.load_state_dict(q_net.state_dict())
    optimizer = torch.optim.Adam(q_net.parameters(), lr=args.lr)

    # Replay buffer
    rb = ReplayBuffer(args.buffer_size, obs_dim)

    # Output
    reward_label = 'sparse' if args.sparse else 'dense'
    dirname = f'results_dqn_cleanrl_{reward_label}_{args.n_tau}'
    os.makedirs(dirname, exist_ok=True)

    # TensorBoard
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(dirname)

    # --- Training loop ---
    print(f"TensorBoard: tensorboard --logdir {dirname}")
    obs, _ = env.reset(seed=args.seed)
    episode_return = 0.0
    episode_num = 0
    start_time = time.time()

    for step in range(1, args.num_steps + 1):
        # Epsilon-greedy action
        if rng.random() < args.epsilon:
            action = rng.integers(n_actions)
        else:
            with torch.no_grad():
                action = q_net(torch.from_numpy(obs)).argmax().item()

        # Step
        next_obs, reward, terminated, truncated, _ = env.step(action)
        rb.add(obs, action, reward, next_obs)

        episode_return += reward
        obs = next_obs

        # Episode end
        if terminated or truncated:
            episode_num += 1
            writer.add_scalar('episode/return', episode_return, step)
            if step % 10_000 < 100:
                sps = step / (time.time() - start_time)
                print(f"step={step:>8d}  episode_return={episode_return:.1f}  "
                      f"SPS={sps:.0f}")
            obs, _ = env.reset()
            episode_return = 0.0

        # Optimize
        if rb.size >= args.learning_starts:
            s, a, r, s1 = rb.sample(args.batch_size, rng)
            with torch.no_grad():
                target = r + args.gamma * q_target(s1).max(dim=1).values
            predicted = q_net(s).gather(1, a.unsqueeze(1).long()).squeeze(1)
            loss = F.smooth_l1_loss(predicted, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % 1000 == 0:
                writer.add_scalar('train/loss', loss.item(), step)

        # Target network update
        if step % args.target_update_interval == 0:
            q_target.load_state_dict(q_net.state_dict())

    # --- Save model ---
    model_path = os.path.join(dirname, 'q_net.pt')
    torch.save({
        'q_net': q_net.state_dict(),
        'obs_dim': obs_dim,
        'n_actions': n_actions,
        'args': vars(args),
    }, model_path)

    # --- Record video ---
    rec_env = RecordVideo(
        pendulum.DiscretePendulumWrapper(
            pendulum.PendulumEnv(render_mode='rgb_array', sparse_reward=args.sparse),
            n_tau=args.n_tau,
        ),
        video_folder=dirname,
        episode_trigger=lambda ep: True,
        disable_logger=True,
    )
    obs, _ = rec_env.reset()
    done = False
    while not done:
        with torch.no_grad():
            action = q_net(torch.from_numpy(obs)).argmax().item()
        obs, _, terminated, truncated, _ = rec_env.step(action)
        done = terminated or truncated
    rec_env.close()
    writer.close()

    sps = args.num_steps / (time.time() - start_time)
    print(f"\nDone! {args.num_steps} steps in {time.time() - start_time:.1f}s "
          f"({sps:.0f} SPS)")
    print(f"Results saved to: {dirname}")
