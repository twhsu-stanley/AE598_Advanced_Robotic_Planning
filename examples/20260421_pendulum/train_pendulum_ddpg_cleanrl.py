"""
DDPG in CleanRL style for the simple pendulum.

This is a self-contained, single-file DDPG implementation following the
conventions of CleanRL (https://github.com/vwxyzjn/cleanrl). It is
intentionally minimal — no agent class, no helper modules, just a flat
training loop with inline logging.

Key differences from DQN:
  - Continuous actions (no DiscretePendulumWrapper)
  - Actor-critic: separate policy (actor) and Q (critic) networks
  - Exploration via Gaussian noise added to the actor's output
  - Soft target updates (Polyak averaging) instead of hard copy

Usage:
    python train_pendulum_ddpg_cleanrl.py
    python train_pendulum_ddpg_cleanrl.py --num_steps 500000 --sparse
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
    parser = argparse.ArgumentParser(description='DDPG (CleanRL-style) for the simple pendulum.')
    parser.add_argument('--sparse', action='store_true')
    parser.add_argument('--gamma', type=float, default=0.95)
    parser.add_argument('--noise_std', type=float, default=0.1,
                        help='std of Gaussian exploration noise (default: 0.1)')
    parser.add_argument('--num_steps', type=int, default=500_000)
    parser.add_argument('--buffer_size', type=int, default=100_000)
    parser.add_argument('--learning_starts', type=int, default=5_000)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--tau', type=float, default=0.005,
                        help='Polyak averaging coefficient for target networks (default: 0.005)')
    parser.add_argument('--actor_lr', type=float, default=1e-3)
    parser.add_argument('--critic_lr', type=float, default=1e-3)
    parser.add_argument('--seed', type=int, default=None)
    return parser.parse_args()


class Actor(nn.Module):
    """Deterministic policy: maps observation to a continuous action in [-max_tau, max_tau]."""
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
    """Q-function: maps (observation, action) to a scalar Q-value."""
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + act_dim, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, 1),
        ).double()

    def forward(self, obs, action):
        return self.net(torch.cat([obs, action], dim=-1)).squeeze(-1)


class ReplayBuffer:
    def __init__(self, capacity, obs_dim, act_dim):
        self.capacity = capacity
        self.pos = 0
        self.size = 0
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float64)
        self.actions = np.zeros((capacity, act_dim), dtype=np.float64)
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

    # Environment (continuous obs and continuous actions — no wrapper)
    env = pendulum.PendulumEnv(sparse_reward=args.sparse)
    obs_dim = env.observation_space.shape[0]   # 2
    act_dim = env.action_space.shape[0]        # 1
    max_tau = env.max_tau

    # Networks
    actor = Actor(obs_dim, act_dim, max_tau)
    critic = Critic(obs_dim, act_dim)
    actor_target = Actor(obs_dim, act_dim, max_tau)
    critic_target = Critic(obs_dim, act_dim)
    actor_target.load_state_dict(actor.state_dict())
    critic_target.load_state_dict(critic.state_dict())
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=args.critic_lr)

    # Replay buffer
    rb = ReplayBuffer(args.buffer_size, obs_dim, act_dim)

    # Output
    reward_label = 'sparse' if args.sparse else 'dense'
    dirname = f'results_ddpg_cleanrl_{reward_label}'
    os.makedirs(dirname, exist_ok=True)

    # TensorBoard
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(dirname)

    # --- Training loop ---
    print(f"TensorBoard: tensorboard --logdir {dirname}")
    obs, _ = env.reset(seed=args.seed)
    episode_return = 0.0
    start_time = time.time()

    for step in range(1, args.num_steps + 1):
        # Select action: actor + exploration noise (or random during warmup)
        if step <= args.learning_starts:
            action = rng.uniform(-max_tau, max_tau, size=(act_dim,))
        else:
            with torch.no_grad():
                action = actor(torch.from_numpy(obs)).numpy()
            action += args.noise_std * max_tau * rng.standard_normal(act_dim)
            action = np.clip(action, -max_tau, max_tau)

        # Step
        next_obs, reward, terminated, truncated, _ = env.step(action)
        rb.add(obs, action, reward, next_obs)

        episode_return += reward
        obs = next_obs

        # Episode end
        if terminated or truncated:
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

            # --- Critic update ---
            with torch.no_grad():
                target = r + args.gamma * critic_target(s1, actor_target(s1))
            q_pred = critic(s, a)
            critic_loss = F.mse_loss(q_pred, target)
            critic_optimizer.zero_grad()
            critic_loss.backward()
            critic_optimizer.step()

            # --- Actor update ---
            # Maximize Q by adjusting the actor: loss = -mean(Q(s, actor(s)))
            actor_loss = -critic(s, actor(s)).mean()
            actor_optimizer.zero_grad()
            actor_loss.backward()
            actor_optimizer.step()

            # --- Soft target update (Polyak averaging) ---
            with torch.no_grad():
                for p, p_tgt in zip(actor.parameters(), actor_target.parameters()):
                    p_tgt.data.mul_(1 - args.tau)
                    p_tgt.data.add_(args.tau * p.data)
                for p, p_tgt in zip(critic.parameters(), critic_target.parameters()):
                    p_tgt.data.mul_(1 - args.tau)
                    p_tgt.data.add_(args.tau * p.data)

            if step % 1000 == 0:
                writer.add_scalar('train/critic_loss', critic_loss.item(), step)
                writer.add_scalar('train/actor_loss', actor_loss.item(), step)

    # --- Save model ---
    model_path = os.path.join(dirname, 'ddpg.pt')
    torch.save({
        'actor': actor.state_dict(),
        'critic': critic.state_dict(),
        'obs_dim': obs_dim,
        'act_dim': act_dim,
        'max_tau': max_tau,
        'args': vars(args),
    }, model_path)

    # --- Record video ---
    rec_env = RecordVideo(
        pendulum.PendulumEnv(render_mode='rgb_array', sparse_reward=args.sparse),
        video_folder=dirname,
        episode_trigger=lambda ep: True,
        disable_logger=True,
    )
    obs, _ = rec_env.reset()
    done = False
    while not done:
        with torch.no_grad():
            action = actor(torch.from_numpy(obs)).numpy()
        obs, _, terminated, truncated, _ = rec_env.step(action)
        done = terminated or truncated
    rec_env.close()
    writer.close()

    sps = args.num_steps / (time.time() - start_time)
    print(f"\nDone! {args.num_steps} steps in {time.time() - start_time:.1f}s "
          f"({sps:.0f} SPS)")
    print(f"Results saved to: {dirname}")
