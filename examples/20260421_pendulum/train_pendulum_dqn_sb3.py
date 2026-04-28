"""
DQN with Stable-Baselines3 on the simple pendulum.

Usage:
    python train_pendulum_dqn_sb3.py
    python train_pendulum_dqn_sb3.py --num_steps 500000 --sparse
"""

import argparse
import os
import numpy as np
import pendulum
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from gymnasium.wrappers import RecordVideo


def main():
    parser = argparse.ArgumentParser(description='DQN (SB3) for the simple pendulum.')
    parser.add_argument('--n_tau', type=int, default=31)
    parser.add_argument('--sparse', action='store_true')
    parser.add_argument('--gamma', type=float, default=0.95)
    parser.add_argument('--num_steps', type=int, default=500_000)
    parser.add_argument('--seed', type=int, default=None)
    args = parser.parse_args()

    reward_label = 'sparse' if args.sparse else 'dense'
    dirname = f'results_dqn_sb3_{reward_label}_{args.n_tau}'
    os.makedirs(dirname, exist_ok=True)

    # Training environment (continuous obs, discrete actions)
    env = pendulum.DiscretePendulumWrapper(
        pendulum.PendulumEnv(sparse_reward=args.sparse),
        n_tau=args.n_tau,
    )

    # Evaluation environment (for periodic eval during training)
    eval_env = Monitor(pendulum.DiscretePendulumWrapper(
        pendulum.PendulumEnv(sparse_reward=args.sparse),
        n_tau=args.n_tau,
    ))

    model = DQN(
        'MlpPolicy',
        env,
        gamma=args.gamma,
        learning_rate=1e-3,
        buffer_size=100_000,
        learning_starts=5_000,
        batch_size=32,
        target_update_interval=1_000,
        exploration_fraction=0.1,
        exploration_final_eps=0.1,
        verbose=1,
        seed=args.seed,
        tensorboard_log=dirname,
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=dirname,
        log_path=dirname,
        eval_freq=10_000,
        n_eval_episodes=10,
        deterministic=True,
    )

    print(f"Training SB3 DQN for {args.num_steps} steps...")
    print(f"TensorBoard: tensorboard --logdir {dirname}")
    model.learn(total_timesteps=args.num_steps, callback=eval_callback,
                log_interval=100)
    model.save(os.path.join(dirname, 'final_model'))

    # Record a video of the final policy
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
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = rec_env.step(int(action))
        done = terminated or truncated
    rec_env.close()

    print(f"Done! Results saved to: {dirname}")


if __name__ == '__main__':
    main()
