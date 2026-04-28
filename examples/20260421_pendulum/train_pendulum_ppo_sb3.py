"""
PPO with Stable-Baselines3 on the simple pendulum.

Usage:
    python train_pendulum_ppo_sb3.py
    python train_pendulum_ppo_sb3.py --num_steps 1000000 --sparse
"""

import argparse
import os
import pendulum
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from gymnasium.wrappers import RecordVideo


def main():
    parser = argparse.ArgumentParser(description='PPO (SB3) for the simple pendulum.')
    parser.add_argument('--sparse', action='store_true')
    parser.add_argument('--gamma', type=float, default=0.95)
    parser.add_argument('--num_steps', type=int, default=1_000_000)
    parser.add_argument('--seed', type=int, default=None)
    args = parser.parse_args()

    reward_label = 'sparse' if args.sparse else 'dense'
    dirname = f'results_ppo_sb3_{reward_label}'
    os.makedirs(dirname, exist_ok=True)

    # Training environment (continuous obs, continuous actions)
    env = pendulum.PendulumEnv(sparse_reward=args.sparse)

    # Evaluation environment
    eval_env = Monitor(pendulum.PendulumEnv(sparse_reward=args.sparse))

    model = PPO(
        'MlpPolicy',
        env,
        gamma=args.gamma,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
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

    print(f"Training SB3 PPO for {args.num_steps} steps...")
    print(f"TensorBoard: tensorboard --logdir {dirname}")
    model.learn(total_timesteps=args.num_steps, callback=eval_callback,
                log_interval=10)
    model.save(os.path.join(dirname, 'final_model'))

    # Record a video of the final policy
    rec_env = RecordVideo(
        pendulum.PendulumEnv(render_mode='rgb_array', sparse_reward=args.sparse),
        video_folder=dirname,
        episode_trigger=lambda ep: True,
        disable_logger=True,
    )
    obs, _ = rec_env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = rec_env.step(action)
        done = terminated or truncated
    rec_env.close()

    print(f"Done! Results saved to: {dirname}")


if __name__ == '__main__':
    main()
