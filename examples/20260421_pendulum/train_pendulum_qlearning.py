import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import pendulum
from gymnasium.wrappers import RecordVideo


###############################################################################
# Q-learning agent
###############################################################################

class QLearningAgent:
    """
    Tabular Q-learning agent for a discrete-state, discrete-action environment.

    Parameters
    ----------
    n_states, n_actions : int
        Size of the Q-table.
    alpha : float
        Learning rate.
    gamma : float
        Discount factor.
    epsilon : float
        Probability of choosing a random action (epsilon-greedy).
    seed : int or None
        Random seed for reproducibility.
    """

    def __init__(self, n_states, n_actions, alpha=0.1, gamma=0.95, epsilon=0.1,
                 seed=None):
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.rg = np.random.default_rng(seed)
        self.Q = self.rg.standard_normal(size=(n_states, n_actions))

    @property
    def policy(self):
        """Greedy policy derived from the current Q-table."""
        return np.argmax(self.Q, axis=1)

    @property
    def V(self):
        """Value function derived from the current Q-table."""
        return np.max(self.Q, axis=1)

    def choose_action(self, s):
        """Epsilon-greedy action selection."""
        if self.rg.random() < self.epsilon:
            return self.rg.integers(self.n_actions)
        return np.argmax(self.Q[s, :])

    def update(self, s, a, r, s1):
        """One-step Q-learning update."""
        td_target = r + self.gamma * np.max(self.Q[s1, :])
        self.Q[s, a] += self.alpha * (td_target - self.Q[s, a])

    def train(self, env, num_steps=500_000, snapshot_interval=10_000,
              snapshot_fn=None):
        """
        Train for a given number of total simulation steps.

        snapshot_fn(step, agent) is called periodically for logging.
        Returns a dict of training history.
        """
        history = {'step': [], 'return_disc': [], 'return_undisc': []}
        step = 0

        while step < num_steps:
            s, _ = env.reset()
            episode_rewards = []
            done = False

            while not done:
                a = self.choose_action(s)
                s1, r, terminated, truncated, _ = env.step(a)
                done = terminated or truncated
                self.update(s, a, r, s1)
                s = s1
                episode_rewards.append(r)
                step += 1

            # Record episode return
            G_disc = sum(r * self.gamma**t for t, r in enumerate(episode_rewards))
            history['step'].append(step)
            history['return_disc'].append(G_disc)
            history['return_undisc'].append(sum(episode_rewards))

            # Periodic snapshot — fire if this episode crossed a snapshot boundary
            # (i.e., step jumped over a multiple of snapshot_interval during the episode)
            if snapshot_fn is not None:
                step_before_episode = step - len(episode_rewards)
                crossed_boundary = step // snapshot_interval > step_before_episode // snapshot_interval
                if crossed_boundary:
                    snapshot_fn(step, self)

        return history


###############################################################################
# Rollout and evaluation (same as in train_pendulum_vi.py)
###############################################################################

def rollout(env, policy, gamma=0.95):
    """Roll out a discrete policy on a DiscretePendulumWrapper."""
    obs, _ = env.reset()
    cont_s = env.unwrapped._get_obs()
    traj = {'t': [0], 's': [cont_s.copy()], 'a': [], 'r': []}
    done = False
    while not done:
        a = int(policy[obs])
        obs, r, terminated, truncated, _ = env.step(a)
        done = terminated or truncated
        cont_s = env.unwrapped._get_obs()
        traj['t'].append(traj['t'][-1] + 1)
        traj['s'].append(cont_s.copy())
        traj['a'].append(env.tau_centers[a])
        traj['r'].append(r)
    disc_return = sum(r * gamma**t for t, r in enumerate(traj['r']))
    return traj, disc_return, sum(traj['r'])


def evaluate_policy(env, policy, gamma=0.95, num_episodes=100):
    """Mean discounted and undiscounted returns over many episodes."""
    disc, undisc = [], []
    for _ in range(num_episodes):
        _, d, u = rollout(env, policy, gamma)
        disc.append(d)
        undisc.append(u)
    return np.mean(disc), np.mean(undisc)


###############################################################################
# Plotting helpers
###############################################################################

def save_policy_plot(grid, policy, filename, title=''):
    """Heatmap of tau vs (theta, thetadot)."""
    n_s = grid['n_theta'] * grid['n_thetadot']
    u = np.array([grid['tau_centers'][policy[s]] for s in range(n_s)])
    u = u.reshape(grid['n_theta'], grid['n_thetadot']).T

    plt.figure()
    plt.pcolor(grid['theta_edges'], grid['thetadot_edges'], u,
               shading='flat', vmin=-grid['max_tau'], vmax=grid['max_tau'])
    plt.xlabel('theta')
    plt.ylabel('thetadot')
    plt.xlim(-np.pi, np.pi)
    plt.ylim(-grid['max_thetadot'], grid['max_thetadot'])
    plt.colorbar(label='tau')
    if title:
        plt.title(title)
    plt.savefig(filename)
    plt.close()


def save_value_function_plot(grid, V, filename, title=''):
    """Heatmap of V vs (theta, thetadot), with percentile-based color limits."""
    V_grid = V.reshape(grid['n_theta'], grid['n_thetadot']).T

    plt.figure()
    plt.pcolor(grid['theta_edges'], grid['thetadot_edges'], V_grid,
               shading='flat', vmin=np.percentile(V, 2), vmax=np.percentile(V, 98))
    plt.xlabel('theta')
    plt.ylabel('thetadot')
    plt.xlim(-np.pi, np.pi)
    plt.ylim(-grid['max_thetadot'], grid['max_thetadot'])
    plt.colorbar(label='V(s)')
    if title:
        plt.title(title)
    plt.savefig(filename)
    plt.close()


def save_trajectory_plot(traj, filename, title=''):
    """3-panel plot of (theta, thetadot), tau, and reward vs time step."""
    s_arr = np.array(traj['s'])
    t = np.array(traj['t'])

    fig, ax = plt.subplots(3, 1, figsize=(10, 10))
    ax[0].plot(t, s_arr[:, 0], label='theta')
    ax[0].plot(t, s_arr[:, 1], label='thetadot')
    ax[0].legend()
    ax[0].grid(True)
    ax[1].plot(t[:-1], traj['a'], label='tau')
    ax[1].legend()
    ax[1].grid(True)
    ax[2].plot(t[:-1], traj['r'], label='r')
    ax[2].legend()
    ax[2].grid(True)
    ax[2].set_xlabel('time step')
    if title:
        fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def save_learning_curve(history, filename):
    """Plot of mean episode return (rolling average) vs training steps."""
    import pandas as pd

    fig, ax = plt.subplots(2, 1, figsize=(10, 8))
    window = max(1, len(history['return_disc']) // 50)

    for i, (key, label) in enumerate([
        ('return_disc', 'discounted return'),
        ('return_undisc', 'undiscounted return'),
    ]):
        series = pd.Series(history[key])
        mean = series.rolling(window).mean()
        std = series.rolling(window).std()
        ax[i].plot(history['step'], history[key], '.', markersize=1,
                   alpha=0.1, color='C0')
        ax[i].plot(history['step'], mean, '-', linewidth=2, color='C1',
                   label=label)
        ax[i].fill_between(history['step'], mean - std, mean + std,
                           alpha=0.3, color='C1')
        ax[i].grid(True)
        ax[i].legend()
        ax[i].set_xlim(0, history['step'][-1])

    ax[1].set_xlabel('training steps')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def record_video(env_eval, policy, video_dir):
    """Run one episode under the given policy and save an mp4 via RecordVideo."""
    rec_env = RecordVideo(
        pendulum.DiscretePendulumWrapper(
            pendulum.PendulumEnv(
                render_mode='rgb_array',
                sparse_reward=env_eval.unwrapped.sparse_reward,
            ),
            n_theta=env_eval.n_theta,
            n_thetadot=env_eval.n_thetadot,
            n_tau=env_eval.n_tau,
        ),
        video_folder=video_dir,
        episode_trigger=lambda ep: True,
        disable_logger=True,
    )
    obs, _ = rec_env.reset()
    done = False
    while not done:
        obs, _, terminated, truncated, _ = rec_env.step(int(policy[obs]))
        done = terminated or truncated
    rec_env.close()


###############################################################################
# Grid metadata (needed for plotting — matches the wrapper's discretization)
###############################################################################

def make_grid(env):
    """Build grid metadata dict from a DiscretePendulumWrapper for plotting."""
    base = env.unwrapped
    theta_edges = np.linspace(-np.pi, np.pi, env.n_theta + 1)
    thetadot_edges = np.linspace(-base.max_thetadot, base.max_thetadot,
                                 env.n_thetadot + 1)
    return {
        'n_theta': env.n_theta,
        'n_thetadot': env.n_thetadot,
        'n_tau': env.n_tau,
        'tau_centers': env.tau_centers,
        'theta_edges': theta_edges,
        'thetadot_edges': thetadot_edges,
        'max_thetadot': base.max_thetadot,
        'max_tau': base.max_tau,
    }


###############################################################################
# Main
###############################################################################

def main():
    parser = argparse.ArgumentParser(
        description='Q-learning for the simple pendulum.',
    )
    parser.add_argument('--n_theta', type=int, default=31,
                        help='grid points in theta (default: 31, should be odd)')
    parser.add_argument('--n_thetadot', type=int, default=31,
                        help='grid points in thetadot (default: 31, should be odd)')
    parser.add_argument('--n_tau', type=int, default=31,
                        help='grid points in tau (default: 31, should be odd)')
    parser.add_argument('--sparse', action='store_true',
                        help='use sparse reward instead of dense quadratic reward')
    parser.add_argument('--alpha', type=float, default=0.1,
                        help='learning rate (default: 0.1)')
    parser.add_argument('--gamma', type=float, default=0.95,
                        help='discount factor (default: 0.95)')
    parser.add_argument('--epsilon', type=float, default=0.1,
                        help='exploration rate (default: 0.1)')
    parser.add_argument('--num_steps', type=int, default=500_000,
                        help='total training steps (default: 500000)')
    parser.add_argument('--seed', type=int, default=None,
                        help='random seed (default: None, i.e., random)')
    args = parser.parse_args()

    n_theta = args.n_theta
    n_thetadot = args.n_thetadot
    n_tau = args.n_tau
    sparse_reward = args.sparse

    reward_label = 'sparse' if sparse_reward else 'dense'
    print(f"Reward: {reward_label}")
    print(f"Hyperparameters: alpha={args.alpha}, gamma={args.gamma}, "
          f"epsilon={args.epsilon}")

    dirname = (f'results_qlearning_{reward_label}_{n_theta}_{n_thetadot}_{n_tau}'
               f'_a{args.alpha}_g{args.gamma}_e{args.epsilon}')
    for sub in ['policy', 'valuefunction', 'trajectory', 'video']:
        os.makedirs(os.path.join(dirname, sub), exist_ok=True)

    # --- Create environment and agent ---
    env = pendulum.DiscretePendulumWrapper(
        pendulum.PendulumEnv(sparse_reward=sparse_reward),
        n_theta=n_theta, n_thetadot=n_thetadot, n_tau=n_tau,
    )
    grid = make_grid(env)

    agent = QLearningAgent(
        n_states=env.observation_space.n,
        n_actions=env.action_space.n,
        alpha=args.alpha,
        gamma=args.gamma,
        epsilon=args.epsilon,
        seed=args.seed,
    )

    # --- Train ---
    def snapshot(step, agent):
        mean_d, mean_u = evaluate_policy(env, agent.policy, gamma=args.gamma,
                                         num_episodes=10)
        print(f"  step {step:>8d} / {args.num_steps}:  "
              f"disc={mean_d:.3f}  undisc={mean_u:.3f}")

    print(f"Training for {args.num_steps} steps...")
    history = agent.train(env, num_steps=args.num_steps,
                          snapshot_interval=50_000, snapshot_fn=snapshot)
    print()

    # --- Evaluate final policy ---
    policy = agent.policy
    mean_disc, mean_undisc = evaluate_policy(env, policy, gamma=args.gamma)
    print(f"Final policy evaluation (100 episodes):")
    print(f"  Mean discounted return:   {mean_disc:.3f}")
    print(f"  Mean undiscounted return: {mean_undisc:.3f}")
    print()

    # --- Save results ---
    save_policy_plot(grid, policy,
                     os.path.join(dirname, 'policy', 'policy.png'),
                     title='Q-learning — policy')
    save_value_function_plot(grid, agent.V,
                             os.path.join(dirname, 'valuefunction',
                                          'valuefunction.png'),
                             title='Q-learning — value function')
    save_learning_curve(history,
                        os.path.join(dirname, 'learning_curve.png'))

    traj, _, _ = rollout(env, policy, gamma=args.gamma)
    save_trajectory_plot(traj,
                         os.path.join(dirname, 'trajectory', 'trajectory.png'),
                         title='Q-learning — example trajectory')

    print("Saving pendulum video...")
    record_video(env, policy, os.path.join(dirname, 'video'))

    print(f"Done! Results saved to: {dirname}")


if __name__ == '__main__':
    main()
