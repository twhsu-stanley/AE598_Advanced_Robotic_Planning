import numpy as np
import scipy.integrate
import matplotlib
import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium import spaces


class PendulumEnv(gym.Env):
    """
    Simple pendulum swing-up environment.

    The state is (theta, thetadot) where theta is the angle from the
    upward vertical (theta=0 is upright, theta=pi is hanging down)
    and thetadot is the angular velocity. The action is a scalar torque.

    Observation space: Box([-pi, -max_thetadot], [pi, max_thetadot])
    Action space:      Box([-max_tau], [max_tau])

    The episode runs for max_num_steps steps and then truncates.
    """

    metadata = {'render_modes': ['rgb_array'], 'render_fps': 10}

    def __init__(self, render_mode=None, sparse_reward=False, max_num_steps=100):
        super().__init__()
        self.render_mode = render_mode

        # Physical parameters
        self.params = {
            'm': 1.0,   # mass
            'g': 9.8,   # acceleration of gravity
            'l': 1.0,   # length
            'b': 0.1,   # coefficient of viscous friction
        }
        self.max_thetadot = 15.0
        self.max_theta_for_upright = 0.1 * np.pi
        self.max_thetadot_for_init = 5.0
        self.max_tau = 5.0
        self.dt = 0.1
        self.max_num_steps = max_num_steps
        self.sparse_reward = sparse_reward

        # Spaces
        obs_high = np.array([np.pi, self.max_thetadot], dtype=np.float64)
        self.observation_space = spaces.Box(
            low=-obs_high, high=obs_high, dtype=np.float64,
        )
        self.action_space = spaces.Box(
            low=-self.max_tau, high=self.max_tau, shape=(1,), dtype=np.float64,
        )

        # Internal state (set by reset)
        self._state = None
        self._num_steps = 0

    def _dxdt(self, x, u):
        theta_ddot = (
            u
            - self.params['b'] * x[1]
            + self.params['m'] * self.params['g'] * self.params['l'] * np.sin(x[0])
        ) / (self.params['m'] * self.params['l']**2)
        return np.array([x[1], theta_ddot])

    def _wrap_theta(self, theta):
        return ((theta + np.pi) % (2 * np.pi)) - np.pi

    def _get_obs(self):
        return np.array([self._wrap_theta(self._state[0]), self._state[1]])

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        if options is not None and 'x0' in options:
            self._state = np.array(options['x0'], dtype=np.float64)
        else:
            self._state = self.np_random.uniform(
                low=[-np.pi, -self.max_thetadot_for_init],
                high=[np.pi, self.max_thetadot_for_init],
            )

        self._num_steps = 0
        return self._get_obs(), {}

    def step(self, action):
        action = np.asarray(action, dtype=np.float64).flatten()
        u = np.clip(action[0], -self.max_tau, self.max_tau)

        # Integrate dynamics
        sol = scipy.integrate.solve_ivp(
            fun=lambda t, x: self._dxdt(x, u),
            t_span=[0, self.dt],
            y0=self._state,
            t_eval=[self.dt],
        )
        self._state = sol.y[:, 0]

        obs = self._get_obs()
        theta, thetadot = obs

        # Reward
        if abs(thetadot) > self.max_thetadot:
            reward = -100.0
        elif self.sparse_reward:
            reward = 1.0 if abs(theta) < self.max_theta_for_upright else 0.0
        else:
            reward = float(max(-100.0, -theta**2 - 0.01 * thetadot**2 - 0.01 * u**2))

        # This is a continuing task — episodes end by truncation, not termination
        self._num_steps += 1
        terminated = False
        truncated = (self._num_steps >= self.max_num_steps)

        return obs, reward, terminated, truncated, {}

    def render(self):
        if self.render_mode != 'rgb_array':
            return None

        theta = self._wrap_theta(self._state[0])

        fig, ax = plt.subplots(figsize=(4, 4), dpi=50)
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)
        ax.set_aspect('equal')
        ax.grid(True)
        ax.plot([0, -np.sin(theta)], [0, np.cos(theta)], 'o-', lw=2)
        ax.set_title(f'time = {self._num_steps * self.dt:.1f}')
        fig.canvas.draw()
        frame = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
        plt.close(fig)
        return frame


class DiscretePendulumWrapper(gym.Wrapper):
    """
    Wraps a continuous PendulumEnv to discretize observation and/or action spaces.

    Usage:
        # Continuous obs, discrete actions (for DQN)
        env = DiscretePendulumWrapper(PendulumEnv(), n_tau=31)

        # Discrete obs and actions (for Q-learning)
        env = DiscretePendulumWrapper(PendulumEnv(), n_theta=31, n_thetadot=31, n_tau=31)

    When observations are discretized, obs is a single integer (flat index).
    When actions are discretized, action is a single integer.
    """

    def __init__(self, env, n_theta=None, n_thetadot=None, n_tau=None):
        super().__init__(env)

        # --- Discretize actions ---
        self.discrete_actions = (n_tau is not None)
        if self.discrete_actions:
            self.n_tau = n_tau
            self.tau_centers = np.linspace(-env.max_tau, env.max_tau, n_tau)
            self.action_space = spaces.Discrete(n_tau)

        # --- Discretize observations ---
        self.discrete_obs = (n_theta is not None and n_thetadot is not None)
        if self.discrete_obs:
            self.n_theta = n_theta
            self.n_thetadot = n_thetadot
            self.theta_centers = np.linspace(-np.pi, np.pi, n_theta, endpoint=False)
            self.theta_centers += (self.theta_centers[1] - self.theta_centers[0]) / 2
            self.thetadot_centers = np.linspace(
                -env.max_thetadot, env.max_thetadot, n_thetadot, endpoint=False,
            )
            self.thetadot_centers += (self.thetadot_centers[1] - self.thetadot_centers[0]) / 2
            self.observation_space = spaces.Discrete(n_theta * n_thetadot)

    def _discretize_obs(self, obs):
        theta, thetadot = obs
        d_theta = self.theta_centers[1] - self.theta_centers[0]
        d_thetadot = self.thetadot_centers[1] - self.thetadot_centers[0]
        i = int(np.clip(np.round((theta - self.theta_centers[0]) / d_theta),
                        0, self.n_theta - 1)) % self.n_theta
        j = int(np.clip(np.round((thetadot - self.thetadot_centers[0]) / d_thetadot),
                        0, self.n_thetadot - 1))
        return i * self.n_thetadot + j

    def _action_to_torque(self, action):
        return np.array([self.tau_centers[action]])

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        if self.discrete_obs:
            obs = self._discretize_obs(obs)
        return obs, info

    def step(self, action):
        if self.discrete_actions:
            action = self._action_to_torque(action)
        obs, reward, terminated, truncated, info = self.env.step(action)
        if self.discrete_obs:
            obs = self._discretize_obs(obs)
        return obs, reward, terminated, truncated, info
