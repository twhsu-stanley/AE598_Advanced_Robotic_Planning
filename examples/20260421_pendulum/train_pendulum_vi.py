import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import argparse
import os
import pendulum
from gymnasium.wrappers import RecordVideo


###############################################################################
# Build the transition model
###############################################################################

def build_transition_model(env, n_theta, n_thetadot, n_tau, method='bilinear'):
    """
    Build a discretized transition model for the continuous pendulum.

    For every (s, a) pair on the grid, we call env.reset / env.step to
    simulate one time step, then map the resulting continuous state back
    onto the grid using either nearest-neighbor or bilinear interpolation.

    Parameters
    ----------
    env : pendulum.PendulumEnv
        Continuous pendulum environment (its sparse_reward flag determines
        which reward function is used).
    n_theta, n_thetadot, n_tau : int
        Grid resolution in each dimension (should be odd).
    method : 'nearest' or 'bilinear'

    Returns
    -------
    P    : ndarray (n_s, n_a, n_s) — transition probabilities
    R    : ndarray (n_s, n_a)      — expected rewards
    grid : dict                    — grid metadata for plotting
    """
    max_thetadot = env.max_thetadot
    max_tau = env.max_tau

    theta_edges = np.linspace(-np.pi, np.pi, n_theta + 1)
    theta_centers = 0.5 * (theta_edges[:-1] + theta_edges[1:])
    thetadot_edges = np.linspace(-max_thetadot, max_thetadot, n_thetadot + 1)
    thetadot_centers = 0.5 * (thetadot_edges[:-1] + thetadot_edges[1:])
    tau_centers = np.linspace(-max_tau, max_tau, n_tau)

    d_theta = theta_centers[1] - theta_centers[0]
    d_thetadot = thetadot_centers[1] - thetadot_centers[0]

    n_s = n_theta * n_thetadot
    n_a = n_tau
    P = np.zeros((n_s, n_a, n_s))
    R = np.zeros((n_s, n_a))

    def ij_to_s(i, j):
        return int(i) * n_thetadot + int(j)

    for s in range(n_s):
        i, j = s // n_thetadot, s % n_thetadot
        x0 = np.array([theta_centers[i], thetadot_centers[j]])

        for a in range(n_a):
            env.reset(options={'x0': x0})
            s_next, r, _, _, _ = env.step(np.array([tau_centers[a]]))
            R[s, a] = r

            theta_1 = s_next[0]
            thetadot_1 = np.clip(s_next[1], -max_thetadot, max_thetadot)

            if method == 'nearest':
                fi = (theta_1 - theta_centers[0]) / d_theta
                fj = (thetadot_1 - thetadot_centers[0]) / d_thetadot
                ci = int(np.clip(np.round(fi), 0, n_theta - 1)) % n_theta
                cj = int(np.clip(np.round(fj), 0, n_thetadot - 1))
                P[s, a, ij_to_s(ci, cj)] = 1.0

            elif method == 'bilinear':
                fi = (theta_1 - theta_centers[0]) / d_theta
                i_lo = int(np.floor(fi))
                alpha_i = fi - i_lo
                i_lo, i_hi = i_lo % n_theta, (i_lo + 1) % n_theta

                fj = np.clip((thetadot_1 - thetadot_centers[0]) / d_thetadot,
                             0, n_thetadot - 1)
                j_lo = min(int(np.floor(fj)), n_thetadot - 1)
                j_hi = min(j_lo + 1, n_thetadot - 1)
                alpha_j = fj - int(np.floor(fj))

                for ci, wi in [(i_lo, 1 - alpha_i), (i_hi, alpha_i)]:
                    for cj, wj in [(j_lo, 1 - alpha_j), (j_hi, alpha_j)]:
                        if wi * wj > 0:
                            P[s, a, ij_to_s(ci, cj)] += wi * wj

    assert np.allclose(P.sum(axis=2), 1.0), "P rows don't sum to 1"

    grid = {
        'n_theta': n_theta, 'n_thetadot': n_thetadot, 'n_tau': n_tau,
        'theta_centers': theta_centers, 'thetadot_centers': thetadot_centers,
        'tau_centers': tau_centers, 'theta_edges': theta_edges,
        'thetadot_edges': thetadot_edges, 'max_thetadot': max_thetadot,
        'max_tau': max_tau,
    }
    return P, R, grid


###############################################################################
# Value Iteration
###############################################################################

def value_iteration(P, R, gamma=0.95, tol=1e-8, max_iters=10000, callback=None):
    """
    Standard value iteration.

    callback(iteration, V, policy) is called after each sweep to allow
    recording snapshots (e.g., for a convergence video).
    """
    n_s = P.shape[0]
    n_a = P.shape[1]
    V = np.zeros(n_s)
    info = {'delta': [], 'iterations': 0}

    for iteration in range(max_iters):
        # This is what we want to compute:
        #   Q[s, a] = R[s, a] + gamma * sum_{s'} P[s, a, s'] * V[s']
        #
        # Here is one way to do it (compute Q column by column):
        Q = np.empty((n_s, n_a))
        for a in range(n_a):
            Q[:, a] = R[:, a] + gamma * P[:, a, :] @ V
        #
        # Here is a different way to do it, with broadcasting:
        #   Q = R + gamma * (P @ V)
        #
        # Here is another way to do it, with einstein summation:
        #   Q = R + gamma * np.einsum('san,n->sa', P, V)
        #
        # We could check that all these methods produce the same results:
        #   assert(np.allclose(Q, R + gamma * (P @ V)))
        #   assert(np.allclose(Q, R + gamma * np.einsum('san,n->sa', P, V)))
        #
        # It is likely that both broadcasting and einstein summation are
        # faster - they also loop to compute Q (more or less), but do it
        # in C rather than in python. However, the code may be harder to
        # understand and debug at first.
        #
        # Now, for each s, we want to compute this maximum:
        #   max_a Q[s, a]
        #
        # Here is one way to do it, by maximizing each row over all columns:
        V_new = np.max(Q, axis=1)

        # Compute the maximum change in the value function
        delta = np.max(np.abs(V_new - V))
        info['delta'].append(delta)

        # Update the value function
        V = V_new

        if callback is not None:
            callback(iteration, V.copy(), np.argmax(Q, axis=1))

        if delta < tol:
            info['iterations'] = iteration + 1
            break
    else:
        info['iterations'] = max_iters
        print(f"Warning: VI did not converge in {max_iters} iters (delta={delta:.2e})")

    return V, np.argmax(Q, axis=1), info


###############################################################################
# Rollout and evaluation
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


def td0(env, policy, n_s, gamma=0.95, alpha=0.1, num_steps=100000, snapshot_interval=10000):
        V = np.zeros(n_s)
        step = 0
        while step < num_steps:
            s, _ = env.reset()
            done = False
            while not done:
                a = int(policy[s])
                s1, r, terminated, truncated, _ = env.step(a)
                done = terminated or truncated
                delta = (r + gamma * V[s1]) - V[s]
                V[s] += alpha * delta
                s = s1
                step += 1
                if step % snapshot_interval == 0:
                    print(f' {step:10d} / {num_steps:10d}')
        
        return V


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


def save_convergence_plot(deltas, filename):
    """Semilogy plot of Bellman residual vs iteration."""
    plt.figure()
    plt.semilogy(deltas)
    plt.xlabel('Iteration')
    plt.ylabel('Bellman residual (max |V_new − V|)')
    plt.title('Value Iteration Convergence')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def save_vi_video(grid, vi_snapshots, filename, writer='ffmpeg', skip=1):
    """Animated video of the value function evolving over VI iterations."""
    frames = vi_snapshots[::skip]
    V_final = frames[-1][1]
    vmin, vmax = np.percentile(V_final, 2), np.percentile(V_final, 98)

    fig, ax = plt.subplots()
    V_grid = frames[0][1].reshape(grid['n_theta'], grid['n_thetadot']).T
    mesh = ax.pcolormesh(grid['theta_edges'], grid['thetadot_edges'], V_grid,
                         shading='flat', vmin=vmin, vmax=vmax)
    ax.set_xlabel('theta')
    ax.set_ylabel('thetadot')
    ax.set_xlim(-np.pi, np.pi)
    ax.set_ylim(-grid['max_thetadot'], grid['max_thetadot'])
    fig.colorbar(mesh, ax=ax, label='V(s)')
    title = ax.set_title('')

    def animate(i):
        iteration, V = frames[i]
        mesh.set_array(V.reshape(grid['n_theta'], grid['n_thetadot']).T.ravel())
        title.set_text(f'Value Iteration — iteration {iteration}')
        return mesh, title

    anim = animation.FuncAnimation(fig, animate, len(frames),
                                   interval=100, blit=False, repeat=False)
    anim.save(filename, writer=writer, fps=10)
    plt.close()


###############################################################################
# Record a pendulum video using gymnasium.wrappers.RecordVideo
###############################################################################

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
# Main
###############################################################################

def main():
    parser = argparse.ArgumentParser(
        description='Value iteration for the simple pendulum.',
    )
    parser.add_argument('--n_theta', type=int, default=31,
                        help='grid points in theta (default: 31, should be odd)')
    parser.add_argument('--n_thetadot', type=int, default=31,
                        help='grid points in thetadot (default: 31, should be odd)')
    parser.add_argument('--n_tau', type=int, default=31,
                        help='grid points in tau (default: 31, should be odd)')
    parser.add_argument('--sparse', action='store_true',
                        help='use sparse reward instead of dense quadratic reward')
    parser.add_argument('--method', type=str, default='bilinear',
                        choices=['bilinear', 'nearest'],
                        help='interpolation method (default: bilinear)')
    args = parser.parse_args()

    n_theta = args.n_theta
    n_thetadot = args.n_thetadot
    n_tau = args.n_tau
    sparse_reward = args.sparse
    method = args.method
    gamma = 0.95

    reward_label = 'sparse' if sparse_reward else 'dense'
    print(f"Reward: {reward_label}")

    dirname = f'results_vi_{method}_{reward_label}_{n_theta}_{n_thetadot}_{n_tau}'
    for sub in ['policy', 'valuefunction', 'trajectory', 'video']:
        os.makedirs(os.path.join(dirname, sub), exist_ok=True)

    # --- Build transition model (uses the base continuous env) ---
    env = pendulum.PendulumEnv(sparse_reward=sparse_reward)

    print(f"Building transition model ({n_theta}x{n_thetadot}x{n_tau}, {method})...")
    P, R, grid = build_transition_model(env, n_theta, n_thetadot, n_tau, method)
    n_s = n_theta * n_thetadot
    print(f"  {n_s} states, {n_tau} actions")
    print(f"  Max successor states per (s,a): {int(np.max(np.sum(P > 0, axis=2)))}")
    print()

    # --- Value Iteration ---
    print("Running value iteration...")
    vi_snapshots = []
    V, policy, info = value_iteration(
        P, R, gamma=gamma,
        callback=lambda it, V, pi: vi_snapshots.append((it, V)),
    )
    print(f"  Converged in {info['iterations']} iterations")
    print(f"  Final Bellman residual: {info['delta'][-1]:.2e}")

    # Evaluate via the discrete wrapper
    env_eval = pendulum.DiscretePendulumWrapper(
        pendulum.PendulumEnv(sparse_reward=sparse_reward),
        n_theta=n_theta, n_thetadot=n_thetadot, n_tau=n_tau,
    )
    mean_disc, mean_undisc = evaluate_policy(env_eval, policy, gamma=gamma)
    print(f"  Mean discounted return:   {mean_disc:.3f}")
    print(f"  Mean undiscounted return: {mean_undisc:.3f}")
    print()

    # Find the value function with TD(0)
    print(f"Running TD(0) to (re-)estimate value function for optimal policy...")
    V_td0 = td0(env_eval, policy, n_s, gamma=gamma, alpha=0.1, num_steps=1000000)
    save_value_function_plot(grid, V_td0,
                             os.path.join(dirname, 'valuefunction', 'valuefunction_td0.png'),
                             title='Value Iteration — value function by TD(0)')
    print()

    # --- Save results ---
    save_policy_plot(grid, policy,
                     os.path.join(dirname, 'policy', 'policy.png'),
                     title='Value Iteration — policy')
    save_value_function_plot(grid, V,
                             os.path.join(dirname, 'valuefunction', 'valuefunction.png'),
                             title='Value Iteration — value function')
    save_convergence_plot(info['delta'],
                          os.path.join(dirname, 'convergence.png'))

    traj, _, _ = rollout(env_eval, policy, gamma=gamma)
    save_trajectory_plot(traj,
                         os.path.join(dirname, 'trajectory', 'trajectory.png'),
                         title='Value Iteration — example trajectory')

    print("Saving pendulum video...")
    record_video(env_eval, policy, os.path.join(dirname, 'video'))

    print("Saving VI convergence video...")
    skip = max(1, len(vi_snapshots) // 100)
    save_vi_video(grid, vi_snapshots,
                  os.path.join(dirname, 'video', 'vi_convergence.mp4'),
                  skip=skip)

    print(f"Done! Results saved to: {dirname}")


if __name__ == '__main__':
    main()
