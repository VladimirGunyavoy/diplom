"""
Trajectories - Generate SporeIntegrator trajectories for all control patterns
===============================================================================

1D double integrator: state = [x, x_dot], control = u (scalar).

Returns a list of trajectories. Each trajectory is a list of states [x, x_dot].
"""

import numpy as np
from itertools import permutations
from ..math import SporeIntegrator


def _gen_taus(N, pattern_length, tau, seed=None):
    switch_num = pattern_length - 1  # внутренних точек переключения
    rng = np.random.default_rng(seed)
    mid = rng.uniform(0.0, tau, size=(N, switch_num))
    out = np.empty((N, switch_num + 2), dtype=float)
    out[:, 0] = 0.0
    out[:, 1:1 + switch_num] = mid
    out[:, -1] = float(tau)
    out = np.sort(out, axis=1)
    out = out[:, 1:] - out[:, :-1]
    return out


def _gen_patterns(length):
    """
    Generate all valid control sequences of given length.
    Controls: 2D vectors from {(1,0), (-1,0), (0,1), (0,-1)}.
    Constraint: each coordinate of the next control != same coordinate of the previous.
    """
    controls = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    result = []
    for combo in permutations(controls, length):
        valid = all(
            combo[i][0] != combo[i + 1][0] and combo[i][1] != combo[i + 1][1]
            for i in range(length - 1)
        )
        if valid:
            result.append(combo)
    return result


def generate_trajectories(start_state, pattern_length, tau, N, seed=None, pattern_index=None):
    """
    Generate trajectories for control patterns of given length.

    Args:
        start_state:    [x, x_dot]
        pattern_length: number of controls in each pattern
        tau:            total time per trajectory
        N:              number of random tau sets per pattern
        seed:           random seed
        pattern_index:  if given, use only that one pattern; otherwise use all patterns

    Returns:
        list of trajectories, each = list of np.array([x, x_dot])
    """
    all_patterns = _gen_patterns(pattern_length)
    patterns = [all_patterns[pattern_index]] if pattern_index is not None else all_patterns
    taus_list = _gen_taus(N=N, pattern_length=pattern_length, tau=tau, seed=seed)

    integrator = SporeIntegrator()
    trajectories = []

    for pattern in patterns:
        for taus in taus_list:
            integrator.reset(*start_state)
            states = [np.array(start_state, dtype=float)]
            for i, u in enumerate(pattern):
                next_state = integrator.step(u, taus[i])
                states.append(next_state.copy())
            trajectories.append(states)

    return trajectories


def run_pattern(start_state, pattern_index, taus):
    """
    Run a single pattern with explicitly specified time intervals.

    Args:
        start_state:   [x, x_dot]
        pattern_index: index of the pattern from _gen_patterns(len(taus))
        taus:          list of time durations, one per control step

    Returns:
        list of np.array([x, x_dot]) — states at each step including start
    """
    pattern = _gen_patterns(len(taus))[pattern_index]
    integrator = SporeIntegrator()
    integrator.reset(*start_state)
    states = [np.array(start_state, dtype=float)]
    for u, dt in zip(pattern, taus):
        states.append(integrator.step(u, dt).copy())
    return states


def generate_endpoints(start_state, pattern_length, tau, N, seed=None, pattern_index=None):
    """
    Generate only the final states of all trajectories.

    Returns:
        list of np.array([x, x_dot]) — one per trajectory
    """
    trajectories = generate_trajectories(
        start_state=start_state,
        pattern_length=pattern_length,
        tau=tau,
        N=N,
        seed=seed,
        pattern_index=pattern_index,
    )
    return [traj[-1] for traj in trajectories]
