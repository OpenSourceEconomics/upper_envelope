from functools import partial
from typing import Tuple

import jax
from jax import numpy as jnp
from upper_envelope.fues_jax.check_conditions import (
    create_indicator_if_value_function_is_switched,
)
from upper_envelope.math_funcs import calc_gradient


def forward_scan(
    value: jnp.ndarray,
    policy: jnp.array,
    endog_grid: jnp.ndarray,
    endog_grid_j: float,
    policy_j: float,
    idx_to_scan_from: int,
    n_points_to_scan: int,
    jump_thresh: float,
) -> Tuple[float, int]:
    """Find next point on same value function as most recent point on upper envelope.

    We use the forward scan to find the next point that lies on the same value
    function segment as the most recent point on the upper envelope (j).
    Then we calculate the gradient between the point found on the same value function
    segment and the point we currently inspect at idx_to_scan_from.

    Args:
        value (np.ndarray): 1d array of shape (n_grid_wealth,) containing the
            unrefined value correspondence.
        policy (np.ndarray): 1d array of shape (n_grid_wealth,) containing the
            unrefined policy correspondence.
        endog_grid (np.ndarray): 1d array of shape (n_grid_wealth,) containing the
            unrefined endogenous wealth grid.
        endog_grid_j (float): Endogenous grid point that corresponds to the most recent
            value function point that lies on the upper envelope (j).
        policy_j (float): Point of the policy function that corresponds to the most
            recent value function point that lies on the upper envelope (j).
        idx_to_scan_from (int): Index of the point we want to scan from. This should
            be the current point we inspect.
        n_points_to_scan (int): Number of points to scan.
        jump_thresh (float): Threshold for the jump in the value function.

    Returns:
        tuple:

        - grad_next_forward (float): The gradient of the next point on the same
            value function.
        - idx_on_same_value (int): Index of next point on the value function.

    """
    (
        grad_next_on_same_value,
        idx_on_same_value,
    ) = back_and_forward_scan_wrapper(
        endog_grid_to_calculate_gradient=endog_grid[idx_to_scan_from],
        value_to_calculate_gradient=value[idx_to_scan_from],
        endog_grid_to_scan_from=endog_grid_j,
        policy_to_scan_from=policy_j,
        endog_grid=endog_grid,
        value=value,
        policy=policy,
        idx_to_scan_from=idx_to_scan_from,
        n_points_to_scan=n_points_to_scan,
        jump_thresh=jump_thresh,
        direction="forward",
    )

    return (
        grad_next_on_same_value,
        idx_on_same_value,
    )


def backward_scan(
    value: jnp.ndarray,
    policy: jnp.ndarray,
    endog_grid: jnp.ndarray,
    endog_grid_j,
    value_j,
    idx_to_scan_from: int,
    n_points_to_scan: int,
    jump_thresh: float,
) -> Tuple[float, int]:
    """Find previous point on same value function as idx_to_scan_from.

    We use the backward scan to find the preceding point that lies on the same value
    function segment as the point we inspect. Then we calculate the gradient between
    the point found and the most recent point on the upper envelope (j).

    Args:
        value (np.ndarray): 1d array of shape (n_grid_wealth,) containing the
            unrefined value correspondence.
        policy (np.ndarray): 1d array of shape (n_grid_wealth,) containing the
            unrefined policy correspondence.
        endog_grid (np.ndarray): 1d array of shape (n_grid_wealth,) containing the
            unrefined endogenous wealth grid.
        endog_grid_j (float): Endogenous grid point that corresponds to the most recent
            value function point that lies on the upper envelope (j).
        value_j (float): Point of the value function that corresponds to the most recent
            value function point that lies on the upper envelope (j).
        idx_to_scan_from (int): Index of the point we want to scan from. This should
            be the current point we inspect.
        n_points_to_scan (int): Number of points to scan.
        jump_thresh (float): Threshold for the jump in the value function.

    Returns:
        tuple:

        - grad_before_on_same_value (float): The gradient of the previous point on
            the same value function.
        - is_before_on_same_value (int): Indicator for whether we have found a
            previous point on the same value function.

    """
    (
        grad_before_on_same_value,
        idx_point_before_on_same_value,
    ) = back_and_forward_scan_wrapper(
        endog_grid_to_calculate_gradient=endog_grid_j,
        value_to_calculate_gradient=value_j,
        endog_grid_to_scan_from=endog_grid[idx_to_scan_from],
        policy_to_scan_from=policy[idx_to_scan_from],
        endog_grid=endog_grid,
        value=value,
        idx_to_scan_from=idx_to_scan_from,
        policy=policy,
        n_points_to_scan=n_points_to_scan,
        jump_thresh=jump_thresh,
        direction="backward",
    )

    return grad_before_on_same_value, idx_point_before_on_same_value


def back_and_forward_scan_wrapper(
    endog_grid_to_calculate_gradient: float | jnp.ndarray,
    value_to_calculate_gradient: float | jnp.ndarray,
    endog_grid_to_scan_from: float | jnp.ndarray,
    policy_to_scan_from: float | jnp.ndarray,
    endog_grid: float | jnp.ndarray,
    value,
    policy,
    idx_to_scan_from,
    n_points_to_scan,
    jump_thresh,
    direction,
):
    """Wrapper function to execute the backwards and forward scan.

    Args:
        endog_grid_to_calculate_gradient (float): The endogenous grid point to calculate
            the gradient from.
        value_to_calculate_gradient (float): The value function point to calculate the
            gradient from.
        endog_grid_to_scan_from (float): The endogenous grid point to scan from. We want
            to find the grid point which is on the same value function segment as the
            point we scan from.
        policy_to_scan_from (float): The policy function point to scan from. We want to
            find the grid point which is on the same value function segment as the point
            we scan from.
        idx_to_scan_from (int): Index of the point we want to scan from. This should
            be the current point we inspect.
        endog_grid (np.ndarray): 1d array of shape (n_grid_wealth,) containing the
            unrefined endogenous wealth grid.
        value (np.ndarray): 1d array of shape (n_grid_wealth,) containing the
            unrefined value correspondence.
        policy (np.ndarray): 1d array of shape (n_grid_wealth,) containing the
            unrefined policy correspondence.
        n_points_to_scan (int): Number of points to scan.
        jump_thresh (float): Threshold for the jump in the value function.
        direction (str): The direction of the scan. Either 'forward' or 'backward'.

    Returns:
        tuple:

        - grad_we_search_for (float): The gradient we search for.
        - idx_on_same_value (int): The index of the point on the same value function
            segment as the point we scan from.

    """
    # Prepare body function by partialing in, everything except carry and counter
    partial_body = partial(
        back_and_forward_scan_body,
        endog_grid_to_calculate_gradient=endog_grid_to_calculate_gradient,
        value_to_calculate_gradient=value_to_calculate_gradient,
        endog_grid_to_scan_from=endog_grid_to_scan_from,
        policy_to_scan_from=policy_to_scan_from,
        endog_grid=endog_grid,
        value=value,
        policy=policy,
        jump_thresh=jump_thresh,
        direction=direction,
    )

    if direction == "forward":
        max_index = idx_to_scan_from + n_points_to_scan

        def cond_func(carry):
            (
                is_on_same_value,
                idx_on_same_value,
                grad_we_search_for,
                current_index,
            ) = carry
            return (
                ~is_on_same_value
                & (current_index < max_index)
                & (current_index < len(endog_grid))
            )

        start_index = idx_to_scan_from + 1

    elif direction == "backward":
        min_index = idx_to_scan_from - n_points_to_scan

        def cond_func(carry):
            (
                is_on_same_value,
                idx_on_same_value,
                grad_we_search_for,
                current_index,
            ) = carry
            return (
                ~is_on_same_value & (current_index > min_index) & (current_index >= 0)
            )

        start_index = idx_to_scan_from - 1
    else:
        raise ValueError("Direction must be either 'forward' or 'backward'.")

    # Initialize starting values
    is_on_same_value = False
    idx_on_same_value = 0
    grad_we_search_for = 0.0

    # These values will be updated each iteration.
    carry_to_update = (
        is_on_same_value,
        idx_on_same_value,
        grad_we_search_for,
        start_index,
    )

    # Execute scan function. The result is the final carry value.
    final_carry = jax.lax.while_loop(
        cond_fun=cond_func, body_fun=partial_body, init_val=carry_to_update
    )

    # Read out final carry.
    (
        is_on_same_value,
        idx_on_same_value,
        grad_we_search_for,
        start_index,
    ) = final_carry

    return (
        grad_we_search_for,
        idx_on_same_value,
    )


def back_and_forward_scan_body(
    carry,
    endog_grid_to_calculate_gradient,
    value_to_calculate_gradient,
    endog_grid_to_scan_from,
    policy_to_scan_from,
    endog_grid,
    value,
    policy,
    jump_thresh,
    direction,
):
    """The scan body to be executed at each iteration of the backwards and forward scan
    function.

    Args:
        carry (tuple): The carry value passed from the previous iteration. This is a
            tuple containing the variables that are updated in each iteration.
        endog_grid_to_calculate_gradient (float): The endogenous grid point to calculate
            the gradient from.
        value_to_calculate_gradient (float): The value function point to calculate the
            gradient from.
        endog_grid_to_scan_from (float): The endogenous grid point to scan from. We want
            to find the grid point which is on the same value function segment as the
            point we scan from.
        policy_to_scan_from (float): The policy function point to scan from. We want to
            find the grid point which is on the same value function segment as the point
            we scan from.
        endog_grid (np.ndarray): 1d array of shape (n_grid_wealth,) containing the
            unrefined endogenous wealth grid.
        value (np.ndarray): 1d array of shape (n_grid_wealth,) containing the
            unrefined value correspondence.
        policy (np.ndarray): 1d array of shape (n_grid_wealth,) containing the
            unrefined policy correspondence.
        jump_thresh (float): Threshold for the jump in the value function.
        direction (str): The direction of the scan. Either 'forward' or 'backward'.

    Returns:
        tuple:

        - carry (tuple): The updated carry value passed to the next iteration.
        - None: Dummy value to be returned.

    """
    (
        _,
        _,
        _,
        current_index_to_scan,
    ) = carry

    is_not_on_same_value = create_indicator_if_value_function_is_switched(
        endog_grid_1=endog_grid_to_scan_from,
        policy_1=policy_to_scan_from,
        endog_grid_2=endog_grid[current_index_to_scan],
        policy_2=policy[current_index_to_scan],
        jump_thresh=jump_thresh,
    )
    is_on_same_value = ~is_not_on_same_value

    grad_to_idx_to_scan = calc_gradient(
        x1=endog_grid_to_calculate_gradient,
        y1=value_to_calculate_gradient,
        x2=endog_grid[current_index_to_scan],
        y2=value[current_index_to_scan],
    )

    # Update if we found the point we search for
    idx_on_same_value = current_index_to_scan * is_on_same_value

    # Update the first time a new point is found
    grad_we_search_for = grad_to_idx_to_scan * is_on_same_value
    if direction == "forward":
        current_index_to_scan += 1
    elif direction == "backward":
        current_index_to_scan -= 1

    return (
        is_on_same_value,
        idx_on_same_value,
        grad_we_search_for,
        current_index_to_scan,
    )
