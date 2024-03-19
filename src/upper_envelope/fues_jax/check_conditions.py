from jax import numpy as jnp
from upper_envelope.fues_jax.scan_funcs import backward_scan
from upper_envelope.fues_jax.scan_funcs import forward_scan
from upper_envelope.math_funcs import calc_gradient


def determine_cases_and_conduct_necessary_scans(
    point_to_inspect,
    points_j_and_k,
    value,
    policy,
    endog_grid,
    idx_to_scan_from,
    n_points_to_scan,
    last_point_was_intersect,
    is_final_point_on_grid,
    jump_thresh,
):
    """Determine cases and if the index to be scanned this iteration should be updated.

    This function is crucial for the optimality of the FUES. We want to have a clear
    documentation of how the cases determined here map into the into the situations
    on how the candidate solutions after solving the euler equation can look like.
    We need to do more work here!

    Args:
        point_to_inspect (tuple): Tuple containing the value, policy and endogenous grid
            of the point to be inspected.
        points_j_and_k (tuple): Tuple containing the value, policy, and endogenous grid
            point of the most recent point that lies on the upper envelope (j) and
            the point before (k).
        last_point_was_intersect (bool): Indicator if the last point was an
            intersection point.
        is_final_point_on_grid (bool): Indicator if this is the final point on the grid.
        jump_thresh (float): Jump detection threshold.

    Returns:
        tuple:

        - cases (tuple): Tuple containing the indicators for the different cases.
        - update_idx (bool): Indicator if the index should be updated.

    """
    value_k_and_j, policy_k_and_j, endog_grid_k_and_j = points_j_and_k
    value_to_inspect, _policy_to_inspect, endog_grid_to_inspect = point_to_inspect

    (
        grad_next_forward,
        idx_next_on_lower_curve,
    ) = forward_scan(
        value=value,
        policy=policy,
        endog_grid=endog_grid,
        endog_grid_j=endog_grid_k_and_j[1],
        policy_j=policy_k_and_j[1],
        idx_to_scan_from=idx_to_scan_from,
        n_points_to_scan=n_points_to_scan,
        jump_thresh=jump_thresh,
    )

    (
        grad_next_backward,
        idx_before_on_upper_curve,
    ) = backward_scan(
        value=value,
        policy=policy,
        endog_grid=endog_grid,
        value_j=value_k_and_j[1],
        endog_grid_j=endog_grid_k_and_j[1],
        idx_to_scan_from=idx_to_scan_from,
        n_points_to_scan=n_points_to_scan,
        jump_thresh=jump_thresh,
    )

    grad_before = calc_gradient(
        x1=endog_grid_k_and_j[1],
        y1=value_k_and_j[1],
        x2=endog_grid_k_and_j[0],
        y2=value_k_and_j[0],
    )

    # gradient with leading index to be checked
    grad_next = calc_gradient(
        x1=endog_grid_to_inspect,
        y1=value_to_inspect,
        x2=endog_grid_k_and_j[1],
        y2=value_k_and_j[1],
    )

    suboptimal_cond, does_the_value_func_switch = check_for_suboptimality(
        points_j_and_k=points_j_and_k,
        point_to_inspect=point_to_inspect,
        grad_next=grad_next,
        grad_next_forward=grad_next_forward,
        grad_before=grad_before,
        jump_thresh=jump_thresh,
    )

    next_point_past_intersect = (grad_before > grad_next) | (
        grad_next < grad_next_backward
    )
    point_j_past_intersect = grad_next > grad_next_backward

    # Generate cases. They are exclusive in ascending order, i.e. if 1 is true the
    # rest can't be and 2 can only be true if 1 isn't.
    # Start with checking if last iteration was case_5, and we need
    # to add another point to the refined grid.
    case_1 = last_point_was_intersect
    case_2 = is_final_point_on_grid & ~case_1
    case_3 = suboptimal_cond & ~case_1 & ~case_2
    case_4 = ~does_the_value_func_switch * ~case_1 * ~case_2 * ~case_3
    case_5 = next_point_past_intersect & ~case_1 & ~case_2 & ~case_3 & ~case_4
    case_6 = point_j_past_intersect & ~case_1 & ~case_2 & ~case_3 & ~case_4 & ~case_5

    in_case_134 = case_1 | case_3 | case_4
    update_idx = in_case_134 | (~in_case_134 & suboptimal_cond)

    return (
        (case_1, case_2, case_3, case_4, case_5, case_6),
        update_idx,
        idx_next_on_lower_curve,
        idx_before_on_upper_curve,
    )


def check_for_suboptimality(
    points_j_and_k,
    point_to_inspect,
    grad_next,
    grad_next_forward,
    grad_before,
    jump_thresh,
):
    """Check if current point is sub-optimal.

    Even if the function returns False, the point may still be suboptimal.
    That is iff, in the next iteration, we find that this point actually lies after a
    switching point.

    Here, we check if the point fulfills one of three conditions:
    1) Either the value function is decreasing with decreasing value function,
    2) the value function is not montone in consumption, or
    3) if the gradient of the index we inspect and the point j (the most recent point
        on the same choice-specific policy) is shallower than the gradient joining
        the i+1 and j. If True, delete the j'th point.

    If the point to inspect is the same as point j, this is always false and we
    switch the value function as well. Therefore, the third if is chosen.

    Args:
        points_j_and_k (tuple): Tuple containing the value, policym and endogenous grid
            point of the most recent point on the upper envelope (j) and the point
            before (k).
        point_to_inspect (tuple): Tuple containing the value, policy, and endogenous
            grid point of the point we inspect.
        grad_next (float): The gradient between the most recent point that lies on the
            upper envelope (j) and the point we inspect.
        grad_next_forward (float): The gradient between the next point on the same value
            function segment as j and the current point we inspect.
        grad_before (float): The gradient between the most recent point on the upper
            envelope (j) and the point before (k).
        jump_thresh (float): Jump detection threshold.

    Returns:
        tuple:

        - suboptimal_cond (bool): Indicator if the point is suboptimal.
        - does_the_value_func_switch (bool): Indicator if the value function switches.

    """
    value_k_and_j, policy_k_and_j, endog_grid_k_and_j = points_j_and_k
    value_to_inspect, policy_to_inspect, endog_grid_to_inspect = point_to_inspect

    does_the_value_func_switch = create_indicator_if_value_function_is_switched(
        endog_grid_1=endog_grid_k_and_j[1],
        policy_1=policy_k_and_j[1],
        endog_grid_2=endog_grid_to_inspect,
        policy_2=policy_to_inspect,
        jump_thresh=jump_thresh,
    )
    switch_value_func_and_steep_increase_after = (
        grad_next < grad_next_forward
    ) & does_the_value_func_switch

    decreasing_value = value_to_inspect < value_k_and_j[1]

    are_savings_non_monotone = check_for_non_monotone_savings(
        endog_grid_j=endog_grid_k_and_j[1],
        policy_j=policy_k_and_j[1],
        endog_grid_idx_to_inspect=endog_grid_to_inspect,
        policy_idx_to_inspect=policy_to_inspect,
    )

    # Aggregate the three cases
    suboptimal_cond = (
        switch_value_func_and_steep_increase_after
        | decreasing_value
        # Do we need the grad condition next?
        | (are_savings_non_monotone & (grad_next < grad_before))
    )

    return suboptimal_cond, does_the_value_func_switch


def check_for_non_monotone_savings(
    endog_grid_j, policy_j, endog_grid_idx_to_inspect, policy_idx_to_inspect
):
    """Check if savings are a non-monotone in wealth.

    Check the grid between the most recent point on the upper envelope (j)
    and the current point we inspect.

    Args:
        endog_grid_j (float): The endogenous grid point of the most recent point
            on the upper envelope.
        policy_j (float): The value of the policy function of the most recent point
            on the upper envelope.
        endog_grid_idx_to_inspect (float): The endogenous grid point of the point we
            check.
        policy_idx_to_inspect (float): The policy index of the point we check.

    Returns:
        non_monotone_policy (bool): Indicator if the policy is non-monotone in wealth
            between the most recent point on the upper envelope and the point we check.

    """
    exog_grid_j = endog_grid_j - policy_j
    exog_grid_idx_to_inspect = endog_grid_idx_to_inspect - policy_idx_to_inspect
    are_savings_non_monotone = exog_grid_idx_to_inspect < exog_grid_j

    return are_savings_non_monotone


def create_indicator_if_value_function_is_switched(
    endog_grid_1: float | jnp.ndarray,
    policy_1: float | jnp.ndarray,
    endog_grid_2: float | jnp.ndarray,
    policy_2: float | jnp.ndarray,
    jump_thresh: float | jnp.ndarray,
):
    """Create boolean to indicate whether value function switches between two points.

    Args:
        endog_grid_1 (float): The first endogenous wealth grid point.
        policy_1 (float): The policy function at the first endogenous wealth grid point.
        endog_grid_2 (float): The second endogenous wealth grid point.
        policy_2 (float): The policy function at the second endogenous wealth grid
            point.
        jump_thresh (float): Jump detection threshold.

    Returns:
        bool: Indicator if value function is switched.

    """
    exog_grid_1 = endog_grid_1 - policy_1
    exog_grid_2 = endog_grid_2 - policy_2
    gradient_exog_grid = calc_gradient(
        x1=endog_grid_1,
        y1=exog_grid_1,
        x2=endog_grid_2,
        y2=exog_grid_2,
    )
    gradient_exog_abs = jnp.abs(gradient_exog_grid)
    is_switched = gradient_exog_abs > jump_thresh

    return is_switched
