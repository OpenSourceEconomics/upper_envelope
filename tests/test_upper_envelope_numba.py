from functools import partial
from pathlib import Path
from typing import Callable

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal as aaae
from upper_envelope.upper_envelope_numba import fast_upper_envelope
from upper_envelope.upper_envelope_numba import fast_upper_envelope_wrapper

from tests.utils.fast_upper_envelope_org import fast_upper_envelope_wrapper_org
from tests.utils.upper_envelope_fedor import upper_envelope

# Obtain the test directory of the package.
TEST_DIR = Path(__file__).parent

# Directory with additional resources for the testing harness
TEST_RESOURCES_DIR = TEST_DIR / "resources"


def calc_current_value(
    consumption: np.ndarray,
    next_period_value: np.ndarray,
    choice: int,
    discount_factor: float,
    compute_utility: Callable,
) -> np.ndarray:
    """Compute the agent's current value.

    We only support the standard value function, where the current utility and
    the discounted next period value have a sum format.

    Args:
        consumption (np.ndarray): Level of the agent's consumption.
            Array of shape (n_quad_stochastic * n_grid_wealth,).
        next_period_value (np.ndarray): The value in the next period.
        choice (int): The current discrete choice.
        compute_utility (callable): User-defined function to compute the agent's
            utility. The input ``params``` is already partialled in.
        discount_factor (float): The discount factor.

    Returns:
        np.ndarray: The current value.

    """
    utility = compute_utility(consumption, choice)
    value = utility + discount_factor * next_period_value

    return value


def utility_crra(consumption: np.array, choice: int, params: dict) -> np.array:
    """Computes the agent's current utility based on a CRRA utility function.

    Args:
        consumption (jnp.array): Level of the agent's consumption.
            Array of shape (i) (n_quad_stochastic * n_grid_wealth,)
            when called by :func:`~dcgm.call_egm_step.map_exog_to_endog_grid`
            and :func:`~dcgm.call_egm_step.get_next_period_value`, or
            (ii) of shape (n_grid_wealth,) when called by
            :func:`~dcgm.call_egm_step.get_current_period_value`.
        choice (int): Choice of the agent, e.g. 0 = "retirement", 1 = "working".
        params_dict (dict): Dictionary containing model parameters.
            Relevant here is the CRRA coefficient theta.

    Returns:
        utility (jnp.array): Agent's utility . Array of shape
            (n_quad_stochastic * n_grid_wealth,) or (n_grid_wealth,).

    """
    utility_consumption = (consumption ** (1 - params["rho"]) - 1) / (1 - params["rho"])

    utility = utility_consumption - (1 - choice) * params["delta"]

    return utility


@pytest.fixture
def setup_model():
    max_wealth = 50
    n_grid_wealth = 500
    exog_savings_grid = np.linspace(0, max_wealth, n_grid_wealth)

    params = {}
    params["beta"] = 0.95  # discount_factor
    params["rho"] = 1.95
    params["delta"] = 0.35

    state_choice_vec = {"choice": 0}

    compute_utility = partial(utility_crra, params=params)
    compute_value = partial(
        calc_current_value,
        discount_factor=params["beta"],
        compute_utility=compute_utility,
    )

    return params, state_choice_vec, exog_savings_grid, compute_utility, compute_value


@pytest.mark.parametrize("period", [2, 4, 9, 10, 18])
def test_fast_upper_envelope_wrapper(period, setup_model):
    value_egm = np.genfromtxt(
        TEST_RESOURCES_DIR / f"upper_envelope_period_tests/val{period}.csv",
        delimiter=",",
    )
    policy_egm = np.genfromtxt(
        TEST_RESOURCES_DIR / f"upper_envelope_period_tests/pol{period}.csv",
        delimiter=",",
    )
    value_refined_fedor = np.genfromtxt(
        TEST_RESOURCES_DIR / f"upper_envelope_period_tests/expec_val{period}.csv",
        delimiter=",",
    )
    policy_refined_fedor = np.genfromtxt(
        TEST_RESOURCES_DIR / f"upper_envelope_period_tests/expec_pol{period}.csv",
        delimiter=",",
    )
    policy_expected = policy_refined_fedor[
        :, ~np.isnan(policy_refined_fedor).any(axis=0)
    ]
    value_expected = value_refined_fedor[
        :,
        ~np.isnan(value_refined_fedor).any(axis=0),
    ]

    (
        _params,
        state_choice_vec,
        exog_savings_grid,
        _compute_utility,
        compute_value,
    ) = setup_model

    endog_grid_refined, policy_refined, value_refined = fast_upper_envelope_wrapper(
        endog_grid=policy_egm[0, 1:],
        policy=policy_egm[1, 1:],
        value=value_egm[1, 1:],
        expected_value_zero_savings=value_egm[1, 0],
        exog_grid=np.append(0, exog_savings_grid),
        choice=state_choice_vec["choice"],
        compute_value=compute_value,
    )
    endog_grid_got = endog_grid_refined[~np.isnan(endog_grid_refined)]
    policy_got = policy_refined[~np.isnan(policy_refined)]
    value_got = value_refined[~np.isnan(value_refined)]

    aaae(endog_grid_got, policy_expected[0])
    aaae(policy_got, policy_expected[1])
    value_expected_interp = np.interp(
        endog_grid_got, value_expected[0], value_expected[1]
    )
    aaae(value_got, value_expected_interp)


def test_fast_upper_envelope_against_org_fues(setup_model):
    policy_egm = np.genfromtxt(
        TEST_RESOURCES_DIR / "upper_envelope_period_tests/pol10.csv", delimiter=","
    )
    value_egm = np.genfromtxt(
        TEST_RESOURCES_DIR / "upper_envelope_period_tests/val10.csv", delimiter=","
    )

    (
        _params,
        state_choice_vec,
        exog_savings_grid,
        compute_utility,
        _compute_value,
    ) = setup_model

    endog_grid_refined, value_refined, policy_refined = fast_upper_envelope(
        endog_grid=policy_egm[0],
        value=value_egm[1],
        policy=policy_egm[1],
        exog_grid=np.append(0, exog_savings_grid),
    )

    endog_grid_org, policy_org, value_org = fast_upper_envelope_wrapper_org(
        endog_grid=policy_egm[0],
        policy=policy_egm[1],
        value=value_egm[1],
        exog_grid=exog_savings_grid,
        choice=state_choice_vec["choice"],
        compute_utility=compute_utility,
    )

    endog_grid_expected = endog_grid_org[~np.isnan(endog_grid_org)]
    policy_expected = policy_org[~np.isnan(policy_org)]
    value_expected = value_org[~np.isnan(value_org)]

    assert np.all(np.in1d(endog_grid_expected, endog_grid_refined))
    assert np.all(np.in1d(policy_expected, policy_refined))
    assert np.all(np.in1d(value_expected, value_refined))


@pytest.mark.parametrize("period", [2, 4, 10, 9, 18])
def test_fast_upper_envelope_against_fedor(period, setup_model):
    value_egm = np.genfromtxt(
        TEST_RESOURCES_DIR / f"upper_envelope_period_tests/val{period}.csv",
        delimiter=",",
    )
    policy_egm = np.genfromtxt(
        TEST_RESOURCES_DIR / f"upper_envelope_period_tests/pol{period}.csv",
        delimiter=",",
    )

    (
        params,
        state_choice_vec,
        exog_savings_grid,
        compute_utility,
        compute_value,
    ) = setup_model

    _policy_fedor, _value_fedor = upper_envelope(
        policy=policy_egm,
        value=value_egm,
        exog_grid=exog_savings_grid,
        state_choice_vec=state_choice_vec,
        params=params,
        compute_utility=compute_utility,
    )
    policy_expected = _policy_fedor[:, ~np.isnan(_policy_fedor).any(axis=0)]
    value_expected = _value_fedor[
        :,
        ~np.isnan(_value_fedor).any(axis=0),
    ]

    _endog_grid_fues, _policy_fues, _value_fues = fast_upper_envelope_wrapper(
        endog_grid=policy_egm[0, 1:],
        policy=policy_egm[1, 1:],
        value=value_egm[1, 1:],
        expected_value_zero_savings=value_egm[1, 0],
        exog_grid=np.append(0, exog_savings_grid),
        choice=state_choice_vec["choice"],
        compute_value=compute_value,
    )
    endog_grid_got = _endog_grid_fues[~np.isnan(_endog_grid_fues)]
    policy_got = _policy_fues[~np.isnan(_policy_fues)]
    value_got = _value_fues[~np.isnan(_value_fues)]

    aaae(endog_grid_got, policy_expected[0])
    aaae(policy_got, policy_expected[1])
    value_expected_interp = np.interp(
        endog_grid_got, value_expected[0], value_expected[1]
    )
    aaae(value_got, value_expected_interp)
