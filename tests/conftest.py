import glob
import os
import sys
from pathlib import Path

import jax
import pandas as pd
import pytest
import yaml

# Obtain the test directory of the package
TEST_DIR = Path(__file__).parent

# Directory with additional resources for the testing harness
REPLICATION_TEST_RESOURCES_DIR = TEST_DIR / "resources" / "replication_tests"

WEALTH_GRID_POINTS = 100

NUMBA_DISABLE_JIT = 1


# Add the utils directory to the path so that we can import helper functions.
sys.path.append(os.path.join(os.path.dirname(__file__), "utils"))


def pytest_sessionstart(session):  # noqa: ARG001
    jax.config.update("jax_enable_x64", val=True)


def pytest_configure(config):
    """Called after command line options have been parsed."""
    os.environ["NUMBA_DISABLE_JIT"] = "1"


def pytest_unconfigure(config):
    """Called before test process is exited."""
    os.environ.pop("NUMBA_DISABLE_JIT", None)


@pytest.hookimpl(tryfirst=True)
def pytest_sessionfinish(session, exitstatus):  # noqa: ARG001
    # Get the current working directory
    cwd = os.getcwd()

    # Search for .npy files that match the naming pattern
    pattern = os.path.join(cwd, "[endog_grid_, policy_, value_]*.npy")
    npy_files = glob.glob(pattern)

    # Delete the matching .npy files
    for file in npy_files:
        os.remove(file)


@pytest.fixture(scope="session")
def load_example_model():
    def load_options_and_params(model):
        """Return parameters and options of an example model."""
        params = pd.read_csv(
            REPLICATION_TEST_RESOURCES_DIR / f"{model}" / "params.csv",
            index_col=["category", "name"],
        )
        params = (
            params.reset_index()[["name", "value"]].set_index("name")["value"].to_dict()
        )
        options = yaml.safe_load(
            (REPLICATION_TEST_RESOURCES_DIR / f"{model}" / "options.yaml").read_text()
        )
        return params, options

    return load_options_and_params
