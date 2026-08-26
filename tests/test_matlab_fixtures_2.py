# test_matlab_fixtures_2.py
# Tests parameter recovery equivalence between MATLAB and Python code, based on a simulated dataset from `cfc_example_2.m` in original cfc repository (linked in README.md).


import pickle

import numpy as np
import pytest
import scipy.io as sio

from cfc_python.cfc_fit import cfc_fit


fixture_dir = "tests"

def load_matlab_fixture(grouped_data_path, cfc_struct_path):
    gd_mat = sio.loadmat(grouped_data_path, squeeze_me=True, struct_as_record=False)
    grouped_data = gd_mat['grouped_data']

    struct_mat = sio.loadmat(cfc_struct_path, squeeze_me=True, struct_as_record=False)
    cfc_struct_matlab = struct_mat['cfc_struct']  # adjust key if named differently

    return grouped_data, cfc_struct_matlab

# Prints MATLAB vs. Python values for `field`, failing the test if they diverge more than `tol`value.
# Use assert_close=False for quantities known to be non-identifiable (e.g. noise/boost trade-off at joint optimum). Default is assert_close=True
def compare(matlab_struct, python_struct, field, tol=1e-2, label=None, assert_close=True):

    label = label or field
    try:
        mval = np.atleast_1d(getattr(matlab_struct, field))
    except AttributeError:
        pytest.skip(f"MATLAB struct has no field '{field}'")
        return

    pval = np.atleast_1d(python_struct[field])
    assert mval.shape == pval.shape, (
        f"{label}: shape mismatch, MATLAB {mval.shape} vs Python {pval.shape}"
    )

    diff = np.abs(mval - pval)
    print(f"  {label}: MATLAB={mval}  Python={pval}  diff={diff}")

    if assert_close:
        assert np.all(diff < tol), (
            f"{label}: MATLAB={mval} Python={pval} diff={diff} exceeds tol={tol}"
        )


@pytest.fixture(scope="module")
def ex2_fit():
    grouped_data, matlab_struct = load_matlab_fixture(
        f"{fixture_dir}/exp2_grouped_data.mat",
        f"{fixture_dir}/exp2_cfc_struct.mat",
    )

    model_parameters = {
        'sens_noise': [1, 2],
        'sens_crit': [3, 4],
        'conf_noise': [5, 6],
        'conf_boost': [7, 8],
        'conf_bias': [0, 9],
    }

    python_struct = cfc_fit(
        grouped_data.astype(float),
        model_parameters=model_parameters,
        boost_init_list=[0.0, 0.2, 0.5, 0.8, 1.0],
        skip_efficiency=False,
        verbose=2,
    )

    with open(f"{fixture_dir}/ex2_python_struct.pkl", "wb") as f:
        pickle.dump(python_struct, f)

    return matlab_struct, python_struct


def test_ex2_sens_noise(ex2_fit):
    matlab_struct, python_struct = ex2_fit
    compare(matlab_struct, python_struct, 'sens_noise', tol=1e-2)


def test_ex2_sens_crit(ex2_fit):
    matlab_struct, python_struct = ex2_fit
    compare(matlab_struct, python_struct, 'sens_crit', tol=1e-2)


def test_ex2_efficiency(ex2_fit):
    matlab_struct, python_struct = ex2_fit
    compare(matlab_struct, python_struct, 'efficiency', tol=5e-2)


def test_ex2_loglike(ex2_fit):
    matlab_struct, python_struct = ex2_fit
    compare(matlab_struct, python_struct, 'loglike', tol=5.0)


def test_ex2_conf_noise(ex2_fit):
    matlab_struct, python_struct = ex2_fit
    compare(matlab_struct, python_struct, 'conf_noise', tol=1e-2, assert_close=False)


def test_ex2_conf_boost(ex2_fit):
    matlab_struct, python_struct = ex2_fit
    compare(matlab_struct, python_struct, 'conf_boost', tol=1e-2, assert_close=False)