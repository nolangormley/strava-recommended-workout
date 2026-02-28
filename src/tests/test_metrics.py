import pytest
import pandas as pd
import numpy as np
from src.metrics import calculate_ctl_atl, calculate_tsb, get_target_category, calculate_vo2max_from_df, clean_val, calculate_acwr

def test_calculate_ctl_atl():
    loads = [100.0, 50.0, 150.0]
    ctl, atl = calculate_ctl_atl(loads, ctl_decay=42, atl_decay=7)
    assert len(ctl) == 3
    assert len(atl) == 3
    # First load
    assert ctl[0] == 100.0 / 42.0
    assert atl[0] == 100.0 / 7.0

def test_calculate_tsb():
    ctl = [10.0, 20.0]
    atl = [15.0, 10.0]
    tsb = calculate_tsb(ctl, atl)
    assert tsb == [-5.0, 10.0]

def test_calculate_tsb_scalar():
    tsb = calculate_tsb(50.0, 70.0)
    assert tsb == -20.0

def test_calculate_acwr():
    # scalar division
    assert calculate_acwr(50.0, 70.0) == 1.4
    # zero ctl handles safely
    assert calculate_acwr(0.0, 50.0) == 0.0
    # list division
    acwr = calculate_acwr([50.0, 0.0], [70.0, 50.0])
    assert acwr == [1.4, 0.0]

def test_get_target_category():
    assert get_target_category(10) == "Anaerobic"
    assert get_target_category(0) == "VO2Max"
    assert get_target_category(-20) == "Aerobic"
    assert get_target_category(-40) == "Recovery"

def test_calculate_vo2max_from_df():
    data = {'speed': [2.0, 3.0], 'hr': [130, 170]}
    df = pd.DataFrame(data)
    result = calculate_vo2max_from_df(df, hr_max=190, hr_rest=60)
    assert 'vo2_max_est' in result.columns
    assert len(result) == 2

def test_calculate_vo2max_zero_hrr():
    # If HR is at or below rest, pct_hrr <= 0, which gets filtered
    data = {'speed': [2.0], 'hr': [60]}
    df = pd.DataFrame(data)
    result = calculate_vo2max_from_df(df, hr_max=190, hr_rest=60)
    assert result is None or result.empty

def test_clean_val():
    assert clean_val(10.123, 1) == 10.1
    assert clean_val(pd.NA) is None
    assert clean_val(np.nan) is None
    assert clean_val(None) is None
