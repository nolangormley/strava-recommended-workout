import pandas as pd
import numpy as np

def calculate_ctl_atl(loads, ctl_decay=42, atl_decay=7):
    """
    Calculates the Chronic Training Load (CTL) and Acute Training Load (ATL)
    using Exponentially Weighted Moving Average (EWMA) over a list of daily loads.
    Returns two lists (ctl, atl) of the same length as `loads`.
    """
    ctl = [0.0]
    atl = [0.0]
    for i in range(len(loads)):
        daily_strain = loads[i]
        curr_ctl = ctl[-1] + (daily_strain - ctl[-1]) / ctl_decay
        curr_atl = atl[-1] + (daily_strain - atl[-1]) / atl_decay
        ctl.append(curr_ctl)
        atl.append(curr_atl)
    
    return ctl[1:], atl[1:]

def calculate_tsb(ctl, atl):
    """
    Calculates Training Stress Balance (TSB) from CTL and ATL.
    Also known as 'Form'.
    """
    if isinstance(ctl, list) and isinstance(atl, list):
        return [c - a for c, a in zip(ctl, atl)]
    return ctl - atl

def calculate_acwr(ctl, atl):
    """
    Calculates Acute-to-Chronic Workload Ratio (ACWR) from CTL and ATL.
    Also known as ATL/CTL. Returns 0 if CTL is 0.
    """
    if isinstance(ctl, pd.Series) and isinstance(atl, pd.Series):
         return np.where(ctl > 0, atl / ctl, 0.0)
    if isinstance(ctl, list) and isinstance(atl, list):
        return [(a / c if c > 0 else 0.0) for c, a in zip(ctl, atl)]
    return atl / ctl if ctl > 0 else 0.0

def get_target_category(tsb):
    """
    Maps TSB to a target training category.
    """
    if tsb > 5:
        return "Anaerobic"
    elif -10 <= tsb <= 5:
        return "VO2Max"
    elif -30 <= tsb < -10:
        return "Aerobic"
    else:
        return "Recovery"

def calculate_vo2max_from_df(df, hr_max=190, hr_rest=60):
    """
    Given a DataFrame with 'speed' and 'hr' columns, estimates VO2 max.
    Returns a DataFrame with 'vo2_max_est' calculated.
    """
    df_calc = df.copy()
    df_calc['vo2_current'] = (0.2 * (df_calc['speed'] * 60)) + 3.5
    df_calc['pct_hrr'] = (df_calc['hr'] - hr_rest) / (hr_max - hr_rest)
    # avoid division by zero or negative
    df_calc = df_calc[df_calc['pct_hrr'] > 0]
    
    if df_calc.empty:
        return None

    df_calc['vo2_max_est'] = ((df_calc['vo2_current'] - 3.5) / df_calc['pct_hrr']) + 3.5
    return df_calc

def clean_val(val, decimals=1):
    """
    Cleans a value (usually from pandas) for JSON serialization, handling NaNs
    and rounding it.
    """
    if pd.isna(val) or val is None:
        return None
    return round(float(val), decimals)
