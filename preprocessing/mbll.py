"""
Author:			Alex Karim El Adl
Project:		Multimodal-BCI-Glasses
File:			/preprocessing/mbll.py

Description:  Custom implementation of Modified Beer-Lambert Law and MNE wrapper
    to convert NIRS light intensity measurements to hemoglobin concentrations
"""

import mne
import numpy as np
import pandas as pd
from typing import Dict, List
from mne.preprocessing import nirs
from mne_nirs.channels import get_long_channels
from mne_nirs.signal_enhancement import (
    enhance_negative_correlation,
    short_channel_regression,
)

from config import *


def dpf(age: int, wl: int) -> float:
    """
    calculates differential pathlength factor using wavelength specific formulas
    equation from [DOI: 10.1117/1.JBO.18.10.105004]
    """
    a = 223.3
    b = 0.05624
    c = 0.8493
    d = -5.723e-7
    e = 0.001245
    f = -0.9025

    dpf = a + b * age**c + d * wl**3 + e * wl**2 + f * wl
    return dpf


def optical_density(df: pd.DataFrame, devices=NIRS_DEVICES):
    """
    calculates optical density changes based on measured light intensity
    and applies short channel regression to remove extracortical components
    """
    od_df = df.copy()
    for dev in devices:
        for wl in NIRS_WAVELEN:
            for opt in ['10', '27']:
                I = df.loc[:, (dev, f'{wl}nm{opt}mm')].to_numpy() + 1  # raw light intensity values
                od_df.loc[:, (dev, f'{wl}nm{opt}mm')] = -1 * np.log(I / I.mean())

            # short channel regression according to
            # Scholkmann, Metz, and Wolf 2014 https://www.zora.uzh.ch/id/eprint/105754/4/ZORA105754.pdf
            factor = 1.000000001  # scales adjustment
            od_long = od_df.loc[:, (dev, f'{wl}nm27mm')].to_numpy()
            od_short = od_df.loc[:, (dev, f'{wl}nm10mm')].to_numpy()

            alpha = (np.dot(od_short, od_long) / np.dot(od_short, od_short)) * factor
            od_df.loc[:, (dev, f'{wl}nm')] = od_long * factor - od_short * factor * alpha

    return od_df


def hb_concentration(od_df: pd.DataFrame, age: int, wls: List[str], return_all=False, coeff='my', devices=NIRS_DEVICES):
    """
    calculates HbO and HbR concentration changes using Modified Beer-Lambert Law
    """
    hb_df = od_df.copy()

    # extinction coefficients for HbO and HbR (respectively) at each wavelength
    E = np.array([EXTINCTIONS[coeff][wl] for wl in wls])
    Et = np.transpose(E)
    EtE = np.dot(Et, E)
    extinction = np.dot(np.linalg.inv(EtE), Et)

    for dev in devices:
        # calculate hbo/r for corrected dOD and each optode seperately
        for opt in ['', '10', '27']:
            sds = opt + 'mm' if opt != '' else opt
            dist = 1.0 if opt == '10' else 2.7
            density = np.array([od_df.loc[:, (dev, f'{wl}nm{sds}')].to_numpy() / (dist * dpf(age, wl)) for wl in wls])

            dC = np.dot(extinction, density)  # change in concentrations
            hb_df.loc[:, (dev, f'hbo{opt}')] = dC[0]
            hb_df.loc[:, (dev, f'hbr{opt}')] = dC[1]

    drop_cols = []
    for col in hb_df.columns.to_list():
        if 'nm' in col[1]:
            drop_cols.append(col)
        if not return_all and ('10' in col[1] or '27' in col[1]):
            drop_cols.append(col)

    return hb_df.drop(columns=drop_cols)


def mbll_mne(
    mne_raw: mne.io.Raw, 
    sci_thresh=None,
    regress_short=True,
    tddr=True, 
    neg_corr=True, 
    drop_short=True
):# -> tuple[mne.io.Raw, Dict[str, float]]:
    """
    Applies modified beer-lambert law (MBLL) to convert measured light intensity
    to hemoglobin concentrations and corrects major extracerebral artifacts
    Args:
        raw (mne.Raw): part data
        regress (bool, optional): apply short ch regression. Defaults to True.
        tddr (bool, optional): appl temporal derivative dist repair. Defaults to True.
        enhance (bool, optional): negative corr enhancment. Defaults to True.
        drop_short (bool, optional): only return long separation channels. Defaults to True.
    """
    od = nirs.optical_density(mne_raw)
    # sci = nirs.scalp_coupling_index(od)

    # od.info["bads"] = list(np.compress(od.ch_names, sci < sci_thresh))
    # od.interpolate_bads()

    if regress_short:
        od = short_channel_regression(od)
    if tddr:
        od = nirs.temporal_derivative_distribution_repair(od)

    hb = nirs.beer_lambert_law(od)

    hb = hb.rename_channels(HB_CH_MAP)
    
    if drop_short:
        hb = get_long_channels(hb)

    if neg_corr:
        hb = enhance_negative_correlation(hb)
    
    return hb#, dict(zip(hb.ch_names, sci[::2]))
