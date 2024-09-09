"""
Author:			Alex Karim El Adl
Project:		Multimodal-BCI-Glasses
File:			/preprocessing/artifacts.py

Description: 
"""

import numpy as np
import mne
from config import *
from autoreject import AutoReject

def autoreject_eeg( subject: str,
                    n_interp = np.array([1, 2]),
                    consensus = np.linspace(0, 1.0, 11),
                    method = 'bayesian_optimization'):
    #! AutoReject EEG Epochs
    for part in EXP_PARTS:
        part_dir = PROC_DIR / subject / part
        
        eeg_epochs = mne.read_epochs(part_dir / 'mne/eeg_epo.fif')
        
        ar = AutoReject(n_interp, thresh_func=method, random_state=42)
        eeg_epochs_clean, reject_log = ar.fit_transform(eeg_epochs, return_log=True)
        eeg_epochs_clean.save(part_dir / 'mne/eeg_clean_epo.fif', overwrite=True)
