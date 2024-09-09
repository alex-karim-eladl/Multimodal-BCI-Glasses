from pathlib import Path

# dats directories
DATA_DIR = Path(__file__).parent / 'data'
RAW_DIR = DATA_DIR / 'raw'
PROC_DIR = DATA_DIR / 'processed'
MNE_DIR = DATA_DIR / 'mne'
FEAT_DIR = DATA_DIR / 'features'

SUBJECTS = [
    'sven',
    'zoey',
    'david',
    'daniel',
    'khiem',
    'michelle',
    'nader',
    'sam',
    'isabelle',
    'nina', 
    'harry', 
    'tina', 
    'claire', 
    'natasha', 
    'samuel'
]

AGES = {
    'claire': 21,
    'daniel': 22,
    'david': 24,
    'harry': 21,
    'isabelle': 22,
    'khiem': 25,
    'michelle': 22,
    'nader': 21,
    'natasha': 22,
    'nina': 24,
    'sam': 28,
    'samuel': 22,
    'tina': 23,
    'sven': 24,
    'zoey': 19,
}


TASK_LEN = 28.0
REGULAR_PART_LEN = 560.0
RANDOM_PART_LEN = 840.0


EEG_FS = 250.0
EEG_POS = ['T7', 'T8', 'Pz']
EEG_CH = ['CH1', 'CH2', 'CH3']

IMU_FS = 125.0
IMU_CH = ['GYROX', 'GYROY', 'GYROZ', 'ACCX', 'ACCY', 'ACCZ']
IMU_CH_MAP = {  # to conform with snirf standard
    'ACCX': 'ACCEL_X',
    'ACCY': 'ACCEL_Y',
    'ACCZ': 'ACCEL_Z',
    'GYROX': 'GYRO_X',
    'GYROY': 'GYRO_Y',
    'GYROZ': 'GYRO_Z',
}

NIRS_FS = 25.0
NIRS_DEVICES = ['f7', 'f8', 'fp']
NIRS_OPTODES = ['10', '27']  # mm separation btwn detector and emmitter
NIRS_WAVELEN = ['740', '850', '940']  # emitted (nm)
NIRS_DEV_CH = [f'{wl}nm{opt}mm' for wl in NIRS_WAVELEN for opt in NIRS_OPTODES]  # fnirs channels
NIRS_CH = [
    f'{wl}nm{opt}mm_{dev}' for wl in NIRS_WAVELEN for opt in NIRS_OPTODES for dev in NIRS_DEVICES
]  # fnirs channels
NIRS_SOURCE_POS = {'f7': [80, 60, 0], 'f8': [-80, 60, 0], 'fp': [-10, 100, 0]}
NIRS_DETECTOR_POS = {
    'f7': [[80, 50, 0], [80, 33, 0]],  # S1_D1, S1_D2
    'f8': [[-80, 50, 0], [-80, 33, 0]],  # S2_D3, S2_D4
    'fp': [[0, 100, 0], [17, 100, 0]],  # S3_D5, S3_D6
}


HB_CH = [f'{hb}_{dev}' for dev in NIRS_DEVICES for hb in ['hbo', 'hbr']]
HB_CH_MAP = {  # custom channel labels from mne defaults
    'S1_D1 hbo': 'hbo_f7_short',
    'S1_D2 hbo': 'hbo_f7',
    'S1_D1 hbr': 'hbr_f7_short',
    'S1_D2 hbr': 'hbr_f7',
    'S2_D3 hbo': 'hbo_f8_short',
    'S2_D4 hbo': 'hbo_f8',
    'S2_D3 hbr': 'hbr_f8_short',
    'S2_D4 hbr': 'hbr_f8',
    'S3_D5 hbo': 'hbo_fp_short',
    'S3_D6 hbo': 'hbo_fp',
    'S3_D5 hbr': 'hbr_fp_short',
    'S3_D6 hbr': 'hbr_fp',
}


# experiment structure
TIMINGS = ['regular', 'random']
STIMULI = ['visual', 'auditory', 'imagined']
EVENTS = {'math': 1, 'rest': 2}
ROUNDS = 10
TASKS = ['math', 'rest']

EXP_PARTS = [f'{timing}_{stimulus}' for timing in TIMINGS for stimulus in STIMULI]

# 'wray': data from S. Wray et al., 1988
# 'cope': data from M. Cope, 1991
# 'gratzer': data from W.B. Gratzer and K. Kollias compiled by S. Prahl (https://omlc.org/spectra/hemoglobin/summary.html)
# 'moaveni': data from M.K. Moaveni and J.M. Schmitt compiled by S. Prahl (https://omlc.org/spectra/hemoglobin/moaveni.html)
# 'takatani': data from S. Takatani and M.D. Graham compiled by S. Prahl (https://omlc.org/spectra/hemoglobin/takatani.html)
EXTINCTIONS = {  # unit: 1/(cm*M)
    'cope': {740: [492.0, 1341.1], 850: [1159.6, 786.1], 940: [1352.0, 787.4]},
    'gratzer': {740: [446, 1115.88], 850: [1058, 691.32], 940: [1214, 693.44]},
    'moaveni': {740: [440, 1520], 850: [1060, 800], 940: [1200, 800]},
    'takatani': {740: [480, 1616], 850: [1068, 820], 940: [1212, 756]},
    'wray': {740: [561, 1349], 850: [1097, 781], 940: [1258, 778]},
    'my': {740: [450, 1500], 850: [750, 1100], 940: [1350, 900]},
}
# 740.0,446.0,1115.88
# 850.0,1058.0,691.32
# 940.0,1214.0,693.44


EEG_BANDS = {
    'delta': (0.5, 4),
    'theta': (4, 8),
    'alpha': (8, 12),
    'sigma': (12, 16),
    'beta1': (16, 22),
    'beta2': (22, 30),
    # 'gamma': (30, 40),
}