# Multimodal BCI Glasses

## references
## 1) Local Setup:

 * clone this repository on your local system with `git clone https://github.com/alex-karim-eladl/Multimodal-BCI-Glasses.git`


## 2) Create and Activate Virtual Environment:

## General Python Tips

1. Install `pyenv` and use virtual environments.
2. Use `.venv` for all virtual environment names (ensure you install pyenv first, then use venv to create the virtual environment)

```
eval "$(pyenv init --path)"
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -) "
pyenv install 3.10.4
pyenv local 3.10.4 # in root

# in specific folder
python3 -m venv .venv && source .venv/bin/activate

# Confirm with:
which python
# Should be Multimodal-BCI-Glasses/.venv/bin/python
pip install -r requirements.txt
```
### Using Venv:
 * run `python3.10 -m venv .venv && source .venv/bin/activate`

### Using Conda:
 * run `conda create -n [venv_name] python=3.10 pip`
 * run `conda activate [venv_name]`

 * Note: If `which python3` still returns local python from within the conda environment you should execute `conda deactivate` twice to deactivate (base) and rerun `conda activate [venv_name]`. This will set the correct python path.


## 3) Install Dependencies:
 * run `pip install -r requirements.txt`
 * run `python3.10 -m ipykernel install --user --name=.venv`


## 4) Run experiment:

 `python run.py [subject_id] [trial_id]`

 * [subject_id] should be subject name or some other unique identifier
 * [trial_id] should uniquely identify the trial if the same subject does the experiment more than once (e.g. 1,2,3,...)

## Modifying Timings:
edit lines 14-20 of run.py to change length of breaks, number of math/rest sets per part, number of math problems per math period, and length of each math problem

## 5) Data:
* data will be saved in data/[subject_id] as [trial_id]L.csv, [trial_id]R.csv, and [trial_id]F.csv from left, right, and forehead devices respectively
* timing information is saved after each part in data/[subject_id]/[trial_id].pkl
* if a file already exists the data will be appended


## Test Recording:
 * `python record.py -a [MAC address] -u [subject_id] -r [trial_id]`
 * ex: `python record.py -a 3E8847D9-3D52-7A2F-B913-2FBD9F63ECF3 -u nat -r 4`

## Test Connection:
 * `python test_connection.py` attempts to connect to all three devices

## Blueberry Device MAC Addresses:
to see MAC addresses run `python ble_devices.py` and edit lines 91-96 of record.py to reflect what they are on your device

Karim's Laptop:
 * left side of glasses: 3E8847D9-3D52-7A2F-B913-2FBD9F63ECF3
 * right side of glasses: 737E96F8-345B-0B18-66D0-B72FF7301397
 * headband: CAC8BEEB-CDC7-ED2F-04D7-48A3A88C2614

Lab Mac:
 * 95C2D4B6-C213-2AE0-7E83-C19F17DD5A2E: blueberry-0B33
 * 59169A01-1E11-5D0E-0BEF-E62E4CC1D877: blueberry-0B27
 * FCDACC9F-A3CE-F013-1DA3-00CB438003B0: blueberry-0B28


# Multimodal-BCI-Glasses
 ### Local Setup

> Prerequisite: Google Cloud SDK (Used for Data Lake storage)

- <https://cloud.google.com/sdk/docs/install#deb>

Use a virtual environment per project (one for ingest, one for mindstream). Activate it every time.

```bash
python3 -m venv .venv
source .venv/bin/activate
```

You can also use conda:

```bash
conda create -n ingest -y python=3.9
conda activate ingest
```

Install required dependencies with:

```bash
cd ingest
pip install -e ".[dev]"
```

Install in development mode with:

```bash
python setup.py develop
```

Install the pre-commit push hooks to ensure code quality.

```bash
pre-commit install --hook-type pre-push

# Run manually
pre-commit run --all-files
```
