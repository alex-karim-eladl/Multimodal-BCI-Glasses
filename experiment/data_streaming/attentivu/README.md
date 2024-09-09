# AttentivU Streamer
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

This script provides basic cross-platform functionality to pull data from the AttentivU glasses. Note that as new versions of the glasses are made, the notification handler in this script will need to be updated with any changes made to the structure of the BLE packets. Any changes to UUIDs will need to be updated as well. The code style is black, please keep it this way.

Note: The Streamer **does not currently support Windows** due to how the asyncio event loop is handled. Linux 20.04 and MacOS Big Sur are supported. Python version must be at least 3.9.0.

## Installation
This script only uses bleak, a platform-agnostic python implementation of the Bluetooth stack, and numpy.

```
pip install -r requirements.txt
```

## Usage
If running on MacOS, make sure your terminal emulator has access to Bluetooth in Security & Privacy settings. Otherwise, the process can get [aborted](https://github.com/hbldh/bleak/issues/438).

```
python3 Streamer.py

# MacOS 12 may require additional env flag to ensure that the python compile version number is not used
SYSTEM_VERSION_COMPAT=0 python3 Streamer.py
```

When the script is started it will automatically connect to the glasses and immediately begin streaming the data to a local csv file. To stop the streaming, press `Control+C`.

## Architecture
### File Structure
Each run will generate a folder inside the data directory. It will either be named the date and time that run started ("mm-dd-yyyy-HHMMSS") or the name given by the user at runtime. Any timeseries data the streamer collects will be saved in separate CSV files labelled with the corresponding data type (ex. "exg.csv").

### CSV Structure
The CSV structure varies based on the type of data collected and the number of channels. Generally, the files have column names across the top, with a timestamp column occupying the leftmost spot and channels coming afterward. All files are comma-delimited. Note that two files that are similarly labelled might have different internal structure because channels can be enabled or disabled.

### Services.json
This file describes the set of services and characteristics the Connection class will try to connect to. As characteristics are added, the Connection class will look for functions that can handle them based on the service they're in. For example, currently both **exg** and **imu** characteristics are part of the **dat** service. A single handler, called **dat_handler**, will be used dynamically to process this data. When adding new services, make sure there's functionality in the Connection class to handle them or else it will break.

The general structure is as follows:
```
{
    "service1" : {
        "char1": "99999999-9999-9999-9999-999999999996",
        "char2": "99999999-9999-9999-9999-999999999997"

    }
    "service2" : {
        "char3": "99999999-9999-9999-9999-999999999998",
        "char4": "99999999-9999-9999-9999-999999999999"

    }
}
```

### Connection.py
This holds the Connection class that can be dropped into other implementations like ML pipelines, etc. It must be called from an event loop (see Streamer.py) and be passed a dictionary of services to subscribe too (see Services.json), a callback function to handle the parsed data (see Utils.py), and the number of samples to buffer.

Note that the number of samples to buffer does not determine the final size and is just a threshold. Therefore callback functions will need to be able to handle variable length arrays. If this is a problem, like in the case of certain ML algorithms, consider padding or truncating the final array.

## Troubleshooting
```
[1]    4900 abort      SYSTEM_VERSION_COMPAT=0 python3 Streamer.py
```
On MacOS, this most likely indicates that the terminal emulator has no access to Bluetooth. Fix it by adding your terminal emulator app to System Preferences -> Security & Privacy -> Privacy -> Bluetooth. Find a similar issue [here](https://github.com/hbldh/bleak/issues/438).
