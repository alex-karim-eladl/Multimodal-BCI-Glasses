import argparse
import asyncio
import json
import os
import signal
import time
from datetime import datetime
from pathlib import Path

from blueberry import Blueberry

global bby, bby_task, save_file


def save_csv(data_1, data_2, data_3):
    ms_device = data_1['ms_device']
    ambient = data_1['ambient']
    led740nm10mm = data_1['led740nm10mm']
    led940nm10mm = data_1['led940nm10mm']
    led850nm10mm = data_1['led850nm10mm']
    led740nm27mm = data_1['led740nm27mm']
    led940nm27mm = data_1['led940nm27mm']
    led850nm27mm = data_1['led850nm27mm']

    timeNow = datetime.time()  # time.time()

    save_file.write(
        '{},{},{},{},{},{},{},{},{}\n'.format(
            timeNow,
            ms_device,
            ambient,
            led740nm10mm,
            led940nm10mm,
            led850nm10mm,
            led740nm27mm,
            led940nm27mm,
            led850nm27mm,
        )
    )

    ms_device = data_2['ms_device']
    ambient = data_2['ambient']
    led740nm10mm = data_2['led740nm10mm']
    led940nm10mm = data_2['led940nm10mm']
    led850nm10mm = data_2['led850nm10mm']
    led740nm27mm = data_2['led740nm27mm']
    led940nm27mm = data_2['led940nm27mm']
    led850nm27mm = data_2['led850nm27mm']

    save_file.write(
        '{},{},{},{},{},{},{},{},{}\n'.format(
            timeNow + 0.04,
            ms_device,
            ambient,
            led740nm10mm,
            led940nm10mm,
            led850nm10mm,
            led740nm27mm,
            led940nm27mm,
            led850nm27mm,
        )
    )

    ms_device = data_3['ms_device']
    ambient = data_3['ambient']
    led740nm10mm = data_3['led740nm10mm']
    led940nm10mm = data_3['led940nm10mm']
    led850nm10mm = data_3['led850nm10mm']
    led740nm27mm = data_3['led740nm27mm']
    led940nm27mm = data_3['led940nm27mm']
    led850nm27mm = data_3['led850nm27mm']

    save_file.write(
        '{},{},{},{},{},{},{},{},{}\n'.format(
            timeNow + 0.08,
            ms_device,
            ambient,
            led740nm10mm,
            led940nm10mm,
            led850nm10mm,
            led740nm27mm,
            led940nm27mm,
            led850nm27mm,
        )
    )


async def main():
    global blueberry, blueberry_task, save_file

    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--address', help='MAC address of the blueberry')
    parser.add_argument('-d', '--debug', help='debug', action='store_true')
    parser.add_argument('-u', '--subject', help='subject identifier')
    parser.add_argument('-r', '--runname', help='runname')
    args = parser.parse_args()

    file_path = f'./data/{args.subject}/{args.runname}'
    with Path(__file__).joinpath('devices.json').open() as f:
        devices = json.load(f)

    for dev, mac in devices.items():
        if mac == args.address:
            file_path += f'{dev}.csv'

    if not file_path.endswith('.csv'):
        raise Exception('Invalid MAC address.')

    if os.path.isfile(file_path):
        print(f'Data file {file_path} exists! Data will be appended.')

    save_file = open(file_path, 'a+')
    save_file.write('timestamp,ms_device,ambient,740nm10mm,940nm10mm,850nm10mm,740nm27mm,940nm27mm,850nm27mm\n')

    # create blueberry instance
    blueberry = Blueberry(args.address, callback=save_csv)

    # connect to and listen to notification from the blueberry
    blueberry_task = asyncio.create_task(blueberry.run())

    await blueberry_task
    save_file.close()


async def shutdown():
    global blueberry, blueberry_task
    await blueberry.stop()
    await blueberry_task


# create asyncio event loop and start program
loop = asyncio.get_event_loop()

# handle kill events (Ctrl-C)
for signame in ('SIGINT', 'SIGTERM'):
    loop.add_signal_handler(getattr(signal, signame), lambda: asyncio.ensure_future(shutdown()))

# start program loop
try:
    loop.run_until_complete(main())
finally:
    loop.close()
