import argparse
import asyncio
import json
import os

# import platform
from datetime import datetime

# import click
from connection import Connection
from utils import get_human_version, write_arr
from pathlib import Path

# FIXME: correct relative paths / project structure
ROOT = Path(__file__).parent.parent / 'data/raw/'  # "../../data"
VERSION = '1.3.1'


start_time = datetime.now()
timestamp = start_time.strftime('%m-%d-%y-%H%M%S')


# @click.command()
# @click.option(
#     "-n",
#     "--name",
#     prompt="Enter run name",
#     default=timestamp,
#     help="e",
# )
# @click.option(
#     "-d",
#     "--descr",
#     prompt="Enter run description",
#     default="None",
#     help="The run description",
# )
def run_streamer(name: str, descr: str):
    # os.mkdir(f"{ROOT}/{name}/attentivu")
    subdir = ROOT / name / 'attentivu'
    subdir.mkdir(parents=True, exist_ok=True)

    # with open(f"{ROOT}/log.txt", "a+") as logfile:
    #     logfile.write(f"{timestamp}:\n\tName: {name}\n\tDescription: {descr}\n")

    # with open(f"{ROOT}/{name}/attentivu/details.txt", "w+") as detailfile:
    with (subdir / 'details.txt').open('w+') as detailfile:
        detailfile.write(f'Name: {name},\nTimestamp: {timestamp},\nDescription: {descr},\n')

    conn = start_connection(subdir, write_arr)

    with (subdir / 'details.txt').open('a+') as detailfile:
        detailfile.write(f'Duration: {str(datetime.now()-start_time)},\n')
        for f in [x for x in os.listdir(subdir) if x.endswith('.csv')]:
            samp = count_lines(subdir / f)
            detailfile.write(f"{f.split('.csv')[0].title()} Samples: {samp},\n")

        detailfile.write(f'Device Name: {conn.device.name},\n')
        detailfile.write(f'Device UUID: {conn.device.address},\n')

        for field in ['EXG Mask', 'IMU Mask', 'EXG FS', 'IMU FS']:
            key = field.lower().replace(' ', '_')
            detailfile.write(f'{field}: {getattr(conn, key, None)},\n')


def count_lines(path: Path):
    with path.open() as f:
        for i, l in enumerate(f):
            pass
    return i + 1


def start_connection(path, write_handler):
    # Loads the json file to pass to the Connection class
    # Would be nice to not have to use this and just have the device tell us
    with Path(__file__).joinpath('DefinedServices.json').open() as f:
        services = json.load(f)

    loop = asyncio.get_event_loop()
    connection = Connection(path, services, write_handler)

    try:
        fut_con = asyncio.ensure_future(connection.connect())
        loop.run_until_complete(fut_con)
    except KeyboardInterrupt:
        print('\nUser stopped program.')
    finally:
        print('Stopping...')
        loop.run_until_complete(connection.cleanup())

    return connection


if __name__ == '__main__':
    #     # Banners
    #     with open("./banner.txt") as b:
    #         banner_text = b.read() \
    #             .replace("{{VERSION}}", VERSION) \
    #             .replace("{{OS}}", platform.system()) \
    #             .replace("{{OS_VERSION}}", get_human_version())
    #         print(banner_text)

    # parse arguments
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('name', type=str, help='The run name')
    parser.add_argument('descr', type=str, help='The run description')
    args = parser.parse_args()

    run_streamer(args.name, args.descr)
