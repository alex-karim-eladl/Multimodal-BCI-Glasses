"""
Blueberry, cayden
"""

# import sys
# import logging
import asyncio

# import platform
# import bitstring
import argparse
import time
import signal

# import atexit
import os

# import click

from bleak import BleakClient
from bleak import _logger as logger
from pathlib import Path

from blueberry import Blueberry

global bby, bby_task, bby_killer_task, save_file


# ROOT = "./data/blueberry"
ROOT = Path(__file__)

#####################################


def my_callback(data):
    """
    This is called and passed the blueberry data evertime a new data point is received by the Blueberry over bluetooth

    All we do here is save the data to a csv
    """
    idx = data['packet_index']
    path = data['path']
    c1 = data['channel1']
    c2 = data['channel2']
    c3 = data['channel3']

    if data['path'] == 'long_path' and data['big'] == True:
        save_file.write('{},{},{},{},{},{}\n'.format(time.time(), idx, path, c1, c2, c3))
    else:
        save_file.write('{},{},{},{},{},{}\n'.format(time.time(), idx, path, c1, c2, c3))


async def sleeper(bby, totaldur):
    """
    This runs for 40 seconds then kills the blueberry connection. It's just here for an example of how to asynchronously terminate the Blueberry.
    """
    length = int(totaldur) + 30
    for i in range(0, length):
        await asyncio.sleep(1)
    await bby.stop()


async def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--address', help='MAC address of the blueberry')
    parser.add_argument('-d', '--debug', help='debug', action='store_true')
    parser.add_argument('-u', '--userid')
    parser.add_argument('-r', '--rno')
    parser.add_argument('-o', '--totaldur')
    args = parser.parse_args()

    userid = args.userid
    rno = args.rno
    totaldur = args.totaldur

    path = ROOT / userid / rno

    global bby, bby_task, bby_killer_task, save_file

    # get address
    mac = args.address

    # get debug
    debug = args.debug

    with (path / 'exg.csv').open('w+') as f:
        f.write('timestamp,idx,path,c1,c2,c3\n')

    # create blueberry instance
    bby = Blueberry(mac, callback=my_callback, debug=debug)

    # start a task to connect to and stream from the blueberry
    bby_task = asyncio.create_task(bby.run())
    # start a task that waits 15 seconds and then terminates the blueberry task
    bby_killer_task = asyncio.create_task(sleeper(bby, totaldur))
    # setup program killer
    # signal.signal(signal.SIGINT, signal_handler)

    await bby_task, bby_killer_task
    save_file.close()


async def shutdown():
    global bby, bby_task, bby_killer_task
    bby_killer_task.cancel()
    await bby.stop()
    await bby_task, bby_killer_task


if __name__ == '__main__':

    # create asyncio event loop and start program
    loop = asyncio.get_event_loop()

    # handle kill events (Ctrl-C)
    for signame in ('SIGINT', 'SIGTERM'):
        loop.add_signal_handler(getattr(signal, signame), lambda: asyncio.ensure_future(shutdown()))
    # start program loop
    try:
        loop.run_until_complete(main())
    except:
        raise Exception('Stream ended')
    finally:
        loop.close()
