import argparse
import asyncio
from bleak import BleakScanner

discovered = []


async def scan(name: str):
    while True:
        devices = await BleakScanner.discover()

        for dev in devices:
            if dev.name is not None:
                if name in dev.name and dev.name not in discovered:
                    print(dev)
                    discovered.append(dev.name)


if __name__ == '__main__':
    # command line arguments
    parser = argparse.ArgumentParser(description='Scan for BLE devices.')
    parser.add_argument('-t', '--timeout', type=int, default=30, help='Timeout in seconds.')
    parser.add_argument(
        '-n', '--name', type=str, default='', help='Only search for devices whose name includes this substring.'
    )
    args = parser.parse_args()

    # setup asyncio event loop and timeout
    loop = asyncio.get_event_loop()
    task = loop.create_task(scan(args.name))
    loop.call_later(args.timeout, task.cancel)

    try:
        loop.run_until_complete(task)
    except KeyboardInterrupt:
        print('\nUser stopped program.')
    except asyncio.CancelledError:
        print(f'\n{args.timeout}s timeout reached.')
