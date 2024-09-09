import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Callable

import click
import numpy as np
from bleak import BleakClient, BleakScanner
from bleak.backends.device import BLEDevice
from tqdm import tqdm
from utils import get_macos_version

UINT24_MAX = 2**24 - 1
ZERO_OFFSET = UINT24_MAX // 2 + 1  # 8388608


def char_name_to_type(name: str) -> str:
    return name[:3]


@dataclass
class Connection:
    """
    Class describing the connection to the glasses
    """

    # The path to the folder where data should be saved
    root: str
    # Dictionary describing services and characteristics
    services: dict
    # Called when lists grow past output_size length
    write_handler: Callable
    # Set the last event number to reverse mod % 256
    last_event_number: int = 0
    # Determines threshold of write_handler call
    output_size: int = 512

    # Bluetooth Connection Fields
    client: BleakClient = None
    connected: bool = False
    device: BLEDevice = None

    # Post Initialisation Helpers
    uuid_to_name: dict[str, str] = field(init=False)
    uuid_to_cols: dict[str, list[str]] = field(init=False)
    known_services: set[str] = field(init=False)

    def __post_init__(self):
        # Builds a flat reverse lookups from UUID to display name / columns
        self.uuid_to_name = {}
        self.uuid_to_cols = {}
        self.known_services = set()
        for service in self.services:
            self.known_services.add(service['uuid'].lower())
            for character in service['characteristics']:
                self.uuid_to_name[character['uuid']] = character['name']
                self.uuid_to_cols[character['uuid']] = character.get('cols', [])

    async def discover(self, shouldUserSelect=True):
        indent = '\n> '
        service_uuids = []
        if get_macos_version()[0] and int(get_macos_version()[0]) >= 12:
            # In macOS 12 Monterey the service_uuids need to be specified. As a
            # workaround for this example program, we scan for all known UUIDs to
            # increse the chance of at least something showing up. However, in a
            # "real" program, only the device-specific advertised UUID should be
            # used. Devices that don't advertize at least one service UUID cannot
            # currently be detected.
            service_uuids.extend(self.known_services)

        scanTime = datetime.now()
        print(
            f"[{scanTime}] Looking for devices{f' with Service UUIDs {indent}' if service_uuids else ''}{indent.join(service_uuids)}"
        )
        devices = await BleakScanner.discover(service_uuids=service_uuids)
        validDevices = {}
        for d in devices:
            try:
                if (
                    'AttentivU' in d.name  # Legacy
                    or d.name == 'ESP32'  # Very legacy
                    or 'AtU' in d.name  # Shortname fallback (should work with uuid)
                    or (set(self.known_services) & set(u.lower() for u in d.metadata['uuids']))
                ):
                    validDevices[f"{d.name} '{d.address}'"] = d
            except:
                pass

        if len(validDevices) == 0:
            print(f'[{scanTime}] AttentivU Not Found')
            return

        validDeviceIds = sorted(list(validDevices.keys()))
        chosen = validDeviceIds[0]

        if not shouldUserSelect:
            for idx, deviceId in enumerate(validDeviceIds):
                print(f'[{scanTime}] - ({idx}) {deviceId}')
            return

        if len(validDevices) != 1:
            print('Found multiple valid devices, choose one:')
            for idx, deviceId in enumerate(validDeviceIds):
                print(f'[{idx}] {deviceId}')
            idx = click.prompt('Device', type=click.IntRange(0, len(validDeviceIds) - 1), default=0)
            chosen = validDeviceIds[idx]

        self.device = validDevices[chosen]

    async def connect(self):
        await self.discover()

        if self.device is None:
            print('AttentivU Not Selected')
            exit(1)

        self.client = BleakClient(self.device)
        try:
            print('Connecting...')
            self.connected = await self.client.connect(device=self.device, timeout=20)
            if self.connected:
                print(f'Connected to AttentivU - {self.device.name}')
                self.client.set_disconnected_callback(self.on_disconnect)

                services_array = await self.client.get_services()
                # print("Got services", services_array.services)
                self.handle_map = services_array.characteristics

                device_services = {f'{s.uuid}': s for s in self.client.services}

                # All services arrive as a dict of services which are dicts of chars
                # For each independent service, subscribe its respective handler to each
                # characteristic inside of it if it a subscribe type. If it is a read
                # type, read it once at the beginning and pass the value to a callback.
                for service in self.services:
                    device_service = device_services.get(service['uuid'], None)
                    if not device_service:
                        print(f"> Service '{service['uuid']}' could not be found.")
                        continue

                    device_characteristics = [c.uuid.lower() for c in device_service.characteristics]

                    for character in service['characteristics']:
                        if character['uuid'] not in device_characteristics:
                            print(f"> Character '{character['uuid']}' could not be found.")
                            continue

                        if service['type'] == 'read':
                            val = await self.client.read_gatt_char(character['uuid'])
                            getattr(self, f"{service['name']}_handler")(character, val)

                        if service['type'] == 'subscribe':
                            await self.client.start_notify(
                                character['uuid'],
                                getattr(self, f"{service['name']}_handler"),
                            )

                print('Starting Streaming')
                self.exg_progress = tqdm(desc='EXG', unit='samples')
                self.imu_progress = tqdm(desc='IMU', unit='samples')
                self.exg_packets_progress = tqdm(desc='Pkt', unit='packets')
                while True:
                    await asyncio.sleep(0.5)
                    if not self.connected:
                        self.exg_progress.close()
                        self.imu_progress.close()
                        self.exg_packets_progress.close()
                        break
            else:
                print('Failed to Connect to AttentivU')
        except Exception as e:
            print(e)

    def on_disconnect(self, client):
        self.connected = False
        print(f'Disconnected from AttentivU')

    async def cleanup(self):
        if self.client:
            for service in self.services:
                if service['type'] != 'subscribe':
                    continue

                for character in service['characteristics']:
                    try:
                        await self.client.stop_notify(character['uuid'])
                    except:
                        pass

            await self.client.disconnect()

    def config_handler(self, char, val: bytes):
        # print(f"{char['name']}: {val}")
        # TODO: It would be nice to not have these _mask options hardcoded
        char_name = char['name']
        if '_mask' in char_name:
            setattr(self, char_name, [bool(val[0] & (1 << n)) for n in range(8)])
        else:
            setattr(self, char_name, int.from_bytes(val, 'big'))

    def dat_handler(self, sender, data):
        # Translate handle number into UUID and map it to known services
        uuid = self.handle_map[sender].uuid
        char_name = self.uuid_to_name[uuid]

        try:
            if (
                getattr(self, f'{char_name_to_type(char_name)}_fs') is None
                or getattr(self, f'{char_name_to_type(char_name)}_mask') is None
            ):
                return
        except AttributeError:
            return

        # All data packets follow the following format:
        # |firstSampleNumber(1B)|sampleCount(1B)|S1CH1(XB)|S1CH2(XB)...|S2CH1(XB)...
        # data is transmitted in big-endian order
        # X = 3 for EXG
        # X = 2 for IMU
        first_sample_idx = 0

        first_sample_number = int(data[0])
        first_sample_idx += 1

        num_samp = int(data[1])
        first_sample_idx += 1

        if char_name_to_type(char_name) == 'exg' and hasattr(self, 'exg_packets_progress'):
            self.exg_packets_progress.update(1)
        if char_name_to_type(char_name) == 'exg' and hasattr(self, 'exg_progress'):
            self.exg_progress.update(num_samp)
        if char_name_to_type(char_name) == 'imu' and hasattr(self, 'imu_progress'):
            self.imu_progress.update(num_samp)

        mask = getattr(self, f'{char_name_to_type(char_name)}_mask').copy()
        try:
            # Take the first 4 for exg0, and the second batch of 4 for exg1
            if len(char_name) > 3:
                idx = 4 * int(char_name[3:])
                mask = mask[idx : idx + 4]
        except IndexError as err:
            print(f"[WARNING] mask '{mask}' is too small for '{char_name}'", err)

        channel_count = sum(mask)

        # Handle the different types of data differently
        # TODO: See if there's a way to make this dynamic like everything else
        if 'exg' in char_name:
            as_list = list(data)

            # Desired datatype is u4
            dt = np.dtype('>u4')

            # Total data points: 3B chunks
            # = Number of Channels * Number of Samples In Packet
            count = len(as_list[first_sample_idx:]) // 3

            if channel_count * num_samp != count:
                print(f'[WARNING] missing datapoints from claimed packet size: {channel_count} {num_samp} {count}')

            # Pad out 3 bytes to 4 bytes in the data segment of this packet
            #    Index, Samples, A2, A1, A0, B2, B1 ...
            # => Index, Samples, 0, A2, A1, A0, 0, B2, B1 ...
            for i in np.arange(count) * 3 + np.arange(count) + first_sample_idx:
                as_list.insert(i, 0)
            data = bytes(as_list)
        elif char_name == 'imu':
            dt = np.dtype('>u2')
        else:
            print(f"[WARNING] Unknown charater name '{char_name}' for datatype of datapoints, assuming >u4")
            dt = np.dtype('>u4')

        # Convert it into a numpy array and reshape to the number of unmasked channels
        # DP x 1 => Samples x Channels
        # VITAL: Convert to a signed format after parsing from u4, as we want negative values as well
        arr = np.frombuffer(data[first_sample_idx:], dtype=dt).astype(np.int32)
        arr = arr.reshape((-1, channel_count))

        if 'exg' in char_name:
            # Convert unsigned back to signed
            arr = arr - ZERO_OFFSET

            # # Convert to float
            # arr = arr.astype(np.float32)
            # # # Normalise value to approach 'true' voltage readings
            # arr *= 5.0 / UINT24_MAX

        if len(arr) != num_samp:
            print('[WARNING] claimed packet size mismatch')

        # Assign Sample number to all samples in the packet
        # Rollaround at 256
        sample_numbers = np.remainder(np.arange(0, num_samp) + first_sample_number, 256)
        final = np.hstack((np.expand_dims(sample_numbers, 1), arr))

        # Interpolate the timestamps in seconds for each sample given the sampling rate
        step = int(1 / getattr(self, f'{char_name_to_type(char_name)}_fs') * 1e3)
        utc_now = datetime.now()
        utc_times = np.array([timedelta(milliseconds=int(x * step)) + utc_now for x in np.arange(0, num_samp)])
        utc_times = [x.strftime('"%Y-%m-%d %H:%M:%S.%f"') for x in utc_times]
        final = np.hstack((np.expand_dims(utc_times, 1), final))

        # Append to the relevant array if it exists
        try:
            past = getattr(self, char_name)
        except AttributeError:
            past = None
            setattr(self, char_name, None)
        setattr(self, char_name, np.vstack((past, final)) if past is not None else final)

        # When the data buffered is greater than some output_size, pass it to the handler
        # with the corresponding column names. Then clean up that array.
        if past is not None:
            if len(past) + len(final) >= self.output_size:
                self.flush_char(char_name, mask, uuid)

    def flush_char(self, char_name, mask, uuid):
        masked_cols = ['UTC_TIME', 'SAMPLE_NUMBER'] + [
            col
            for col, m in zip(
                self.uuid_to_cols[uuid],
                mask,
            )
            if m
        ]
        if len(masked_cols) == 2:
            print('[ERROR] Masked Columns is only 2', self.uuid_to_cols[uuid], mask)
            return

        self.write_handler(
            f'{self.root}/{char_name}.csv',
            getattr(self, char_name),
            masked_cols,
        )
        setattr(self, char_name, None)
