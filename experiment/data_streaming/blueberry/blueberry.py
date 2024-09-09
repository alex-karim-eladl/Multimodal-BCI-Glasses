"""
MIT License

Copyright (c) 2020, Henrik Blidh

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

Blueberry X Technologies Inc. - modifications for firmare version 20220401
"""

import sys
import logging
import asyncio
import platform
import bitstring
import argparse
import time
import signal
import atexit

from bleak import BleakClient
from bleak import _logger as logger


class Blueberry:
    def __init__(self, device_address, callback=None, debug=False):
        self.device_address = device_address
        self.callback = callback
        self.debug = debug

        self.client = None

        # GATT server characteristics information
        self.bbxService = {'name': 'blueberry service', 'uuid': '0f0e0d0c-0b0a-0908-0706-050403020100'}
        self.bbxchars = {
            'commandCharacteristic': {
                'name': 'write characteristic',
                'uuid': '1f1e1d1c-1b1a-1918-1716-151413121110',
                'handles': [None],
            },
            'fnirsCharacteristic': {
                'name': 'fnirs',
                'uuid': '4f4e4d4c-4b8a-8988-8786-858483428180',
                'handles': [19, 20, 27, 47],
            },
        }
        self.stream = False
        self.logger = None

        # logging
        self.l = None
        self.h = None

    def _cleanup(self):
        for connection in self._connections.copy():
            connection.disconnect()

    # unpack fNIRS data
    def unpack_fnirs(self, sender, packet):
        data_1 = dict()
        data_2 = dict()
        data_3 = dict()
        # unpack packet
        byte_packet = bitstring.Bits(bytes=packet)
        pattern = 'intbe:32,intbe:32,intbe:32,intbe:32,intbe:32,intbe:32,intbe:32,intbe:32,intbe:32,intbe:32,intbe:32,intbe:32,intbe:32,intbe:32,intbe:32,intbe:32,intbe:32,intbe:32,intbe:32,intbe:32,intbe:32,intbe:32,intbe:32,intbe:32'
        res = byte_packet.unpack(pattern)
        # print(res)
        data_1['ms_device'] = res[0]
        data_2['ms_device'] = res[8]
        data_3['ms_device'] = res[16]
        data_1['ambient'] = res[1]
        data_2['ambient'] = res[9]
        data_3['ambient'] = res[17]
        data_1['led740nm10mm'] = res[2]
        data_2['led740nm10mm'] = res[10]
        data_3['led740nm10mm'] = res[18]
        data_1['led940nm10mm'] = res[3]
        data_2['led940nm10mm'] = res[11]
        data_3['led940nm10mm'] = res[19]
        data_1['led850nm10mm'] = res[4]
        data_2['led850nm10mm'] = res[12]
        data_3['led850nm10mm'] = res[20]
        data_1['led740nm27mm'] = res[5]
        data_2['led740nm27mm'] = res[13]
        data_3['led740nm27mm'] = res[21]
        data_1['led940nm27mm'] = res[6]
        data_2['led940nm27mm'] = res[14]
        data_3['led940nm27mm'] = res[22]
        data_1['led850nm27mm'] = res[7]
        data_2['led850nm27mm'] = res[15]
        data_3['led850nm27mm'] = res[23]

        return data_1, data_2, data_3

    def notification_handler(self, sender, data):
        """Simple notification handler which prints the data received."""
        data_1, data_2, data_3 = self.unpack_fnirs(sender, data)
        if data_1 is None:
            return
        ms_device = data_1['ms_device']
        ambient = data_1['ambient']
        led740nm10mm = data_1['led740nm10mm']
        led940nm10mm = data_1['led940nm10mm']
        led850nm10mm = data_1['led850nm10mm']
        led740nm27mm = data_1['led740nm27mm']
        led940nm27mm = data_1['led940nm27mm']
        led850nm27mm = data_1['led850nm27mm']

        ms_device = data_2['ms_device']
        ambient = data_2['ambient']
        led740nm10mm = data_2['led740nm10mm']
        led940nm10mm = data_2['led940nm10mm']
        led850nm10mm = data_2['led850nm10mm']
        led740nm27mm = data_2['led740nm27mm']
        led940nm27mm = data_2['led940nm27mm']
        led850nm27mm = data_2['led850nm27mm']

        ms_device = data_3['ms_device']
        ambient = data_3['ambient']
        led740nm10mm = data_3['led740nm10mm']
        led940nm10mm = data_3['led940nm10mm']
        led850nm10mm = data_3['led850nm10mm']
        led740nm27mm = data_3['led740nm27mm']
        led940nm27mm = data_3['led940nm27mm']
        led850nm27mm = data_3['led850nm27mm']

        if self.callback is not None:
            self.callback(data_1, data_2, data_3)
            # print("receiving data")

    async def stop(self):
        # stop stream
        print('stopping connection and disconnect')
        self.stream = False
        await asyncio.sleep(0.2)
        if self.client != None:
            await self.client.disconnect()

    async def run(self):
        self.stream = True
        # log = logging.getLogger(__name__)
        # log.setLevel(logging.WARNING)
        # h = logging.StreamHandler(sys.stdout)
        # h.setLevel(logging.WARNING)
        # log.addHandler(h)

        # connect and stream
        FNIRS_CHAR_UUID = self.bbxchars['fnirsCharacteristic']['uuid']

        print('Trying to connect...')
        async with BleakClient(self.device_address) as self.client:
            x = await self.client.is_connected()
            print('Connected to: {0}'.format(self.device_address))
            logger.setLevel(logging.WARNING)

            await self.client.start_notify(FNIRS_CHAR_UUID, self.notification_handler)
            while self.stream:
                # set to 40ms interval
                await asyncio.sleep(0.04)
            await self.client.stop_notify(FNIRS_CHAR_UUID)
        print('Blueberry disconnected.')
