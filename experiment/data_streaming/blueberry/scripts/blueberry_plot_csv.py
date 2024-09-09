"""
MIT License

Copyright (c) 2022, Blueberry

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

Blueberry X Technologies Inc. - modifications for firmare version 20220401
"""

import matplotlib.pyplot as plt
import matplotlib

matplotlib.axes.Axes.legend
matplotlib.pyplot.legend
import pandas as pd
import numpy as np
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # example commmand
    # python blueberry_plot_csv.py -f "blueberry_data_2022-07-10_13_37_42.csv"
    parser.add_argument('-f', help='csv filename', required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.f)

    timestamps = df['timestamp'].to_numpy()
    led740nm10mm = df['740nm10mm'].to_numpy()
    led940nm10mm = df['940nm10mm'].to_numpy()
    led850nm10mm = df['850nm10mm'].to_numpy()
    led740nm27mm = df['740nm27mm'].to_numpy()
    led940nm27mm = df['940nm27mm'].to_numpy()
    led850nm27mm = df['850nm27mm'].to_numpy()
    ambient27mm = df['ambient'].to_numpy()

    fig, axs = plt.subplots(3, 1, sharex=True)
    fig.tight_layout(h_pad=2)

    # plot data
    # axs[0].plot(led740nm10mm, label="740nm10mm")
    # axs[0].plot(led940nm10mm, label="940nm10mm")
    axs[0].plot(led850nm10mm, label='850nm10mm')
    axs[0].set_title('850nm10mm')

    # axs[1].plot(led740nm27mm, label="740nm27mm")
    # axs[1].plot(led940nm27mm, label="940nm27mm")
    axs[1].plot(led850nm27mm, label='850nm27mm')
    axs[1].set_title('850nm27mm')

    # axs[2].plot(timestamps, label="timestamp")
    axs[2].plot(ambient27mm, label='ambient27mm')
    axs[2].set_title('ambient27mm')
    plt.show()
