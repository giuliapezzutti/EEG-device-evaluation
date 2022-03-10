import os
import sys
import matplotlib.pyplot as plt
import numpy as np
from numpy import arange
from pyxdf import pyxdf
from scipy.signal import butter, lfilter
from scipy.fft import fft
import tkinter


def get_path():
    name = 'tkinter'

    if name in sys.modules:
        from tkinter import filedialog
        path_selected = filedialog.askopenfilename(initialdir=os.getcwd(), title="Select a File",
                                                   filetypes=(("xdf files", "*.xdf*"),))
    else:
        path_selected = input("Not able to use tkinter to select the file. Insert here the file path and press ENTER:\n")

    return path_selected


def getfilename(path):
    base = os.path.basename(path)
    file = os.path.splitext(base)[0]
    return file


def load_xdf(path):
    dat = pyxdf.load_xdf(path)

    orn_signal, eeg_signal, marker_signal, eeg_frequency = None, None, None, None

    for i in range(len(dat[0])):
        stream_name = dat[0][i]['info']['name']

        if stream_name == ['Explore_CA46_ORN']:
            orn_signal = dat[0][i]['time_series']
        if stream_name == ['Explore_CA46_ExG']:
            eeg_signal = dat[0][i]['time_series']
            eeg_frequency = int(dat[0][i]['info']['nominal_srate'][0])
        if stream_name == ['Explore_CA46_Marker']:
            marker_signal = dat[0][i]['time_series']

    return orn_signal, eeg_signal, marker_signal, eeg_frequency


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    # noinspection PyTupleAssignmentBalance
    b, a = butter(order, (low, high), btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


if __name__ == '__main__':

    path = get_path()
    filename = getfilename(path)
    [_, eeg, _, eeg_freq] = load_xdf(path)

    eeg = np.asmatrix(eeg)
    interval_removal = 60  # in seconds
    eeg = eeg[interval_removal * eeg_freq:-interval_removal * eeg_freq]

    eeg = eeg - np.mean(eeg, axis=0)

    fig, axs = plt.subplots(eeg.shape[1], 4, figsize=(25, 16), gridspec_kw={'width_ratios': [3, 3, 3, 3]})
    fig.subplots_adjust(wspace=0.5)

    for channel in range(eeg.shape[1]):
        eeg_current = eeg[:, channel]

        eeg_filt = butter_bandpass_filter(eeg_current, lowcut=0.1, highcut=40, fs=eeg_freq, order=8)[:, 0]

        x = np.arange(len(eeg_filt)) / eeg_freq
        axs[channel, 0].set_xlabel('Time (s)', fontsize=10)
        axs[channel, 0].set_ylabel('Amplitude (uV)', fontsize=10)
        axs[channel, 0].yaxis.set_label_coords(-0.2, 0.5)
        axs[channel, 0].plot(x, eeg_filt)

        x = np.arange(500) / eeg_freq
        axs[channel, 1].set_xlabel('Time (s)', fontsize=10)
        axs[channel, 1].set_ylabel('Amplitude (uV)', fontsize=10)
        axs[channel, 1].yaxis.set_label_coords(-0.2, 0.5)
        axs[channel, 1].plot(x, eeg_filt[0:500])

        bins = np.linspace(np.min(eeg_filt), np.max(eeg_filt), 100)
        axs[channel, 2].set_xlabel('Amplitude', fontsize=10)
        axs[channel, 2].set_ylabel('Frequency', fontsize=10)
        axs[channel, 2].yaxis.set_label_coords(-0.2, 0.5)
        axs[channel, 2].hist(eeg_filt, bins, alpha=0.5, histtype='bar', ec='black')

        n = len(eeg_filt)  # length of the signal
        k = arange(n)
        T = n / eeg_freq
        frq = k / T  # two sides frequency range
        frq = frq[range(int(n / 2))]  # one side frequency range

        Y = fft(eeg_filt) / n  # fft computing and normalization
        Y = Y[range(int(n / 2))]

        # axs[channel, 3].plot(frq, abs(Y), 'r')  # plotting the spectrum
        axs[channel, 3].magnitude_spectrum(eeg_filt, Fs=eeg_freq)
        axs[channel, 3].set_xlabel('Freq (Hz)', fontsize=10)
        axs[channel, 3].set_ylabel('Amplitude (uV)', fontsize=10)
        axs[channel, 3].yaxis.set_label_coords(-0.2, 0.5)
        axs[channel, 3].set_xlim(-2, 42)

    fig.suptitle(filename, fontsize=30)
    plt.savefig('images/{}.jpg'.format(filename))
    fig.show()
