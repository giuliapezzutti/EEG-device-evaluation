import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal
from numpy import arange
from pyxdf import pyxdf
from scipy.signal import butter, lfilter, sosfilt
from scipy.fft import fft
import tkinter
import mne
from sklearn.metrics import mean_squared_error


def get_path():
    name = 'tkinter'

    if name in sys.modules:
        from tkinter import filedialog
        path_selected = filedialog.askopenfilename(initialdir=os.getcwd(), title="Select a File",
                                                   filetypes=(("xdf files", "*.xdf*"),))
    else:
        path_selected = input(
            "Not able to use tkinter to select the file. Insert here the file path and press ENTER:\n")

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


def butter_bandpass(lowcut, highcut, fs, order=8):
    low = lowcut / fs
    high = highcut / fs
    sos = butter(order, [low, high], analog=False, btype='band', output='sos')
    return sos


def butter_bandpass_filter(data, lowcut, highcut, fs, order=8):
    sos = butter_bandpass(lowcut, highcut, fs, order=order)
    y = sosfilt(sos, data)

    b, a = scipy.signal.iirnotch(50, Q=150, fs=fs)
    y = lfilter(b, a, y)
    return y


def get_rms_smoothed(x):
    print('RMS: ')
    print(mean_squared_error(x, [0 for _ in x], squared=False))


if __name__ == '__main__':

    path = get_path()
    # path = 'C:/Users/giuli/Documents/Università/Traineeship/device-evaluation/data/sub-test_without_gel/ses-S001/eeg/sub-test_without_gel_ses-S001_task-Default_run-001_eeg.xdf'
    filename = getfilename(path)
    [_, eeg, _, eeg_freq] = load_xdf(path)

    eeg = np.asmatrix(eeg)
    interval_removal = 60  # in seconds
    eeg = eeg[interval_removal * eeg_freq:-interval_removal * eeg_freq]

    eeg = eeg - np.mean(eeg, axis=0)

    fig, axs = plt.subplots(eeg.shape[1], 4, figsize=(25, 16), gridspec_kw={'width_ratios': [3, 3, 3, 3]})
    fig.subplots_adjust(wspace=0.5)

    for channel in range(eeg.shape[1]):
        eeg_current = np.array(eeg[:, channel]).flatten()

        eeg_filt = butter_bandpass_filter(eeg_current, lowcut=0.1, highcut=40, fs=eeg_freq, order=8)

        x = np.arange(len(eeg_filt)) / eeg_freq
        axs[channel, 0].set_xlabel('Time (s)', fontsize=10)
        axs[channel, 0].set_ylabel('Amplitude (uV)', fontsize=10)
        axs[channel, 0].yaxis.set_label_coords(-0.2, 0.5)
        axs[channel, 0].plot(x, eeg_filt)

        get_rms_smoothed(eeg_filt)

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

        # axs[channel, 3].plot(frq, abs(Y), 'r')  # plotting the spectrum
        axs[channel, 3].magnitude_spectrum(eeg_filt, Fs=eeg_freq)
        axs[channel, 3].set_xlabel('Frequency (Hz)', fontsize=10)
        axs[channel, 3].set_ylabel('Amplitude (uV)', fontsize=10)
        axs[channel, 3].yaxis.set_label_coords(-0.2, 0.5)
        axs[channel, 3].set_xlim(-2, 42)

    fig.suptitle(filename, fontsize=30)
    plt.savefig('images/{}.jpg'.format(filename))
    fig.show()
