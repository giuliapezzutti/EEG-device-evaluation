import matplotlib.pyplot as plt
import numpy as np
from pyxdf import pyxdf


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


if __name__ == '__main__':

    [_, eeg, _, eeg_freq] = load_xdf(
        'C:/Users/Giulia Pezzutti/Documents/project-selina/data/sub-test_with_water/ses-S001/eeg'
        '/sub-test_with_water_ses-S001_task-Default_run-001_eeg.xdf')

    eeg = np.asmatrix(eeg)
    interval_removal = 60  # in seconds
    eeg = eeg[interval_removal*eeg_freq:-interval_removal*eeg_freq]

    eeg = eeg - np.mean(eeg, axis=0)
    plt.plot(eeg[:, 4])

    scale_factor = 1
    xmin, xmax = plt.xlim()
    ymin, ymax = plt.ylim()
    plt.xlim(xmin * scale_factor, xmax * scale_factor)
    plt.ylim(ymin * scale_factor, ymax * scale_factor)

    plt.show()

    print(np.min(eeg))
    print(np.max(eeg))

    bins = np.linspace(np.min(eeg[:, 4]), np.max(eeg[:, 4]), 100)
    plt.title('Relative Amplitude', fontsize=30)
    plt.xlabel('Random Histogram')
    plt.ylabel('Frequency', fontsize=30)
    plt.hist(eeg[:, 4], bins, alpha=0.5, histtype='bar', ec='black')

    plt.legend(loc='upper right', fontsize=30)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.show()


