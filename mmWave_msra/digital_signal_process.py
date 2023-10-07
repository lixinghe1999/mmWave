import os
import sys
import argparse
import numpy as np
import scipy.signal as signal
import scipy.fft as fft
import matplotlib.pyplot as plt
from radar_config import RadarConfig
from adc_data import ADCData


def music_estimate(X : np.ndarray, lamda : float, k : int = 2):
    '''
    X: Rx signal with shape (Nm, samples)
    lamda: wavelength
    k: tunable parameter for MUSIC
    '''
    assert len(X.shape) == 2
    sep = lamda / 2
    Nm = X.shape[0]
    W = np.eye(Nm)
    C = X @ X.T
    C = C + np.conj(np.flip(np.flip(C, 0), 1))
    C = W.T @ C @ W

    w, v = np.linalg.eig(C)
    v_sort = v[:, np.argsort(-np.abs(w))]
    NN = v_sort[:, k : Nm]

    theta = np.linspace(-90, 90, 181)
    Pmusic = np.zeros(len(theta))
    for k in range(len(theta)):
        SS = np.exp(1j * 2 * np.pi * sep * np.arange(0, Nm) * np.sin(theta[k] / 180 * np.pi) / lamda)
        PP = SS @ NN @ NN.T @ SS.T
        Pmusic[k] = abs(1.0 / PP)
    return theta, Pmusic


class DSP:
    """
    1. load config and binary data
    >>> dsp = DSP(xml_file_path)
    >>> sig = dsp.load_bin(bin_file_path)
    """

    def __init__(self, xml_file : str, bin_file : str = None):
        self.cfg = RadarConfig(xml_file)
        self.adc_data = ADCData(self.cfg)
        self.bin_file = bin_file
            
    def load_bin(self):
        self.adc_data.load_bin(self.bin_file)
        print('Loaded ADC Data with Shape:', self.adc_data.adc_data.shape)
    
    def adc_data_raw(self):
        # return format: (frame, chirp, Tx, Rx, sample)
        return self.adc_data.adc_data
    
    def adc_data_tx_rx_first(self):
        if self.adc_data.adc_data is None:
            return None
        # return format: (Tx, Rx, frame, chirp, sample)
        return np.transpose(self.adc_data.adc_data, (2, 3, 0, 1, 4))
    
    def adc_data_frame_loop_merged(self):
        if self.adc_data.adc_data is None:
            return None
        x = np.transpose(self.adc_data.adc_data, (2, 3, 0, 1, 4))
        # return format: (Tx, Rx, frame * chirp, sample)
        return x.reshape(x.shape[0], x.shape[1], -1, x.shape[-1])
    
    def adc_data_virtual_antenna(self):
        x = self.adc_data_frame_loop_merged()
        if x is None:
            return None
        x = x[[1, 0]]  # select and swap TX3 and TX1
        x = x.reshape(x.shape[0] * x.shape[1], x.shape[2], x.shape[3])
        return x


def parse_args():
    parser = argparse.ArgumentParser('Digital Signal Processing.')
    parser.add_argument('config_file', type = str, help = 'path to xml format config file.')
    parser.add_argument('adc_data', type = str, help = 'path to adc_data.bin.')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    dsp = DSP(args.config_file)
    max_distance = dsp.cfg.freq2dist(0.5 * dsp.cfg.digOutSampleRate)
    print('max distance = ', max_distance)

    adc_data = dsp.load_bin(args.adc_data)
    print('ADC Data Shape:', adc_data.shape)
    adc_data = dsp.adc_data_tx_rx_first()
    print('Transposed ADC Data Shape:', adc_data.shape)
    adc_data = dsp.adc_data_frame_loop_merged()
    print('Frame-Loop Merged ADC Data Shape:', adc_data.shape)

    # spectrum
    x = adc_data[0][0]
    print('One TX & RX signal:', x.shape)
    hamming = signal.get_window('hamming', x.shape[-1])
    # y = fft.fft(x)
    y = fft.fft(x * hamming)
    f = fft.fftfreq(x.shape[-1], 1.0 / dsp.cfg.digOutSampleRate)
    y, f = fft.fftshift(y, axes = 1), fft.fftshift(f)
    d = dsp.cfg.freq2dist(f)

    # mag = np.abs(y[0])
    mag = np.mean(np.abs(y), axis = 0)
    plt.plot(d, mag)
    plt.title('Frequency Spectrum')
    plt.xlabel('distance (m)')
    plt.ylabel('magnitude')
    plt.show()
