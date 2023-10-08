from mmwave.dsp import range_processing, doppler_processing
from digital_signal_process import DSP
import os
import numpy as np
import scipy.fft as fft
from analyzeRadar import comp_win_func, comp_avg
def phase_freq(range_fft):
    # range_fft: (num_chirps_per_frame, num_rx_antennas, num_range_bins)
    phase = np.angle(range_fft)
    diff_phase = np.diff(phase, axis=0)
    phase = np.unwrap(diff_phase, period=np.pi)
    freq_phase = fft.fft(phase, axis=0)
    freq_phase = np.mean(np.abs(freq_phase), axis=1)
    return freq_phase


config_path = 'awr1843_cfg.xml'
data_path = "data/previous"
data_path = "data/2023-10-08_13;53;20"
data_cache = os.path.join(data_path, 'cache_chirp.npy')
dsp = DSP(config_path, data_path)
if not os.path.exists(data_cache):
    dsp.load_bin()
    adc_data = dsp.adc_data_tx_rx_first()
    chirp = adc_data[0] # TX0, RX all
    print('adc_data.shape:', adc_data.shape)
    chosen_len = min(100, chirp.shape[1])
    chirp = chirp[:, :chosen_len]
    np.save(data_cache, chirp)
else:
    chirp = np.load(data_cache)
print('chirp.shape:', chirp.shape)  # RX, frames, chirps, samples (4, 100, 128, 512)

f = fft.fftfreq(512, 1.0 / dsp.cfg.digOutSampleRate)
f = fft.fftshift(f)
d = dsp.cfg.freq2dist(f)

chirp = np.transpose(chirp, (1, 2, 0, 3))
from mmwave.dsp.utils import Window
import matplotlib.pyplot as plt
locations = [0, 60]
for frame in chirp:
    fig, axs = plt.subplots(1, 2)

    radar_cube = range_processing(frame, window_type_1d=Window.BLACKMAN)
    radar_vis = np.sum(np.abs(np.fft.fftshift(radar_cube, axes=-1)), axis=1)
    radar_vis /= np.max(radar_vis)
    axs[0].plot(d, radar_vis.T)

    # det_matrix = phase_freq(radar_cube)
    radar_cube = radar_cube[:, :, locations[0]:locations[1]]
    det_matrix, aoa_input = doppler_processing(np.angle(radar_cube), num_tx_antennas=1, interleaved=False,
                                                clutter_removal_enabled=True, 
                                                window_type_2d=Window.HAMMING)
    det_matrix_vis = np.fft.fftshift(det_matrix, axes=1)
    det_matrix_vis /= np.max(det_matrix_vis)
    axs[1].imshow(det_matrix_vis, aspect='auto')
    plt.show()    
    # break