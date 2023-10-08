import numpy as np
import scipy.signal as signal
import scipy.fft as fft
import matplotlib.pyplot as plt
from digital_signal_process import DSP
from scipy.signal import stft, correlate, find_peaks, butter, filtfilt

def comp_win_func(chirp,d):
    x = chirp
    wins = ['boxcar', 'hamming', 'hann', 'blackman', 'blackmanharris', 'bartlett']
    for w in wins:
        win = signal.get_window(w, x.shape[-1])
        y = fft.fft(x * win)
        y = fft.fftshift(y, axes=1)
        mag = np.mean(np.abs(y), axis=0)
        plt.plot(d, mag, lw=1.0)
    # plt.xlim(-1.0, 3.0)
    plt.xlabel('distance (m)')
    plt.ylabel('magnitude')
    plt.legend(wins)
    plt.title('Frequency Spectrum')
    plt.savefig('compare_window.png')

def comp_avg(chirp,d):
    x = chirp
    win = signal.get_window('hamming', x.shape[-1])
    y = fft.fft(x * win)
    y = fft.fftshift(y, axes=1)
    mag = np.mean(np.abs(y), axis=0)
    plt.plot(d, np.abs(y[0]), lw=1.0)  # only one piece
    plt.plot(d, mag, lw=1.0)  # average over all rows

    plt.xlabel('distance (m)')
    plt.ylabel('magnitude')
    plt.legend(['single-piece', 'averaged-all'])
    plt.title('Frequency Spectrum')
    plt.savefig('compare_average.png')

def tot_comp(chirp,d):
    x = chirp
    # no window
    y = fft.fft(x)
    y = fft.fftshift(y, axes=1)
    mag = np.mean(np.abs(y), axis=0)
    plt.plot(d, np.abs(y[0]), lw=1.0)  # only one piece
    plt.plot(d, mag, lw=1.0)  # average over all rows

    # hamming window
    win = signal.get_window('hamming', x.shape[-1])
    y = fft.fft(x * win)
    y = fft.fftshift(y, axes=1)
    mag = np.mean(np.abs(y), axis=0)
    plt.plot(d, np.abs(y[0]), lw=1.0)  # only one piece
    plt.plot(d, mag, lw=1.0)  # average over all rows

    plt.xlabel('distance (m)')
    plt.ylabel('magnitude')
    plt.legend(['rect-single-piece', 'rect-averaged-all', 'hamming-single-piece', 'hamming-averaged-all'])
    plt.title('Frequency Spectrum')
    plt.savefig('compare_overall.png')

def analyze(config_path, data_path, enable_save = True,):
    dsp = DSP(config_path, data_path)
    if enable_save:
        dsp.load_bin()
        adc_data = dsp.adc_data_frame_loop_merged()
        chirp = adc_data[0][0]         # use the data at Rx 0  from Tx 0
        print('adc_data.shape:', adc_data.shape)
        print('chirp.shape:', chirp.shape)        # (# of chirp loops * # of chirp per loop, # of samples per chirp)
        chosen_len = min(1000,chirp.shape[0])
        chirp = chirp[:chosen_len,:].real
        # unit_d = dsp.cfg.digOutSampleRate/dsp.cfg.numAdcSamples * dsp.cfg.speedOfLight / (2*dsp.cfg.freqSlope)
        # hardware_delay = int(hardware_offset / unit_d)
        # for i in range(chirp.shape[0]):
        #     chirp[i,:] = np.concatenate((np.zeros(hardware_delay),chirp[i,:]))[:dsp.cfg.numAdcSamples]
        np.save('cache_chirp.npy',chirp)
        print("saved ", chirp.shape)
    else:
        chirp = np.load('cache_chirp.npy')

    pad_len = 1000
    new_chirp = np.zeros((chirp.shape[0],pad_len),dtype=np.float64)
    for i in range(chirp.shape[0]):
        new_chirp[i,:] = np.concatenate((chirp[i,:].real, np.zeros(pad_len - chirp.shape[1])))
        # plt.plot(np.abs(fft.fft(new_chirp[i,:])))
        # plt.show()
    f = fft.fftfreq(new_chirp.shape[-1], 1.0 / dsp.cfg.digOutSampleRate)
    f = fft.fftshift(f)
    d = dsp.cfg.freq2dist(f)

    comp_win_func(new_chirp,d)
    #comp_avg(new_chirp,d)
    #tot_comp(new_chirp,d)

if __name__ == '__main__':
    cfg_path = 'awr1843_cfg.xml'
    bin_path = "data/previous"
    enable_save = True
    analyze(cfg_path, bin_path, enable_save)