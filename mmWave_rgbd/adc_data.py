import os
import sys
import argparse
from typing import Union
import numpy as np
import scipy.io as io
from radar_config import RadarConfig
from tqdm import tqdm


def ti_dca1000_real2complex(x:np.array):
    """ deal with the format like I0 I1 Q0 Q1 I2 I3 Q2 Q3 ...
    >>> x = np.array(range(1,13))
    >>> print(x)
    [ 1  2  3  4  5  6  7  8  9 10 11 12]
    >>> ti_dca1000_real2complex(x)
    [ 1. +3.j  2. +4.j  5. +7.j  6. +8.j  9.+11.j 10.+12.j]
    """
    x = x.reshape(-1,4)
    xx1 = x[:,0] + 1j * x[:,2]
    xx2 = x[:,1] + 1j * x[:,3]
    return np.vstack((xx1, xx2)).transpose().reshape(-1)


class ADCData:
    '''
    class for reading ADC data.
    '''
    def __init__(self, cfg : Union[RadarConfig, str]):
        '''
        cfg: must be either a RadarConfig instance or path to xml file.
        '''
        if isinstance(cfg, RadarConfig):
            self.cfg = cfg
        else:
            self.cfg = RadarConfig(cfg)
        self.adc_data = None
    
    def load_bin(self, bin_file : str, save_raw = False):
        """
        data format described in : mmwave_studio_user_guide.pdf
        Page.80, Section 24.8, DCA1000 EVM capture format (xWR16xx complex, 4 channel, 2 lanes [NonInterleaved])
        Also see : 
        D:\\ti\\mmwave_studio_02_01_01_00\mmWaveStudio\MatlabExamples\singlechip_raw_data_reader_example\rawDataReader.m
        dimensions for each frame: (Chirp, Channel, Sample)
        return ndarray with shape (-1, self.numChirpsPerFrame, self.numRxChan, self.numAdcSamples)
        """
        bin_files = os.listdir(bin_file)
        bin_files = [f for f in bin_files if f.endswith('.bin')]
        bin_files = [bin_file + '/' + f for f in bin_files]
        x = []
        for bin in bin_files:
            x.append(np.fromfile(bin, dtype = np.int16).astype('float'))
        x = np.concatenate(x)
        
        # always complex data
        x = ti_dca1000_real2complex(x)
        # return format: (frame, chirp, Tx, Rx, sample)
        x = x.reshape(-1, self.cfg.frameCfg_numLoops, self.cfg.numChirpsPerLoop, self.cfg.numRxChan, self.cfg.numAdcSamples)
        if save_raw:
            self.adc_data = np.copy(x)  # retain a copy of adc data within object
        else:
            self.adc_data = x  # save a reference to the adc data
        return x
    
    def save_to_mat(self, save_path : str):
        if self.adc_data is None:
            return
        io.savemat(save_path, {'adc_data' : self.adc_data, 'format' : '(frame, chirp, Tx, Rx, sample)'}, do_compression = True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('ADC Data Reader.')
    parser.add_argument('--config_file', required=False, type = str, default = 'awr1843_cfg.xml', help = 'path to xml format config file.')
    parser.add_argument('--adc_data',  required=False, type = str, default = 'data', help = 'path to adc_data.bin.')
    parser.add_argument('--save-mat', required=False, default = None, help = 'save to MATLAB format (.mat) data.')
    args = parser.parse_args()

    print('Config file:', args.config_file)
    print('ADC Data:', args.adc_data)
    if args.save_mat:
        print('Save to MATLAB format:', args.save_mat)
    if not os.path.exists(args.config_file):
        print('Error:', args.config_file, 'does not exist.')
        sys.exit(1)
    if not os.path.exists(args.adc_data):
        print('Error:', args.adc_data, 'does not exist.')
        sys.exit(1)
    adc_data = ADCData(args.config_file)
    x = adc_data.load_bin(args.adc_data)
    print('ADC Data Shape:', x.shape)
    if args.save_mat:
        adc_data.save_to_mat(args.save_mat)
