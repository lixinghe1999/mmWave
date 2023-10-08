import os
import sys
from dataclasses import dataclass
import xml.etree.ElementTree as ET
import numpy as np

@dataclass
class UnitScale:
    us = 1e-6
    KSps = 1e3
    GHz = 1e9
    MHz = 1e6


def convert_param_node_to_dict(node):
    params = {}
    for item in node:
        params[item.get('name')] = item.get('value')
    return params


def read_config_from_xml(xml_file):
    """read the config and convert them to standard unit:
        distance (m), frequency (Hz), time (sec),
        sample rate (sps, sample per sec.)
        frequency slope (Hz / sec.)
    """
    tree = ET.parse(xml_file)
    root = tree.getroot()
    cfg = {}
    frame_cfg = convert_param_node_to_dict(root.find('apiname_frame_cfg'))
    cfg['frameCfg_chirpStartIdx'] = int(frame_cfg['fchirpStartIdx'])
    cfg['frameCfg_chirpEndIdx'] = int(frame_cfg['fchirpEndIdx'])
    cfg['frameCfg_numLoops'] = int(frame_cfg['loopCount'])
    profile_cfg = convert_param_node_to_dict(root.find('apiname_profile_cfg'))
    cfg['gAdcOneSampleSize'] = 4  # (16bits ADC, complex data)
    cfg['numAdcSamples'] = int(profile_cfg['numAdcSamples'])
    cfg['numRxChan'] = 4
    cfg['digOutSampleRate'] = float(profile_cfg['digOutSampleRate']) * UnitScale.KSps  # ! kps
    cfg['freqSlope'] = float(profile_cfg['freqSlopeConst']) * UnitScale.MHz / UnitScale.us  # ! MHz/us
    cfg['adcStartTime'] = float(profile_cfg['adcStartTimeConst']) * UnitScale.us # ! us
    cfg['startFreq'] = float(profile_cfg['startFreqConst']) * UnitScale.GHz # ! GHz
    cfg['idleTime'] = float(profile_cfg['idleTimeConst']) * UnitScale.us # ! us
    cfg['rampEndTime'] = float(profile_cfg['rampEndTime']) * UnitScale.us # ! us
    # calculated params
    cfg['numChirpsPerLoop'] = cfg['frameCfg_chirpEndIdx'] - cfg['frameCfg_chirpStartIdx'] + 1
    cfg['numChirpsPerFrame'] = cfg['frameCfg_numLoops'] * cfg['numChirpsPerLoop']
    cfg['dataSizeOneChirp'] = cfg['gAdcOneSampleSize'] * cfg['numAdcSamples'] * cfg['numRxChan']
    cfg['dataSizeOneFrame'] = cfg['dataSizeOneChirp'] * cfg['numChirpsPerFrame']
    cfg['bandWidth'] = float(cfg['numAdcSamples']) / cfg['digOutSampleRate'] * cfg['freqSlope']  # GHz
    cfg['speedOfLight'] = 299792458   # m/s
    return cfg


class RadarConfig:
    '''
    Class for parsing and storing configs.
    '''
    def __init__(self, xml_file : str = None, **kwargs):
        if xml_file:
            cfg = read_config_from_xml(xml_file)
        else:
            cfg = kwargs
        for k, v in cfg.items():
            setattr(self, k, v)
        self.cfg_dict = cfg

    def freq2dist(self, f):
        """convert frequency to distance by d = f * C / (2 * slope)
        """
        return f * self.speedOfLight / (2 * self.freqSlope)

    def dist2freq(self, d):
        """convert distance to frequency
        """
        return d * (2 * self.freqSlope) / self.speedOfLight

    def gen_time(self, n : int = None, step : float = UnitScale.us):
        """return time series for ploting """
        n = n or self.numAdcSamples
        return np.array(range(n)) / self.digOutSampleRate / step


if __name__ == '__main__':
    print('Radar Config Reader.')
    if len(sys.argv) != 2:
        print('usage:', sys.argv[0], 'path_to_xml_file')
        sys.exit(0)
    xml_file = sys.argv[1]
    if not os.path.exists(xml_file):
        print('Error:', xml_file, 'does not exist.')
        sys.exit(1)
    cfg = RadarConfig(xml_file)
    print('Configs:')
    for k, v in cfg.cfg_dict.items():
        print(k, ':', v)
