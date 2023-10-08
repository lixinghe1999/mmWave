import sys
import numpy as np
import mmwave.dsp as dsp
import mmwave.clustering as clu
from mmwave.dataloader import DCA1000

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from numpy import linalg
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.colors as colors

plt.close('all')

# QOL settings
loadData = True

numFrames = 300
numADCSamples = 128
numTxAntennas = 3
numRxAntennas = 4
numLoopsPerFrame = 128
numChirpsPerFrame = numTxAntennas * numLoopsPerFrame

numRangeBins = numADCSamples
numDopplerBins = numLoopsPerFrame
numAngleBins = 64

range_resolution, bandwidth = dsp.range_resolution(numADCSamples)
doppler_resolution = dsp.doppler_resolution(bandwidth)
plotRangeDopp = True  
plot2DscatterXY = False  
plot2DscatterXZ = False  
singFrameView = True


if __name__ == '__main__':
    ims = []
    max_size = 0

    # (1) Reading in adc data
    if loadData:
        adc_data = np.fromfile('./data/1_person_walking_128loops.bin', dtype=np.uint16)
        print(adc_data.shape)
        adc_data = adc_data.reshape(numFrames, -1)
        adc_data = np.apply_along_axis(DCA1000.organize, 1, adc_data, num_chirps=numChirpsPerFrame,
                                       num_rx=numRxAntennas, num_samples=numADCSamples)
        print("Data Loaded!")

    # (1.5) Required Plot Declarations
    if plotRangeDopp:
        fig = plt.figure()
    else:
        fig, axes = plt.subplots(1, 2)

    # (1.6) Optional single frame view
    if singFrameView:
        dataCube = np.zeros((1, numChirpsPerFrame, 4, 128), dtype=complex)
        dataCube[0, :, :, :] = adc_data[100]
    else:
        dataCube = adc_data

    for i, frame in enumerate(dataCube):
#        print(i,end=',') # Frame tracker
        # (2) Range Processing
        from mmwave.dsp.utils import Window

        radar_cube = dsp.range_processing(frame, window_type_1d=Window.BLACKMAN)
        assert radar_cube.shape == (
        numChirpsPerFrame, numRxAntennas, numADCSamples), "[ERROR] Radar cube is not the correct shape!"

        # (3) Doppler Processing 
        det_matrix, aoa_input = dsp.doppler_processing(radar_cube, num_tx_antennas=3, 
                                                       clutter_removal_enabled=True, 
                                                       window_type_2d=Window.HAMMING)

        # (4) Object Detection
        # --- CFAR, SNR is calculated as well.
        fft2d_sum = det_matrix.astype(np.int64)
        thresholdDoppler, noiseFloorDoppler = np.apply_along_axis(func1d=dsp.ca_,
                                                                  axis=0,
                                                                  arr=fft2d_sum.T,
                                                                  l_bound=1.5,
                                                                  guard_len=4,
                                                                  noise_len=16)

        thresholdRange, noiseFloorRange = np.apply_along_axis(func1d=dsp.ca_,
                                                              axis=0,
                                                              arr=fft2d_sum,
                                                              l_bound=2.5,
                                                              guard_len=4,
                                                              noise_len=16)

        thresholdDoppler, noiseFloorDoppler = thresholdDoppler.T, noiseFloorDoppler.T
        det_doppler_mask = (det_matrix > thresholdDoppler)
        det_range_mask = (det_matrix > thresholdRange)

        # Get indices of detected peaks
        full_mask = (det_doppler_mask & det_range_mask)
        det_peaks_indices = np.argwhere(full_mask == True)

        # peakVals and SNR calculation
        peakVals = fft2d_sum[det_peaks_indices[:, 0], det_peaks_indices[:, 1]]
        snr = peakVals - noiseFloorRange[det_peaks_indices[:, 0], det_peaks_indices[:, 1]]

        dtype_location = '(' + str(numTxAntennas) + ',)<f4'
        dtype_detObj2D = np.dtype({'names': ['rangeIdx', 'dopplerIdx', 'peakVal', 'location', 'SNR'],
                                   'formats': ['<i4', '<i4', '<f4', dtype_location, '<f4']})
        detObj2DRaw = np.zeros((det_peaks_indices.shape[0],), dtype=dtype_detObj2D)
        detObj2DRaw['rangeIdx'] = det_peaks_indices[:, 0].squeeze()
        detObj2DRaw['dopplerIdx'] = det_peaks_indices[:, 1].squeeze()
        detObj2DRaw['peakVal'] = peakVals.flatten()
        detObj2DRaw['SNR'] = snr.flatten()

        # Further peak pruning. This increases the point cloud density but helps avoid having too many detections around one object.
        detObj2DRaw = dsp.prune_to_peaks(detObj2DRaw, det_matrix, numDopplerBins, reserve_neighbor=True)

        # --- Peak Grouping
        detObj2D = dsp.peak_grouping_along_doppler(detObj2DRaw, det_matrix, numDopplerBins)
        SNRThresholds2 = np.array([[2, 23], [10, 11.5], [35, 16.0]])
        peakValThresholds2 = np.array([[4, 275], [1, 400], [500, 0]])
        detObj2D = dsp.range_based_pruning(detObj2D, SNRThresholds2, peakValThresholds2, numRangeBins, 0.5, range_resolution)
        azimuthInput = aoa_input[detObj2D['rangeIdx'], :, detObj2D['dopplerIdx']]
        x, y, z = dsp.naive_xyz(azimuthInput.T)
        xyzVecN = np.zeros((3, x.shape[0]))
        xyzVecN[0] = x * range_resolution * detObj2D['rangeIdx']
        xyzVecN[1] = y * range_resolution * detObj2D['rangeIdx']
        xyzVecN[2] = z * range_resolution * detObj2D['rangeIdx']

        Psi, Theta, Ranges, xyzVec = dsp.beamforming_naive_mixed_xyz(azimuthInput, detObj2D['rangeIdx'],
                                                                     range_resolution, method='Bartlett')
        # (6) Visualization
        if plotRangeDopp:

            det_matrix_vis = np.fft.fftshift(det_matrix, axes=1)
            plt.imshow(det_matrix_vis / det_matrix_vis.max(), aspect='equal',
                       extent=[-doppler_resolution * numDopplerBins/2, doppler_resolution * numDopplerBins/2,
                               -range_resolution * numRangeBins / 2, range_resolution * numRangeBins / 2,] )
            plt.ylabel('Range')
            plt.xlabel('Doppler')
            plt.title("Range-Doppler plot " + str(i))
            if singFrameView:
                plt.show()
            else:
                plt.pause(0.05)
                plt.clf()
        elif plot2DscatterXY or plot2DscatterXZ:
            if plot2DscatterXY:
                xyzVec = xyzVec[:, (np.abs(xyzVec[2]) < 1.5)]
                xyzVecN = xyzVecN[:, (np.abs(xyzVecN[2]) < 1.5)]
                axes[0].set_ylim(bottom=0, top=10)
                axes[0].set_ylabel('Range')
                axes[0].set_xlim(left=-4, right=4)
                axes[0].set_xlabel('Azimuth')
                axes[0].grid(b=True)

                axes[1].set_ylim(bottom=0, top=10)
                axes[1].set_xlim(left=-4, right=4)
                axes[1].set_xlabel('Azimuth')
                axes[1].grid(b=True)
                axes[0].scatter(xyzVec[0], xyzVec[1], c='r', marker='o', s=3)
                axes[1].scatter(xyzVecN[0], xyzVecN[1], c='b', marker='o', s=3)
            elif plot2DscatterXZ:
                axes[0].set_ylim(bottom=-5, top=5)
                axes[0].set_ylabel('Elevation')
                axes[0].set_xlim(left=-4, right=4)
                axes[0].set_xlabel('Azimuth')
                axes[0].grid(b=True)

                axes[1].set_ylim(bottom=-5, top=5)
                axes[1].set_xlim(left=-4, right=4)
                axes[1].set_xlabel('Azimuth')
                axes[1].grid(b=True)
                axes[0].scatter(xyzVec[0], xyzVec[2], c='r', marker='o', s=3)
                axes[1].scatter(xyzVecN[0], xyzVecN[2], c='b', marker='o', s=3)
            if singFrameView:
                plt.show()
            else:
                plt.pause(0.1)
                axes[0].clear()
                axes[1].clear()
        else:
            sys.exit("Unknown plot options.")
       