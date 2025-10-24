import numpy as np
import matplotlib.pyplot as plt
from myDIP.filters.frequency import FreqFilter
from myDIP.fourier import Fourier2D
from myDIP.general import adjustRange



def laplacianFreq(input_img, c=-1):

    input_img = adjustRange(input_img, (0,255), (0,1))
    FFT = Fourier2D(input_img)
    FFT.fft()
    fft_magnitude = FFT.getMagnitude()

    FqFilter = FreqFilter(input_img.shape)

    lpc_filter = FqFilter.getLaplacian()

    lpc = (fft_magnitude*lpc_filter)
    # lpc = c*(fft_magnitude*lpc_filter)

    FFT.setMagnitude(lpc)
    # FFT.showMagnitude(log_scale=False)
    FFT.ifft()
    edge_img = FFT.getOutputImg()
    # edge_img = adjustRange(edge_img, (edge_img.min(), edge_img.max()), (-1,1))
    edge_img = edge_img/np.abs(edge_img).max()

    # print(edge_img.min(), edge_img.max())

    # plt.figure()
    # plt.imshow(input_img, cmap='gray')
    plt.figure()
    plt.imshow(lpc_filter, cmap='gray')
    plt.figure()
    plt.imshow(edge_img, cmap='gray')

    output_img = input_img + (c*edge_img)

    output_img = np.clip(output_img, 0, 1)
    output_img = adjustRange(output_img, (0, 1), (0,255)).astype(np.uint8)

    return output_img

# def laplacianFreq(input_img, c=-1):

#     input_img = adjustRange(input_img, (0,255), (0,1))
#     FFT = Fourier2D(input_img)
#     FFT.fft()
#     fft_magnitude = FFT.getMagnitude()
#     fft_magnitude = fft_magnitude/fft_magnitude.max()

#     FqFilter = FreqFilter(input_img.shape)

#     lpc_filter = FqFilter.getLaplacian()
#     # print(lpc_filter.min(), lpc_filter.max())
#     # lpc_filter = lpc_filter/lpc_filter.max()

#     # lpc_filter = adjustRange(lpc_filter, (lpc_filter.min(), lpc_filter.max()), (-1,1))

#     sharpening_magnitude = (1+(c*lpc_filter))*fft_magnitude
#     # plt.figure()
#     # plt.imshow(sharpening_magnitude, cmap='hot')


#     # lpc = lpc/np.abs(lpc).max()

#     # print(lpc.min(), lpc.max())

#     FFT.setMagnitude(sharpening_magnitude)
#     FFT.ifft()
#     output_img = FFT.getOutputImg()
#     # edge_img = adjustRange(edge_img, (edge_img.min(), edge_img.max()), (-1,1))
#     # edge_img = edge_img/np.abs(edge_img).max()

#     print(output_img.min(), output_img.max())

#     plt.figure()
#     plt.imshow(lpc_filter, cmap='gray')
#     plt.title("filter")
#     plt.figure()
#     plt.imshow(output_img, cmap='gray')
#     # plt.figure()
#     # plt.imshow(edge_img, cmap='gray')

#     # output_img = input_img + edge_img

#     # output_img = np.clip(output_img, 0, 1)
#     # output_img = adjustRange(output_img, (0, 1), (0,255)).astype(np.uint8)

#     return output_img

