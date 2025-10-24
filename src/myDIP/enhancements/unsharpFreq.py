import numpy as np
from myDIP.fourier import Fourier2D
from myDIP.filters.frequency import FreqFilter


def unsharpFreq(input_img, freq_cutoff, f_type="Ideal", n_order=2, k=1):

    FFT = Fourier2D(input_img)
    FFT.fft()
    fft_magnitude = FFT.getMagnitude()

    FqFilter = FreqFilter(input_img.shape)
    
    hp_filter = FqFilter.getHPF(freq_cutoff, f_type, n_order)

    ifft_magnitude = (1 + k*hp_filter) * fft_magnitude

    FFT.setMagnitude(ifft_magnitude)
    FFT.ifft()

    output_img = FFT.getOutputImg()

    output_img = np.clip(output_img, 0, 255).astype(np.uint8)

    return output_img