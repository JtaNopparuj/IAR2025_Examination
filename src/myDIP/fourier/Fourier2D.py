import numpy as np
from scipy import fftpack
import matplotlib.pyplot as plt
from myDIP.intensityTransform import logTransform
from scipy.signal.windows import gaussian, tukey

class Fourier2D:

    # def __init__(self, input_img):
    #     self.__input_img = input_img

    def __init__(self, input_img, zero_mean=False, window_func=None):
        self.__input_img = input_img
        self.__zero_mean = zero_mean
        self.__window_func = window_func

        self.__img_height = input_img.shape[0]
        self.__img_width = input_img.shape[1]

    def __zeroMean(self):

        if self.__zero_mean:
            output_img = self.__input_img - self.__input_img.mean()
        else:
            output_img = self.__input_img.copy()

        return output_img
    
    def __windowing(self, input_img):

        if self.__window_func is None:
            output_img = input_img
        elif self.__window_func == "Gaussian":
            gauss_std = lambda kernel_size : 0.3*((kernel_size-1)*0.5)+0.8
             # -> 1D Window Function
            winfunc_vert = gaussian(self.__img_height, gauss_std(self.__img_height)).reshape((1, -1))
            winfunc_horz = gaussian(self.__img_width, gauss_std(self.__img_width)).reshape((1, -1))
            # winfunc_vert = gaussian(self.__img_height, gauss_std(self.__img_height)).reshape((-1, 1))
            # winfunc_horz = gaussian(self.__img_width, gauss_std(self.__img_width)).reshape((-1, 1))

            # print(winfunc_vert.shape)
            # print(winfunc_horz.shape)
            self.__window = winfunc_horz * winfunc_vert.T
            # self.__window = winfunc_vert* winfunc_horz.T
            # print(self.__window.shape)
            output_img = input_img * self.__window

            # plt.figure()
            # plt.imshow(self.__window, cmap='gray')
            # plt.figure()
            # plt.imshow(output_img, cmap='gray')

        elif self.__window_func == "Tukey":
            winfunc_vert = tukey(self.__img_height, 0.5).reshape((1, -1))
            winfunc_horz = tukey(self.__img_width, 0.5).reshape((1, -1))
            self.__window = winfunc_horz* winfunc_vert.T
            output_img = input_img * self.__window

            # plt.figure()
            # plt.imshow(output_img, cmap='gray')

        return output_img

    def fft(self):
        
        preproc_img = self.__zeroMean()
        preproc_img = self.__windowing(preproc_img)

        # -> Fast Fourier Transform
        fft_complex = fftpack.fft2(preproc_img)

        # -> Split Magnitude and Phase
        self.__fft_magnitude = np.abs(fft_complex)
        self.__fft_phase = np.arctan2(fft_complex.imag, fft_complex.real)

        # -> Shift Quadrant
        self.__fft_magnitude = fftpack.fftshift(self.__fft_magnitude)

    # def fft(self):
    #     # -> Fast Fourier Transform
    #     fft_complex = fftpack.fft2(self.__input_img)

    #     # -> Split Magnitude and Phase
    #     self.__fft_magnitude = np.abs(fft_complex)
    #     self.__fft_phase = np.arctan2(fft_complex.imag, fft_complex.real)

    #     # -> Shift Quadrant
    #     self.__fft_magnitude = fftpack.fftshift(self.__fft_magnitude)

    def ifft(self):

        # -> Shift back Quadrant
        ifft_magnitude = fftpack.ifftshift(self.__fft_magnitude)

        # -> Combine Magnitude and Phase
        ifft_real = ifft_magnitude * np.cos(self.__fft_phase) 
        ifft_imag = ifft_magnitude * np.sin(self.__fft_phase)

        ifft_complex = ifft_real + (ifft_imag * 1j)

        # -> Invert FFT
        output_complex = fftpack.ifft2(ifft_complex)

        # -> Get only real part
        self.__output_img = output_complex.real

    def getOutputImg(self):
        return self.__output_img
    
    def getMagnitude(self):
        return self.__fft_magnitude
    
    def setMagnitude(self, fft_magnitude):
        self.__fft_magnitude = fft_magnitude

    def showMagnitude(self, log_scale=False):

        display_img = self.__fft_magnitude.copy()
        if log_scale:
            # display_img = display_img/display_img.max()
            display_img = logTransform(display_img, to_uint8=False)

        plt.figure()
        plt.imshow(display_img, cmap='hot')
        # plt.show()

