import numpy as np
from myDIP.general import adjustRange
import matplotlib.pyplot as plt

class FreqFilter:

    def __init__(self, filter_size):
        
        self.__img_height = filter_size[0]
        self.__img_width = filter_size[1]

    
    # def __distanceMap(self):

    #     v = np.arange(0, self.__img_height)
    #     u = np.arange(0, self.__img_width)
    #     # v = np.arange(0.5*((self.__img_height+1)%2), 0.5*((self.__img_height+1)%2)+self.__img_height)
    #     # u = np.arange(0.5*((self.__img_width+1)%2), 0.5*((self.__img_width+1)%2)+self.__img_width)

    #     uv, vv = np.meshgrid(u, v)  # (u, v)

    #     center_v, center_u = (self.__img_height//2, self.__img_width//2)

    #     self.__distance_map = np.sqrt((vv - center_v)**2 + (uv - center_u)**2)

    def __distanceMap(self, center_pos=None, scale=True):

        v = np.arange(0, self.__img_height)
        u = np.arange(0, self.__img_width)
        # v = np.arange(0.5*((self.__img_height+1)%2), 0.5*((self.__img_height+1)%2)+self.__img_height)
        # u = np.arange(0.5*((self.__img_width+1)%2), 0.5*((self.__img_width+1)%2)+self.__img_width)

        uv, vv = np.meshgrid(u, v)  # (u, v)

        # scale = self.__img_width / self.__img_height
        if scale:
            scale_v = self.__img_height / max(self.__img_height, self.__img_width)
            scale_u = self.__img_width / max(self.__img_height, self.__img_width)
        else:
            scale_v, scale_u = 1,1

        if center_pos is None:
            center_v, center_u = (self.__img_height//2, self.__img_width//2)
        else:
            center_v, center_u = center_pos

        self.__distance_map = np.sqrt(((vv - center_v)/scale_v)**2 + ((uv - center_u)/scale_u)**2)

    def __LPF(self, freq_cutoff, f_type="Ideal", n_order=2):

        if f_type == "Ideal":
            self.__filter = np.where(self.__distance_map <= freq_cutoff, 1, 0)

        elif f_type == "Gaussian":
            self.__filter = np.exp(-self.__distance_map**2 / (2*freq_cutoff**2))

        elif f_type == "Butterworth":
            self.__filter = 1 / (1 + (self.__distance_map/freq_cutoff)**(2*n_order))

    def __BPF(self, band_center, band_width, f_type="ideal", n_order=2):

        eps = 1e-6

        if f_type == "Ideal":
            self.__filter = np.where((self.__distance_map >= (band_center-band_width/2)) & \
                                     (self.__distance_map <= (band_center+band_width/2)),
                                     1, 0)
        elif f_type == "Gaussian":
            self.__filter = np.exp(-((self.__distance_map**2-band_center**2) / ((self.__distance_map*band_width)+eps))**2)

        elif f_type == "Butterworth":
            self.__filter = 1 / (1 + ((self.__distance_map**2-band_center**2) / ((self.__distance_map*band_width)+eps))**(2*n_order))


    def __laplacian(self):

        # self.__filter = -4 * (np.pi**2) * (self.__distance_map**2)
        self.__filter = 4 * (np.pi**2) * (self.__distance_map**2)
        # self.__filter = -(self.__distance_map**2)


    def getLPF(self, freq_cutoff, f_type="Ideal", n_order=2, center_pos=None, scale=True):

        self.__distanceMap(center_pos, scale)

        self.__LPF(freq_cutoff, f_type, n_order)

        return self.__filter
    
    def getHPF(self, freq_cutoff, f_type="Ideal", n_order=2, center_pos=None, scale=True):

        lowpass_filter = self.getLPF(freq_cutoff, f_type, n_order, center_pos, scale)

        self.__filter = 1 - lowpass_filter

        return self.__filter
    
    def getBPF(self, band_center, band_width, f_type="ideal", n_order=2):

        self.__distanceMap()

        self.__BPF(band_center, band_width, f_type, n_order)

        # print(self.__distance_map.min())

        return self.__filter
    
    def getBRF(self, band_center, band_width, f_type="ideal", n_order=2):

        bandpass_filter = self.getBPF(band_center, band_width, f_type, n_order)

        self.__filter = 1 - bandpass_filter

        return self.__filter
    

    def getSelectiveFilter(self, pos_list, radius_list, pass_filter=False, f_type="Ideal", n_order=2):

        # -> Check if radius or pos are not in List form
        assert isinstance(radius_list, list), "\"radius_list\" must be in list form, e.g., [r1, r2, ...]."
        assert isinstance(pos_list, list), "\"pos_list\" must be in list form, e.g., [(pos_v, pos_u)]."

        select_filter = np.ones((self.__img_height, self.__img_width))

        for pos, r in zip(pos_list, radius_list):
            # print(r[0], pos)
            hp_filter = self.getHPF(r, f_type, n_order, pos, False)

            mpos_v = self.__img_height - pos[0] - (1 if self.__img_height % 2 != 0 else 0)
            mpos_u = self.__img_width - pos[1] - (1 if self.__img_width % 2 != 0 else 0)

            # if self.__img_height%2 == 0:
            #     mpos_y = self.__img_height-pos[0]

            # else:
            #     mpos_y = self.__img_height-pos[0]-1

            # if self.__img_width%2 == 0:
            #     mpos_x = self.__img_width-pos[1]

            # else:
            #     mpos_x = self.__img_width-pos[1]-1

            mirror_pos = (mpos_v, mpos_u)

            hp_mfilter = self.getHPF(r, f_type, n_order, mirror_pos, False)

            select_filter = select_filter * hp_filter * hp_mfilter

        if pass_filter:
            select_filter = 1 - select_filter

        return select_filter



    
    def getLaplacian(self):

        self.__distanceMap()

        self.__laplacian()

        # self.__filter = self.__filter/np.abs(self.__filter).max()

        return self.__filter


    def getDistanceMap(self):

        return self.__distance_map

        

