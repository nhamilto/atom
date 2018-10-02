import importlib
import os
import numpy as np
import inspect
import pandas as pd
import datetime as dt
import scipy.signal as sps
import scipy as sp
from atom_functions import meta_data
"""
Library of functions and classes to proces data from the acoustic tomography array at
the National Wind Technology Center.

speaker and microphone time series data are in yyyymmddHHMMSS_AcouTomMainData.txt
Sonic anemometer and T/H probe time series data are in yyyymmddHHMMSS_AcouTomaux_dataa.txt


TDSI - time dependent stochastic inversion #TODO
==============================================
from the observed travel times, create over-determined least-squares regression to
detmine fluctuation velocity and temperature fields in the array.

"""


class meanField(object):
    """
    mean flow field object calucated from acoustic travel times
    """

    ####################################
    def __init__(self, travel_times=None):
        print('new mean field')

        if travel_times is None:
            print('need to supply travel times...')
        if isinstance(travel_times, str):
            self.travel_times = pd.read_csv(travel_times)
        else:
            self.travel_times = travel_times

        self.travel_times.index.name = 'pathID'
        self.travel_times.set_index(
            ['speaker', 'mic', self.travel_times.index], inplace=True)

        # extract a bit of meta data
        self.speakers = self.travel_times.index.unique(level=0).values
        self.nspeakers = len(self.speakers)

        self.mics = self.travel_times.index.unique(level=1).values
        self.nmics = len(self.mics)

        self.pathIDs = self.travel_times.index.unique(level=2).values
        self.npaths = len(self.pathIDs)

        self.frames = self.travel_times.columns.values
        self.nframes = len(self.frames)

        # path to calibration data and constants
        caldatapath = os.path.join(
            os.path.dirname(__file__), '..',
            'calibration_data/')  #TODO this should not be hardcoded!
        self = self.get_meta(caldatapath)
        self = self.get_calibration_info(caldatapath)

    ####################################
    def get_meta(self, constantspath):
        """
        get meta data for experiment

        Parameters:

            constantspath: directory path to file containing meta data
        """
        self.meta = meta_data()
        constants = self.get_constants(constantspath)
        self.meta.from_data(constants)

        return self

    ####################################
    def get_constants(self, constantspath):
        """
        get values of constants used in experiment

        Parameters:

            constantspath: str
                path to directory containing 'constants.py'
        """
        meta = importlib.machinery.SourceFileLoader(
            'constants', constantspath + 'constants.py').load_module()

        return meta

    ####################################
    def get_calibration_info(self, caldatapath):
        """
        get locations of speakers and mics, signal latency between particular devices,
        and sound propagation delays from speakers as a function of azimuth

        Parameters:
            caldatapath: str
                path to directory containing raw data
                    - 'average_latency_yymmdd.csv'
                    - 'mic_locations_yymmdd.csv'
                    - 'speaker_locations_yymmdd.csv'
                and/or containing processed data to import
                    - 'offsets.py'
        """
        # offsets = importlib.import_module(caldatapath)
        offsets = importlib.machinery.SourceFileLoader(
            'offsets', caldatapath + 'offsets.py').load_module()

        # latency 8 rows (speakers) by 8 columns (mics)
        self.latency = np.array(offsets.latency)

        # mic_locations(northing, easting, elevation) in feet
        self.mic_locations = np.array(offsets.mic_locations)
        self.mic_xy_m = self.mic_locations[:, 0:2] * 0.3048

        # speaker_locations(northing, easting, elevation) in feet
        self.speaker_locations = np.array(offsets.speaker_locations)
        self.speaker_xy_m = self.speaker_locations[:, 0:2] * 0.3048

        # physical distance between instruments
        # east-west distance
        mx = self.mic_xy_m[:, 0]
        mx = mx[:, np.newaxis].repeat(8, axis=1)
        sx = self.speaker_xy_m[:, 0]
        sx = sx[np.newaxis, :].repeat(8, axis=0)
        distx = mx - sx
        # north-south distance
        my = self.mic_xy_m[:, 1]
        my = my[:, np.newaxis].repeat(8, axis=1)
        sy = self.speaker_xy_m[:, 1]
        sy = sy[np.newaxis, :].repeat(8, axis=0)
        disty = my - sy
        # euclidean distance between each speaker/mic combo
        # instrument spacing is an 8x8 array (nspeakers, nmics)
        self.path_lengths = np.sqrt(distx**2 + disty**2)

        return self

    ####################################
    def calc_mean_field(self):
        """
        calculate the mean virtual temperature and velocity fields from recorded travel times
        """

        path_velocity = self.travel_times / self.path_lengths
