# import numpy as np
# import tkinter as tk
# from tkinter import filedialog
# import glob
# from tqdm import tqdm
# import math
# import scipy as sp
# import scipy.signal as sps

import importlib
import os
import numpy as np
import inspect
import pandas as pd
import datetime as dt
import scipy.signal as sps
import scipy as sp

# import sys
# sys.path.append('calibration_data/')
"""
Library of functions and classes to proces data from the acoustic tomography array at
the National Wind Technology Center.

speaker and microphone time series data are in yyyymmddHHMMSS_AcouTomMainData.txt
Sonic anemometer and T/H probe time series data are in yyyymmddHHMMSS_AcouTomaux_dataa.txt

key stops:
___________

generate file list


travel time extraction
=======================
calculate the propagation time of acoustic signals between speakers and microphones in
the array. With known transducer locations, the 'expected' travel time can be calculated
with the current estimate of speed of sound (c) from the sonic anemometer.

TDSI - time dependent stochastic inversion
==============================================
from the observed travel times, create over-determined least-squares regression to
detmine fluctuation velocity and temperature fields in the array.

"""


####################################
class dataset(object):
    """
    dataset is the object class for raw data. It should contain a directory in where
    raw data are to be found, data I/O routines, experiement constants, array
    calibration info, etc.

    """

    ####################################
    def __init__(self, datapath):
        self.datapath = datapath
        filelist = os.listdir(datapath)
        self.mainfiles = [x for x in filelist if 'main' in x.lower()]
        self.auxfiles = [x for x in filelist if 'aux' in x.lower()]
        # sanity check
        if len(self.mainfiles) is not len(self.auxfiles):
            print('Number of main files does not match number of aux files.')

        # path to calibration data and constants
        caldatapath = os.path.join(
            os.path.dirname(__file__), 'calibration_data/')
        self = self.get_meta(caldatapath)
        self = self.get_calibration_info(caldatapath)

    ####################################
    def get_meta(self, constantspath):
        """
        get meta data for experiment
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
        ____________
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
        ____________
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
        self.instrument_spacing = np.sqrt(distx**2 + disty**2)

        return self

    ####################################
    def load_aux(self, sampletime):  #maindatapath
        """
        load data file into dataset object

        Parameters:
        ____________
        aux_dataapath: str
            path to directory containing aux data
        """

        if type(sampletime) is int:
            sampletime = self.datapath + self.auxfiles[sampletime]

        # column names for the data
        colnames = ['vx', 'vy', 'vz', 'c', 'T', 'H']
        # load into pd.DataFrame
        aux_data = pd.read_csv(
            sampletime, skiprows=4, names=colnames, delim_whitespace=True)
        # get the timestamp from each filename
        timestamp = sampletime.split('/')[-1].split('_')[0]
        # calculate and assign timestamp as index
        aux_data.set_index(
            pd.DatetimeIndex(
                freq='0.05S',
                start=dt.datetime.strptime(timestamp, '%Y%m%d%H%M%S'),
                periods=len(aux_data.index)),
            inplace=True)

        self.aux_data = aux_data

    ####################################
    def load_main(self, sampletime):  #maindatapath
        """
        load data file into dataset object

        Parameters:
        ____________
        maindatapath: str
            path to directory containing main data
        """

        if type(sampletime) is int:
            sampletime = self.datapath + self.mainfiles[sampletime]

        # column names for the data
        colnames = ['S{}'.format(x)
                    for x in range(8)] + ['M{}'.format(x) for x in range(8)]

        # load into pd.DataFrame
        main_data = pd.read_csv(
            sampletime, skiprows=4, names=colnames, delim_whitespace=True)

        # get the timestamp from each filename
        timestamp = sampletime.split('/')[-1].split('_')[0]

        # calculate and assign timestamp as index
        main_data.set_index(
            pd.DatetimeIndex(
                freq='50U',
                start=dt.datetime.strptime(timestamp, '%Y%m%d%H%M%S'),
                periods=len(main_data.index)),
            inplace=True)

        # calculate the number of records within the file
        nrec = int(len(main_data) / 10000)
        recindex = [
            main_data.index[0] + ii * pd.Timedelta(value=0.5, unit='s')
            for ii in range(nrec)
        ]
        # add record number as a series
        recdata = ['record {}'.format(ii) for ii in range(nrec)]
        recordseries = pd.Series(data=recdata, index=recindex)
        main_data['record'] = recordseries
        main_data['record'].ffill(inplace=True)

        # reindex by both the record number and the time index
        main_data.set_index(['record', main_data.index], inplace=True)
        # self.main_data = main_data

        # split into speaker data and mic data
        self.speaker_data = main_data[[
            x for x in main_data.columns if 'S' in x
        ]]
        self.mic_data = main_data[[x for x in main_data.columns if 'M' in x]]

    ####################################
    def load_data_sample(self, fileID):
        """
        load data file into dataset object

        Parameters:
        ____________
        fileID: int
            integer value of main and aux data in respective lists
        """
        if fileID > len(self.mainfiles):
            print('Ran out of data files ...')
            pass

        self.load_main(fileID)

        self.load_aux(fileID)

        # return self

    ####################################
    def upsample_data(self, upsamplefactor):  #TODO
        """
        artificially upsample data to provide the desired resolution
        new_timedelta = oldtimedelta / upsamplefactor
        upsamplefactor > 1 ==> increase in time resolution
        upsamplefactor < 1 ==> decrease in time resolution

        Parameters:
        ____________
        upsamplefactor: float or int
            scale by which to resample data

        """
        # cubic interpolation of aux_data
        delta_t = self.aux_data.index[1] - self.aux_data.index[0]
        rule = delta_t / upsamplefactor

        self.aux_data = self.aux_data.resample('{}U'.format(
            rule.microseconds)).interpolate(method='cubic')

        # cubic interpolation of speaker data
        delta_t = self.speaker_data.index[1] - self.speaker_data.index[0]
        rule = delta_t / upsamplefactor

        self.speaker_data = self.speaker_data.resample('{}U'.format(
            rule.microseconds)).interpolate(method='cubic')

        # cubic interpolation of micdata
        delta_t = self.mic_data.index[1] - self.mic_data.index[0]
        rule = delta_t / upsamplefactor

        self.mic_data = self.mic_data.resample('{}U'.format(
            rule.microseconds)).interpolate(method='cubic')

    ####################################
    def time_info(self):
        """
        print time resolution of main and aux data
        """

        main_delta_t = self.speaker_data.index[1] - self.speaker_data.index[0]
        print('main data data resolution = {}us'.format(
            main_delta_t.microseconds))
        main_span_t = self.speaker_data.index[-1] - self.speaker_data.index[0]
        print('main data spans {}s'.format(main_span_t.seconds))

        main_delta_t = self.mic_data.index[1] - self.mic_data.index[0]
        print('main data data resolution = {}us'.format(
            main_delta_t.microseconds))
        main_span_t = self.mic_data.index[-1] - self.mic_data.index[0]
        print('main data spans {}s'.format(main_span_t.seconds))

        aux_delta_t = self.aux_data.index[1] - self.aux_data.index[0]
        print('auxiliary data resolution = {}us'.format(
            aux_delta_t.microseconds))
        aux_span_t = self.aux_data.index[-1] - self.aux_data.index[0]
        print('auxiliary data spans {}s'.format(aux_span_t.seconds))

    ####################################
    def signal_ETAs(self):
        """
        imports the current speaker (i) and microphone (j) as well as the temperature array for the current signal period
        """

        c0 = self.aux_data['c'].mean()

        self.ETAs_time = self.instrument_spacing / c0
        self.ETAs_index = (
            self.ETAs_time / self.meta.main_delta_t).round().astype(int)

    ####################################
    def extract_travel_times(self,
                             upsamplefactor=10,
                             searchLag=None,
                             filterflag=True,
                             verbose=False):
        """
        The guts.
        """

        # width of search window in index value
        if searchLag is None:
            searchLag = 3 * self.meta.chirp_record_length * upsamplefactor
        if verbose:
            print('searchLag = ', searchLag)

        # get record ID's from speakers
        records = list(self.speaker_data.index.unique(level=0))
        nrecords = len(records)

        # estimate signal travel times based on instrument locations and the
        # average speed of sound reported in auxdata
        self.signal_ETAs()

        # maximum resolved frequency of microphone
        max_freq = self.meta.main_f_range.max()

        # get a speaker signals from a single record
        speakersamp = self.speaker_data.xs(records[0], level=0)
        # if upsampling is required
        if upsamplefactor is not 1:
            speakersamp = upsample(speakersamp, upsamplefactor)

        # get indices speaker signal offsets
        speaker_signal_delay = get_speaker_signal_delay(speakersamp)

        # get speaker signals and emission times
        speakersigs, speakerstarttime = signalOnSpeaker(
            speakersamp, searchLag, self.meta.chirp_N_records)
        # adjust time index of channel three
        speakersigs['S3'] = rollchannel(speakersigs['S3'],
                                        int(self.meta.chirp_N_records / 2))

        if verbose:
            print('speakerstarttime:', speakerstarttime)

        # estimate arrival time of signals from speaker emission and spacing
        signalETAs = (speakerstarttime + self.ETAs_index).astype(int)
        if verbose:
            print('signalETAs (index):', signalETAs)

        # allocate space for travel times between each speaker/mic combo
        travel_times = np.zeros((self.meta.nspeakers, self.meta.nmics,
                                 nrecords))

        # allocate space for received signals (nspeakers, nmics, ndata, nrecords)
        ATom_signals = np.zeros((self.meta.nspeakers, self.meta.nmics,
                                 searchLag, nrecords))

        # cycle through each record:
        # detect speaker signal emissions
        # detect microphone signal receptions
        for nrec, record in enumerate(records):

            # extract a single record
            micsamp = self.mic_data.xs(record, level=0)

            # if upsampling is required
            if upsamplefactor is not 1:
                micsamp = upsample(micsamp, upsamplefactor)

            # get microphone singals and reception times
            micsigs, time_received = signalOnMic(
                micsamp, speakersigs, signalETAs, speaker_signal_delay,
                searchLag, self.meta.chirp_N_records)

            ATom_signals[..., nrec] = micsigs
            travel_times[..., nrec] = time_received - speaker_signal_delay

        return ATom_signals, travel_times


####################################
class meta_data(object):
    """
    Base class for instrument or data record meta data.
    Takes in a list of parameters and values.
    """

    def __init__(self):
        return

    def from_data(self, data):
        """
        pass data from constants file,
        create attribute/value pairs
        use to create meta data attributes
        """
        keys = [
            x for x in data.__dict__.keys() if '__' not in x
            if not inspect.ismodule(data.__dict__[x])
        ]
        values = [data.__dict__[x] for x in keys]
        for (key, value) in zip(keys, values):
            self.__dict__[key] = value
        return self

    def from_lists(self, keys, values):
        """
        instantiate meta_data object from lists of attribute/value pairs
        """
        for (key, value) in zip(keys, values):
            self.__dict__[key] = value
        return self

    def from_file(self, keys, values):
        """
        instantiate meta_data object from lists of file pairs
        """
        for (key, value) in zip(keys, values):
            self.__dict__[key] = value
        return self

    def to_list(self):
        """
        pretty print list of attributes
        """
        for k, v in self.__dict__.items():
            print('{:25} {}'.format(k, v, type(v)))


####################################
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = sps.butter(order, [low, high], btype='bandpass')
    return b, a


####################################
def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):

    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = sps.lfilter(b, a, data, axis=0)
    return y


####################################
def upsample(datasample, upsamplefactor):
    """
    upsample data by desired factor

    """
    delta_t = datasample.index[1] - datasample.index[0]
    rule = delta_t / upsamplefactor
    newdatasample = datasample.resample('{}U'.format(
        rule.microseconds)).interpolate(method='cubic')
    return newdatasample


####################################
def signalOnSpeaker(speakersamp, searchLag, chirp_N_records):

    # expected delay (in index) of each acoustic signal
    speaker_signal_delay = get_speaker_signal_delay(speakersamp)

    # beginning index of speaker signal to extract
    signalstarts = [
        int(s) if s > 0 else 0
        for s in (speaker_signal_delay + chirp_N_records / 2) - searchLag / 2
    ]

    # end index of speaker signal to extract
    signalends = [s + searchLag for s in signalstarts]

    # make dict, sample data, return new dataframe
    sample = {
        speakersamp.columns[i]: range(signalstarts[i], signalends[i])
        for i in range(len(speaker_signal_delay))
    }
    speakersigs = {k: speakersamp[k].iloc[v].values for k, v in sample.items()}
    speakersigs = pd.DataFrame.from_dict(speakersigs)

    return speakersigs, speaker_signal_delay


####################################
def signalOnMic_depricated(micsamp, speakersigs, signalETAs,
                           speaker_signal_delay, searchLag, chirp_N_records):

    # mic_signal_delay = speaker_signal_delay + signalETAs

    signalstarts = np.array([
        int(s) if s > 0 else 0
        for s in (signalETAs.flatten() + chirp_N_records) - 0.35 * searchLag
    ]).reshape(8, 8)

    signalends = signalstarts + searchLag
    micsigs = np.zeros((8, 8, searchLag))

    for mi, mic in enumerate(micsamp.columns):

        sample = {
            micsamp.columns[i]: range(signalstarts[:, mi][i],
                                      signalends[:, mi][i])
            for i in range(8)
        }

        received_signals = {
            k: micsamp[mic].iloc[v].values
            for k, v in sample.items()
        }
        received_signals = pd.DataFrame.from_dict(received_signals)

        tmp = received_signals.rolling(
            window=int(len(received_signals) / 4), center=True).cov(
                speakersigs, pairwise=True)

        micsigs[:, mi, :] = received_signals.T.values
        time_received = tmp.unstack().idxmax().unstack()

    return micsigs, time_received


####################################
def signalOnMic(micsamp, speakersigs, signalETAs, speaker_signal_delay,
                searchLag, chirp_N_records):

    # mic_signal_delay = speaker_signal_delay + signalETAs

    signalstarts = np.array([
        int(s) if s > 0 else 0
        for s in (signalETAs.flatten() + chirp_N_records) - 0.35 * searchLag
    ]).reshape(8, 8)

    signalends = signalstarts + searchLag
    micsigs = np.zeros((8, 8, searchLag))

    for mi, mic in enumerate(micsamp.columns):

        sample = {
            micsamp.columns[i]: range(signalstarts[:, mi][i],
                                      signalends[:, mi][i])
            for i in range(8)
        }

        received_signals = {
            k: micsamp[mic].iloc[v].values
            for k, v in sample.items()
        }
        received_signals = pd.DataFrame.from_dict(received_signals)

        covar = covariance(received_signals, speakersigs)

        micsigs[:, mi, :] = received_signals.T.values

        time_received = np.zeros((8, 8))
        for ii in range(8):
            for jj in range(8):
                time_received[ii, jj] = np.argmax(covar[ii, jj, :])

    return micsigs, time_received


####################################
def get_speaker_signal_delay(speakersamp):
    """
    extract the speaker signal delays from a single record

    Parameters
    ___________
    speakersamp: pd.DataFrame
        speaker time series data for a single record

    Outputs
    ----------
    speaker_signal_delay: np.array
        index corresponding to detected speaker signal delays
    """
    # allocate space for signal delays
    nspeakers = len(speakersamp.columns)
    speaker_signal_delay = np.zeros(nspeakers)

    # get first index of non-zero value
    for ic, col in enumerate(speakersamp.columns):
        speaker_signal_delay[ic] = speakersamp[col].nonzero()[0][0] - 2
    speaker_signal_delay = speaker_signal_delay.astype(int)

    return speaker_signal_delay


####################################
def crosscorr(datax, datay, lag=0):
    """ Lag-N cross correlation.

    Parameters
    ----------
    lag : int, default 0
    datax, datay : pandas.Series objects of equal length

    Returns
    ----------
    covar : float
    """
    covar = np.abs(datax.corr(datay.shift(lag)))
    return covar


####################################
def covariance(micdat, speakerdat):
    """ Lag-N cross correlation.
    Parameters
    ----------

    Returns
    ----------
    covar : float
    """
    if micdat.shape != speakerdat.shape:
        print('size mismatch')

    covar = np.zeros((micdat.shape[1], micdat.shape[1], micdat.shape[0]))
    ncols = micdat.shape[1]

    for ii in range(ncols):
        for jj in range(ncols):

            covar[ii, jj, :] = np.correlate(
                micdat.values[:, ii], speakerdat.values[:, jj], mode='same')**2

    return covar


####################################
def rollchannel(data, rollval):
    """
    shifts values of data forward by rollval index

    Parameters
    ----------
    rolval : int, default 0
    data : pandas.Series

    Returns
    ----------
    covar : float
    """
    return np.roll(data.values, int(rollval))
