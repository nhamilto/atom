import importlib
import os
import numpy as np
import inspect
import pandas as pd
import datetime as dt
import scipy.signal as sps
import scipy as sp
import atom_functions as ATF
"""
Library of functions and classes to proces data from the acoustic tomography array at
the National Wind Technology Center.

speaker and microphone time series data are in yyyymmddHHMMSS_AcouTomMainData.txt
Sonic anemometer and T/H probe time series data are in yyyymmddHHMMSS_AcouTomaux_dataa.txt

key stops:


instantiate dataset object
    this is the master class for acoustic tomography projects.
    a dataset contains all file paths, loads data, identifies acoustic
    signals, and determines signal travel times

add meta data
    data crucial to acoustic tomography is loaded into dataset.meta
    meta data includes acoustic signal design info, data collection parameters,
    speaker and microphone locations, etc.

travel time extraction
    calculate the propagation time of acoustic signals between speakers and microphones in
    the array. With known transducer locations, the 'expected' travel time can be calculated
    with the current estimate of speed of sound (c) from the sonic anemometer.

TDSI - time dependent stochastic inversion #TODO
==============================================
from the observed travel times, create over-determined least-squares regression to
detmine fluctuation velocity and temperature fields in the array.

"""


####################################
class meta_data(object):
    """
    Base class for instrument or experiment meta data.
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

        # savefile names
        self.save_tt = [x.split('_')[0] + '_tt.csv' for x in self.auxfiles]
        self.save_filtered_tt = [
            x.split('_')[0] + 'filtered_tt.csv' for x in self.auxfiles
        ]
        self.save_signals = [
            x.split('_')[0] + '_signals.csv' for x in self.auxfiles
        ]

        # sanity check
        if len(self.mainfiles) is not len(self.auxfiles):
            print('Number of main files does not match number of aux files.')

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
        self.mic_xy_m = np.fliplr(self.mic_locations[:, 0:2] * 0.3048)

        # speaker_locations(northing, easting, elevation) in feet
        self.speaker_locations = np.array(offsets.speaker_locations)
        self.speaker_xy_m = np.fliplr(self.speaker_locations[:, 0:2] * 0.3048)

        # physical distance between instruments
        # east-west distance
        mx = self.mic_xy_m[:, 0]
        mx = mx[:, np.newaxis].repeat(8, axis=1)
        sx = self.speaker_xy_m[:, 0]
        sx = sx[np.newaxis, :].repeat(8, axis=0)
        self.distx = mx - sx

        # north-south distance
        my = self.mic_xy_m[:, 1]
        my = my[:, np.newaxis].repeat(8, axis=1)
        sy = self.speaker_xy_m[:, 1]
        sy = sy[np.newaxis, :].repeat(8, axis=0)
        self.disty = my - sy
        # euclidean distance between each speaker/mic combo
        # instrument spacing is an 8x8 array (nspeakers, nmics)
        self.path_lengths = np.sqrt(self.distx**2 + self.disty**2)

        # Ad-hoc tuning of signal ETAs. #TODO figure this shit out.
        self.ETA_index_offsets = offsets.ETA_index_offsets
        return self

    ####################################
    def load_aux(self, sampletime):  #maindatapath
        """
        load data file into dataset object

        Parameters:
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
            pd.TimedeltaIndex(
                freq='50U', start=0, periods=len(main_data.index)),
            inplace=True)

        # calculate the number of frames within the file
        nframe = int(len(main_data) / 10000)
        frameindex = [
            main_data.index[0] + ii * pd.Timedelta(value=0.5, unit='s')
            for ii in range(nframe)
        ]
        # add frame number as a series
        framedata = ['frame {}'.format(ii) for ii in range(nframe)]
        frameseries = pd.Series(data=framedata, index=frameindex)
        main_data['frame'] = frameseries
        main_data['frame'].ffill(inplace=True)

        # reindex by both the frame number and the time index
        main_data.set_index(['frame', main_data.index], inplace=True)

        # split into speaker data and mic data
        self.speaker_data = main_data[[
            x for x in main_data.columns if 'S' in x
        ]]
        self.mic_data = main_data[[x for x in main_data.columns if 'M' in x]]

        self.micnames = self.mic_data.columns.tolist()
        self.speakernames = self.speaker_data.columns.tolist()

    ####################################
    def load_data_sample(self, fileID):
        """
        load data file into dataset object

        Parameters:
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
    def estimate_travel_times(self):
        """
        imports the current speaker (i) and microphone (j) as well as the temperature
        array for the current signal period
        """

        c0 = self.aux_data['c'].mean()

        self.expected_tt_time = self.path_lengths / c0
        self.expected_tt_index = (
            (self.path_lengths / c0) / (self.meta.main_delta_t / 1000)
        ).round().astype(int) + self.ETA_index_offsets

    ####################################
    def mic_signal_window(self, searchLag, upsamplefactor, window_width):
        """
        calculate search windows for each microphone

        Parameters
            signal_ETA_index: np.array
                index of expected speaker signal arrival time

            searchLag: int
                length of search window in samples

        Returns:
            window_indices: np.array
                indices of search windows
        """

        windowshift = (window_width - 1) / (2 * window_width)

        expected_tt_index = (self.expected_tt_index +
                             self.meta.speaker_signal_delay[:, np.newaxis]
                             .repeat(8, 1)) * upsamplefactor

        micsamp = self.mic_data.xs('frame 0', level=0)
        if upsamplefactor > 1:
            micsamp = ATF.upsample(micsamp, upsamplefactor)

        # Beginning of search window for each microphone
        signal_starts = (
            expected_tt_index - searchLag * windowshift).astype(int)
        # End of search window for each microphone
        signal_ends = signal_starts + searchLag

        # adjust signals in case search window extends past
        # beginning or end of recording
        signal_ends[signal_starts < 0] += abs(signal_starts[signal_starts < 0])
        signal_starts[signal_starts < 0] = 0

        signal_starts[signal_ends > len(
            micsamp)] -= signal_ends[signal_ends > len(micsamp)] - len(micsamp)
        signal_ends[signal_ends > len(micsamp)] = len(micsamp)

        nspeakers, nmics = expected_tt_index.shape

        # make dict of dict of window limits
        window_indices = {
            self.micnames[mi]: {
                self.speakernames[si]: (signal_starts[si, mi],
                                        signal_ends[si, mi])
                for si in range(nspeakers)
            }
            for mi in range(nmics)
        }

        return window_indices

    ####################################
    def extract_travel_times(self,
                             upsamplefactor=10,
                             searchLag=None,
                             window_width=3,
                             filterflag='fft',
                             verbose=False):
        """
        Main processing step of raw data.

        Acoustic chirps are identified in speaker and microphone signals.
        Travel time from each speaker to each mic are calculated.

        Parameters:
            upsamplefactor: int
                degree to which acoustic signals are upsampled. This is needed to
                increase precision of travel time estimate

            searchLag: int
                acoustic signal window width. If none is provided, a default window
                width is assigned of `searchLag = 3 * self.meta.chirp_record_length * upsamplefactor`

            filterflag: bool
                implement frequency filter to microphone signals to remove spurious
                spectral contributions. Band-pass filter with acoustic chirp bandwidth
                around the central frequency of the acoustic chip, with the bandwidth

            verbose: bool
                determine output text. used to debug.

        Returns:
            ATom_signals: np.ndarray [nspeakers, nmics, searchLag, nframes]
                acoustic chirps received by the microphones

            travel_times: np.ndarray [nspeakers, nmics, nframes]
                travel times (ms) of chirps between each speaker and mic for each frame

            travel_inds: np.ndarray [nspeakers, nmics, nframes]
                travel times (samples) of chirps between each speaker and mic for each frame
        """

        # width of search window in index value
        if searchLag is None:
            searchLag = window_width * self.meta.chirp_record_length * upsamplefactor
        if verbose:
            print('searchLag = ', searchLag)

        # get frame ID's from speakers
        frames = list(self.speaker_data.index.unique(level=0))
        nframes = len(frames)
        if verbose:
            print('working with {} frames '.format(nframes))
        # maximum resolved frequency of microphone
        max_freq = self.meta.main_f_range.max()

        # get a speaker signals from a single frame
        speakersamp = self.speaker_data.xs(frames[0], level=0)
        if verbose:
            print('speakerstarttime:', speaker_signal_delay)
        # if upsampling is required
        if upsamplefactor is not 1:
            speakersamp = ATF.upsample(speakersamp, upsamplefactor)

        # get indices speaker signal offsets
        speaker_signal_delay = self.meta.speaker_signal_delay * upsamplefactor
        # speaker_signal_delay = get_speaker_signal_delay(speakersamp)

        # estimate signal travel times based on instrument locations and the
        # average speed of sound reported in auxdata
        self.estimate_travel_times()
        # estimate arrival time of signals from speaker emission and spacing
        self.signal_ETA_index = (
            self.expected_tt_index.T * upsamplefactor + speaker_signal_delay).T
        if verbose:
            print('self.signal_ETA_index (index):', self.signal_ETA_index)

        ############# speaker signals
        # get speaker signals and emission times
        speakersigs = ATF.signalOnSpeaker(speakersamp, searchLag, window_width,
                                          speaker_signal_delay, upsamplefactor)

        self.speakersigs = speakersigs
        # # adjust time index of channel three
        # speakersigs['S3'] = rollchannel(
        #     speakersigs['S3'],
        #     int(self.meta.chirp_record_length * upsamplefactor))

        # Calculate search windows
        # these are the same for every frame, every recording,
        # so onle needs to be done once.
        window_indices = self.mic_signal_window(searchLag, upsamplefactor,
                                                window_width)

        # allocate space for travel times between each speaker/mic combo
        travel_times = np.zeros((self.meta.nspeakers, self.meta.nmics,
                                 nframes))
        travel_time_inds = np.zeros((self.meta.nspeakers, self.meta.nmics,
                                     nframes))
        # allocate space for received signals (nspeakers, nmics, ndata, nframes)
        ATom_signals = np.zeros((self.meta.nspeakers, self.meta.nmics,
                                 searchLag, nframes))
        offsets = np.zeros((self.meta.nspeakers, self.meta.nmics, nframes))
        ############# Mic signals
        # cycle through each frame:
        # detect speaker signal emissions
        # detect microphone signal receptions
        for nframe, frame in enumerate(frames):
            if verbose:
                print('extracting from ' + frame)

            # extract a single frame
            micsamp = self.mic_data.xs(frame, level=0)

            # filter mic signals to exclude frequencies outside desired range
            if filterflag == 'fft':
                micsamp = ATF.freq_filter(micsamp, self.meta.filter_freq_inds)

            elif filterflag == 'butter':
                lowcut = 1000 * (
                    self.meta.chirp_freq - 1 * self.meta.chirp_bandWidth / 2)
                hicut = 1000 * (
                    self.meta.chirp_freq + 1 * self.meta.chirp_bandWidth / 2)
                fs = self.meta.main_sampling_freq
                micfilter = ATF.butter_bandpass_filter(micsamp, lowcut, hicut,
                                                       fs)
                micsamp = pd.DataFrame(
                    data=micfilter,
                    index=micsamp.index,
                    columns=micsamp.columns)

            # if upsampling is required
            if upsamplefactor is not 1:
                micsamp = ATF.upsample(micsamp, upsamplefactor)

            # get microphone singals and reception times
            micsigs, index_received, offset = ATF.signalOnMic(
                micsamp, speakersigs, window_indices, searchLag,
                self.signal_ETA_index)

            # store extracted microphone signals, travel times, indices
            ATom_signals[..., nframe] = micsigs
            travel_time_inds[
                ..., nframe] = self.expected_tt_index * upsamplefactor + offset
            #+ (offset / upsamplefactor).round().astype(int)
            # time = index * 1000 / (samplingfreq * upsamplingfactor) to put into [ms]
            travel_times[..., nframe] = travel_time_inds[..., nframe] / (
                self.meta.main_sampling_freq * upsamplefactor) * 1000

            offsets[..., nframe] = offset

        # convert travel times and microphone signals into multi-index
        # dataframes for easy storage
        travel_times = ATF.tt_to_multiindex(travel_times)
        ATom_signals = ATF.atomsigs_to_multiindex(ATom_signals, upsamplefactor,
                                                  self.meta.main_delta_t)

        return ATom_signals, travel_times, travel_time_inds  #, offsets

    ####################################
    def meanfield(self, travel_times):
        """
        caluclate the mean field values of velocity temperature etc.

        Parameters:
            travel_times: np.ndarray [nspeakers, nmics, nframes]
                travel times (ms) of chirps between each speaker and mic for each frame

        Returns:

        """
