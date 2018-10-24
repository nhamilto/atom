import importlib
import os
import numpy as np
import inspect
import pandas as pd
import datetime as dt
import scipy.signal as sps
import scipy as sp

# """
# Library of functions and classes to proces data from the acoustic tomography array at
# the National Wind Technology Center.

# speaker and microphone time series data are in yyyymmddHHMMSS_AcouTomMainData.txt
# Sonic anemometer and T/H probe time series data are in yyyymmddHHMMSS_AcouTomaux_dataa.txt

# key stops:

# instantiate dataset object
#     this is the master class for acoustic tomography projects.
#     a dataset contains all file paths, loads data, identifies acoustic
#     signals, and determines signal travel times

# add meta data
#     data crucial to acoustic tomography is loaded into dataset.meta
#     meta data includes acoustic signal design info, data collection parameters,
#     speaker and microphone locations, etc.

# travel time extraction
#     calculate the propagation time of acoustic signals between speakers and microphones in
#     the array. With known transducer locations, the 'expected' travel time can be calculated
#     with the current estimate of speed of sound (c) from the sonic anemometer.

# TDSI - time dependent stochastic inversion #TODO
# ==============================================
# from the observed travel times, create over-determined least-squares regression to
# detmine fluctuation velocity and temperature fields in the array.

# """

# ####################################
# class meta_data(object):
#     """
#     Base class for instrument or experiment meta data.
#     Takes in a list of parameters and values.
#     """

#     def __init__(self):
#         return

#     def from_data(self, data):
#         """
#         pass data from constants file,
#         create attribute/value pairs
#         use to create meta data attributes
#         """
#         keys = [
#             x for x in data.__dict__.keys() if '__' not in x
#             if not inspect.ismodule(data.__dict__[x])
#         ]
#         values = [data.__dict__[x] for x in keys]
#         for (key, value) in zip(keys, values):
#             self.__dict__[key] = value
#         return self

#     def from_lists(self, keys, values):
#         """
#         instantiate meta_data object from lists of attribute/value pairs
#         """
#         for (key, value) in zip(keys, values):
#             self.__dict__[key] = value
#         return self

#     def from_file(self, keys, values):
#         """
#         instantiate meta_data object from lists of file pairs
#         """
#         for (key, value) in zip(keys, values):
#             self.__dict__[key] = value
#         return self

#     def to_list(self):
#         """
#         pretty print list of attributes
#         """
#         for k, v in self.__dict__.items():
#             print('{:25} {}'.format(k, v, type(v)))

# ####################################
# class dataset(object):
#     """
#     dataset is the object class for raw data. It should contain a directory in where
#     raw data are to be found, data I/O routines, experiement constants, array
#     calibration info, etc.

#     """

#     ####################################
#     def __init__(self, datapath):
#         self.datapath = datapath
#         filelist = os.listdir(datapath)
#         self.mainfiles = [x for x in filelist if 'main' in x.lower()]
#         self.auxfiles = [x for x in filelist if 'aux' in x.lower()]

#         # savefile names
#         self.save_tt = [x.split('_')[0] + '_tt.csv' for x in self.auxfiles]
#         self.save_filtered_tt = [
#             x.split('_')[0] + 'filtered_tt.csv' for x in self.auxfiles
#         ]
#         self.save_signals = [
#             x.split('_')[0] + '_signals.csv' for x in self.auxfiles
#         ]

#         # sanity check
#         if len(self.mainfiles) is not len(self.auxfiles):
#             print('Number of main files does not match number of aux files.')

#         # path to calibration data and constants
#         caldatapath = os.path.join(
#             os.path.dirname(__file__), '..',
#             'calibration_data/')  #TODO this should not be hardcoded!
#         self = self.get_meta(caldatapath)
#         self = self.get_calibration_info(caldatapath)

#     ####################################
#     def get_meta(self, constantspath):
#         """
#         get meta data for experiment

#         Parameters:

#             constantspath: directory path to file containing meta data
#         """
#         self.meta = meta_data()
#         constants = self.get_constants(constantspath)
#         self.meta.from_data(constants)

#         return self

#     ####################################
#     def get_constants(self, constantspath):
#         """
#         get values of constants used in experiment

#         Parameters:

#             constantspath: str
#                 path to directory containing 'constants.py'
#         """
#         meta = importlib.machinery.SourceFileLoader(
#             'constants', constantspath + 'constants.py').load_module()

#         return meta

#     ####################################
#     def get_calibration_info(self, caldatapath):
#         """
#         get locations of speakers and mics, signal latency between particular devices,
#         and sound propagation delays from speakers as a function of azimuth

#         Parameters:
#             caldatapath: str
#                 path to directory containing raw data
#                     - 'average_latency_yymmdd.csv'
#                     - 'mic_locations_yymmdd.csv'
#                     - 'speaker_locations_yymmdd.csv'
#                 and/or containing processed data to import
#                     - 'offsets.py'
#         """
#         # offsets = importlib.import_module(caldatapath)
#         offsets = importlib.machinery.SourceFileLoader(
#             'offsets', caldatapath + 'offsets.py').load_module()

#         # latency 8 rows (speakers) by 8 columns (mics)
#         self.latency = np.array(offsets.latency)

#         # mic_locations(northing, easting, elevation) in feet
#         self.mic_locations = np.array(offsets.mic_locations)
#         self.mic_xy_m = self.mic_locations[:, 0:2] * 0.3048

#         # speaker_locations(northing, easting, elevation) in feet
#         self.speaker_locations = np.array(offsets.speaker_locations)
#         self.speaker_xy_m = self.speaker_locations[:, 0:2] * 0.3048

#         # physical distance between instruments
#         # east-west distance
#         mx = self.mic_xy_m[:, 0]
#         mx = mx[:, np.newaxis].repeat(8, axis=1)
#         sx = self.speaker_xy_m[:, 0]
#         sx = sx[np.newaxis, :].repeat(8, axis=0)
#         distx = mx - sx
#         # north-south distance
#         my = self.mic_xy_m[:, 1]
#         my = my[:, np.newaxis].repeat(8, axis=1)
#         sy = self.speaker_xy_m[:, 1]
#         sy = sy[np.newaxis, :].repeat(8, axis=0)
#         disty = my - sy
#         # euclidean distance between each speaker/mic combo
#         # instrument spacing is an 8x8 array (nspeakers, nmics)
#         self.path_lengths = np.sqrt(distx**2 + disty**2)

#         # Ad-hoc tuning of signal ETAs. #TODO figure this shit out.
#         self.ETA_index_offsets = offsets.ETA_index_offsets
#         return self

#     ####################################
#     def load_aux(self, sampletime):  #maindatapath
#         """
#         load data file into dataset object

#         Parameters:
#             aux_dataapath: str
#                 path to directory containing aux data
#         """

#         if type(sampletime) is int:
#             sampletime = self.datapath + self.auxfiles[sampletime]

#         # column names for the data
#         colnames = ['vx', 'vy', 'vz', 'c', 'T', 'H']
#         # load into pd.DataFrame
#         aux_data = pd.read_csv(
#             sampletime, skiprows=4, names=colnames, delim_whitespace=True)
#         # get the timestamp from each filename
#         timestamp = sampletime.split('/')[-1].split('_')[0]
#         # calculate and assign timestamp as index
#         aux_data.set_index(
#             pd.DatetimeIndex(
#                 freq='0.05S',
#                 start=dt.datetime.strptime(timestamp, '%Y%m%d%H%M%S'),
#                 periods=len(aux_data.index)),
#             inplace=True)

#         self.aux_data = aux_data

#     ####################################
#     def load_main(self, sampletime):  #maindatapath
#         """
#         load data file into dataset object

#         Parameters:
#             maindatapath: str
#                 path to directory containing main data
#         """

#         if type(sampletime) is int:
#             sampletime = self.datapath + self.mainfiles[sampletime]

#         # column names for the data
#         colnames = ['S{}'.format(x)
#                     for x in range(8)] + ['M{}'.format(x) for x in range(8)]

#         # load into pd.DataFrame
#         main_data = pd.read_csv(
#             sampletime, skiprows=4, names=colnames, delim_whitespace=True)

#         # get the timestamp from each filename
#         timestamp = sampletime.split('/')[-1].split('_')[0]

#         # calculate and assign timestamp as index
#         main_data.set_index(
#             pd.TimedeltaIndex(
#                 freq='50U', start=0, periods=len(main_data.index)),
#             inplace=True)

#         # calculate the number of frames within the file
#         nframe = int(len(main_data) / 10000)
#         frameindex = [
#             main_data.index[0] + ii * pd.Timedelta(value=0.5, unit='s')
#             for ii in range(nframe)
#         ]
#         # add frame number as a series
#         framedata = ['frame {}'.format(ii) for ii in range(nframe)]
#         frameseries = pd.Series(data=framedata, index=frameindex)
#         main_data['frame'] = frameseries
#         main_data['frame'].ffill(inplace=True)

#         # reindex by both the frame number and the time index
#         main_data.set_index(['frame', main_data.index], inplace=True)

#         # split into speaker data and mic data
#         self.speaker_data = main_data[[
#             x for x in main_data.columns if 'S' in x
#         ]]
#         self.mic_data = main_data[[x for x in main_data.columns if 'M' in x]]

#         self.micnames = self.mic_data.columns.tolist()
#         self.speakernames = self.speaker_data.columns.tolist()

#     ####################################
#     def load_data_sample(self, fileID):
#         """
#         load data file into dataset object

#         Parameters:
#             fileID: int
#                 integer value of main and aux data in respective lists
#         """
#         if fileID > len(self.mainfiles):
#             print('Ran out of data files ...')
#             pass

#         self.load_main(fileID)

#         self.load_aux(fileID)

#         # return self

#     ####################################
#     def time_info(self):
#         """
#         print time resolution of main and aux data
#         """

#         main_delta_t = self.speaker_data.index[1] - self.speaker_data.index[0]
#         print('main data data resolution = {}us'.format(
#             main_delta_t.microseconds))
#         main_span_t = self.speaker_data.index[-1] - self.speaker_data.index[0]
#         print('main data spans {}s'.format(main_span_t.seconds))

#         main_delta_t = self.mic_data.index[1] - self.mic_data.index[0]
#         print('main data data resolution = {}us'.format(
#             main_delta_t.microseconds))
#         main_span_t = self.mic_data.index[-1] - self.mic_data.index[0]
#         print('main data spans {}s'.format(main_span_t.seconds))

#         aux_delta_t = self.aux_data.index[1] - self.aux_data.index[0]
#         print('auxiliary data resolution = {}us'.format(
#             aux_delta_t.microseconds))
#         aux_span_t = self.aux_data.index[-1] - self.aux_data.index[0]
#         print('auxiliary data spans {}s'.format(aux_span_t.seconds))

#     ####################################
#     def estimate_travel_times(self):
#         """
#         imports the current speaker (i) and microphone (j) as well as the temperature
#         array for the current signal period
#         """

#         c0 = self.aux_data['c'].mean()

#         self.expected_tt_time = self.path_lengths / c0
#         self.expected_tt_index = (
#             (self.path_lengths / c0) / (self.meta.main_delta_t / 1000)
#         ).round().astype(int) + self.ETA_index_offsets

#     ####################################
#     def mic_signal_window(self, searchLag, upsamplefactor, window_width):
#         """
#         calculate search windows for each microphone

#         Parameters
#             signal_ETA_index: np.array
#                 index of expected speaker signal arrival time

#             searchLag: int
#                 length of search window in samples

#         Returns:
#             window_indices: np.array
#                 indices of search windows
#         """

#         windowshift = (window_width - 1) / (2 * window_width)

#         expected_tt_index = (self.expected_tt_index +
#                              self.meta.speaker_signal_delay[:, np.newaxis]
#                              .repeat(8, 1)) * upsamplefactor

#         micsamp = self.mic_data.xs('frame 0', level=0)
#         if upsamplefactor > 1:
#             micsamp = upsample(micsamp, upsamplefactor)

#         # Beginning of search window for each microphone
#         signal_starts = (
#             expected_tt_index - searchLag * windowshift).astype(int)
#         # End of search window for each microphone
#         signal_ends = signal_starts + searchLag

#         # adjust signals in case search window extends past
#         # beginning or end of recording
#         signal_ends[signal_starts < 0] += abs(signal_starts[signal_starts < 0])
#         signal_starts[signal_starts < 0] = 0

#         signal_starts[signal_ends > len(
#             micsamp)] -= signal_ends[signal_ends > len(micsamp)] - len(micsamp)
#         signal_ends[signal_ends > len(micsamp)] = len(micsamp)

#         nspeakers, nmics = expected_tt_index.shape

#         # make dict of dict of window limits
#         window_indices = {
#             self.micnames[mi]: {
#                 self.speakernames[si]: (signal_starts[si, mi],
#                                         signal_ends[si, mi])
#                 for si in range(nspeakers)
#             }
#             for mi in range(nmics)
#         }

#         return window_indices

#     ####################################
#     def extract_travel_times(self,
#                              upsamplefactor=10,
#                              searchLag=None,
#                              window_width=3,
#                              filterflag='fft',
#                              verbose=False):
#         """
#         Main processing step of raw data.

#         Acoustic chirps are identified in speaker and microphone signals.
#         Travel time from each speaker to each mic are calculated.

#         Parameters:
#             upsamplefactor: int
#                 degree to which acoustic signals are upsampled. This is needed to
#                 increase precision of travel time estimate

#             searchLag: int
#                 acoustic signal window width. If none is provided, a default window
#                 width is assigned of `searchLag = 3 * self.meta.chirp_record_length * upsamplefactor`

#             filterflag: bool
#                 implement frequency filter to microphone signals to remove spurious
#                 spectral contributions. Band-pass filter with acoustic chirp bandwidth
#                 around the central frequency of the acoustic chip, with the bandwidth

#             verbose: bool
#                 determine output text. used to debug.

#         Returns:
#             ATom_signals: np.ndarray [nspeakers, nmics, searchLag, nframes]
#                 acoustic chirps received by the microphones

#             travel_times: np.ndarray [nspeakers, nmics, nframes]
#                 travel times (ms) of chirps between each speaker and mic for each frame

#             travel_inds: np.ndarray [nspeakers, nmics, nframes]
#                 travel times (samples) of chirps between each speaker and mic for each frame
#         """

#         # width of search window in index value
#         if searchLag is None:
#             searchLag = window_width * self.meta.chirp_record_length * upsamplefactor
#         if verbose:
#             print('searchLag = ', searchLag)

#         # get frame ID's from speakers
#         frames = list(self.speaker_data.index.unique(level=0))
#         nframes = len(frames)
#         if verbose:
#             print('working with {} frames '.format(nframes))
#         # maximum resolved frequency of microphone
#         max_freq = self.meta.main_f_range.max()

#         # get a speaker signals from a single frame
#         speakersamp = self.speaker_data.xs(frames[0], level=0)
#         if verbose:
#             print('speakerstarttime:', speaker_signal_delay)
#         # if upsampling is required
#         if upsamplefactor is not 1:
#             speakersamp = upsample(speakersamp, upsamplefactor)

#         # get indices speaker signal offsets
#         speaker_signal_delay = self.meta.speaker_signal_delay * upsamplefactor
#         # speaker_signal_delay = get_speaker_signal_delay(speakersamp)

#         # estimate signal travel times based on instrument locations and the
#         # average speed of sound reported in auxdata
#         self.estimate_travel_times()
#         # estimate arrival time of signals from speaker emission and spacing
#         self.signal_ETA_index = (
#             self.expected_tt_index.T * upsamplefactor + speaker_signal_delay).T
#         if verbose:
#             print('self.signal_ETA_index (index):', self.signal_ETA_index)

#         ############# speaker signals
#         # get speaker signals and emission times
#         speakersigs = signalOnSpeaker(speakersamp, searchLag, window_width,
#                                       speaker_signal_delay, upsamplefactor)

#         self.speakersigs = speakersigs
#         # # adjust time index of channel three
#         # speakersigs['S3'] = rollchannel(
#         #     speakersigs['S3'],
#         #     int(self.meta.chirp_record_length * upsamplefactor))

#         # Calculate search windows
#         # these are the same for every frame, every recording,
#         # so onle needs to be done once.
#         window_indices = self.mic_signal_window(searchLag, upsamplefactor,
#                                                 window_width)

#         # allocate space for travel times between each speaker/mic combo
#         travel_times = np.zeros((self.meta.nspeakers, self.meta.nmics,
#                                  nframes))
#         travel_time_inds = np.zeros((self.meta.nspeakers, self.meta.nmics,
#                                      nframes))
#         # allocate space for received signals (nspeakers, nmics, ndata, nframes)
#         ATom_signals = np.zeros((self.meta.nspeakers, self.meta.nmics,
#                                  searchLag, nframes))
#         offsets = np.zeros((self.meta.nspeakers, self.meta.nmics, nframes))
#         ############# Mic signals
#         # cycle through each frame:
#         # detect speaker signal emissions
#         # detect microphone signal receptions
#         for nframe, frame in enumerate(frames):
#             if verbose:
#                 print('extracting from ' + frame)

#             # extract a single frame
#             micsamp = self.mic_data.xs(frame, level=0)

#             # filter mic signals to exclude frequencies outside desired range
#             if filterflag == 'fft':
#                 micsamp = freq_filter(micsamp, self.meta.filter_freq_inds)

#             elif filterflag == 'butter':
#                 lowcut = 1000 * (
#                     self.meta.chirp_freq - 1 * self.meta.chirp_bandWidth / 2)
#                 hicut = 1000 * (
#                     self.meta.chirp_freq + 1 * self.meta.chirp_bandWidth / 2)
#                 fs = self.meta.main_sampling_freq
#                 micfilter = butter_bandpass_filter(micsamp, lowcut, hicut, fs)
#                 micsamp = pd.DataFrame(
#                     data=micfilter,
#                     index=micsamp.index,
#                     columns=micsamp.columns)

#             # if upsampling is required
#             if upsamplefactor is not 1:
#                 micsamp = upsample(micsamp, upsamplefactor)

#             # get microphone singals and reception times
#             micsigs, index_received, offset = signalOnMic(
#                 micsamp, speakersigs, window_indices, searchLag,
#                 self.signal_ETA_index)

#             # store extracted microphone signals, travel times, indices
#             ATom_signals[..., nframe] = micsigs
#             travel_time_inds[
#                 ..., nframe] = self.expected_tt_index * upsamplefactor + offset
#             #+ (offset / upsamplefactor).round().astype(int)
#             # time = index * 1000 / (samplingfreq * upsamplingfactor) to put into [ms]
#             travel_times[..., nframe] = travel_time_inds[..., nframe] / (
#                 self.meta.main_sampling_freq * upsamplefactor) * 1000

#             offsets[..., nframe] = offset

#         # convert travel times and microphone signals into multi-index
#         # dataframes for easy storage
#         travel_times = tt_to_multiindex(travel_times)
#         ATom_signals = atomsigs_to_multiindex(ATom_signals, upsamplefactor,
#                                               self.meta.main_delta_t)

#         return ATom_signals, travel_times, travel_time_inds  #, offsets


####################################
def signalOnSpeaker(speakersamp, searchLag, window_width, speaker_signal_delay,
                    upsamplefactor):
    """
    Extract chirps from the speakers. These are generated signals, and
    so are clean, consistent, and spaced by known amounts. Speaker
    signals are compared against microphone signals to determine the
    actual trasit time of acoustic chirps across the array.

    Parameters:
        speakersamp: pd.DataFrame
            speaker signals for a single frame

        searchLag: int
            length of search window in samples

        window_width: int
            multiple of the chirp length used for

        speaker_signal_delay: array
            each speaker chirp is delayed by a specified amount
            to offset the chirps in time
            base values = [2480, 2080, 4080,    0, 3200, 4000,  800, 2880]
            multiplied by upsample factor

    Returns:
        speakersigs: pd.DataFrame
            extracted speaker chirps, centered in a window of length searchLag
    """

    # beginning index of speaker signal to extract
    windowshift = (window_width - 1) / (2 * window_width)

    signalstarts = (speaker_signal_delay - searchLag * windowshift).astype(int)
    signalstarts[signalstarts < 0] = 0

    # end index of speaker signal to extract
    signalends = (signalstarts + searchLag).astype(
        int)  #[s + searchLag for s in signalstarts]

    # make dict, sample data, return new dataframe
    sample = {
        speakersamp.columns[i]: range(signalstarts[i], signalends[i])
        for i in range(len(speaker_signal_delay))
    }
    speakersigs = {k: speakersamp[k].iloc[v].values for k, v in sample.items()}
    speakersigs = pd.DataFrame.from_dict(speakersigs)

    speakersigs.index = pd.TimedeltaIndex(
        freq='{}U'.format(50 / upsamplefactor),
        start=0,
        periods=len(speakersigs)).microseconds
    speakersigs.index.name = 'time_us'

    return speakersigs


####################################
def signalOnMic(micsamp, speakersigs, window_indices, searchLag,
                signal_ETA_index):
    """
    Extract chirps from the microphones. Each microphone receives
    (nominally 8) chirps emitted by each speaker. Known speaker/mic locations,
    along with known speaker chirp emission times, together provide
    expected travel times from each speaker to each mic. Time correlation
    between signals determines the precise time of chirp arrival and adds
    to the expected signal travel times.

    Parameters:
        micsamp: pd.DataFrame
            microphone signals for a single frame

        speakersigs: pd.DataFrame
            extracted acoustic chirps from 'signalOnSpeaker'

        window_indices: dict
            dictionary describing the search windows for each
            speaker signal within each mic recording
            e.g. 'Mi': 'Sj': (start, end) ...

        searchLag: int
            length of search window in samples

        signal_ETA_index: np.array
            index corresponding to the expecter arrival time of each
            speakersignal on each microhpone

    Returns:
        micsigs: pd.DataFrame
            extracted speaker chirps, centered in a window of length searchLag
            should be nSpeaker signals for each microphone

        time_received_record: np.array
            transit time of each acoustic signal in samples
    """

    nmics = len(micsamp.columns)
    nspeakers = len(speakersigs.columns)

    # allocate space for extracted mic signals
    micsigs = np.zeros((nspeakers, nmics, searchLag))
    index_received = np.zeros((nspeakers, nmics))
    offset = np.zeros((nspeakers, nmics))

    for mi, mic in enumerate(micsamp.columns):

        # extract seach windows for each speaker signal
        # from a mic recording
        for si, spk in enumerate(speakersigs.columns):
            micsigs[si, mi, :] = micsamp[mic].iloc[window_indices[mic][spk][0]:
                                                   window_indices[mic][spk][1]]

        # micsigs[:, mi, :] = micsamp[mic].iloc
        # calculate covariance between extracted mic sample and speaker signal
        covar = covariance(micsigs[:, mi, :].T, speakersigs.values)

        # difference between expected and observed arrival times
        offset[:, mi] = np.argmax(covar, axis=0) - searchLag / 2

        index_received[:, mi] = signal_ETA_index[:, mi] + offset[:, mi]

    return micsigs, index_received, offset.astype(int)


####################################
def get_speaker_signal_delay(speakersamp):
    """
    extract the speaker signal delays from a single frame

    Parameters
        speakersamp: pd.DataFrame
            speaker time series data for a single frame

    Returns:
        speaker_signal_delay: np.array
            index corresponding to detected speaker signal delays
    """
    # allocate space for signal delays
    nspeakers = len(speakersamp.columns)
    speaker_signal_delay = np.zeros(nspeakers)

    # get first index of non-zero value
    for ic, col in enumerate(speakersamp.columns):
        speaker_signal_delay[ic] = speakersamp[col].round(2).nonzero()[0][0]
    speaker_signal_delay = speaker_signal_delay.astype(int)

    return speaker_signal_delay


####################################
def covariance(micdat, speakerdat):
    """
    Lag-N cross correlation between two signals.
    Only the correlation between a speaker chirp and its respective
    signal in each microphone frame sample is required.

    Parameters:
        micdat: np.array
            extracted microphone data containing received acoustic signals

        speakerdat: np.array
            extracted speaeker acoustic signals

    Returns
        covar: np.array
            time-lag correlation between micdat and speakerdat

    """
    if micdat.shape != speakerdat.shape:
        print('size mismatch')

    covar = np.zeros(micdat.shape)

    for ii in range(8):
        covar[:, ii] = np.correlate(
            micdat[:, ii], speakerdat[:, ii], mode='same')

    return covar


####################################
def rollchannel(data, rollval):
    """
    shifts values of data forward by rollval index

    Parameters:
        data: pandas.Series
            data to shift in time

        rollval: int
            default 0

    """
    return np.roll(data.values, int(rollval))


####################################
def butter_bandpass(lowcut, highcut, fs, order=5):
    """
    create band-pass frequency filter to isolate chirp frequency in mic singals

    Parameters:
        lowcut: float
            lower limit of frequency band

        highcut: float
            upper limit of frequency band

        fs: float
            sampling frequency of data

        order: int
            filter order, default = 5

    Returns:
        b, a: ndarray, ndarray
            Numerator (b) and denominator (a) polynomials of the IIR filter. Only returned if output='ba'.
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = sps.butter(order, [low, high], btype='bandpass')
    return b, a


####################################
def butter_bandpass_filter(data,
                           lowcut,
                           highcut,
                           fs,
                           order=5,
                           fixtimedelay=True):
    """
    implement fileter on input data

    Parameters:
        data: np.arary
            acoustic signal to be filtered (microphone data)

        lowcut: float
            lower limit of frequency band, passed to `butter_bandpass`

        highcut: float
            upper limit of frequency band, passed to `butter_bandpass`

        fs: float
            sampling frequency of data, passed to `butter_bandpass`

        order: int
            filter order, default = 5, passed to `butter_bandpass`

    Returns:
        y: np.array
            frequency-filtered data

    """
    # determine expected time delay from 'ideal' filter
    td_exp = {
        k: v
        for k, v in zip(
            np.arange(2, 11),
            np.array([
                0.225, 0.318, 0.416, 0.515, 0.615, 0.715, 0.816, 0.917, 1.017
            ]))
    }
    td_exp = td_exp[order] / highcut
    rollval = int(np.round(td_exp * fs))

    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = sps.lfilter(b, a, data, axis=0)

    if fixtimedelay:
        y = np.roll(y, -rollval)

    return y


####################################
def upsample(datasample, upsamplefactor, method='cubic'):
    """
    upsample data by desired factor using specified method

    Parameters:
        datasample: pd.DataFrame
            microphone or speaker data to upsample

        upsamplefactor: int, float
            factor by which to upsample data
            upsamplefactor > 1 ==> increase in time resolution
            upsamplefactor < 1 ==> decrease in time resolution

        method: str
            method by which to interpolate data:

            - pandas - built-in interp method, slow

            - linear - linear interpolation

            - cubic (default) - cubic interpolation

    Returns:
        newdatasample: pd.DataFrame
            upsampled data
    """
    delta_t = datasample.index[1] - datasample.index[0]
    rule = delta_t / upsamplefactor

    if method is 'pandas':
        newdatasample = datasample.resample('{}U'.format(
            rule.microseconds)).interpolate(method='cubic')
    elif method is 'linear':
        xold = np.arange(len(datasample))
        xnew = np.arange(0, len(datasample), 1 / upsamplefactor)
        xnew = xnew[:-(upsamplefactor - 1)]

        interpolant = sp.interpolate.interp1d(
            np.arange(len(datasample)), datasample.values, axis=0, kind=method)
        newdatasample = interpolant(xnew)

        newdatasample = pd.DataFrame(
            newdatasample,
            columns=datasample.columns,
            index=pd.DatetimeIndex(
                start=0,
                freq='{}U'.format(rule.microseconds),
                periods=len(xnew)))

    elif method is 'cubic':
        xold = np.arange(len(datasample))
        xnew = np.arange(0, len(datasample), 1 / upsamplefactor)
        xnew = xnew[:-(upsamplefactor - 1)]

        interpolant = sp.interpolate.interp1d(
            np.arange(len(datasample)), datasample.values, axis=0, kind=method)
        newdatasample = interpolant(xnew)

        newdatasample = pd.DataFrame(
            newdatasample,
            columns=datasample.columns,
            index=pd.DatetimeIndex(
                start=0,
                freq='{}U'.format(rule.microseconds),
                periods=len(xnew)))

    return newdatasample


####################################
def freq_filter(datasample, filter_freq_inds):
    """
    frequency filter to isolate chirp signal in microphones
    brute force method takes FFT of datasample, sets frequecies
    outside specified windows to zero, implements IFFT.

    Probably produces ringing in data. Should probably use proper filter.

    Parameters:
        datasample: np.array
            data to filter

        filter_freq_inds: np.array
            key frequencies used in filter design.
    """
    ftmp = sp.fftpack.fft(datasample, axis=0)

    ftmp[range(filter_freq_inds[0], filter_freq_inds[1]), :] = 0
    ftmp[range(filter_freq_inds[2], filter_freq_inds[3]), :] = 0
    ftmp[range(filter_freq_inds[4], filter_freq_inds[5]), :] = 0

    mfilt = sp.fftpack.ifft(ftmp, axis=0)

    datasample = pd.DataFrame(
        np.real(mfilt), columns=datasample.columns, index=datasample.index)

    return datasample


####################################
def tt_to_multiindex(tt):
    """
    Convert 3D array of travel times to a pandas MultiIndex

    Parameters:
        tt: np.ndarray
            3D array of nspeakers x nmics x nframes

    Returns:
        ttdf: pd.MultiIndex
            MultiIndex dataFrame of travel times
    """
    nspeakers, nmics, nframes = tt.shape

    snames = ['S{}'.format(x) for x in range(nspeakers)]
    mnames = ['M{}'.format(x) for x in range(nmics)]
    fnames = ['frame {}'.format(x) for x in range(nframes)]

    for ii in range(8):
        temp = tt[:, ii, :]
        tdf = pd.DataFrame(index=snames, data=temp, columns=fnames)
        tdf.index.name = 'speaker'
        tdf['mic'] = [mnames[ii] for x in range(8)]
        tdf.set_index([tdf.index, 'mic'], inplace=True)
        if ii == 0:
            ttdf = tdf
        else:
            ttdf = pd.concat([ttdf, tdf])

    return ttdf


####################################
def atomsigs_to_multiindex(atomsigs, upsamplefactor, main_delta_t):
    """
    Convert 4D array of acoustic signals to a pandas MultiIndex

    Parameters:
        atomsigs: np.ndarray
            4D array of nspeakers x nmics x searchLag x nframes
        upsamplefactor: int
            factor by which original signals have been upsampled
         main_delta_t: float
            time delta (in milliseconds) between acoustic signal samples

    Returns:
        sigdf: pd.MultiIndex
            MultiIndex dataFrame of acoustic signals

    """

    nspeakers, nmics, nsamp, nframes = atomsigs.shape

    snames = ['S{}'.format(x) for x in range(nspeakers)]
    mnames = ['M{}'.format(x) for x in range(nmics)]
    fnames = ['frame {}'.format(x) for x in range(nframes)]
    # get freq into us
    freq = main_delta_t / 1000 / upsamplefactor * 1000000
    # make a date time index
    time_us = pd.DatetimeIndex(
        start=0, periods=nsamp, freq='{}U'.format(freq)).microsecond

    for jj in range(len(snames)):
        for ii in range(len(mnames)):
            # data for one mic/speaker pair, all times, all frames
            temp = atomsigs[jj, ii, :, :]
            tdf = pd.DataFrame(data=temp, columns=fnames)
            tdf.set_index(time_us, inplace=True)
            tdf.index.name = 'time_us'

            # add mic and speaker columns
            tdf['mic'] = [mnames[ii] for x in range(len(tdf))]
            tdf['speaker'] = [snames[jj] for x in range(len(tdf))]

            # triple MultiIndex (time, speaker, mic)
            tdf.set_index([tdf.index, 'speaker', 'mic'], inplace=True)

            if ii == 0 and jj == 0:
                sigdf = tdf
            else:
                sigdf = pd.concat([sigdf, tdf])

    return sigdf


####################################
def get_sample(dataset, fn=0):
    """
    extract speaker and mic samples for testing
    """
    fn = 'frame {}'.format(fn)

    micsamp = dataset.mic_data.xs(fn, level=0)
    speakersamp = dataset.mic_data.xs(fn, level=0)

    return micsamp, speakersamp


####################################
def nan_helper(y):
    """Helper to handle indices and logical indices of NaNs.

    Parameters:
        y: 1d numpy array with possible NaNs

    Returns:
        nans: bool
            logical indices of NaNs
        x: a function
            signature indices= index(logical_indices),
            to convert logical indices of NaNs to 'equivalent' indices
    Example:
        >>> # linear interpolation of NaNs
        >>> nans, x= nan_helper(y)
        >>> y[nans]= np.interp(x(nans), x(~nans), y[~nans])
    """

    return np.isnan(y), lambda z: z.nonzero()[0]


####################################
def ttfilter(y):
    """
    filter of travel times of acoustic signals

    Parameters:
        y: 1d numpy array with possible NaNs

    Returns:
        ytmp: filtered tt series
    """
    ytmp = y.copy()

    b, a = sp.signal.butter(3, 0.1)
    yfilt = sp.signal.filtfilt(b, a, ytmp)

    pout = np.abs(ytmp - yfilt)

    outlierinds = np.argwhere(pout > 0.3)

    ytmp[outlierinds] = np.NaN
    nans, x = nan_helper(ytmp)

    # replace outliers
    ytmp[nans] = np.interp(x(nans), x(~nans), ytmp[~nans])

    return ytmp


####################################
def filter_travel_times(travel_times):
    """
    implements a basic low-pass filter on recorded travel times
    identifies outliers, removes them, interpolates over missing values

    Parameters:
        travel_times: pd.DataFrame
            travel times of acoustic tomography signals

    Returns:
        travel_times_filtered: pd.DataFrame
            filtered travel times of acoustic tomography signals
    """
    travel_times_filtered = pd.DataFrame(
        index=travel_times.index, columns=travel_times.columns)

    for path in travel_times.index.values:
        travel_times_filtered.loc[path] = ttfilter(travel_times.loc[path])

    return travel_times_filtered