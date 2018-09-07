import importlib
import os
import numpy as np
import inspect
import pandas as pd
import datetime as dt
import scipy.signal as sps
import scipy as sp
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
        self.instrument_spacing = np.sqrt(distx**2 + disty**2)

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
    def signal_ETAs(self):
        """
        imports the current speaker (i) and microphone (j) as well as the temperature
        array for the current signal period
        """

        c0 = self.aux_data['c'].mean()

        self.ETAs_time = self.instrument_spacing / c0
        self.ETAs_index = ((self.instrument_spacing / c0) /
                           (self.meta.main_delta_t / 1000)).round().astype(int)

    ####################################
    def extract_travel_times(self,
                             upsamplefactor=10,
                             searchLag=None,
                             filterflag=True,
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
            ATom_signals: np.ndarray [nspeakers, nmics, searchLag, nrecords]
                acoustic chirps received by the microphones

            travel_times: np.ndarray [nspeakers, nmics, nrecords]
                travel times (ms) of chirps between each speaker and mic for each record

            travel_inds: np.ndarray [nspeakers, nmics, nrecords]
                travel times (samples) of chirps between each speaker and mic for each record
        """

        # width of search window in index value
        if searchLag is None:
            searchLag = 3 * self.meta.chirp_record_length * upsamplefactor
        if verbose:
            print('searchLag = ', searchLag)

        # get record ID's from speakers
        records = list(self.speaker_data.index.unique(level=0))
        nrecords = len(records)

        # maximum resolved frequency of microphone
        max_freq = self.meta.main_f_range.max()

        # get a speaker signals from a single record
        speakersamp = self.speaker_data.xs(records[0], level=0)
        # get indices speaker signal offsets
        speaker_signal_delay = get_speaker_signal_delay(
            speakersamp) * upsamplefactor
        if verbose:
            print('speakerstarttime:', speakerstarttime)
        # if upsampling is required
        if upsamplefactor is not 1:
            speakersamp = upsample(speakersamp, upsamplefactor)

        # estimate signal travel times based on instrument locations and the
        # average speed of sound reported in auxdata
        self.signal_ETAs()
        # estimate arrival time of signals from speaker emission and spacing
        self.signalETAs = (
            self.ETAs_index.T * upsamplefactor + speaker_signal_delay).T
        if verbose:
            print('self.signalETAs (index):', self.signalETAs)

        ############# speaker signals
        # get speaker signals and emission times
        speakersigs = signalOnSpeaker(
            speakersamp, searchLag,
            self.meta.chirp_record_length * upsamplefactor,
            speaker_signal_delay)
        # adjust time index of channel three
        speakersigs['S3'] = rollchannel(
            speakersigs['S3'],
            int(self.meta.chirp_record_length * upsamplefactor))

        # allocate space for travel times between each speaker/mic combo
        travel_times = np.zeros((self.meta.nspeakers, self.meta.nmics,
                                 nrecords))
        travel_inds = np.zeros((self.meta.nspeakers, self.meta.nmics,
                                nrecords))
        # allocate space for received signals (nspeakers, nmics, ndata, nrecords)
        ATom_signals = np.zeros((self.meta.nspeakers, self.meta.nmics,
                                 searchLag, nrecords))

        ############# Mic signals
        # cycle through each record:
        # detect speaker signal emissions
        # detect microphone signal receptions
        for nrec, record in enumerate(records):

            # extract a single record
            micsamp = self.mic_data.xs(record, level=0)

            # filter mic signals to exclude frequencies outside desired range
            if filterflag:
                micsamp = freq_filter(micsamp, self.meta.filter_freq_inds)

            # if upsampling is required
            if upsamplefactor is not 1:
                micsamp = upsample(micsamp, upsamplefactor)

            # get microphone singals and reception times
            micsigs, time_received = signalOnMic(
                micsamp, speakersigs, self.signalETAs, searchLag,
                self.meta.chirp_record_length * upsamplefactor)

            # store extracted microphone signals, travel times, indices
            ATom_signals[..., nrec] = micsigs
            travel_times[..., nrec] = time_received * (
                self.meta.main_delta_t / upsamplefactor)
            travel_inds[..., nrec] = time_received

        return ATom_signals, travel_times, travel_inds


####################################
def signalOnSpeaker(speakersamp, searchLag, chirp_record_length,
                    speaker_signal_delay):
    """
    Extract chirps from the speakers. These are generated signals, and
    so are clean, consistent, and spaced by known amounts. Speaker
    signals are compared against microphone signals to determine the
    actual trasit time of acoustic chirps across the array.

    Parameters:
        speakersamp: pd.DataFrame
            speaker signals for a single record

        searchLag: int
            length of search window in samples

        chirp_record_length: int
            length of acoustic chirp in samples
            base value = 116
            multiplied by upsample factor

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
    signalstarts = [
        int(s) if s > 0 else 0
        for s in (speaker_signal_delay + chirp_record_length / 2) -
        searchLag / 2
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

    return speakersigs


####################################
def signalOnMic(micsamp, speakersigs, signalETAs, searchLag,
                chirp_record_length):
    """
    Extract chirps from the microphones. Each microphone receives
    (nominally 8) chirps emitted by each speaker. Known speaker/mic locations,
    along with known speaker chirp emission times, together provide
    expected travel times from each speaker to each mic. Time correlation
    between signals determines the precise time of chirp arrival and adds
    to the expected signal travel times.

    Parameters:
        micsamp: pd.DataFrame
            microphone signals for a single record

        speakersigs: pd.DataFrame
            extracted acoustic chirps from 'signalOnSpeaker'

        searchLag: int
            length of search window in samples

        chirp_record_length: int
            length of acoustic chirp in samples
            base value = 116
            multiplied by upsample factor

    Returns:
        micsigs: pd.DataFrame
            extracted speaker chirps, centered in a window of length searchLag
            should be nSpeaker signals for each microphone

        time_received_record: np.array
            transit time of each acoustic signal in samples
    """

    signal_starts = (
        signalETAs + (chirp_record_length - searchLag) / 2).astype(int)
    signal_starts[signal_starts < 0] = 0
    signal_ends = signal_starts + searchLag

    micsigs = np.zeros((8, 8, searchLag))
    time_received_record = np.zeros((8, 8))

    for mi, mic in enumerate(micsamp.columns):

        sample = {
            speakersigs.columns[i]: range(signal_starts[i, mi],
                                          signal_ends[i, mi])
            for i in range(8)
        }

        received_signals = {
            k: micsamp[mic].iloc[v].values
            for k, v in sample.items()
        }
        received_signals = pd.DataFrame.from_dict(received_signals)
        micsigs[:, mi, :] = received_signals.T.values

        covar = covariance(received_signals.values, speakersigs.values)
        offset = np.zeros(8)
        time_received = np.zeros(8)
        for ii in range(8):
            offset[ii] = np.argmax(covar)
            time_received[ii] = signalETAs[ii, mi] - offset[ii]

        time_received_record[:, mi] = time_received

    return micsigs, time_received_record


####################################
def get_speaker_signal_delay(speakersamp):
    """
    extract the speaker signal delays from a single record

    Parameters
        speakersamp: pd.DataFrame
            speaker time series data for a single record

    Returns:
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
def covariance(micdat, speakerdat):
    """
    Lag-N cross correlation between two signals.
    Only the correlation between a speaker chirp and its respective
    signal in each microphone record sample is required.

    Parameters:
        micdat: pd.DataFrame
            extracted microphone data containing received acoustic signals

        speakerdat: pd.DataFrame
            extracted speaeker acoustic signals

    Returns
        covar: np.array
            time-lag correlation between micdat and speakerdat

    """
    if micdat.shape != speakerdat.shape:
        print('size mismatch')

    covar = np.zeros(micdat.shape)

    for ii in range(8):
        covar = np.correlate(micdat[:, ii], speakerdat[:, ii], mode='same')

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
def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
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
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = sps.lfilter(b, a, data, axis=0)
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
def signalOnMic_depricated(micsamp, speakersigs, signalETAs,
                           speaker_signal_delay, searchLag,
                           chirp_record_length):

    signalstarts = np.array([
        int(s) if s > 0 else 0
        for s in (signalETAs.flatten() + chirp_record_length) -
        0.35 * searchLag
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
def covariance_depricated(micdat, speakerdat):
    """ Lag-N cross correlation.
    Parameters
        micdat: pd.DataFrame
            extracted microphone data containing received acoustic signals
        speakerdat: pd.DataFrame
            extracted speaeker acoustic signals

    Returns
        covar: float
    """

    if micdat.shape != speakerdat.shape:
        print('size mismatch')

    covar = np.zeros((micdat.shape[1], micdat.shape[1], micdat.shape[0]))
    ncols = micdat.shape[1]

    for ii in range(ncols):
        for jj in range(ncols):

            covar[jj, ii, :] = np.correlate(
                micdat.values[:, ii], speakerdat.values[:, jj], mode='same')

    # allocate space for signal delays
    nspeakers = len(speakersamp.columns)
    speaker_signal_delay = np.zeros(nspeakers)

    # get first index of non-zero value
    for ic, col in enumerate(speakersamp.columns):
        speaker_signal_delay[ic] = speakersamp[col].nonzero()[0][0] - 2
    speaker_signal_delay = speaker_signal_delay.astype(int)

    return covar


####################################
def crosscorr_depricated(datax, datay, lag=0):
    """ Lag-N cross correlation.

    Parameters
        lag: int
            default 0

        datax, datay: pandas.Series
            objects of equal length

    Returns
        covar: float
    """
    covar = np.abs(datax.corr(datay.shift(lag)))
    return covar


####################################
def upsample_data_deptricated(self, upsamplefactor):  #TODO
    """
    artificially upsample data to provide the desired resolution
    new_timedelta = oldtimedelta / upsamplefactor
    upsamplefactor > 1 ==> increase in time resolution
    upsamplefactor < 1 ==> decrease in time resolution

    Parameters:
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