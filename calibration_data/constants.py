# import glob
import numpy as np
"""
The following initializes various constants used in the program based on experimental condiditons and theoretical constants
"""

nspeakers = 8
nmics = 8
ntravelpaths = nspeakers * nmics

########## Sampling paramters/constants
# main data comes from speakers and mics recorded by labview
# auxiliar data comes from sonic anememometer and T/H probe
# main data (audio) sampling frequency, Hz
main_sampling_freq = 20000
# main data time resolution (ms)
main_delta_t = 1000 / main_sampling_freq

# auxiliary data sampling frequency, Hz
aux_sampling_freq = 20
# auxiliary data time resolution (ms)
aux_delta_t = 1000 / aux_sampling_freq

######### record info
# each record is a single cylce of signal emission and reception. Each speaker emits a chirp every half-second, each mic receives 8 signals in the same period.
# time of a single record, ms
record_length_time = 500  # ms (depricated?)
# converting time of a single record from ms to s
record_time_delta = record_length_time / 1000  # (depricated?)
# main data (audio) samples in each data file
main_record_length = round(record_length_time / main_delta_t)
# aux data (sonic, T/H) samples in each data file
# converting length of a single record to samples
aux_record_length = round(record_length_time / aux_delta_t)

######### Acoustic signal parameters/constants
# Each chirp persists for a fixed time (T_signal)
chirp_time_length = 5.8  # time of transmission of each chirp in ms
chirp_record_length = round(
    chirp_time_length /
    main_delta_t)  # length of the transmitted signal in samples (depricated?)
# chirp_N_records = 116  # number of transmissions / receptions in a file
chirp_freq = 1.2  # kHz, central frequency of the chirp signal
chirp_bandWidth = 0.7  # signal's half bandwidth, kHz

# filtration of whole record
main_delta_f = 1 / main_record_length / main_delta_t  # kHz, spectral resolution in the frequency domain
main_f_range = np.round(
    np.arange(1, main_record_length / 2 + 1) * main_delta_f, 3)

# frequencies for signal filtration
filter_freq_inds = np.squeeze(
    np.array([
        0,
        np.argwhere(main_f_range == chirp_freq - chirp_bandWidth),
        np.argwhere(main_f_range == chirp_freq + chirp_bandWidth),
        main_record_length -
        np.argwhere(main_f_range == chirp_freq + chirp_bandWidth),
        main_record_length -
        np.argwhere(main_f_range == chirp_freq - chirp_bandWidth),
        main_record_length
    ])).astype(int)

# filter_freqs = np.array([main_f_range[x] for x in filter_freq_inds])

# # get index of signal emission from each speaker
# speaker_signal_delay = np.zeros(self.meta.nspeakers)
# for ic, col in enumerate(test.speaker_data.columns):
#     speaker_signal_delay[ic] = test.speaker_data[col].nonzero()[0][0] - 1
# speaker_signal_delay = speaker_signal_delay.astype(int)

############ Path related things, maybe exclude?
# These can be put into another data dir module if I need them, but at this point they don't provide a lot of benefit.
# paths and the number of files in each directory are easy to find along the way, and probably don't belong with the constants

# m_file_path = file_path + '/*Mainaux_delta_t.txt' #adds MainData.txt to the end of the selected file path, specifying what type of file the program should search for
# a_file_path = file_path + '/*AuxData.txt' #add AuxData.txt to the end of the selected file path, specifying what type of file the program should search for
# Fm = (
#     glob.glob(m_file_path)
# )  #retrieves all MainData.txt files in the selected directory and saves the names as strings in a list
# Fa = (
#     glob.glob(a_file_path)
# )  #retrieves all AuxData.txt files in the selected directory and saves the names as strings in a list
# ok_flag = np.zeros(
#     (len(Fm), 1)
# )  #ok_flag is now a vector of length Fm filled with zeroes.if two files are found, [0 0]