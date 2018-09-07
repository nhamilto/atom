import glob
import numpy as np
"""
The following initializes various constants used in the program based on experimental condiditons and theoretical constants
"""

########## Sampling paramters/constants
# main data comes from speakers and mics recorded by labview
# auxiliar data comes from sonic anememometer and T/H probe
# main data (audio) sampling frequency, Hz
fsm = 20000
# auxiliary data sampling frequency, Hz
fsa = 20
# main data time resolution (ms)
dtm = 1000 / fs
# auxiliary data time resolution (ms)
dta = 1000 / fsa

######### record info
# each record is a single cylce of signal emission and reception. Each speaker emits a chirp every half-second, each mic receives 8 signals in the same period.
# time of a single record, ms
T_record = 500 (depricated?)
# converting time of a single record from ms to s
dtt = T_record / 1000  # (depricated?)
# main data (audio) samples in each data file
N_record = round(T_record / dtm)
# main data (audio) samples in each data file
#converting length of a single record to samples
Na_record = round(T_record / dta)

######### Acoustic signal parameters/constants
# Each chirp persists for a fixed time (T_signal)
T_signal = 5.8  #time of transmission of each chirp in ms
N_signal = round(T_signal / dtm)  # length of the transmitted signal in samples (depricated?)
M = 120  # number of transmissions / receptions in a file
Fc = 1.2  # kHz, central frequency of the chirp signal
bandWidth = 0.7  #signal's half bandwidth, kHz

############ Path related things, maybe exclude?
# These can be put into another data dir module if I need them, but at this point they don't provide a lot of benefit.
# paths and the number of files in each directory are easy to find along the way, and probably don't belong with the constants

# m_file_path = file_path + '/*MainDta.txt' #adds MainData.txt to the end of the selected file path, specifying what type of file the program should search for
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