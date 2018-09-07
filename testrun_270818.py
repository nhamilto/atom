import sys
sys.path.append('/Users/nhamilto/Documents/ATom/coderepo/')
import atom_functions_v1 as atom
import numpy as np

test = atom.dataset('/Users/nhamilto/Documents/ATom/data/new_data/')
test.load_data_sample(0)

ATom_signals, travel_times = test.extract_travel_times(upsamplefactor=1)

np.save('../test_ATom_signals_280818.npy', ATom_signals)
np.save('../test_travel_times_280818.npy', travel_times)