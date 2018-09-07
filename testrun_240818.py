import sys
sys.path.append('/Users/nhamilto/Documents/ATom/coderepo/')
import atom_functions as atom
import numpy as np

test = atom.dataset('/Users/nhamilto/Documents/ATom/data/new_data/')
test.load_data_sample(0)

ATom_signals, travel_times = test.extract_travel_times(upsamplefactor=1)

# np.save('../test_output_240818.npy', ATom_signals)