# coding: utf-8

# In[3]:

import pandas as pd
import datetime as dt
import numpy as np
import scipy as sp
import os
import datetime
# get_ipython().run_line_magic('load_ext', 'autoreload')
# get_ipython().run_line_magic('autoreload', '2')

# data path, constants, etc.
import sys
datapath = '/Users/nhamilto/Documents/ATom/coderepo/ATom/'
sys.path.append(datapath)

# plot things
# import matplotlib.pyplot as plt
# plt.style.use('seaborn-deep')

# Acoustic tomography functions
import atom_functions as ATF
import atom_classes as ATC

# In[4]:

adat = ATC.dataset(
    '/Users/nhamilto/Documents/ATom/data/20181005_Data_collection/')
today = datetime.datetime.today().strftime('%Y%m%d')
savedir = '/Users/nhamilto/Documents/ATom/data/{}_processed_data'.format(today)
if not os.path.exists(savedir):
    os.makedirs(savedir)

# In[ ]:

for ii in range(len(adat.auxfiles)):

    print('loading file {}: {}'.format(ii, adat.mainfiles[ii]))
    adat.load_data_sample(ii)
    atomsigs, tt, _ = adat.extract_travel_times(upsamplefactor=10)

    savefile = os.path.join(savedir, adat.save_tt[ii])
    tt.to_csv(savefile)

    tfilt = ATF.filter_travel_times(tt)
    savefile = os.path.join(savedir, adat.save_filtered_tt[ii])
    tfilt.to_csv(savefile)
