# offsets.py
"""
The acoustic tomography array requires that time delays in the system be
tracked to within a few ms, the order of differences between observed travel
times and those predicted by the theory.

Delays come from a few sources:
 - Latency in signal transmission throughout the AT array
 - inhomogeneity in sound propagation from speakers

Predicted travel times are calculated by knowing the precise locations of
transducers in the array, and the speed of sound reported by the sonic
anemometer. If necessary, the speed of sound can also be calculated from the
temperature/humidity probe.
"""
# latency 8 rows (speakers) by 8 columns (mics)
latency = [
    [1.05875, 1.06001389, 1.06030556, 1.06488889, 1.06494444, 1.06097222, 1.06158333, 1.06022222],
    [1.12, 1.12, 1.12, 1.12434722, 1.12206944, 1.125, 1.125, 1.12],
    [1.11554167, 1.11765278, 1.11593056, 1.11819444, 1.11965278, 1.11826389, 1.11522222, 1.11504167],
    [1.04314257, 1.044875, 1.05921592, 1.06113889, 1.05920833, 1.06051389, 1.060875, 1.04394444],
    [1.11075, 1.11193056, 1.11, 1.11398611, 1.11497222, 1.11497222, 1.11405556, 1.11004167],
    [1.0805, 1.08470833, 1.085125, 1.08493056, 1.08505556, 1.08504167,1.08465278, 1.07798611],
    [1.13984722, 1.13931944, 1.13798611, 1.14, 1.14004167, 1.13959722, 1.139875, 1.13877778],
    [1.09001389, 1.0905, 1.10404167, 1.09431944, 1.09194444, 1.09, 1.09, 1.08895833]
    ] # yapf: disable

# mic_locations(northing, easting, elevation) in feet
mic_locations = [
        [757342.515,  76460.405,   6094.11 ],
        [757304.419,  76557.597,   6094.15 ],
        [757223.495,  76551.17 ,   6093.872],
        [757095.2  ,  76497.122,   6094.042],
        [757081.445,  76420.334,   6094.116],
        [757129.534,  76299.993,   6094.066],
        [757220.544,  76273.092,   6094.115],
        [757349.609,  76324.109,   6094.266]
        ]  # yapf: disable

# speaker_locations(northing, easting, elevation) in feet
speaker_locations = [
        [757343.68675,  76458.39725,   6094.07125],
        [757306.6325 ,  76556.9445 ,   6094.13125],
        [757225.796  ,  76551.59625,   6094.06675],
        [757096.06975,  76499.3025 ,   6094.20125],
        [757081.109  ,  76422.58175,   6094.10975],
        [757127.54725,  76301.21125,   6093.9995 ],
        [757218.311  ,  76272.815  ,   6094.05125],
        [757348.7315 ,  76321.955  ,   6094.19575]
        ]  # yapf: disable

ETA_index_offsets = [
        [  15,   60,   44,  -41,  -67, -121, -144, -120],
        [-106,   16,   65,  -26,  -70, -130, -151, -169],
        [-121,  -80,   16,   27,  -18, -106, -123, -146],
        [  12,    8,   26,  148,  176,   81,   36,    1],
        [ -72,  -88,  -83,  -68,   15,   29,  -26,  -85],
        [ -24,  -43,  -53,  -86,  -81,   15,   63,   -9],
        [  10,  -18,  -31,  -78,  -92,  -86,   18, -295],
        [  59,   19,   -1,  -60,  -85, -129, -116,   32]] # yapf: disable