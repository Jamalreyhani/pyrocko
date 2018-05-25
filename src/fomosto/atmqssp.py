# http://pyrocko.org - GPLv3
#
# The Pyrocko Developers, 21st Century
# ---|P------/S----------~Lg----------
from __future__ import absolute_import, division
from builtins import range, zip

import numpy as num
import logging
import os
import shutil
import glob
import copy
import signal
import math
import time

from tempfile import mkdtemp
from subprocess import Popen, PIPE
from os.path import join as pjoin

from pyrocko import trace, util, cake, gf
from pyrocko.guts import Object, Float, String, Bool, Tuple, Int, List
from pyrocko.moment_tensor import MomentTensor, symmat6

guts_prefix = 'pf'

logger = logging.getLogger('pyrocko.fomosto.qssp2017')

# how to call the programs
program_bins = {
    'atmqssp': 'fomosto_qssp2017',
}


def have_backend():
    have_any = False
    for cmd in [[exe] for exe in program_bins.values()]:
        try:
            p = Popen(cmd, stdout=PIPE, stderr=PIPE, stdin=PIPE)
            (stdout, stderr) = p.communicate()
            have_any = True

        except OSError:
            pass

    return have_any


qssp_components = {
    1: '_disp_e _disp_n _disp_z'.split(),
    2: 'ar at ap gr sd tt tp ur ut up vr vt vp'.split(),
}


def str_float_vals(vals):
    return ' '.join(['%e' % val for val in vals])


def cake_model_to_config(mod):
    k = 1000.
    srows = []
    for i, row in enumerate(mod.to_scanlines()):
        depth, vp, vs, rho, qp, qs = row
        row = [depth/k, vp/k, vs/k, rho/k, qp, qs]
        srows.append('%i %s' % (i+1, str_float_vals(row)))

    return '\n'.join(srows), len(srows)


class QSSPSource(Object):
    lat = Float.T(default=0.0)
    lon = Float.T(default=0.0)
    depth = Float.T(default=10.0)
    torigin = Float.T(default=0.0)
    trise = Float.T(default=1.0)

    def string_for_config(self):
        return '%(lat)15e %(lon)15e %(depth)15e %(torigin)15e %(trise)15e' \
            % self.__dict__


class QSSPSourceMT(QSSPSource):
    munit = Float.T(default=1.0)
    mrr = Float.T(default=1.0)
    mtt = Float.T(default=1.0)
    mpp = Float.T(default=1.0)
    mrt = Float.T(default=0.0)
    mrp = Float.T(default=0.0)
    mtp = Float.T(default=0.0)

    def string_for_config(self):
        return '%(munit)15e %(mrr)15e %(mtt)15e %(mpp)15e ' \
               '%(mrt)15e %(mrp)15e %(mtp)15e ' \
            % self.__dict__ + QSSPSource.string_for_config(self)


class QSSPSourceDC(QSSPSource):
    moment = Float.T(default=1.0e9)
    strike = Float.T(default=0.0)
    dip = Float.T(default=90.0)
    rake = Float.T(default=0.0)

    def string_for_config(self):
        return '%(moment)15e %(strike)15e %(dip)15e %(rake)15e ' \
            % self.__dict__ + QSSPSource.string_for_config(self)


class QSSPReceiver(Object):
    lat = Float.T(default=10.0)
    lon = Float.T(default=0.0)
    name = String.T(default='')
    tstart = Float.T(default=0.0)
    distance = Float.T(default=0.0)

    def string_for_config(self):
        return "%(lat)15e %(lon)15e '%(name)s' %(tstart)e" % self.__dict__


class QSSPGreen(Object):
    depth = Float.T(default=10.0)
    filename = String.T(default='GF_10km')
    calculate = Bool.T(default=True)

    def string_for_config(self):
        return "%(depth)15e '%(filename)s' %(calculate)i" % self.__dict__


class QSSPConfig(Object):
    qssp_version = String.T(default='2017')
    time_region = Tuple.T(2, gf.Timing.T(), default=(
        gf.Timing('-10'), gf.Timing('+890')))

    frequency_max = Float.T(optional=True)
    slowness_max = Float.T(default=0.4)
    antialiasing_factor = Float.T(default=0.1)

    switch_turning_point_filter = Int.T(default=0)
    max_pene_d1 = Float.T(default=2891.5)
    max_pene_d2 = Float.T(default=6371.0)
    earth_radius = Float.T(default=6371.0)
    switch_free_surf_reflection = Int.T(default=1)

    lowpass_order = Int.T(default=0, optional=True)
    lowpass_corner = Float.T(default=1.0, optional=True)

    bandpass_order = Int.T(default=0, optional=True)
    bandpass_corner_low = Float.T(default=1.0, optional=True)
    bandpass_corner_high = Float.T(default=1.0, optional=True)

    output_slowness_min = Float.T(default=0.0, optional=True)
    output_slowness_max = Float.T(optional=True)

    spheroidal_modes = Bool.T(default=True)
    toroidal_modes = Bool.T(default=True)

    # only available in 2010beta:
    cutoff_harmonic_degree_sd = Int.T(optional=True, default=0)

    cutoff_harmonic_degree_min = Int.T(default=0)
    cutoff_harmonic_degree_max = Int.T(default=25000)

    crit_frequency_sge = Float.T(default=0.0)
    crit_harmonic_degree_sge = Int.T(default=0)

    include_physical_dispersion = Bool.T(default=False)

    source_patch_radius = Float.T(default=0.0)

    cut = Tuple.T(2, gf.Timing.T(), optional=True)

    fade = Tuple.T(4, gf.Timing.T(), optional=True)
    relevel_with_fade_in = Bool.T(default=False)
    nonzero_fade_in = Bool.T(default=False)
    nonzero_fade_out = Bool.T(default=False)

    def items(self):
        return dict(self.T.inamevals(self))


class QSSPConfigFull(QSSPConfig):
    time_window = Float.T(default=900.0)

    receiver_depth = Float.T(default=0.0)
    sampling_interval = Float.T(default=5.0)

    output_filename = String.T(default='receivers_')
    output_format = Int.T(default=1)
    output_time_window = Float.T(optional=True)
    gf_directory = String.T(default='qssp_green')
    greens_functions = List.T(QSSPGreen.T())

    sources = List.T(QSSPSource.T())
    receivers = List.T(QSSPReceiver.T())

    earthmodel_1d = gf.meta.Earthmodel1D.T(optional=True)

    @staticmethod
    def example():
        conf = QSSPConfigFull()
        conf.sources.append(QSSPSourceMT())
        lats = [20.]
        conf.receivers.extend(QSSPReceiver(lat=lat) for lat in lats)
        conf.greens_functions.append(QSSPGreen())
        return conf

    @property
    def components(self):
        return qssp_components[self.output_format]

    def get_output_filenames(self, rundir):
        return [
            pjoin(rundir, self.output_filename+c+'.dat') for c in self.components]

    def ensure_gf_directory(self):
        util.ensuredir(self.gf_directory)

    def string_for_config(self):

        def aggregate(l):
            return len(l), '\n'.join(x.string_for_config() for x in l)

        assert len(self.greens_functions) > 0
        assert len(self.sources) > 0
        assert len(self.receivers) > 0

        d = self.__dict__.copy()

        if self.output_time_window is None:
            d['output_time_window'] = self.time_window

        if self.output_slowness_max is None:
            d['output_slowness_max'] = self.slowness_max

        if self.frequency_max is None:
            d['frequency_max'] = 0.5/self.sampling_interval

        d['gf_directory'] = os.path.abspath(self.gf_directory) + '/'

        d['n_receiver_lines'], d['receiver_lines'] = aggregate(self.receivers)
        d['n_source_lines'], d['source_lines'] = aggregate(self.sources)
        d['n_gf_lines'], d['gf_lines'] = aggregate(self.greens_functions)
        model_str, nlines = cake_model_to_config(self.earthmodel_1d)
        d['n_model_lines'] = nlines
        d['model_lines'] = model_str

        if len(self.sources) == 0 or isinstance(self.sources[0], QSSPSourceMT):
            d['point_source_type'] = 1
        else:
            d['point_source_type'] = 2

        d['scutoff_doc'] = '''
#    (SH waves), minimum and maximum cutoff harmonic degrees
#    Note: if the near-field static displacement is desired, the minimum
#          cutoff harmonic degree should not be smaller than, e.g., 2000.
'''.strip()

        d['scutoff'] = '%i %i' % (
                self.cutoff_harmonic_degree_min,
                self.cutoff_harmonic_degree_max)

        d['sfilter_doc'] = '''
# 4. selection of order of Butterworth low-pass filter (if <= 0, then no
#    filtering), corner frequency (smaller than the cut-off frequency defined
#    above)
'''.strip()

        if self.bandpass_order != 0:
            raise QSSPError(
                'this version of qssp does not support bandpass '
                'settings, use lowpass instead')

        d['sfilter'] = '%i %f %f' % (
            self.bandpass_order,
            self.bandpass_corner_low,
            self.bandpass_corner_high)

        template = '''# autogenerated QSSP input by qssp.py
#
# This is the input file of FORTRAN77 program "qssp2017" for calculating
# synthetic seismograms of a self-gravitating, spherically symmetric,
# isotropic and viscoelastic earth.
#
# by
# Rongjiang Wang <wang@gfz-potsdam.de>
# Helmholtz-Centre Potsdam
# GFZ German Reseach Centre for Geosciences
# Telegrafenberg, D-14473 Potsdam, Germany
#
# Last modified: Potsdam, October 2017
#
# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
# If not specified, SI Unit System is used overall!
#
# Coordinate systems:
# spherical (r,t,p) with r = radial,
#                        t = co-latitude,
#                        p = east longitude.
# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
#
#	UNIFORM RECEIVER DEPTH
#	======================
# 1. uniform receiver depth [km]
#-------------------------------------------------------------------------------------------
    %(receiver_depth)e
#-------------------------------------------------------------------------------------------
#
#   SPACE-TIME SAMPLING PARAMETERS
#	=========================
# 1. time window [sec], sampling interval [sec]
# 2. max. frequency [Hz] of Green's functions
# 3. max. slowness [s/km] of Green's functions
#    Note: if the near-field static displacement is desired, the maximum slowness should not
#          be smaller than the S wave slowness in the receiver layer
# 4. anti-aliasing factor (> 0 & < 1), if it is <= 0 or >= 1/e (~ 0.4), then
#    default value of 1/e is used (e.g., 0.1 = alias phases will be suppressed
#    to 10%% of their original amplitude)
# 5. switch (1/0 = yes/no) of turning-point filter, the range (d1, d2) of max. penetration
#    depth [km] (d1 is meaningless if it is smaller than the receiver/source depth, and
#    d2 is meaningless if it is equal to or larger than the earth radius)
#
#    Note: The turning-point filter (Line 5) works only for the extended QSSP code (e.g.,
#          qssp2016). if this filter is selected, all phases with the turning point
#          shallower than d1 or deeper than d2 will be filtered.
#
# 6. Earth radius [km], switch of free-surface-reflection filter (1/0 = with/without free
#    surface reflection)
#
#    Note: The free-surface-reflection filter (Line 6) works only for the extended QSSP
#          code (e.g., qssp2016). if this filter is selected, all phases with the turning
#          point shallower than d1 or deeper than d2 will be filtered.
#-------------------------------------------------------------------------------------------
    %(time_window)e   %(sampling_interval)e
    %(frequency_max)e
    %(slowness_max)e
    %(antialiasing_factor)e
    %(switch_turning_point_filter)i   %(max_pene_d1)e   %(max_pene_d2)e
    %(earth_radius)e   %(switch_free_surf_reflection)i
#-------------------------------------------------------------------------------------------
#
#	SELF-GRAVITATING EFFECT
#	=======================
# 1. the critical frequency [Hz] and the critical harmonic degree, below which
#    the self-gravitating effect should be included
#-------------------------------------------------------------------------------------------
    %(crit_frequency_sge)e     %(crit_harmonic_degree_sge)i
#-------------------------------------------------------------------------------------------
#
#	WAVE TYPES
#	==========
# 1. selection (1/0 = yes/no) of speroidal modes (P-SV waves), selection of toroidal modes
%(scutoff_doc)s
#-------------------------------------------------------------------------------------------
    %(spheroidal_modes)i     %(toroidal_modes)i    %(scutoff)s
#-------------------------------------------------------------------------------------------
#	GREEN'S FUNCTION FILES
#	======================
# 1. number of discrete source depths, estimated radius of each source patch [km] and
#    directory for Green's functions
# 2. list of the source depths [km], the respective file names of the Green's
#    functions (spectra) and the switch number (0/1) (0 = do not calculate
#    this Green's function because it exists already, 1 = calculate or update
#    this Green's function. Note: update is required if any of the above
#    parameters is changed)
#-------------------------------------------------------------------------------------------
   %(n_gf_lines)i   %(source_patch_radius)e   '%(gf_directory)s'
   %(gf_lines)s
#--------------------------------------------------------------------------------------------------------
#
#   MULTI-EVENT SOURCE PARAMETERS
#   =============================
# 1. number of discrete point sources and selection of the source data format
#    (1, 2 or 3)
# 2. list of the multi-event sources
#
#    Format 1 (full moment tensor):
#    Unit     Mrr  Mtt  Mpp  Mrt  Mrp  Mtp  Lat   Lon   Depth  T_origin T_rise
#    [Nm]                                   [deg] [deg] [km]   [sec]    [sec]
#
#    Format 2 (double couple):
#    Unit   Strike    Dip       Rake      Lat   Lon   Depth  T_origin T_rise
#    [Nm]   [deg]     [deg]     [deg]     [deg] [deg] [km]   [sec]    [sec]
#
#    Format 3 (single force):
#    Unit      Feast    Fnorth  Fvertical    Lat   Lon   Depth  T_origin T_rise
#    [N]                                     [deg] [deg] [km]   [sec]    [sec]
#
#    Note: for each point source, the default moment (force) rate time function is used, defined by a
#          squared half-period (T_rise) sinusoid starting at T_origin.
#-----------------------------------------------------------------------------------
  %(n_source_lines)i     %(point_source_type)i
%(source_lines)s
#--------------------------------------------------------------------------------------------------------
#
#   RECEIVER PARAMETERS
#   ===================
# 1. select output observables (1/0 = yes/no)
#    Note: the gravity change defined here is space based, i.e., the effect due to free-air
#          gradient and inertial are not included. the vertical component is positve upwards.
# 2. output file name
# 3. output time window [sec] (<= Green's function time window)
%(sfilter_doc)s
# 5. lower and upper slowness cut-off [s/km] (slowness band-pass filter)
# 6. number of receiver
# 7. list of the station parameters
#    Format:
#    Lat     Lon    Name     Time_reduction
#    [deg]   [deg]           [sec]
#    (Note: Time_reduction = start time of the time window)
#--------------------------------------------------------------------------------------------------------
# disp | velo | acce | strain | strain_rate | stress | stress_rate | rotation | rotation_rate | gravity
#--------------------------------------------------------------------------------------------------------
  1      1      1      1        1             1        1             1          1               1        1
  '%(output_filename)s'
  %(output_time_window)e
  %(sfilter)s
  %(output_slowness_min)e    %(output_slowness_max)e
  %(n_receiver_lines)i
%(receiver_lines)s
#-------------------------------------------------------------------------------------------
#
#	                LAYERED EARTH MODEL (IASP91)
#                   ============================
# 1. number of data lines of the layered model and selection for including
#    the physical dispersion according Kamamori & Anderson (1977)
#-------------------------------------------------------------------------------------------
    %(n_model_lines)i    %(include_physical_dispersion)i
#--------------------------------------------------------------------------------------------------------
#
#   MODEL PARAMETERS
#   ================
# no depth[km] vp[km/s] vs[km/s] ro[g/cm^3]      qp     qs
#-------------------------------------------------------------------------------------------
  1      0.000     5.8000     3.2000     2.6000    1478.30     599.99
  2      3.300     5.8000     3.2000     2.6000    1478.30     599.99
  3      3.300     5.8000     3.2000     2.6000    1478.30     599.99
  4     10.000     5.8000     3.2000     2.6000    1478.30     599.99
  5     10.000     6.8000     3.9000     2.9200    1368.02     599.99
  6     18.000     6.8000     3.9000     2.9200    1368.02     599.99
  7     18.000     8.0355     4.4839     3.6410     950.50     394.62
  8     43.000     8.0379     4.4856     3.5801     972.77     403.93
  9     80.000     8.0400     4.4800     3.5020    1008.71     417.59
 10     80.000     8.0450     4.4900     3.5020     182.03      75.60
 11    120.000     8.0505     4.5000     3.4268     182.57      76.06
 12    165.000     8.1750     4.5090     3.3711     188.72      76.55
 13    210.000     8.3007     4.5184     3.3243     200.97      79.40
 14    210.000     8.3007     4.5184     3.3243     338.47     133.72
 15    260.000     8.4822     4.6094     3.3663     346.37     136.38
 16    310.000     8.6650     4.6964     3.4110     355.85     139.38
 17    360.000     8.8476     4.7832     3.4577     366.34     142.76
 18    410.000     9.0302     4.8702     3.5068     377.93     146.57
 19    410.000     9.3601     5.0806     3.9317     413.66     162.50
 20    460.000     9.5280     5.1864     3.9273     417.32     164.87
 21    510.000     9.6962     5.2922     3.9233     419.94     166.80
 22    560.000     9.8640     5.3989     3.9218     422.55     168.78
 23    610.000    10.0320     5.5047     3.9206     425.51     170.82
 24    660.000    10.2000     5.6104     3.9201     428.69     172.93
 25    660.000    10.7909     5.9607     4.2387    1350.54     549.45
 26    710.000    10.9222     6.0898     4.2986    1311.17     543.48
 27    760.000    11.0553     6.2100     4.3565    1277.93     537.63
 28    809.500    11.1355     6.2424     4.4118    1269.44     531.91
 29    859.000    11.2228     6.2799     4.4650    1260.68     526.32
 30    908.500    11.3068     6.3164     4.5162    1251.69     520.83
 31    958.000    11.3897     6.3519     4.5654    1243.02     515.46
 32   1007.500    11.4704     6.3860     4.5926    1234.54     510.20
 33   1057.000    11.5493     6.4182     4.6198    1226.52     505.05
 34   1106.500    11.6265     6.4514     4.6467    1217.91     500.00
 35   1156.000    11.7020     6.4822     4.6735    1210.02     495.05
 36   1205.500    11.7768     6.5131     4.7001    1202.04     490.20
 37   1255.000    11.8491     6.5431     4.7266    1193.99     485.44
 38   1304.500    11.9208     6.5728     4.7528    1186.06     480.77
 39   1354.000    11.9891     6.6009     4.7790    1178.19     476.19
 40   1403.500    12.0571     6.6285     4.8050    1170.53     471.70
 41   1453.000    12.1247     6.6554     4.8307    1163.16     467.29
 42   1502.500    12.1912     6.6813     4.8562    1156.04     462.96
 43   1552.000    12.2558     6.7070     4.8817    1148.76     458.72
 44   1601.500    12.3181     6.7323     4.9069    1141.32     454.55
 45   1651.000    12.3813     6.7579     4.9321    1134.01     450.45
 46   1700.500    12.4427     6.7820     4.9570    1127.02     446.43
 47   1750.000    12.5030     6.8056     4.9817    1120.09     442.48
 48   1799.500    12.5638     6.8289     5.0062    1108.58     436.68
 49   1849.000    12.6226     6.8517     5.0306    1097.16     431.03
 50   1898.500    12.6807     6.8743     5.0548    1085.97     425.53
 51   1948.000    12.7384     6.8972     5.0789    1070.38     418.41
 52   1997.500    12.7956     6.9194     5.1027    1064.23     414.94
 53   2047.000    12.8524     6.9416     5.1264    1058.03     411.52
 54   2096.500    12.9093     6.9625     5.1499    1048.09     406.50
 55   2146.000    12.9663     6.9852     5.1732    1042.07     403.23
 56   2195.500    13.0226     7.0069     5.1963    1032.14     398.41
 57   2245.000    13.0786     7.0286     5.2192    1018.38     392.16
 58   2294.500    13.1337     7.0504     5.2420    1008.79     387.60
 59   2344.000    13.1895     7.0722     5.2646     999.44     383.14
 60   2393.500    13.2465     7.0932     5.2870     990.77     378.79
 61   2443.000    13.3017     7.1144     5.3092     985.63     375.94
 62   2492.500    13.3584     7.1368     5.3313     976.81     371.75
 63   2542.000    13.4156     7.1584     5.3531     968.46     367.65
 64   2591.500    13.4741     7.1804     5.3748     960.36     363.64
 65   2640.000    13.5311     7.2031     5.3962     952.00     359.71
 66   2690.000    13.5899     7.2253     5.4176     940.88     354.61
 67   2740.000    13.6498     7.2485     5.4387     933.21     350.88
 68   2740.000    13.6498     7.2485     5.6934     722.73     271.74
 69   2789.670    13.6533     7.2593     5.7196     726.87     273.97
 70   2839.330    13.6570     7.2700     5.7458     725.11     273.97
 71   2891.500    13.6601     7.2817     5.7721     723.12     273.97
 72   2891.500     8.0000     0.0000     9.9145   57822.00       0.00
 73   2939.330     8.0382     0.0000     9.9942   57822.00       0.00
 74   2989.660     8.1283     0.0000    10.0722   57822.00       0.00
 75   3039.990     8.2213     0.0000    10.1485   57822.00       0.00
 76   3090.320     8.3122     0.0000    10.2233   57822.00       0.00
 77   3140.660     8.4001     0.0000    10.2964   57822.00       0.00
 78   3190.990     8.4861     0.0000    10.3679   57822.00       0.00
 79   3241.320     8.5692     0.0000    10.4378   57822.00       0.00
 80   3291.650     8.6496     0.0000    10.5062   57822.00       0.00
 81   3341.980     8.7283     0.0000    10.5731   57822.00       0.00
 82   3392.310     8.8036     0.0000    10.6385   57822.00       0.00
 83   3442.640     8.8761     0.0000    10.7023   57822.00       0.00
 84   3492.970     8.9461     0.0000    10.7647   57822.00       0.00
 85   3543.300     9.0138     0.0000    10.8257   57822.00       0.00
 86   3593.640     9.0792     0.0000    10.8852   57822.00       0.00
 87   3643.970     9.1426     0.0000    10.9434   57822.00       0.00
 88   3694.300     9.2042     0.0000    11.0001   57822.00       0.00
 89   3744.630     9.2634     0.0000    11.0555   57822.00       0.00
 90   3794.960     9.3205     0.0000    11.1095   57822.00       0.00
 91   3845.290     9.3760     0.0000    11.1623   57822.00       0.00
 92   3895.620     9.4297     0.0000    11.2137   57822.00       0.00
 93   3945.950     9.4814     0.0000    11.2639   57822.00       0.00
 94   3996.280     9.5306     0.0000    11.3127   57822.00       0.00
 95   4046.620     9.5777     0.0000    11.3604   57822.00       0.00
 96   4096.950     9.6232     0.0000    11.4069   57822.00       0.00
 97   4147.280     9.6673     0.0000    11.4521   57822.00       0.00
 98   4197.610     9.7100     0.0000    11.4962   57822.00       0.00
 99   4247.940     9.7513     0.0000    11.5391   57822.00       0.00
100   4298.270     9.7914     0.0000    11.5809   57822.00       0.00
101   4348.600     9.8304     0.0000    11.6216   57822.00       0.00
102   4398.930     9.8682     0.0000    11.6612   57822.00       0.00
103   4449.260     9.9051     0.0000    11.6998   57822.00       0.00
104   4499.600     9.9410     0.0000    11.7373   57822.00       0.00
105   4549.930     9.9761     0.0000    11.7737   57822.00       0.00
106   4600.260    10.0103     0.0000    11.8092   57822.00       0.00
107   4650.590    10.0439     0.0000    11.8437   57822.00       0.00
108   4700.920    10.0768     0.0000    11.8772   57822.00       0.00
109   4751.250    10.1095     0.0000    11.9098   57822.00       0.00
110   4801.580    10.1415     0.0000    11.9414   57822.00       0.00
111   4851.910    10.1739     0.0000    11.9722   57822.00       0.00
112   4902.240    10.2049     0.0000    12.0001   57822.00       0.00
113   4952.580    10.2329     0.0000    12.0311   57822.00       0.00
114   5002.910    10.2565     0.0000    12.0593   57822.00       0.00
115   5053.240    10.2745     0.0000    12.0867   57822.00       0.00
116   5103.570    10.2854     0.0000    12.1133   57822.00       0.00
117   5153.500    10.2890     0.0000    12.1391   57822.00       0.00
118   5153.500    11.0427     3.5043    12.7037     633.26      85.03
119   5204.610    11.0585     3.5187    12.7289     629.89      85.03
120   5255.320    11.0718     3.5314    12.7530     626.87      85.03
121   5306.040    11.0850     3.5435    12.7760     624.08      85.03
122   5356.750    11.0983     3.5551    12.7980     621.50      85.03
123   5407.460    11.1166     3.5661    12.8188     619.71      85.03
124   5458.170    11.1316     3.5765    12.8387     617.78      85.03
125   5508.890    11.1457     3.5864    12.8574     615.93      85.03
126   5559.600    11.1590     3.5957    12.8751     614.21      85.03
127   5610.310    11.1715     3.6044    12.8917     612.62      85.03
128   5661.020    11.1832     3.6126    12.9072     611.12      85.03
129   5711.740    11.1941     3.6202    12.9217     609.74      85.03
130   5762.450    11.2041     3.6272    12.9351     608.48      85.03
131   5813.160    11.2134     3.6337    12.9474     607.31      85.03
132   5863.870    11.2219     3.6396    12.9586     606.26      85.03
133   5914.590    11.2295     3.6450    12.9688     605.28      85.03
134   5965.300    11.2364     3.6498    12.9779     604.44      85.03
135   6016.010    11.2424     3.6540    12.9859     603.69      85.03
136   6066.720    11.2477     3.6577    12.9929     603.04      85.03
137   6117.440    11.2521     3.6608    12.9988     602.49      85.03
138   6168.150    11.2557     3.6633    13.0036     602.05      85.03
139   6218.860    11.2586     3.6653    13.0074     601.70      85.03
140   6269.570    11.2606     3.6667    13.0100     601.46      85.03
141   6320.290    11.2618     3.6675    13.0117     601.32      85.03
142   6371.000    11.2622     3.6678    13.0122     601.27      85.03
#---------------------------------end of all inputs-----------------------------------------
'''  # noqa

        return (template % d).encode('ascii')


class QSSPError(gf.store.StoreError):
    pass


class Interrupted(gf.store.StoreError):
    def __str__(self):
        return 'Interrupted.'


class QSSPRunner(object):

    def __init__(self, tmp=None, keep_tmp=False):

        self.tempdir = mkdtemp(prefix='qssprun-', dir=tmp)
        self.keep_tmp = keep_tmp
        self.config = None

    def run(self, config):
        self.config = config

        input_fn = pjoin(self.tempdir, 'input')

        with open(input_fn, 'wb') as f:
            input_str = config.string_for_config()
            logger.debug('===== begin qssp input =====\n'
                         '%s===== end qssp input =====' % input_str)
            f.write(input_str)

        #program = program_bins['qssp.%s' % config.qssp_version]
        program = program_bins['atmqssp']

        old_wd = os.getcwd()

        os.chdir(self.tempdir)

        interrupted = []

        def signal_handler(signum, frame):
            os.kill(proc.pid, signal.SIGTERM)
            interrupted.append(True)

        original = signal.signal(signal.SIGINT, signal_handler)
        try:
            try:
                proc = Popen(program, stdin=PIPE, stdout=PIPE, stderr=PIPE)
            except OSError:
                os.chdir(old_wd)
                raise QSSPError(
                    '''could not start qssp executable: "%s"
Available fomosto backends and download links to the modelling codes are listed
on

      https://pyrocko.org/current/apps/fomosto/backends.html

''' % program)

            (output_str, error_str) = proc.communicate(b'input\n')

        finally:
            signal.signal(signal.SIGINT, original)

        if interrupted:
            raise KeyboardInterrupt()

        logger.debug('===== begin qssp output =====\n'
                     '%s===== end qssp output =====' % output_str.decode())

        errmess = []
        if proc.returncode != 0:
            errmess.append(
                'qssp had a non-zero exit state: %i' % proc.returncode)
        if error_str:

            logger.warn(
                'qssp emitted something via stderr: \n\n%s'
                % error_str.decode())

            # errmess.append('qssp emitted something via stderr')
        if output_str.lower().find(b'error') != -1:
            errmess.append("the string 'error' appeared in qssp output")

        if errmess:
            os.chdir(old_wd)
            raise QSSPError('''
===== begin qssp input =====
%s===== end qssp input =====
===== begin qssp output =====
%s===== end qssp output =====
===== begin qssp error =====
%s===== end qssp error =====
%s
qssp has been invoked as "%s"'''.lstrip() % (
                input_str.decode(),
                output_str.decode(),
                error_str.decode(),
                '\n'.join(errmess),
                program))

        self.qssp_output = output_str
        self.qssp_error = error_str

        os.chdir(old_wd)

    def get_traces(self):
        fns = self.config.get_output_filenames(self.tempdir)
        traces = []
        for comp, fn in zip(self.config.components, fns):
            data = num.loadtxt(fn, skiprows=1, dtype=num.float)
            nsamples, ntraces = data.shape
            ntraces -= 1
            deltat = (data[-1, 0] - data[0, 0])/(nsamples-1)
            toffset = data[0, 0]
            for itrace in range(ntraces):
                rec = self.config.receivers[itrace]
                tmin = rec.tstart + toffset
                tr = trace.Trace(
                    '', '%04i' % itrace, '', comp,
                    tmin=tmin, deltat=deltat, ydata=data[:, itrace+1],
                    meta=dict(distance=rec.distance))
                traces.append(tr)

        return traces

    def __del__(self):
        if self.tempdir:
            if not self.keep_tmp:
                shutil.rmtree(self.tempdir)
                self.tempdir = None
            else:
                logger.warn(
                    'not removing temporary directory: %s' % self.tempdir)


class QSSPGFBuilder(gf.builder.Builder):
    nsteps = 2

    def __init__(self, store_dir, step, shared, block_size=None, tmp=None,
                 force=False):
        self.gfmapping = [
            (MomentTensor(m=symmat6(1, 0, 0, 1, 0, 0)),
             {'_disp_n': (0, -1), '_disp_e': (3, -1), '_disp_z': (5, -1)}),
            (MomentTensor(m=symmat6(0, 0, 0, 0, 1, 1)),
             {'_disp_n': (1, -1), '_disp_e': (4, -1), '_disp_z': (6, -1)}),
            (MomentTensor(m=symmat6(0, 0, 1, 0, 0, 0)),
             {'_disp_n': (2, -1), '_disp_z': (7, -1)}),
            (MomentTensor(m=symmat6(0, 1, 0, 0, 0, 0)),
             {'_disp_n': (8, -1), '_disp_z': (9, -1)}),
        ]

        self.store = gf.store.Store(store_dir, 'w')

        if step == 0:
            block_size = (1, 1, self.store.config.ndistances)
        else:
            if block_size is None:
                block_size = (1, 1, 51)

        if len(self.store.config.ns) == 2:
            block_size = block_size[1:]

        gf.builder.Builder.__init__(
            self, self.store.config, step, block_size=block_size, force=force)

        baseconf = self.store.get_extra('qssp')

        conf = QSSPConfigFull(**baseconf.items())
        conf.gf_directory = pjoin(store_dir, 'qssp_green')
        conf.earthmodel_1d = self.store.config.earthmodel_1d
        deltat = self.store.config.deltat

        if 'time_window' not in shared:
            d = self.store.make_timing_params(
                conf.time_region[0], conf.time_region[1])

            tmax = math.ceil(d['tmax'] / deltat) * deltat
            tmin = math.floor(d['tmin'] / deltat) * deltat

            shared['time_window'] = tmax - tmin
            shared['tstart'] = tmin

        self.tstart = shared['tstart']
        conf.time_window = shared['time_window']

        self.tmp = tmp
        if self.tmp is not None:
            util.ensuredir(self.tmp)

        util.ensuredir(conf.gf_directory)

        self.qssp_config = conf

    def work_block(self, iblock):
        if len(self.store.config.ns) == 2:
            (sz, firstx), (sz, lastx), (ns, nx) = \
                self.get_block_extents(iblock)

            rz = self.store.config.receiver_depth
        else:
            (rz, sz, firstx), (rz, sz, lastx), (nr, ns, nx) = \
                self.get_block_extents(iblock)

        gf_filename = 'GF_%gkm_%gkm' % (sz/km, rz/km)

        conf = copy.deepcopy(self.qssp_config)

        gf_path = os.path.join(conf.gf_directory, '?_' + gf_filename)

        if self.step == 0 and len(glob.glob(gf_path)) == 7:
            logger.info(
                'Skipping step %i / %i, block %i / %i (GF already exists)'
                % (self.step+1, self.nsteps, iblock+1, self.nblocks))

            return

        logger.info(
            'Starting step %i / %i, block %i / %i' %
            (self.step+1, self.nsteps, iblock+1, self.nblocks))

        tbeg = time.time()

        runner = QSSPRunner(tmp=self.tmp)

        conf.receiver_depth = rz/km
        conf.sampling_interval = 1.0 / self.gf_config.sample_rate

        dx = self.gf_config.distance_delta

        if self.step == 0:
            distances = [firstx]
        else:
            distances = num.linspace(firstx, firstx + (nx-1)*dx, nx)

        conf.receivers = [
            QSSPReceiver(
                lat=90-d*cake.m2d,
                lon=180.,
                tstart=self.tstart,
                distance=d)

            for d in distances]

        if self.step == 0:
            gf_filename = 'TEMP' + gf_filename[2:]

        gfs = [QSSPGreen(
            filename=gf_filename,
            depth=sz/km,
            calculate=(self.step == 0))]

        conf.greens_functions = gfs

        trise = 0.001*conf.sampling_interval  # make it short (delta impulse)

        if self.step == 0:
            conf.sources = [QSSPSourceMT(
                lat=90-0.001*dx*cake.m2d,
                lon=0.0,
                trise=trise,
                torigin=0.0)]

            runner.run(conf)
            gf_path = os.path.join(conf.gf_directory, '?_' + gf_filename)
            for s in glob.glob(gf_path):
                d = s.replace('TEMP_', 'GF_')
                os.rename(s, d)

        else:
            for mt, gfmap in self.gfmapping[
                    :[3, 4][self.gf_config.ncomponents == 10]]:
                m = mt.m_up_south_east()

                conf.sources = [QSSPSourceMT(
                    lat=90-0.001*dx*cake.m2d,
                    lon=0.0,
                    mrr=m[0, 0], mtt=m[1, 1], mpp=m[2, 2],
                    mrt=m[0, 1], mrp=m[0, 2], mtp=m[1, 2],
                    trise=trise,
                    torigin=0.0)]

                runner.run(conf)

                rawtraces = runner.get_traces()

                interrupted = []

                def signal_handler(signum, frame):
                    interrupted.append(True)

                original = signal.signal(signal.SIGINT, signal_handler)
                self.store.lock()
                duplicate_inserts = 0
                try:
                    for itr, tr in enumerate(rawtraces):
                        if tr.channel in gfmap:

                            x = tr.meta['distance']
                            ig, factor = gfmap[tr.channel]

                            if len(self.store.config.ns) == 2:
                                args = (sz, x, ig)
                            else:
                                args = (rz, sz, x, ig)

                            if conf.cut:
                                tmin = self.store.t(conf.cut[0], args[:-1])
                                tmax = self.store.t(conf.cut[1], args[:-1])
                                if None in (tmin, tmax):
                                    continue

                                tr.chop(tmin, tmax)

                            tmin = tr.tmin
                            tmax = tr.tmax

                            if conf.fade:
                                ta, tb, tc, td = [
                                    self.store.t(v, args[:-1])
                                    for v in conf.fade]

                                if None in (ta, tb, tc, td):
                                    continue

                                if not (ta <= tb and tb <= tc and tc <= td):
                                    raise QSSPError(
                                        'invalid fade configuration')

                                t = tr.get_xdata()
                                fin = num.interp(t, [ta, tb], [0., 1.])
                                fout = num.interp(t, [tc, td], [1., 0.])
                                anti_fin = 1. - fin
                                anti_fout = 1. - fout

                                y = tr.ydata

                                sum_anti_fin = num.sum(anti_fin)
                                sum_anti_fout = num.sum(anti_fout)

                                if conf.nonzero_fade_in \
                                        and sum_anti_fin != 0.0:
                                    yin = num.sum(anti_fin*y) / sum_anti_fin
                                else:
                                    yin = 0.0

                                if conf.nonzero_fade_out \
                                        and sum_anti_fout != 0.0:
                                    yout = num.sum(anti_fout*y) / sum_anti_fout
                                else:
                                    yout = 0.0

                                y2 = anti_fin*yin + fin*fout*y + anti_fout*yout

                                if conf.relevel_with_fade_in:
                                    y2 -= yin

                                tr.set_ydata(y2)

                            gf_tr = gf.store.GFTrace.from_trace(tr)
                            gf_tr.data *= factor

                            try:
                                self.store.put(args, gf_tr)
                            except gf.store.DuplicateInsert:
                                duplicate_inserts += 1

                finally:
                    if duplicate_inserts:
                        logger.warn(
                            '%i insertions skipped (duplicates)'
                            % duplicate_inserts)

                    self.store.unlock()
                    signal.signal(signal.SIGINT, original)

                if interrupted:
                    raise KeyboardInterrupt()

        tend = time.time()
        logger.info(
            'Done with step %i / %i, block %i / %i, wallclock time: %.0f s' % (
                self.step+1, self.nsteps, iblock+1, self.nblocks, tend-tbeg))


km = 1000.

def init(store_dir, variant):
#    if variant is None:
#        variant = '2017'

#    if ('qssp.' + variant) not in program_bins:
#        raise gf.store.StoreError('unsupported qssp variant: %s' % variant)
#
    qssp = QSSPConfig(qssp_version=variant)
    qssp.time_region = (
        gf.Timing('begin-50'),
        gf.Timing('end+100'))

    qssp.cut = (
        gf.Timing('begin-50'),
        gf.Timing('end+100'))

    store_id = os.path.basename(os.path.realpath(store_dir))

    config = gf.meta.ConfigTypeA(
        id=store_id,
        ncomponents=10,
        sample_rate=0.2,
        receiver_depth=0*km,
        source_depth_min=10*km,
        source_depth_max=20*km,
        source_depth_delta=10*km,
        distance_min=100*km,
        distance_max=1000*km,
        distance_delta=10*km,
        earthmodel_1d=cake.load_model('ak135-f-average-no-ocean.f'),
        modelling_code_id='atmqssp',
        tabulated_phases=[
            gf.meta.TPDef(
                id='begin',
                definition='p,P,p\\,P\\,Pv_(cmb)p'),
            gf.meta.TPDef(
                id='end',
                definition='2.5'),
            gf.meta.TPDef(
                id='P',
                definition='!P'),
            gf.meta.TPDef(
                id='S',
                definition='!S'),
            gf.meta.TPDef(
                id='p',
                definition='!p'),
            gf.meta.TPDef(
                id='s',
                definition='!s')])

    config.validate()
    return gf.store.Store.create_editables(
        store_dir,
        config=config,
        extra={'qssp': qssp})


def build(
        store_dir,
        force=False,
        nworkers=None,
        continue_=False,
        step=None,
        iblock=None):

    return QSSPGFBuilder.build(
        store_dir, force=force, nworkers=nworkers, continue_=continue_,
        step=step, iblock=iblock)
