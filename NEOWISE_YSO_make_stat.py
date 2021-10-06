'''
This is a code extracting statistical information from NEOWISE photometric data
used in WISE YSO variability analysis (Park et al. 2021)
Created by Wooseok Park
Last updated : July 28th, 2021
'''


import numpy as np
import pandas as pa
from pandas import DataFrame as df
from scipy.optimize import curve_fit
from astropy.timeseries import LombScargle
import csv

# import packages to be used in new round function
from decimal import Decimal, ROUND_HALF_UP, ROUND_HALF_EVEN

# define and use original round rather than using python round
def my_round(in_number, ndigits=0, rounding_rule=1):
    """rounding_rule==1: ROUND_HALF_UP, Round to nearest, ties away from zero
       rounding_rule==2: ROUND_HALF_EVEN, Round to nearest, ties to even """
    # make float expression
    if ndigits > 0:
        expression = '0.' + '0' * ndigits
        number = in_number
    else:  # 0 or negative
        expression = '0'
        number = in_number / (10 ** (-ndigits))

    # round by rounding rule
    if rounding_rule == 2:
        round_number = Decimal(number).quantize(Decimal(expression), rounding=ROUND_HALF_EVEN)
    else:
        round_number = Decimal(number).quantize(Decimal(expression), rounding=ROUND_HALF_UP)

    # return number
    if ndigits > 0:
        return float(round_number)
    else:  # 0 or negative
        return int(round_number * (10 ** (-ndigits)))

# NEOWISE observation time in Julian Dates
tl1 = 56639
tl2 = 60200

print('start extracting data from WISE.....')

# open csvfile to save all the stats
csvfile = open('wise_csv/NEOWISE_YSO_variable_stat.csv', 'w', newline='')
csvdata = csv.writer(csvfile, delimiter=',')
csvdata.writerow(['Index', 'ra', 'dec','dist_sd',
                  'avg_W1', 'stdev_W1', 'avg_eW1',
                  'avg_W2', 'stdev_W2', 'avg_eW2',
                  'N_w1', 'N_w2', 'slope_w1', 'slope_w2',
                  'sd_sdfid_w1_flux', 'sd_sdfid_w2_flux',
                  'sd_sdfid_w1_mag', 'sd_sdfid_w2_mag',
                  'Delta_w1', 'Delta_w2',
                  'Delta_w1_flux', 'Delta_w2_flux',
                  'class', 'catalogue',
                  'Sw1', 'DeltaSw1', 'Sw2', 'DeltaSw2',
                  'SDw1', 'SDfidw1', 'SDw2', 'SDfidw2',
                  'chi2_mean_w1', 'chi2_wmean_w1', 'chi2_slope_w1',
                  'chi2_mean_w2', 'chi2_wmean_w2', 'chi2_slope_w2',
                  'linear_fap_w1', 'linear_fap_w2',
                  'center_mag_w1', 'center_mag_w2',
                  'max_W1', 'min_W1', 'median_W1',
                  'max_W2', 'min_W2', 'median_W2',
                  # Lomb-Scargle Periodogram analysis stats below
                  'period', 'power', 'baluev_fap',
                  'single_fap', 'mod_baluev_fap',
                  'amp', 'sinamp', 'ls_meanjy', 'sd_sdfid_periodrmv', 'sd_periodrmv', 'chi2_period_w2',
                  'linear_fap_periodrmv', 'Deltaw2_periodrmv'
                  ])


source1 = pa.read_csv('wise_csv/ysos_c.csv')


source2 = pa.read_csv('wise_csv/ysos_info.dat',
                                 header=None, skiprows=1, sep="\s+",
                                 names=['index_1', 'ra', 'dec', 'Disk'])

# YSO source informations are from different 3 catalogues. Index < 3505 from Megeath et al. (2012) and >= 3505 from Dunham et al. (2015)
# Index > 10000 from Esplin & Luhman (2019)
for i in range(1,10520):
    try:
        print(i)
        if i < 10000:
            source = source1
            ev_stage = source.loc[source['index'] == i, 'class'].values[0]
            cat = source.loc[source['index'] == i, 'cat'].values[0]
            ra = source.loc[source['index'] == i, 'ra'].values[0]
            dec = source.loc[source['index'] == i, 'dec'].values[0]
        else:
            source = source2
            ev_stage = source.loc[source['index_1'] == i, 'Disk'].values[0]
            cat = 'Esplin&Luhman'
            ra = source.loc[source['index_1'] == i, 'ra'].values[0]
            dec = source.loc[source['index_1'] == i, 'dec'].values[0]

        # call the averaged photometry data
        mddat = pa.read_csv('outlier_cut_data/'
                            + str(i) + '_cavg.csv',
                            names=['mjd', 'mag', 'emag', 'flt', 'class'],
                            skiprows=1)

        xw1 = mddat[mddat['flt'] == 'W1']
        xw2 = mddat[mddat['flt'] == 'W2']

        nw1 = len(xw1)
        nw2 = len(xw2)

        ### *.alld -> outlier removed raw photometry data. Only used for distance STD ###
        mdall = pa.read_csv('outlier_cut_data/'
                            + str(i) + '_alld.csv',
                            names=['mjd', 'mag', 'emag', 'flt', 'class','ra','dec'],
                            skiprows=1)
        mdall2 = mdall[mdall['flt'] == 'W2']

        # distance from mean position
        radist = abs(mdall2.ra - np.mean(mdall2.ra))
        decdist = abs(mdall2.dec - np.mean(mdall2.dec))

        # standard deviation distance of scattered positions
        dist_sd = np.std(np.sqrt(radist ** 2 + decdist ** 2) * 3600)


        #### W2 linear fitting ####

        xw2s = df.sort_values(xw2, 'mjd')
        xw2s = xw2s.reset_index(drop=True)

        if nw2 >= 3:
            w2f = xw2s['mag']
            mjdw2f = xw2s['mjd']
            ew2f = xw2s['emag']

            # magnitude / error to flux / error conversion
            fw2 = 171.85 * 10 ** (-w2f / 2.5)
            ferr2 = ew2f * fw2 / 1.0857


            # comment upper conversion and uncomment below to use magnitude rather than flux
            # fw2 = w2f
            # ferr = ew2f

            # median flux and mean error
            m2f = np.median(fw2)
            em2f = np.mean(ferr2) # just use mean error - it is too complicate to find the value closest to median

            # normalized flux and time
            yfitw2 = (fw2 - m2f) / m2f
            xfitw2 = mjdw2f - mjdw2f[0]




            def func(x, a, b):
                return a * x + b


            # error propagation
            sig = fw2 / m2f * np.sqrt((ferr2 / fw2) ** 2 + (em2f / m2f) ** 2)

            # linear fitting
            resw2, cov = curve_fit(func, xfitw2, yfitw2, sigma=sig,  # ferr/fw2[0]
                                   absolute_sigma=True)

            # resw2 returns [a,b] which is ax + b
            fiterrw2 = np.sqrt(np.diag(cov))  # fit error
            slope_w2 = (resw2[0] / fiterrw2[0])
            wmean_yfitw2 = np.dot(yfitw2, sig ** -2) / sum(sig ** -2)

            # derive chi-square of various fittings
            yfit = np.polyval(resw2, xfitw2)
            chisq_linfit2 = sum(((yfitw2 - yfit) / sig) ** 2)
            chisq_mean2 = sum((((yfitw2 - np.mean(yfitw2)) / sig) ** 2))
            chisq_wmean2 = sum((((yfitw2 - wmean_yfitw2) / sig) ** 2))

        else:
            slope_w2 = -0.1
            resw2 = [0, 0]
            fiterrw2 = [0, 0]
            chisq_linfit2 = 0
            chisq_mean2 = 0
            chisq_wmean2 = 0


        #### slope_W1 ####

        xw1s = df.sort_values(xw1, 'mjd')

        if nw1 >= 3:
            w1f = xw1s['mag']
            mjdw1f = xw1s['mjd']
            ew1f = xw1s['emag']

            fw1 = 309.54 * 10 ** (-w1f / 2.5)  # flux
            #     fw1 = w1f #magnitude

            ferr1 = ew1f * fw1 / 1.0857  # flux
            #     ferr = ew1f #magnitude

            fw1 = fw1.reset_index(drop=True)  # dataframe에서 sort하며 뒤섞인 index 값을 다시 reset 한다
            mjdw1f = mjdw1f.reset_index(drop=True)
            ferr1 = ferr1.reset_index(drop=True)

            m1f = np.median(fw1)
            em1f = np.mean(ferr1)

            yfitw1 = (fw1 - m1f) / m1f
            xfitw1 = mjdw1f - mjdw1f[0]


            def func(x, a, b):
                return a * x + b


            # test - fit with same error value and see the change of chi2
            # sig = np.ones(len(yfitw1)) * 0.1  # even 0.1 for error
            # sig = ferr1/fw1[0] # original error
            # error propagation
            sig = fw1 / m1f * np.sqrt((ferr1 / fw1) ** 2 + (em1f / m1f) ** 2)

            # sig = np.ones(len(yfitw1))*np.mean(ferr / fw1[0])  # mean error -- new constant

            resw1, cov = curve_fit(func, xfitw1, yfitw1, sigma=sig,  # ferr / fw1[0],
                                   absolute_sigma=True)

            fiterrw1 = np.sqrt(np.diag(cov))  # error in fitting
            slope_w1 = (resw1[0] / fiterrw1[0])
            wmean_yfitw1 = np.dot(yfitw1, sig ** -2) / sum(sig ** -2)

            # chi-square
            yfit = np.polyval(resw1, xfitw1)
            chisq_linfit1 = sum(((yfitw1 - yfit) / sig) ** 2)
            chisq_mean1 = sum((((yfitw1 - np.mean(yfitw1)) / sig) ** 2))
            chisq_wmean1 = sum((((yfitw1 - wmean_yfitw1) / sig) ** 2))


        else:
            slope_w1 = -0.1
            resw1 = [0, 0]
            fiterrw1 = [0, 0]
            chisq_linfit1 = 0
            chisq_mean1 = 0
            chisq_wmean1 = 0


        ### Linear FAP ###
        po1 = (chisq_wmean1 - chisq_linfit1) / chisq_wmean1
        lfap1 = (1 - po1) ** (nw1 / 2)

        po2 = (chisq_wmean2 - chisq_linfit2) / chisq_wmean2
        lfap2 = (1 - po2) ** (nw2 / 2)

        # print('slope_w1 is :       ', slope_w1)

        # print all the results as DataFrame

        # csvdata.writerow(['Index', 'avg_W1', 'stdev_W1', 'avg_eW1', 'DeltaW1', 'avg_W2',
        #                   'stdev_W2', 'avg_eW2', 'DeltaW2', 'N_w1', 'N_W2', 'slope_w1',
        #                   'slope_w2', 'sd_sdfid_w1(flux)', 'sd_sdfid_w2(flux)',
        #                   'Deltaall_w1', 'Deltaall_w2', 'class', 'catalogue',
        #                   'dist', 'sdist', 'Sw1', 'DeltaSw1', 'Sw2', 'DeltaSw2', 'SDw1',
        #                   'SDfidw1', 'SDw2', 'SDfidw2' ]) + 'ra', 'dec'

        if nw1 < 1:
            maxxw1mag = 0
            minxw1mag = 0
        else:
            maxxw1mag = max(xw1['mag'])
            minxw1mag = min(xw1['mag'])

        if nw2 < 1:
            maxxw2mag = 0
            minxw2mag = 0
        else:
            maxxw2mag = max(xw2['mag'])
            minxw2mag = min(xw2['mag'])


        '''
        Lomb-Scargle Periodogram
        Detecting Periodicity using Fourier Transform. We used a single sinusoidal fit for a lightcurve here. 
        '''
        lsav = LombScargle(xw2s.mjd, fw2, ferr2)
        frequency, power = lsav.autopower(  # nyquist_factor=1,
            maximum_frequency=1 / 200,  # minimum period > 201days
            #                                           minimum_frequency=0.0001) #0.2 #maximum period 10000days
            minimum_frequency=1 / 4800)  # maximum period < 4000days

        period_days = 1. / frequency
        period_hours = period_days * 24

        best_period = period_days[np.argmax(power)]
        best_frequency = frequency[np.argmax(power)]
        fap = lsav.false_alarm_probability(power)
        sin_fap = lsav.false_alarm_probability(power, method='single')

        # bootstrap fap
        #         bfap = lsav.false_alarm_probability(power,method='bootstrap',
        #                                             method_kwds={'n_bootstraps':1000})

        # amplitude from sinusoid
        arw2m = np.squeeze(np.array([xw2s.mjd]))
        smjd = np.linspace(arw2m[0], arw2m[-1], 1000)
        flux_jmod = lsav.model(smjd, best_frequency)
        amp = (max(flux_jmod) - min(flux_jmod)) * 0.5

        # real sinusoid amplitude
        phase_model = np.linspace(-0.5, 1.5, 100)
        best_frequency = frequency[np.argmax(power)]
        flux_model = lsav.model(phase_model / best_frequency, best_frequency)
        sinamp = (max(flux_model) - min(flux_model)) * 0.5
        # #     print(fap[np.argmax(power)], best_period)

        # amplitude normalized sd/sdfid
        flux_jdot = lsav.model(xw2s.mjd, best_frequency)
        nw2f = fw2 - flux_jdot
        # nw2f = np.log10(w2f/flux_jdot)*2.5
        nw2 = len(xw2s)
        ampsdfid = np.std(nw2f) / np.mean(ferr2)
        ampsd = np.std(nw2f)
        chi2 = sum(((fw2 - flux_jdot) / ferr2) ** 2)


        # mean from best fit periodogram
        flux_para = lsav.model_parameters(best_frequency)
        flux_off = lsav.offset()
        ls_mean = flux_off + flux_para[0]


        # linear fap after removing sinusoid

        presw2, cov = curve_fit(func, xw2s.mjd, nw2f, sigma=ferr2, absolute_sigma=True)
        yfit = np.polyval(presw2, xw2s.mjd)

        wmean_yfitw2 = np.dot(nw2f, ferr2 ** -2) / sum(ferr2 ** -2)
        chisq_linfit = sum(((nw2f - yfit) / ferr2) ** 2)
        chisq_wmean = sum((((nw2f - wmean_yfitw2) / ferr2) ** 2))
        pol = (chisq_wmean - chisq_linfit) / chisq_wmean
        lfap_prmv = (1 - pol) ** (len(xw2s) / 2)

        # amplitude normalized deltaw2 - reconvert flux to magnitude
        nw2m = 2.5 * np.log10(fw2 / flux_jdot)
        dw2_prmv = max(nw2m) - min(nw2m)



        ### Write all the derived statistics to the opened csv file ###
        if nw1 >= 5 and nw2 >= 5:
            csvdata.writerow([i, ra, dec, dist_sd, # index
                              np.mean(w1f), np.std(w1f), np.mean(ew1f),  # maxw1-minw1,
                              np.mean(w2f), np.std(w2f), np.mean(ew2f),  # maxw2-minw2,
                              nw1, nw2, slope_w1, slope_w2,
                              np.std(fw1) / np.mean(ferr1), np.std(fw2) / np.mean(ferr2),
                              np.std(w1f) / np.mean(ew1f), np.std(w2f) / np.mean(ew2f),
                              maxxw1mag - minxw1mag, maxxw2mag - minxw2mag,
                              np.max(fw1) - np.min(fw1), np.max(fw2) - np.min(fw2),
                              ev_stage, cat,
                              # c1, c2,
                              resw1[0], fiterrw1[0], resw2[0], fiterrw2[0],
                              np.std(fw1), np.mean(ferr1), np.std(fw2), np.mean(ferr2),
                              chisq_mean1, chisq_wmean1, chisq_linfit1,
                              chisq_mean2, chisq_wmean2, chisq_linfit2,
                              lfap1, lfap2,
                              .5 * (max(w1f) + min(w1f)), .5 * (max(w2f) + min(w2f)),
                              np.min(w1f), np.max(w1f), np.median(w1f),
                              np.min(w2f), np.max(w2f), np.median(w2f),
                              # LSP
                              best_period, np.max(power), fap[np.argmax(power)],
                              sin_fap[np.argmax(power)], fap[np.argmax(power)]*200/best_period,
                              amp, sinamp, ls_mean, ampsdfid, ampsd, chi2,
                              lfap_prmv, dw2_prmv
                              ])

        if nw1 < 5 and nw2 >= 5:
            csvdata.writerow([i, ra, dec, dist_sd,  # index
                              np.mean(w1f), 0, 0.01,  # maxw1 - minw1,
                              np.mean(w2f), np.std(w2f), np.mean(ew2f),  # maxw2 - minw2,
                              nw1, nw2, slope_w1, slope_w2,
                              0, np.std(fw2) / np.mean(ferr2),
                              0, np.std(w2f) / np.mean(ew2f),
                              maxxw1mag - minxw1mag, maxxw2mag - minxw2mag,
                              np.max(fw1) - np.min(fw1), np.max(fw2) - np.min(fw2),
                              ev_stage, cat,
                              # c1, c2,
                              resw1[0], fiterrw1[0], resw2[0], fiterrw2[0],
                              0, 0, np.std(fw2), np.mean(ferr2),
                              chisq_mean1, chisq_wmean1, chisq_linfit1,
                              chisq_mean2, chisq_wmean2, chisq_linfit2,
                              lfap1, lfap2,
                              .5 * (max(w1f) + min(w1f)), .5 * (max(w2f) + min(w2f)),
                              np.min(w1f), np.max(w1f), np.median(w1f),
                              np.min(w2f), np.max(w2f), np.median(w2f),
                              # LSP
                              best_period, np.max(power), fap[np.argmax(power)],
                              sin_fap[np.argmax(power)], fap[np.argmax(power)] * 200 / best_period,
                              amp, sinamp, ls_mean, ampsdfid, ampsd, chi2,
                              lfap_prmv, dw2_prmv
                              ])

        if nw1 >= 5 and nw2 < 5:
            csvdata.writerow([i, ra, dec, dist_sd, # index
                              np.mean(w1f), np.std(w1f), np.mean(ew1f),  # maxw1 - minw1,
                              np.mean(w2f), 0, 0.01,  # maxw2 - minw2,
                              nw1, nw2, slope_w1, slope_w2,
                              np.std(fw1) / np.mean(ferr1), 0,
                              np.std(w1f) / np.mean(ew1f), 0,
                              maxxw1mag - minxw1mag, maxxw2mag - minxw2mag,
                              np.max(fw1) - np.min(fw1), np.max(fw2) - np.min(fw2),
                              ev_stage, cat,
                              # c1, c2,
                              resw1[0], fiterrw1[0], resw2[0], fiterrw2[0],
                              np.std(fw1), np.mean(ferr1), 0, 0,
                              chisq_mean1, chisq_wmean1, chisq_linfit1,
                              chisq_mean2, chisq_wmean2, chisq_linfit2,
                              lfap1, lfap2,
                              .5 * (max(w1f) + min(w1f)), .5 * (max(w2f) + min(w2f)),
                              np.min(w1f), np.max(w1f), np.median(w1f),
                              np.min(w2f), np.max(w2f), np.median(w2f),
                              # LSP
                              best_period, np.max(power), fap[np.argmax(power)],
                              sin_fap[np.argmax(power)], fap[np.argmax(power)] * 200 / best_period,
                              amp, sinamp, ls_mean, ampsdfid, ampsd, chi2,
                              lfap_prmv, dw2_prmv
                              ])

        if nw1 < 5 and nw2 < 5:
            csvdata.writerow([i, ra, dec, dist_sd, # index
                              np.mean(w1f), 0, 0.01,  # maxw1 - minw1,
                              np.mean(w2f), 0, 0.01,  # maxw2 - minw2,
                              nw1, nw2, slope_w1, slope_w2,
                              0, 0,
                              0, 0,
                              maxxw1mag - minxw1mag, maxxw2mag - minxw2mag,
                              np.max(fw1) - np.min(fw1), np.max(fw2) - np.min(fw2),
                              ev_stage, cat,
                              # c1, c2,
                              resw1[0], fiterrw1[0], resw2[0], fiterrw2[0],
                              0, 0, 0, 0,
                              chisq_mean1, chisq_wmean1, chisq_linfit1,
                              chisq_mean2, chisq_wmean2, chisq_linfit2,
                              lfap1, lfap2,
                              .5 * (max(w1f) + min(w1f)), .5 * (max(w2f) + min(w2f)),
                              np.min(w1f), np.max(w1f), np.median(w1f),
                              np.min(w2f), np.max(w2f), np.median(w2f),
                              # LSP
                              best_period, np.max(power), fap[np.argmax(power)],
                              sin_fap[np.argmax(power)], fap[np.argmax(power)] * 200 / best_period,
                              amp, sinamp, ls_mean, ampsdfid, ampsd, chi2,
                              lfap_prmv, dw2_prmv
                              ])

    except Exception as e:
        print('source extraction error :  ', e)

csvfile.close()
