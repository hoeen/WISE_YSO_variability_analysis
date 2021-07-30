import numpy as np
import pandas as pa
import matplotlib
# matplotlib.rcParams['text.usetex'] = True
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import seaborn as sea
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from astropy.timeseries import LombScargle
from matplotlib.gridspec import GridSpec
from scipy.optimize import curve_fit

#### Changed variability light curve plot ####
def chvar_lin_plot(index):

    fig, ax = plt.subplots(1,2,figsize=(10,3))

    # plt.subplots_adjust

    # increase / decrease / burst / dimming 

    # stochastic / linear / curved / periodic

    # index = [597, 5862, 4939, 6307]
     #669
    # index = [597, 2842, 4939,4688]

    # index = [85,2842,4939,6307]
    v1 = pa.read_csv('/home/wooseok/WISE_data/WISE_scripts/obsb_outcut_md_taurus_NEOWISE_avg2_median_maxmin_4000.csv')
    v2 = pa.read_csv('/home/wooseok/WISE_data/3rdnewwise/3rd_obsb_outcut_md_taurus_NEOWISE_avg2_median_maxmin_4000.csv')

    # periodic candidates:
    # index = [669, 4688,6313,3614]
    # index = [6036,3325,4491,3614]
    for i in range(2):
        if i == 0:
            wavg = pa.read_csv('/home/wooseok/WISE_data/2ndnewwise/outlier_cut_data/'+str(index)+'_cavg.csv',
                             names=['mjd','mag','emag','flt','class'],
                             skiprows=1)
        elif i == 1:
            wavg = pa.read_csv('/home/wooseok/WISE_data/3rdnewwise/outlier_cut_data/'+str(index)+'_cavg.csv',
                               names=['mjd','mag','emag','flt','class'],
                               skiprows=1)

        w2av = wavg[(wavg['mjd'] > 56000) &
                                (wavg['flt'] == 'W2') &
                                (np.isnan(wavg['mag']) == False) &
                                (np.isnan(wavg['emag']) == False)]

        # flux conversion
        w2f = 171.85 * 10 ** (-w2av.mag / 2.5)  # flux 10%-90%
        ew2f = w2av.emag * w2f / 1.0857


    #     ax[i].errorbar(w2av.mjd, w2av.mag, yerr=w2av.emag, fmt='k.')
        ax[i].errorbar(w2av.mjd, w2f, ew2f, fmt='k.')
    #     ax[i].invert_yaxis()

        ytick = np.linspace(min(w2f)-0.1*min(w2f),max(w2f)+0.1*max(w2f),5)
        ylab = np.round(-2.5*np.log10(ytick/171.85),2)

        ax2 = ax[i].twinx()
        ax2.errorbar(w2av.mjd, w2f, ew2f, fmt='k^',ms=10, alpha=0)
        ax2.set_yticks(ytick)
        ax2.set_yticklabels(ylab)
        if i == 1:
            ax2.set_ylabel('W2 magnitude', size=15)

        #slope fit
        if i < 2:   
            def func(x,a,b):
                    return a*x + b

            resw2, cov = curve_fit(func,w2av.mjd,w2f, sigma=ew2f
                                  ,absolute_sigma = True)

            yfit= np.polyval(resw2,w2av.mjd)
            ax[i].plot(w2av.mjd,yfit,label='fit')

        #curve fit
        if i > 1:
            lsav = LombScargle(w2av.mjd, w2f, ew2f)
            frequency, power = lsav.autopower(  # nyquist_factor=5,
                                                maximum_frequency=1/200, #40 days  # 0.004,#minimum period > 250days
                                                minimum_frequency=1/4000)
            best_frequency = frequency[np.argmax(power)]
            arw2m = np.squeeze(np.array([w2av.mjd]))
            smjd = np.linspace(arw2m[0], arw2m[-1], 1000)
            flux_jmod = lsav.model(smjd, best_frequency)
            ax[i].plot(smjd, flux_jmod, color='r', lw=2,
                         alpha=0.8
                      )
    ax[0].set_xlabel('MJD',size=15)
    ax[1].set_xlabel('MJD',size=15)
    ax[0].set_ylabel('W2 flux',size=15)

#     ax[0].set_xticklabels([])
    # ax[1].set_xticklabels([])
    # ax[2].set_xticklabels([])
    # fig.text(0, 0.52, 'W2 magnitude',fontsize=15, va='center',ha='right',rotation='vertical')
    #J05385001-0720184
    # J05412327-0217357
    #  J05420932-0209501
    # J18321599-0234434
    
    ax[0].text(0.04,0.85,'FAP: '+str(v1[v1['Index'] == index].linear_fap_w2.values[0]),fontsize=12,transform=ax[0].transAxes)
    ax[1].text(0.04,0.85,'FAP: '+str(v2[v2['Index'] == index].linear_fap_w2.values[0]),fontsize=12,transform=ax[1].transAxes)
#     ax[0].text(0.04,0.85,'FU Ori',fontsize=9,transform=ax[0].transAxes)
#     ax[1].text(0.1,0.85,'EX Lup',fontsize=9,transform=ax[1].transAxes)

    plt.tight_layout()
    # fig.savefig('/home/wooseok/WISE_data/ws_paper/WISE_figures/stoch_examples.pdf')
    # plt.savefig('/home/wooseok/WISE_data/ws_paper/WISE_figures/secular_examples_avg_fit.pdf')
#     print()
    ax[0].set_title(str(index),size=15)
    # linear fap, baluev fap 를 나타내기.
    

## LSP fit ##
def chvar_cur_plot(index):
    fig, ax = plt.subplots(1,2,figsize=(10,3))

    v1 = pa.read_csv('/home/wooseok/WISE_data/WISE_scripts/obsb_outcut_md_taurus_NEOWISE_avg2_median_maxmin_4000_dist_w1w2_share.csv')
    v2 = pa.read_csv('/home/wooseok/WISE_data/3rdnewwise/3rd_obsb_outcut_md_taurus_NEOWISE_avg2_median_maxmin_4000.csv')

    # plt.subplots_adjust

    # increase / decrease / burst / dimming 

    # stochastic / linear / curved / periodic

    # index = [597, 5862, 4939, 6307]
    # index = [14,24,40] #669
    # index = [597, 2842, 4939,4688]

    # index = [85,2842,4939,6307]

    # periodic candidates:
    # index = [669, 4688,6313,3614]
    # index = [6036,3325,4491,3614]
    for i in range(2):
        if i == 0:
            wavg = pa.read_csv('/home/wooseok/WISE_data/2ndnewwise/outlier_cut_data/'+str(index)+'_cavg.csv',
                             names=['mjd','mag','emag','flt','class'],
                             skiprows=1)
            
        elif i == 1:
            wavg = pa.read_csv('/home/wooseok/WISE_data/3rdnewwise/outlier_cut_data/'+str(index)+'_cavg.csv',
                               names=['mjd','mag','emag','flt','class'],
                               skiprows=1)
            
        w2av = wavg[(wavg['mjd'] > 56000) &
                                (wavg['flt'] == 'W2') &
                                (np.isnan(wavg['mag']) == False) &
                                (np.isnan(wavg['emag']) == False)]

        # flux conversion
        w2f = 171.85 * 10 ** (-w2av.mag / 2.5)  # flux 10%-90%
        ew2f = w2av.emag * w2f / 1.0857


    #     ax[i].errorbar(w2av.mjd, w2av.mag, yerr=w2av.emag, fmt='k.')
        ax[i].errorbar(w2av.mjd, w2f, ew2f, fmt='k.')
    #     ax[i].invert_yaxis()

        ytick = np.linspace(min(w2f)-0.1*min(w2f),max(w2f)+0.1*max(w2f),5)
        ylab = np.round(-2.5*np.log10(ytick/171.85),2)

        ax2 = ax[i].twinx()
        ax2.errorbar(w2av.mjd, w2f, ew2f, fmt='k^',ms=10, alpha=0)
        ax2.set_yticks(ytick)
        ax2.set_yticklabels(ylab)
        if i == 1:
            ax2.set_ylabel('W2 magnitude', size=15)

        #slope fit
        if i < 0:   
            def func(x,a,b):
                    return a*x + b

            resw2, cov = curve_fit(func,w2av.mjd,w2f, sigma=ew2f
                                  ,absolute_sigma = True)

            yfit= np.polyval(resw2,w2av.mjd)
            ax[i].plot(w2av.mjd,yfit,label='fit')

        #curve fit
        else:
            lsav = LombScargle(w2av.mjd, w2f, ew2f)
            frequency, power = lsav.autopower(  # nyquist_factor=5,
                                                maximum_frequency=1/200, #40 days  # 0.004,#minimum period > 250days
                                                minimum_frequency=1/4000)
            best_frequency = frequency[np.argmax(power)]
            arw2m = np.squeeze(np.array([w2av.mjd]))
            smjd = np.linspace(arw2m[0], arw2m[-1], 1000)
            flux_jmod = lsav.model(smjd, best_frequency)
            ax[i].plot(smjd, flux_jmod, color='r', lw=2,
                         alpha=0.8
                      )

    ax[0].set_xlabel('MJD',size=15)
    ax[0].set_ylabel('W2 flux',size=15)

#     ax[0].set_xticklabels([])
#     ax[1].set_xticklabels([])
    # ax[2].set_xticklabels([])
    # fig.text(0, 0.52, 'W2 magnitude',fontsize=15, va='center',ha='right',rotation='vertical')
    #J05385001-0720184
    # J05412327-0217357
    #  J05420932-0209501
    # J18321599-0234434
#     ax[0].text(0.04,0.85,'V733 Cep $\it{Curved}$',fontsize=9,transform=ax[0].transAxes)
#     ax[1].text(0.04,0.1,'HH354 IRS $\it{Curved}$',fontsize=9,transform=ax[1].transAxes)
#     ax[2].text(0.70,0.85,'HBC 340 $\it{Curved}$',fontsize=9,transform=ax[2].transAxes)
    # ax[3].text(0.025,0.85,'EX Lup',fontsize=9,transform=ax[3].transAxes)
    ax[0].text(0.04,0.8,'FAP: '+str((v1[v1['s_index'] == index].baluev_fap\
                                     /v1[v1['s_index'] == index].period).values[0]*200)+'\nPeriod: '+\
               str(v1[v1['s_index'] == index].period.values[0].round()),fontsize=12,transform=ax[0].transAxes)
    ax[1].text(0.04,0.8,'FAP: '+str(v2[v2['Index'] == index].mod_baluev_fap.values[0])+'\nPeriod: '+\
               str(v2[v2['Index'] == index].period.values[0].round()),fontsize=12,transform=ax[1].transAxes)
    plt.tight_layout()
    # fig.savefig('/home/wooseok/WISE_data/FUors/lightcurves/FUor_curved.pdf')
    # plt.savefig('/home/wooseok/WISE_data/ws_paper/WISE_figures/secular_examples_avg_fit.pdf')
    ax[0].set_title(str(index),size=15)
    # linear fap, baluev fap 를 나타내기.
