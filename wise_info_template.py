import numpy as np
import pandas as pa
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from astropy.timeseries import LombScargle
#from analysis_set import WS_wise_analysis
from astropy import units as u
from astropy.coordinates import SkyCoord
# import seaborn as sea
plt.style.use('seaborn-whitegrid')
# import math
# import datetime as dt
# from scipy.optimize import curve_fit


def allplot(index):
    # locate subplots by gridspec
    gl = 50  # grid_length
    gw = 50  # grid_width

    fig = plt.figure(figsize=(20, 15))
    grid = GridSpec(gl, gw,
                    left=0.1, bottom=0.1, right=0.94, top=0.94, wspace=1.0, hspace=3)

    ax1 = fig.add_subplot(grid[2:11, 1:22])  # [y-행,x-열]
    ax2 = fig.add_subplot(grid[12:21, 1:22])
    ax3 = fig.add_subplot(grid[22:31, 1:22])
    ax4 = fig.add_subplot(grid[33:, 1:22])
    ax6 = fig.add_subplot(grid[2:11, 26:35])
    ax7 = fig.add_subplot(grid[2:11, 37:])
    ax8 = fig.add_subplot(grid[13:23, 26:40])
    ax81 = fig.add_subplot(grid[13:23, 41:49])
    axc1 = fig.add_subplot(grid[13:23, 49:50])
    ax9 = fig.add_subplot(grid[26:37, 26:37])
    ax10 = fig.add_subplot(grid[26:37, 39:])
    ax11 = fig.add_subplot(grid[39:48, 26:37])
    ax12 = fig.add_subplot(grid[39:48, 39:])
    axc2 = fig.add_subplot(grid[49:50, 34:43])

    # call NEOWISE outlier removed data by index
    wall = pa.read_csv('outlier_cut_data/'
                       + str(index) + '_alld.csv',
                       names=['mjd', 'mag', 'emag', 'flt', 'class', 'ra', 'dec'],
                       skiprows=1)

    xw1 = wall[(wall['flt'] == 'W1') &
               (np.isnan(wall['mag']) == False) &
               (np.isnan(wall['emag']) == False)
               ]

    xw2 = wall[(wall['flt'] == 'W2') &
               (np.isnan(wall['mag']) == False) &
               (np.isnan(wall['emag']) == False)
               ]

    # W1 - W2
    xm = pa.merge(xw1, xw2, on='mjd', suffixes=('_w1', '_w2'))
    mcolp = xm['mag_w1'] - xm['mag_w2']
    mcolerr = np.sqrt(xm['emag_w1'] ** 2 + xm['emag_w2'] ** 2)

    # call the class of the indexed yso
    # taurus 추가 필요
    if index < 10000:
        a = pa.read_csv('wise_csv/ysos_c.csv')#,
                        # sep="\s+", header=None,
                        # names=["index", "ra", "dec", "class", "cat"])
        ycl = a[a["index"] == index]['class'].array
        yra = a[a['index'] == index]['ra'].array[0]
        ydec = a[a['index'] == index]['dec'].array[0]

    else:
        a = pa.read_csv('wise_csv/ysos_info.dat',
                        header=None, skiprows=1, sep="\s+",
                        names=['index', 'ra', 'dec', 'Disk'])
        ycl = a[a['index'] == index]['Disk'].array[0]
        yra = a[a['index'] == index]['ra'].array[0]
        ydec = a[a['index'] == index]['dec'].array[0]

    # coord conversion
    c = SkyCoord(ra=yra * u.degree, dec=ydec * u.degree)
    hms = c.to_string('hmsdms')

    # cloud information
    cl = pa.read_csv('ref_catalog/dunham_catalogue.txt', skiprows=24, header=None,
                     sep='\s+', usecols=[0, 1], names=['Index', 'Cloud'])

    if index <= 3504:
        cloud = np.array(['Orion'])
    elif index < 10000:
        cloud = cl[cl['Index'] == index - 3504]['Cloud'].array
    else :
        cloud = np.array(['Taurus'])
    if ycl[0] == "P":
        ycl = 'Protostar'
    if ycl[0] == "D":
        ycl = 'Disk'
    if ycl[0] == "E":
        ycl = 'Evolved'
    if ycl[0] == "F":
        ycl = 'Flat'
    if ycl[0] == "FP":
        ycl = 'Faint Candidate Protostar'
    if ycl[0] == "RP":
        ycl = 'Red Candidate Protostar'

    ### W2 / W1 / W1-W2 lightcurve ###
    for i in [ax1, ax2, ax3]:
        i.set_xlim(min(xw2.mjd) - 100, max(xw2.mjd) + 100)

    fig.suptitle('WISE source information : ' + str(index), size=15)
    fig.text(0.13, 0.92, str(index) + "\n" + str(ycl) + "\n" + hms
             + '\n' + 'Cloud : ' + cloud[0], size=13)
    ax1.errorbar(xw2.mjd, xw2.mag, xw2.emag, fmt='ro', label='W2')
    ax1.set_ylabel('W2 magnitude', size=13)
    ax1.invert_yaxis()
    ax2.errorbar(xw1.mjd, xw1.mag, xw1.emag, fmt='bo', label='W1')
    ax2.invert_yaxis()
    ax2.set_ylabel('W1 magnitude', size=13)
    ax3.errorbar(xm.mjd, mcolp, mcolerr, fmt='ko', label='W1 - W2')
    ax3.set_ylabel('W1 - W2', size=13)
    ax3.set_xlabel('MJD', size=13)
    ####################################

    ####### DeltaW2 vs sd/mu(w2) #######
    # the plot only shows P,D,E. We don't show all data here since it is for background only.
    nstat = pa.read_csv('wise_csv/NEOWISE_YSO_variable_stat.csv')
    nstat = nstat[nstat['avg_eW2'] < 0.2]

    pr = nstat[(nstat['class'] == "P")]
    prfl = nstat[((nstat['class'] == "P") | (nstat['class'] == "F"))]
    di = nstat[(nstat['class'] == "D") ]  # & periodic_c]
    ev = nstat[(nstat['class'] == "E") ]  # & periodic_c]

    yso = [prfl,
           di, ev]
    y_label = ['Protostar',
               'Disk', 'Evolved']
    y_color = ['#ee00b8',
               '#f4af1b', '#057dd1']
    y_size = [10, 4, 10]
    y_marker = 'o'

    for i in range(len(yso)):
        ax4.scatter(yso[i].sd_sdfid_w2_flux, yso[i].Delta_w2,
                    s=y_size[i], c=y_color[i], label=y_label[i], marker=y_marker)
    circ_yso = nstat[nstat['Index'] == index]
    ax4.scatter(circ_yso.sd_sdfid_w2_flux, circ_yso.Delta_w2,
                s=500, facecolors='none', edgecolors='r',
                linewidth=4)

    ax4.legend()
    ax4.set_xscale('log')
    ax4.set_xlabel('Flux standard deviation / Mean flux uncertainty', size=12)
    ax4.set_ylabel('DeltaW2 (Max - Min)', size=12)
    ax4.set_xticks([0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 40])
    ax4.set_xticklabels([0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 40])
    ####################################

    ####### Periodogram #######
    wavg = pa.read_csv('outlier_cut_data/' + str(index) + '_cavg.csv',
                       header=None, skiprows=1,
                       names=['mjd', 'mag', 'emag', 'flt', 'class'])
    w2av = wavg[wavg['flt'] == 'W2']
    w2f = 171.85 * 10 ** (-w2av.mag / 2.5)
    ew2f = w2av.emag * w2f / 1.0857

    lsav = LombScargle(w2av.mjd, w2f, ew2f)  # flux lombscargle

    frequency, power = lsav.autopower(  # nyquist_factor=5,
        maximum_frequency=1 / 200,  # 40 days  # 0.004,#minimum period > 250days
        minimum_frequency=1 / 4800)  # 0.0001) #0.2 #maximum period 10000days

    period_days = 1. / frequency
    # period_hours = period_days * 24

    best_period = period_days[np.argmax(power)]
    # phase = (w2av.mjd / best_period) % 1

    print("Best period: {0:.2f} days / power : {1:.3f}".format(best_period, np.max(power)))

    ax6.plot(period_days, power, '-k', rasterized=True)
    ax6.set_xlabel('Period (days)', size=13)
    ax6.set_ylabel('Lomb-Scargle Power', size=13)
    ax6.set_title('Lomb-Scargle Periodogram', size=13)
    ax6.set_xscale('log')
    ax6.set_xticks([250, 400, 800, 1600, 3200, 4800])
    ax6.set_xticklabels([250, 400, 800, 1600, 3200, 4800])


    fap = lsav.false_alarm_probability(power)
    boot_fap = lsav.false_alarm_probability(power, method='bootstrap',
                                            method_kwds={'n_bootstraps': 1000})

    # fap level plot
    # fapl2 = lsav.false_alarm_level(0.001, method='baluev')
    # fapb2 = lsav.false_alarm_level(0.001, method='bootstrap',
    #                                method_kwds={'n_bootstraps': 1000})
    print("baluev FAP is {:5.2e}".format(fap[np.argmax(power)]))
    print("bootstrap FAP is {:.5f}".format(boot_fap[np.argmax(power)]))

    phase_model = np.linspace(-0.5, 1.5, 100)
    best_frequency = frequency[np.argmax(power)]
    flux_model = lsav.model(phase_model / best_frequency, best_frequency)

    fig.text(0.7, 0.92, '[ Periodogram analysis ]' + '\n'
             + 'best period : {:.2f} days '.format(best_period)
             + '({:.2f} years)'.format(best_period / 365.25) + '\n'
             + 'power : {:.2f}'.format(np.max(power)) + '\n'
             + "baluev FAP : {:5.2e}".format(fap[np.argmax(power)]), size=13)
    ##############################





    ###### Phase plot #######
    new_phase = (w2av.mjd - w2av[w2av['mag'] == max(w2av['mag'])].mjd.values[0]) / best_period % 1

    ax7.errorbar(new_phase, w2f, ew2f,
                 fmt='.k', ecolor='gray', capsize=0)
    ax7.set_xlabel('phase', size=13)
    ax7.set_ylabel('W2 flux', size=13)
    ax7.set_title('Phased Data', size=13)
    ax7.set_xlim(0, 1)

    arw2m = np.squeeze(np.array([w2av.mjd]))
    smjd = np.linspace(max(arw2m), min(arw2m), 1000)
    flux_jmod = lsav.model(smjd, best_frequency)
    # flux_jdot = lsav.model(w2av.mjd, best_frequency)

    #     print('amplitude : {:.4f}Jy.'.format((max(flux_jmod) - min(flux_jmod)) * 0.5))
    ##############################





    ###### best-fit sinusoid with distance colored lightcurve ######
    wavgd = pa.read_csv('raw_data/YSOwise_lc_' + str(index) + '_dcut.dat',
                        sep="\s+", header=None, skiprows=1,
                        names=['mjd', 'mag', 'emag', 'flt', 'flag', 'dist', 'ra', 'dec'])


    xw2d = wavgd[(wavgd['flt'] == 'W2') &
                 (wavgd['flag'] != 0) &
                 (np.isnan(wavgd['mag']) == False) &
                 (np.isnan(wavgd['emag']) == False) &
                 (wavgd['mjd'] > 56000)]

    dayd = xw2d.mjd - min(np.array(xw2d.mjd))

    ra2 = xw2d.ra
    dec2 = xw2d.dec
    # dist2 = xw2d.dist  # distance from catalogue position

    radist = abs(ra2 - np.mean(ra2))  # distance from mean position
    decdist = abs(dec2 - np.mean(dec2))

    stoep_flux = 171.85 * 10 ** (-xw2d.mag / 2.5)
    ax8.plot(smjd - min(arw2m), flux_jmod, color='lightgray', lw=2,
                 alpha=0.8
                 )

    ax8.scatter(dayd, stoep_flux, c=np.sqrt(radist ** 2 + decdist ** 2) * 3600, cmap='rainbow')  # /np.std(dist2)
    #     ax4.invert_yaxis()
    ax8.set_xlabel('days after mjd {:5.0f}'.format(min(np.array(xw2d.mjd))), size=13)
    ax8.set_ylabel('W2 Flux [Jy]', size=13)
    ax8.set_title('Lightcurve with best-fit sinusoid')
    ##############################################





    ####### RA dec distance plot ########
    ax81.scatter((ra2 - np.mean(ra2)) * 3600, (dec2 - np.mean(dec2)) * 3600,
                 c=np.sqrt(radist ** 2 + decdist ** 2) * 3600,
                 cmap='rainbow')

    circ1 = plt.Circle((0, 0),
                       6.4 / 2,
                       # /3600, # WISE W2 angular resolution : 6.4 arcsec. Deg = arcsec/3600 ≈ arcsec*0.0002778
                       color='r',
                       fill=False)
    cbar2 = plt.colorbar(ax81.scatter((ra2 - np.mean(ra2)) * 3600, (dec2 - np.mean(dec2)) * 3600,
                                      c=np.sqrt(radist ** 2 + decdist ** 2) * 3600,
                                      cmap='rainbow'), cax=axc1)
    cbar2.ax.set_ylabel('distance ["]', size=12)
    ax81.axis('equal')
    ax81.add_artist(circ1)
    ax81.set_xlim(-4, 4)
    ax81.set_ylim(-4, 4)
    ax81.invert_xaxis()
    ax81.ticklabel_format(useOffset=False)
    ax81.set_xlabel('RA offset["]', size=13)
    ax81.set_ylabel('Dec offset["]', size=13)

    ######################################






    ########### Color and magnitude with phase plot ############
    m2p = xm['mag_w2']
    # m1p = xm['mag_w1']
    m2err = xm['emag_w2']
    # m1err = xm['emag_w1']



    w2f = 171.85 * 10 ** (-m2p / 2.5)  # flux 10%-90%
    ew2f = m2err * w2f / 1.0857

    # phase matched with periodogram phase (averaged by observing block)
    phase = (xm.mjd - w2av[w2av['mag'] == max(w2av['mag'])].mjd.values[0]) / best_period % 1
    print('best period : {:.2f} days'.format(best_period))
    print('power : ', np.max(power))

    fig.text(0.53, 0.51,
             '[ Color and magnitude with phase ]                              *Arrow indicates $A_{K} = 0.5$', size=14,
             wrap=True)
    ax9.scatter(phase, mcolp, c=phase, cmap='jet')
    ax9.set_xlabel('Phase', size=13)
    ax9.set_ylabel('W1 - W2', size=13)
    #             ax[1]

    axc = ax10.scatter(mcolp, m2p, c=phase, cmap='jet')

    ax10.annotate("", xytext=(min(mcolp), np.mean(m2p)), xy=(min(mcolp) + 0.0325, np.mean(m2p) + 0.215),
                  arrowprops=dict(arrowstyle="->, head_length = 1, head_width = .5", lw=2))
    ax10.invert_yaxis()


    ax10.set_xlabel('W1 - W2', size=13)
    ax10.set_ylabel('W2 magnitude', size=13)


    ax11.scatter(phase, m2p, c=phase, cmap='jet')
    ax11.set_ylabel('W2 magnitude', size=13)
    ax11.set_xlabel('Phase', size=13)
    ax11.invert_yaxis()

    ax12.scatter(xm['mjd'], w2f, c=phase, cmap='jet')
    ax12.set_xlabel('MJD', size=13)
    ax12.set_ylabel('W2 flux [Jy]', size=13)
    ax12.plot(smjd, flux_jmod, color='lightgray', lw=2,
              alpha=0.8
              )

    # scatter plot scale -- it is weird.. shouldn't it be automatic?
    if min(w2f) < min(flux_jmod):
        sc_ymin = min(w2f)
    else:
        sc_ymin = min(flux_jmod)

    if max(w2f) < max(flux_jmod):
        sc_ymax = max(flux_jmod)
    else:
        sc_ymax = max(w2f)

    ax12.set_ylim(ax8.get_ylim())
    #########################################################




    fig.colorbar(axc, cax=axc2, orientation='horizontal',
                 shrink=0.3, label='Phase')

    plt.tight_layout()
    plt.show()

# save figure
#     fig.savefig('WISE_info_{}.pdf'.format(index))

