import numpy as np
import pandas as pa
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms



ab = pa.read_csv('wise_csv/NEOWISE_YSO_variable_stat.csv')
abc = ab[(ab.N_w2 > 5) &
         (ab.dist_sd < 0.3) &
         (ab.avg_eW2 < 0.2)]


pr = abc[(abc['class'] == "P") |
        (abc['class'] == "F") |
        (abc['class'] == "FP") |
        (abc['class'] == "0") |
        (abc['class'] == "I") |
        (abc['class'] == "I?")]


di = abc[(abc['class'] == "D") |
        (abc['class'] == "II") |
        (abc['class'] == "full") |
        (abc['class'] == "full?") |
        (abc['class'] == "debris/ev trans") |
        (abc['class'] == "transitional") |
        (abc['class'] == "evolved") |
        (abc['class'] == "ev or trans")
        ]

ev = abc[(abc['class'] == "E") |
        (abc['class'] == "III") 
        ]

yso = [pr,di,ev]
y_label = ['P', 'D', 'PMS+E']
y_color=['#ee00b8', '#f4af1b', '#057dd1']
y_size=[10,10,10]
y_marker = ['o','o','o']


# definitions for the axes
left, width = 0.1, 0.65
bottom, height = 0.1, 0.65
spacing = 0.005


rect_scatter = [left, bottom, width, height]
rect_histx = [left, bottom + height + spacing, width, 0.2]
rect_histy = [left + width + spacing, bottom, 0.2, height]

# start with a rectangular Figure
fig1 = plt.figure(figsize=(9,9))

rot = transforms.Affine2D().rotate_deg(90)

axsc = plt.axes(rect_scatter)
axsc.tick_params(direction='in', top=True, right=True, labelsize=15)
ax_histx = plt.axes(rect_histx)
ax_histx.tick_params(direction='in', labelbottom=False, labelsize=15)
ax_histy = plt.axes(rect_histy)
ax_histy.tick_params(direction='in', labelleft=False, labelsize=15)

for i in range(len(yso)):
    axsc.scatter(yso[i].sd_sdfid_w2_flux, yso[i].Delta_w2,
                s = y_size[i], c=y_color[i],label=y_label[i],marker=y_marker[i])

### histogram_x ###
yso_hist = [pr,di,ev]
y_label = ['P',
           'D', 'PMS+E']
y_color=['#ee00b8', 
         '#f4af1b', '#057dd1']
hist_ret = []

xlim = (0.055,65)

# plt.figure()
for i in range(len(yso_hist)):
    counts, bins = np.histogram(yso_hist[i].sd_sdfid_w2_flux, bins=np.logspace(np.log10(xlim[0]),np.log10(xlim[1]), 30))
    hist_ret.append(ax_histx.hist(bins[:-1], bins, weights=counts/max(counts),histtype='step',
            color = y_color[i], label=y_label[i], linewidth=2))
    

    
### histogram_y ###

hist_ret2=[]
for i in range(len(yso_hist)):
    counts, bins = np.histogram(yso_hist[i].Delta_w2, bins=np.logspace(np.log10(9e-3),np.log10(5), 30))
    hist_ret2.append(ax_histy.hist(bins[:-1],bins, weights=counts/max(counts),histtype='step',
            color = y_color[i], label=y_label[i], linewidth=2, orientation='horizontal'))    
    



axsc.set_xlim(xlim)
axsc.set_xscale('log')
axsc.set_xlabel('SD / $\sigma$',size=20)
axsc.set_ylabel('$\Delta$W2 (Max - Min)',size=20)
axsc.set_xticks([0.1,0.2,0.5,1,2,5,10,20,40])
axsc.set_xticklabels([0.1,0.2,0.5,1,2,5,10,20,40])


ylm = axsc.get_ylim()



ax_histy.set_xticks([0,0.25,0.5,0.75,1])
ax_histy.set_xticklabels([0,0.25,0.5,0.75,1])


ax_histx.set_yticks([0,0.25,0.5,0.75,1])
ax_histx.set_yticklabels([0,0.25,0.5,0.75,1])
ax_histx.set_xlim(axsc.get_xlim())
ax_histx.set_xscale('log')
ax_histx.set_xticks([])
ax_histy.legend(fontsize=13, loc='lower left', bbox_to_anchor=(0.0, 1.0))
axsc.legend(fontsize=13)


fig1.tight_layout()
plt.show()

# fig1.savefig()