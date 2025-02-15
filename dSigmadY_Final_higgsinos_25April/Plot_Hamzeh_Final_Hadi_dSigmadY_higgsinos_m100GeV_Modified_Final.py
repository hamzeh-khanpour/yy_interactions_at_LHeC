
# Final Version -- Febraury 3034 -- Hamzeh Khanpour

# ================================================================================

import mplhep as hep
import numpy as np
import matplotlib.pyplot as plt
import sys

hep.style.use("CMS")
#plt.style.use(hep.style.ROOT)


import matplotlib.ticker as mticker



'''plt.rcParams["axes.linewidth"] = 1.8
plt.rcParams["xtick.major.width"] = 1.8
plt.rcParams["xtick.minor.width"] = 1.8
plt.rcParams["ytick.major.width"] = 1.8
plt.rcParams["ytick.minor.width"] = 1.8

plt.rcParams["xtick.direction"] = "in"
plt.rcParams["ytick.direction"] = "in"

plt.rcParams["xtick.labelsize"] = 15
plt.rcParams["ytick.labelsize"] = 15

plt.rcParams["legend.fontsize"] = 15

plt.rcParams['legend.title_fontsize'] = 'x-large' '''



sys.path.append('./values')
# from syy_1_3_3_0804 import *
# from syy_1_3_4_0805 import *
# from syy_1_4_4_0907 import *

from dSigmadY_higsinos_100_100000_100000_tagged_elastic_m100GeV import *


fig, ax = plt.subplots(figsize = (10.0, 11.0))
plt.subplots_adjust(left=0.15, right=0.95, bottom=0.12, top=0.95)


ax.set_xlim(0.0, 5.0)
ax.set_ylim(0.0001, 0.1)



# Set y-axis to scientific format
formatter = mticker.ScalarFormatter(useMathText=True)
formatter.set_powerlimits((-2, 2))
ax.yaxis.set_major_formatter(formatter)


inel_label = ('$M_N<$ ${{{:g}}}$ GeV').format(inel_1200[0]) + (' ($Q^2_p<$ ${{{:g}}}$ GeV$^2$)').format(inel_1200[2])
title_label = ('$Q^2_e<$ ${{{:g}}}^{{{:g}}}$ GeV$^2$').format(10,np.log10(inel_1200[1]))

plt.plot(wvalues[3][:303], [value * 1000.0 for value in elas_1200[3][:303]], label="elastic - p detected", linestyle="solid", linewidth=3, color="blue")
#plt.plot(wvalues[3][:303], [value * 1000.0 for value in elas_750[3][:303]], label=r"elastic - p detected ($\sqrt{s}=0.75$ TeV)", linestyle=(0, (5, 2, 1, 2, 1, 2)), linewidth=3, color="red")
plt.plot(wvalues[3][:303], [value * 1000.0 for value in inel_1200[3][:303]], label=r"$M_N < 100$ GeV ($Q^2_p < 10^5$ GeV$^2$)", linestyle="dashed", linewidth=3, color="green")
plt.plot(wvalues[3][:303], [value * 1000.0 for value in inel_750[3][:303]], label=r"$M_N < 100$ GeV ($Q^2_p < 10^5$ GeV$^2$) ($\sqrt{s}=0.75$ TeV)", linestyle="dotted", linewidth=3, color="magenta")

#plt.grid()
plt.legend(title = '$Q^2_e<$ ${{{:g}}}^{{{:g}}}$ GeV$^2$')



# Add additional information
#info_text = "$ep \\rightarrow e (\gamma\gamma \\to ZZ) p^*$"
#plt.text(0.650, 0.650, info_text, transform=ax.transAxes, ha='center', va='center', fontsize=25, color='black')

#info_text_2 = "$M_{\widetilde{\ell}}$ = 100 GeV"
#plt.text(0.2, 0.85, info_text_2, transform=ax.transAxes, ha='center', va='center', fontsize=20, color='black')



# Setting y-axis to log scale
# plt.yscale('log')




# Add additional information
info_text = r"$ep \rightarrow e (\gamma\gamma \to \tilde{H}^+\tilde{H}^-) p^*$"
ax.text(0.650, 0.68, info_text, transform=ax.transAxes, ha='center', va='center', color='black')

info_text_2 = r"$M_{\tilde{H}}$ = 100 GeV"
ax.text(0.650, 0.60, info_text_2, transform=ax.transAxes, ha='center', va='center', color='black')





# Set label colors
ax.xaxis.label.set_color('black')
ax.yaxis.label.set_color('black')

# Add legend with specified colors
legend = plt.legend(title = title_label)

#legend.get_texts()[0].set_color("blue")  # Color for 'elastic'
#legend.get_texts()[1].set_color("red")   # Color for inel_label



font1 = {'family':'serif','color':'black','size':24}
font2 = {'family':'serif','color':'black','size':24}




plt.xlabel(r'$Y_{\tilde{H}^+\tilde{H}^-}$')
plt.ylabel(r'$d\sigma/dY_{\tilde{H}^+\tilde{H}^-} \, [fb]$')


plt.savefig("dSigmadY_higgsinos100GeV_25April_JHEP_Krzysztof.pdf")
#plt.savefig("dSigmadY_higgsinos100GeV_25April_JHEP_Krzysztof.jpg")

plt.show()



# ================================================================================


