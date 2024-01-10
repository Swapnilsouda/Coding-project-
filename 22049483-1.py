# importing libraries


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# reading file
file = pd.read_csv("file3-2.csv", names=['salary'])

# derive the distribution of values, bins 32
# ohist contains numbers of entries in each bin, oedge contains bin boundaries

ohist, oedge = np.histogram(file, bins=32, range=[0, 115330])


# calculate bin centre locations and bin widths
xdst = 0.5 * (oedge[1:] + oedge[:-1])
wdst = oedge[1:] - oedge[:-1]


""" normalising  the give  distribution
   ydist contains a discrete PDF
   and creating the  cumulative distribution
"""


ydst = ohist / np.sum(ohist)

cdst = np.cumsum(ydst)


# plotting Probability Density function
plt.figure(figsize=(8, 6))
plt.bar(xdst, ydst, width=0.9 * wdst)

# Mean value
xmean = np.sum(xdst * ydst)

# and plot it
plt.plot([xmean, xmean], [0.0, max(ydst)], c='red')
text = ''' Mean value: {}'''.format(xmean.astype(int))
plt.text(x=20000, y=max(ydst), s=text, fontsize=12, c='red')


# Find the index where the cumulative distribution is closest to 0.67


indx = np.argmin(np.abs(cdst - 0.33))
x_33 = oedge[indx]

# Plotting the histogram up to the index
plt.bar(xdst[:indx], ydst[:indx], width=0.9 * wdst[:indx], color='green')

# plotting x value and

plt.plot([x_33, x_33], [0.0, max(ydst)], c='black')
text = '''X value '''
plt.text(x=40000, y=max(ydst) - 0.01, s=text, fontsize=12, c='black')

# labeling and title
plt.xlabel("Annual Salary (Euros)")
plt.ylabel("Probability Density")
plt.title("Salary Distribution")

#saving and showing
# plt.savefig("22049483.png",dpi =300)
plt.show()
