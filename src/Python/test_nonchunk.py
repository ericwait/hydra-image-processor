from pylab import imshow, show, get_cmap
from matplotlib import pyplot as plt
from skimage import filters

import numpy as np
import HIP

#%%
im = np.random.random((1,1,12,128,128))
#im = np.random.random((128,128,12,1,1))
#im = np.random.rand(128, 512)
#im = np.zeros((128,256), order='C')
#im[20:50,:] = 1
#print(im.shape)

#%%
#imS = HIP.IdentityFilter(im, 0)
#imS = HIP.Gaussian(im, [5,2,12])
kern = np.ones((1,1,7))
imS = HIP.Opener(im,kern)
print(imS.shape)

#%%
fig, axes = plt.subplots(ncols=2, figsize=(8, 2.5))
ax = axes.ravel()
ax[0] = plt.subplot(1, 2, 1)
ax[1] = plt.subplot(1, 2, 2)

ax[0].imshow(im[0,0,5,:,:], cmap=get_cmap("gray"), interpolation='nearest')
ax[1].imshow(imS[0,0,5,:,:], cmap=get_cmap("gray"), interpolation='nearest')

plt.show()

x = 1
