from helper import get_default
import numpy as np
from IPython import embed
from pylab import *
from model import phi, phi_prime


my_phi = get_default("phi")

Us = np.arange(-80,10,0.01)

#subplot(1,3,1)
plot(Us, phi(Us, my_phi))
#subplot(1,3,2)
plot(Us, phi_prime(Us, my_phi))
#subplot(1,3,3)
plot(Us, phi_prime(Us, my_phi)/phi(Us, my_phi))
show()
