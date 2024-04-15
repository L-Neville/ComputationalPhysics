import unittest
import numpy as np
import numpy.testing as npt
import matplotlib.pyplot as plt
from random import randint
from time import time
import json
import cProfile
from numba import njit
from math import e

class IsingModelHoneycombTest(unittest.TestCase):
    def test_init_spins(self):
        temp = IsingModelHoneycomb(1, 1, 3, 3, spin_b=0)
        npt.assert_array_almost_equal(temp.spins, np.array([[-1, -1, -1], [-1, -1, -1], [-1, -1, -1]]))
        temp = IsingModelHoneycomb(1, 1, 3, 3, spin_b=511)
        npt.assert_array_almost_equal(temp.spins, np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]))

    def test_get_magnetization(self):
        temp = IsingModelHoneycomb(1, 1, 3, 3, spin_b=455)
        self.assertAlmostEqual(temp.get_magnetization(), 3)

    def test_get_hamiltonian(self):
        temp = IsingModelHoneycomb(1, 1, 3, 3, spin_b=0)
        self.assertAlmostEqual(temp.get_hamiltonian(), -12)
        temp = IsingModelHoneycomb(1, 1, 4, 4, spin_b=0)
        self.assertAlmostEqual(temp.get_hamiltonian(), -24)

    def test_get_average_energy(self):
        temp = IsingModelHoneycomb(1, 1, 4, 4)
        def ee(x):
            return np.exp(x)
        benchmark = (12*(-1+3*ee(2)-5*ee(4)+3*ee(6)-3*ee(8)+5*ee(10)-3*ee(12)+ee(14)))/((1+ee(2))*(1+ee(4))*(1-4*ee(2)+8*ee(4)-4*ee(6)+ee(8)))
        _, _, energy, _ = temp.get_average()
        self.assertAlmostEqual(energy*temp.row*temp.col, benchmark, delta=.6)

    def test_get_binder_ratio(self): 
        b1 = [IsingModelHoneycomb(1, t, 4, 4).get_average()[3] for t in np.arange(1, 1000, 50)]
        b2 = [IsingModelHoneycomb(1, t, 4, 4).get_average()[3] for t in np.arange(.1, 1, .1)]
        b = b1 + b2
        l, u = min(b), max(b)
        self.assertGreater(l, 1)
        self.assertLess(u, 3)
        
class IsingModelHoneycomb():
    def __init__(self, J, T, row, col, spin_b=None):
        self.J, self.T = J, T
        self.row, self.col = row, col
        self.sites = self.row*self.col
        """
        set the spins automatically or manually.
        to improve efficiency, spins configuration is represented as an integer (and its binary form), instead of containers
        """
        if spin_b is None: # if spin_config is not passed, randomly set each spin
            self.spin_b = randint(0, 2**(self.row*self.col-1))
        else: 
            assert type(spin_b)==int and spin_b>=0 and spin_b<=2**self.sites-1, "spin_config too large or too small or not an integer"
            self.spin_b = spin_b
        spins = np.array([(self.spin_b >> j) & 1 for j in range(self.row*self.col)]) # generate spin configuration from binary number
        spins = 2 * spins - 1 # convert 1&0 to +1&-1
        spins = spins.reshape((self.row, self.col)) # reshape to fit the lttice shape
        self.spins = spins

    def get_magnetization(self):
        ups, n = 0, self.spin_b
        while n: # count the number of '1' in binary m
            n &= n - 1
            ups += 1 
        self.m = 2*ups-self.sites
        return self.m
    
    def get_hamiltonian(self):
        # use np.roll() and element-wise multiplication to calculate the adjacent product, then sum them up
        h = 0
        down = np.concatenate((self.spins[-1:], self.spins[:-1]))
        up = np.concatenate((self.spins[:-1], self.spins[-1:]))
        down_left = np.concatenate((down[:, :-1], down[:, -1:]), axis=1)
        up_right = np.concatenate((up[:, -1:], up[:, :-1]), axis=1)
        down_multiply = np.multiply(self.spins, down)
        up_multiply = np.multiply(self.spins, up)
        down_left_multiply = np.multiply(self.spins, down_left)
        up_right_multiply = np.multiply(self.spins, up_right)
        h += -.5 * self.J * np.sum(down_multiply + up_multiply)
        h += -.5 * self.J * np.sum(down_left_multiply[::4] + down_left_multiply[2::4])
        h += -.5 * self.J * np.sum(up_right_multiply[1::4] + up_right_multiply[3::4])
        self.h = h
        return h

    
    def get_partition(self):
        Z = 0
        for s in range(2**(self.sites)):
            temp = IsingModelHoneycomb(self.J, self.T, self.row, self.col, spin_b=s)
            Z += np.exp(-temp.get_hamiltonian()/temp.T)
        self.Z = Z*2
        return self.Z
    
    def get_average(self):
        time_start = time()
        Z, HZ, H2Z, M2Z, M4Z = 0, 0, 0, 0, 0
        for s in range(2**(self.sites-1)): # both M squared and H are +/- symmetric
            temp = IsingModelHoneycomb(self.J, self.T, self.row, self.col, spin_b=s)
            h, m = temp.get_hamiltonian(), temp.get_magnetization()
            z = np.exp(-h/temp.T)
            Z += z
            HZ += z*h
            H2Z += z*h**2
            M2Z += z*m**2
            M4Z += z*m**4
        time_end = time()
        elapsed_time = time_end - time_start
        # print(f'  energy per spin is {HZ/Z/self.row/self.col}')
        # print(f'  specific heat per spin is {(H2Z/Z-(HZ/Z)**2)/self.T**2}')
        # print(f'  magnetization per spin squared is {M4Z*Z/M2Z**2}')
        # print(f'  overall takes {elapsed_time}s to calculate')
        return HZ/Z/self.row/self.col, (H2Z/Z-(HZ/Z)**2)/self.T**2, M2Z/Z/(self.row*self.col)**2, M4Z*Z/M2Z**2


            
    
    # def get_average_hamiltonion(self):
    #     HZ, Z = 0, 0
    #     for s in range(2**(self.sites)):
    #         temp = IsingModelHoneycomb(self.J, self.T, 4, 4, spin_config=s)
    #         h = temp.get_hamiltonion()
    #         z = np.exp(-h/temp.T)
    #         Z += z
    #         HZ += z*h
    #     return HZ / Z
    
    # def get_average_magnetization_squared(self):
    #     MZ, Z = 0, 0
    #     for s in range(2**(self.sites)):
    #         temp = IsingModelHoneycomb(self.J, self.T, 4, 4, spin_config=s)
    #         m = temp.get_magnetization()
    #         z = np.exp(-temp.get_hamiltonion()/temp.T)
    #         Z += z
    #         MZ += z*m**2
    #     return MZ / Z
    
    # def get_average_magnetization_4squared(self):
    #     MZ, Z = 0, 0
    #     for s in range(2**(self.sites)):
    #         temp = IsingModelHoneycomb(self.J, self.T, 4, 4, spin_config=s)
    #         m = temp.get_magnetization()
    #         z = np.exp(-temp.get_hamiltonion()/temp.T)
    #         Z += z
    #         MZ += z*m**4
    #     return MZ / Z    

# a = time()
# E, C, U_M = [], [], []
# for t in np.arange(.05,4,.05):
#     temp = IsingModelHoneycomb(1,t,4,4,0)
#     e, c, u_m = temp.get_average()
#     E.append(e)
#     C.append(c)
#     U_M.append(u_m)
# b = time()
# print(f'overall takes {b-a}s to calculate under all temperatures')

# data = [E, C, U_M]
# with open('honeycomb8by4.txt', 'w') as file:
#     json.dump(data, file)

with open('honeycomb5by5-2-load.txt', 'r') as file:
    data = json.load(file)
    E, C, M_N, U_M = data[0], data[1], data[2], data[3]

# fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 12))
# plt.subplots_adjust(hspace=.4)
# ax1.plot(np.arange(.05,4,.05), E, label='energy')
# ax1.set_xlabel('$T$', fontsize=22)
# ax1.set_ylabel('$E$', fontsize=22)
# ax1.set_title('energy per spin', fontsize=22)
# ax1.legend(fontsize=22)
# ax2.plot(np.arange(.05,4,.05), C, label='heat capacity')
# ax2.set_xlabel('$T$', fontsize=22)
# ax2.set_ylabel('$C$', fontsize=22)
# ax2.set_title('specific heat per spin', fontsize=22)
# ax2.legend(fontsize=22)
# ax3.plot(np.arange(.05,4,.05), U_M, label='Binder ratio')
# ax3.set_xlabel('$T$', fontsize=22)
# ax3.set_ylabel('$U_M$', fontsize=22)
# ax3.set_title('Binder ratio', fontsize=22)
# ax3.legend(fontsize=22)
# plt.savefig('honeycomb.jpg', dpi=300)
# plt.show()

""" unpack the class and use @njit to accelerate """
# a = time()
# @njit()
# def main():
#     E, C = [], []
#     M_N, U_M = [], []
#     J = 1
#     row, col = (5, 5)
#     sites = row * col
#     for t in np.arange(.05, 4, .05):
#         Z, HZ, H2Z, M2Z, M4Z = 0, 0, 0, 0, 0
#         for s in np.arange(0, 2**(row*col-1), 1):
#             s = int(s)
#             # init
#             spins = np.array([(s >> j) & 1 for j in range(row*col)]) 
#             spins = 2 * spins - 1 
#             spins = spins.reshape((row, col)) 
#             spins = spins
#             # magnetization
#             ups, n = 0, s
#             while n: # count the number of '1' in binary m
#                 n &= n - 1
#                 ups += 1 
#             m = 2*ups-sites
#             # hamiltonian
#             h = 0
#             down = np.concatenate((spins[-1:], spins[:-1]))
#             up = np.concatenate((spins[:-1], spins[-1:]))
#             down_left = np.concatenate((down[:, :-1], down[:, -1:]), axis=1)
#             up_right = np.concatenate((up[:, -1:], up[:, :-1]), axis=1)
#             down_multiply = np.multiply(spins, down)
#             up_multiply = np.multiply(spins, up)
#             down_left_multiply = np.multiply(spins, down_left)
#             up_right_multiply = np.multiply(spins, up_right)
#             h += -.5 * J * np.sum(down_multiply + up_multiply)
#             h += -.5 * J * np.sum(down_left_multiply[::4] + down_left_multiply[2::4])
#             h += -.5 * J * np.sum(up_right_multiply[1::4] + up_right_multiply[3::4])
#             # average 
#             z = np.exp(-h/t)
#             Z += z
#             HZ += z*h
#             H2Z += z*h**2
#             M2Z += z*m**2
#             M4Z += z*m**4
#         e, c, m_n, u_m = HZ/Z/row/col, (H2Z/Z-(HZ/Z)**2)/t**2/row/col, M2Z/Z/(row*col)**2, M4Z*Z/M2Z**2
#         E.append(e)
#         C.append(c)
#         M_N.append(m_n)
#         U_M.append(u_m)
#     return [E, C, M_N, U_M]
# data = main()
# E, C, M_N, U_M = data[0], data[1], data[2], data[3]
# with open('honeycomb5by5-2.txt', 'w') as file:
#     json.dump(data, file)
# b = time()
# print(f'overall takes {b-a}s after introducing @njit')


# fig, (ax1, ax2, ax4, ax3) = plt.subplots(4, 1, figsize=(20, 15))
# plt.subplots_adjust(hspace=.5)
# ax1.plot(np.arange(.05,4,.05), E, label='energy')
# ax1.set_xlabel('$T$', fontsize=22)
# ax1.set_ylabel(r'$\frac{\langle H \rangle}{N}$', fontsize=26)
# ax1.set_title('energy per spin', fontsize=22)
# ax1.legend(fontsize=22)
# ax2.plot(np.arange(.05,4,.05), C, label='heat capacity')
# ax2.set_xlabel('$T$', fontsize=22)
# ax2.set_ylabel(r'$\frac{C}{N}$', fontsize=26)
# ax2.set_title('specific heat per spin', fontsize=22)
# ax2.legend(fontsize=22)
# ax4.plot(np.arange(.05,4,.05), M_N, label='magnetization')
# ax4.set_xlabel('$T$', fontsize=22)
# ax4.set_ylabel(r'$\frac{\langle M^2 \rangle}{N^2}$', fontsize=26)
# ax4.set_title('magnetization per spin squared', fontsize=22)
# ax4.legend(fontsize=22)
# ax3.plot(np.arange(.05,4,.05), U_M, label='Binder ratio')
# ax3.set_xlabel('$T$', fontsize=22)
# ax3.set_ylabel(r'$\frac{\langle M^4 \rangle}{\langle M^2 \rangle ^2}$', fontsize=26)
# ax3.set_title('Binder ratio', fontsize=22)
# ax3.legend(fontsize=22)
# plt.savefig('honeycomb-2.jpg', dpi=300)
# plt.show()

if __name__ == '__main__':
    unittest.main()