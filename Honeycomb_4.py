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
        temp = IsingModelHoneycomb(1, 1, 4, 2, spin_b=2**8-1)
        npt.assert_array_almost_equal(temp.spins, -np.array([[-1, -1], [-1, -1], [-1, -1], [-1, -1]]))
        temp = IsingModelHoneycomb(1, 1, 4, 2, spin_b=1)
        npt.assert_array_almost_equal(temp.spins, np.array([[1, -1], [-1, -1], [-1, -1], [-1, -1]]))


    def test_get_magnetization(self):
        temp = IsingModelHoneycomb(1, 1, 3, 3, spin_b=455)
        self.assertAlmostEqual(temp.get_magnetization(), 3)

    def test_get_hamiltonian(self):
        temp = IsingModelHoneycomb(1, 1, 4, 2, spin_b=0)
        self.assertAlmostEqual(temp.get_hamiltonian(), -12)
        temp = IsingModelHoneycomb(1, 1, 4, 2, 85)
        self.assertAlmostEqual(temp.get_hamiltonian(), -4)
        temp = IsingModelHoneycomb(1, 1, 4, 4, spin_b=0)
        self.assertAlmostEqual(temp.get_hamiltonian(), -24)

    def test_get_average_energy(self):
        temp = IsingModelHoneycomb(1, 1, 4, 2)
        def ee(x):
            return np.exp(x)
        benchmark = -(12*(-1+3*ee(2)-5*ee(4)+3*ee(6)-3*ee(8)+5*ee(10)-3*ee(12)+ee(14)))/((1+ee(2))*(1+ee(4))*(1-4*ee(2)+8*ee(4)-4*ee(6)+ee(8)))
        energy, _, _, _ = temp.get_average()
        self.assertAlmostEqual(energy*temp.row*temp.col, benchmark)

    # def test_get_binder_ratio(self): 
    #     b1 = [IsingModelHoneycomb(1, t, 4, 4).get_average()[3] for t in np.arange(1, 1000, 50)]
    #     b2 = [IsingModelHoneycomb(1, t, 4, 4).get_average()[3] for t in np.arange(.1, 1, .1)]
    #     b = b1 + b2
    #     l, u = min(b), max(b)
    #     self.assertGreater(l, 1)
    #     self.assertLess(u, 3)
        
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
        spins = spins.reshape((self.row, self.col)) # reshape to fit the lattice shape
        self.spins = spins

    def get_magnetization(self):
        ups, n = 0, self.spin_b
        while n: # count the number of '1' in binary m
            n &= n - 1
            ups += 1 
        self.m = 2*ups-self.sites
        return self.m
    
    # def get_hamiltonian(self):
    #     """
    #     the method of calculating hamiltonion:
    #     the spin matrix element-multiplied by itself:
    #     1) shifted up
    #     2) shifted down
    #     3) the odd and even elements of 2 cols corss-swapped
    #     """
    #     """
    #     Warning: this is an unfinished version that only calculates 8-site lattice
    #     """
    #     h = 0
    #     down = np.concatenate((self.spins[-1:], self.spins[:-1]))
    #     up = np.concatenate((self.spins[1:], self.spins[:1]))
    #     h += -.5 * self.J * sum(sum(np.multiply(self.spins, down) + np.multiply(self.spins, up)))
    #     h += -self.J * (self.spins[0, 0] * self.spins[1, 1] + self.spins[2, 0] * self.spins[3, 1] + self.spins[1, 0] * self.spins[0, 1] + self.spins[3, 0] * self.spins[2, 1])
    #     self.h = h
    #     return self.h
    
    def get_hamiltonian(self):
        h = 0
        down = np.concatenate((self.spins[-1:], self.spins[:-1]))
        up = np.concatenate((self.spins[1:], self.spins[:1]))
        h += -.5 * self.J * sum(sum(np.multiply(self.spins, down) + np.multiply(self.spins, up)))
        product = 0
        for i in range(self.row):
            for j in range(self.col-1):
                if i%2 == 0:
                    product += self.spins[i, j] * self.spins[i + 1, j + 1]
                    # print(f'added {self.spins[i, j] * self.spins[i + 1, j + 1]}')
        for i in range(self.row):
            if i%2 == 1:
                product += self.spins[i, 0] * self.spins[i - 1, -1]
                # print(f'added {self.spins[i, 1] * self.spins[i - 1, -1]}')
        h += -self.J * product
        self.h = h
        return h


    
    def get_partition(self):
        Z = 0
        for s in range(2**(self.sites-1)):
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

# """ unpack the class and use @njit to accelerate """
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
#             while n:
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
# with open(r'Data/honeycomb5by5-2.txt', 'w') as file:
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

# scale = [2, 3, 4]
# all_data = {'E':{}, 'C':{}, 'M_N':{}, 'U_M':{}}
# marker = {2:'o', 3:'^', 4:'s', 5:'D'}
# color = {2:'red', 3:'orange', 4:'green', 5:'blue'}
# for n in scale:
#     path = r'Data/honeycomb' + str(n) + 'by' + str(n) + r'-2.txt'
#     with open(path, 'r') as file:
#         data = json.load(file)
#         E, C, M_N, U_M = data[0], data[1], data[2], data[3]
#         all_data['E'][str(n)] = E
#         all_data['C'][str(n)] = C
#         all_data['M_N'][str(n)] = M_N
#         all_data['U_M'][str(n)] = U_M
# fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(24, 18))
# for n in scale:
#     ax1.scatter(np.arange(.05, 4, .05), all_data['E'][str(n)], label=f'${n}$'+r'$\times$'+f'${n}$'+' lattice', marker=marker[n], color=color[n], s=12)
#     ax1.plot(np.arange(.05, 4, .05), all_data['E'][str(n)], color=color[n])
#     ax1.legend(fontsize=22)
#     ax2.scatter(np.arange(.05, 4, .05), all_data['C'][str(n)], label=f'${n}$'+r'$\times$'+f'${n}$'+' lattice', marker=marker[n], color=color[n], s=12)
#     ax2.plot(np.arange(.05, 4, .05), all_data['C'][str(n)], color=color[n])
#     ax2.legend(fontsize=22)
#     ax3.scatter(np.arange(.05, 4, .05), all_data['M_N'][str(n)], label=f'${n}$'+r'$\times$'+f'${n}$'+' lattice', marker=marker[n], color=color[n], s=12)
#     ax3.plot(np.arange(.05, 4, .05), all_data['M_N'][str(n)], color=color[n])
#     ax3.legend(fontsize=22)
#     ax4.scatter(np.arange(.05, 4, .05), all_data['U_M'][str(n)], label=f'${n}$'+r'$\times$'+f'${n}$'+' lattice', marker=marker[n], color=color[n], s=12)
#     ax4.plot(np.arange(.05, 4, .05), all_data['U_M'][str(n)], color=color[n])
#     ax4.legend(fontsize=22)
# ax1.set_xlabel('$T$', fontsize=22)
# ax1.set_ylabel(r'$\frac{\langle H \rangle}{N}$', fontsize=26)
# ax1.set_title('energy per spin', fontsize=22)
# ax2.set_xlabel('$T$', fontsize=22)
# ax2.set_ylabel(r'$\frac{C}{N}$', fontsize=26)
# ax2.set_title('specific heat per spin', fontsize=22)
# ax3.set_xlabel('$T$', fontsize=22)
# ax3.set_ylabel(r'$\frac{\langle M^2 \rangle}{N^2}$', fontsize=26)
# ax3.set_title('magnetization per spin squared', fontsize=22)
# ax4.set_xlabel('$T$', fontsize=22)
# ax4.set_ylabel(r'$\frac{\langle M^4 \rangle}{\langle M^2 \rangle ^2}$', fontsize=26)
# ax4.set_title('Binder ratio', fontsize=22)
# plt.savefig(r'Graphs/honeycomb-2.png', dpi=600)
# plt.show()

# if __name__ == '__main__':
#     unittest.main()