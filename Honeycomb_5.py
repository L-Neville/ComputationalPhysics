import unittest
import numpy as np
import numpy.testing as npt
import matplotlib.pyplot as plt
from random import randint

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
            temp_spins = []
            for _ in range(self.row):
                temp_lst = []
                for _ in range(self.col):
                    temp_lst.append(2*randint(0, 1)-1)
                temp_spins.append(temp_lst)
            self.spins = np.array(temp_spins)
        else: 
            assert type(spin_b)==int and spin_b>=0 and spin_b<=2**self.sites-1, "spin_config too large or too small or not an integer"
            self.spin_b = spin_b
            spins = np.array([(self.spin_b >> j) & 1 for j in range(self.row*self.col)]) # generate spin configuration from binary number
            spins = 2 * spins - 1 # convert 1&0 to +1&-1
            spins = spins.reshape((self.row, self.col)) # reshape to fit the lattice shape
            self.spins = spins

    def get_magnetization(self):
        cumulant = 0
        for row in self.spins:
            for i in row:
                cumulant += i
        self.magnetization = cumulant
    
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
        # print(f'  energy per spin is {HZ/Z/self.row/self.col}')
        # print(f'  specific heat per spin is {(H2Z/Z-(HZ/Z)**2)/self.T**2}')
        # print(f'  magnetization per spin squared is {M4Z*Z/M2Z**2}')
        # print(f'  overall takes {elapsed_time}s to calculate')
        return HZ/Z/self.row/self.col, (H2Z/Z-(HZ/Z)**2)/self.T**2, M2Z/Z/(self.row*self.col)**2, M4Z*Z/M2Z**2

temp = IsingModelHoneycomb(1, 1, 8, 8)
print(type(temp.spins))