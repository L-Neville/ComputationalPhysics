import numpy as np
from Honeycomb_3 import IsingModelHoneycomb

class MonteCarloSimulation():
    def initialize(self, J, T, row, col, mode='Random'):
        if mode=='Random':
            self.lattice = IsingModelHoneycomb(J, T, row, col, np.random.randint(0, 2**(row*col)))
            return self.lattice
        elif mode=='0':
            self.lattice = IsingModelHoneycomb(J, T, row, col, 0)
            return self.lattice            
        elif mode=='1':
            self.lattice = IsingModelHoneycomb(J, T, row, col, 2**(row*col)-1)
            return self.lattice
        else:
            print('mode undefined, failed to initialize')
        
    def flip(self, digit=None):
        if digit is None:
            digit = np.random.randint(0, self.lattice.row*self.lattice.col-1)
        self.latttice.spin_b ^= (1 << digit) # flip the kth digit (starting from 0): 0<==>1
        self.latttice.spins = np.array([(self.lattice.spin_b >> j) & 1 for j in range(self.lattice.row*self.lattice.col)]) 
        self.latttice.spins = 2 * self.latttice.spins - 1 
        self.latttice.spins = self.latttice.spins.reshape((self.lattice.row, self.lattice.col)) 

    def Metropolis_judge(self):
        before = self.lattice.get_hamiltonian()
        d = np.random.randint(0, self.lattice.row*self.lattice.col-1)
        self.flip(d)
        after = self.lattice.get_hamiltonian()
        if before > after:
            pass # keep the change
        else: # there is prob of exp(-(E2-E1)/T) to change
            r = np.random.random()
            if r > np.exp(-(after-before)/self.lattice.T):
                self.flip(d) # flip the lattice back to the original state
