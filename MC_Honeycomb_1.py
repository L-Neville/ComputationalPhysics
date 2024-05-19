import numpy as np
import matplotlib.pyplot as plt
from Honeycomb_4 import IsingModelHoneycomb

class MonteCarloSimulation():
    def __init__(self, J, T, row, col, rounds, mode='Random'):
        self.rounds = rounds
        if mode=='Random':
            self.lattice = IsingModelHoneycomb(J, T, row, col, np.random.randint(0, 2**(row*col)))
        elif mode=='0':
            self.lattice = IsingModelHoneycomb(J, T, row, col, 0)
        elif mode=='1':
            self.lattice = IsingModelHoneycomb(J, T, row, col, 2**(row*col)-1)
        else:
            print('mode undefined, failed to initialize')
        
    def flip(self, digit=None):
        if digit is None:
            digit = np.random.randint(0, self.lattice.row*self.lattice.col-1)
        self.lattice.spin_b ^= (1 << digit) # flip the kth digit (starting from 0): 0<==>1
        self.lattice.spins = np.array([(self.lattice.spin_b >> j) & 1 for j in range(self.lattice.row*self.lattice.col)]) 
        self.lattice.spins = 2 * self.lattice.spins - 1 
        self.lattice.spins = self.lattice.spins.reshape((self.lattice.row, self.lattice.col)) 

    def Metropolis(self):
        before = self.lattice.get_hamiltonian()
        d = np.random.randint(0, self.lattice.row*self.lattice.col-1)
        self.flip(d)
        after = self.lattice.get_hamiltonian()
        if before > after:
            pass # keep the flip
            return after
        else: # there is prob of exp(-(E2-E1)/T) to flip
            r = np.random.random()
            if r > np.exp(-(after-before)/self.lattice.T):
                self.flip(d) # flip the lattice back to the original state
                return before
            return after

    def simulate(self):
        for _ in self.rounds:
            self.Metropolis()

rounds = 10000
simulation_test = MonteCarloSimulation(1, 3, 4, 4, rounds, 'Random')
initial_lattice = simulation_test.lattice
hamiltonian_bf = initial_lattice.get_average()[0]
hamiltonian_bf = np.array([hamiltonian_bf for _ in range(rounds)])
hamiltonian_mc = [initial_lattice.get_hamiltonian()/(initial_lattice.row*initial_lattice.col)]
for r in range(1, rounds): # in a loop, r is the number of trials having been done
    hamiltonian_mc.append(
        simulation_test.Metropolis()/(initial_lattice.row*initial_lattice.col)/(r+1) + hamiltonian_mc[r-1]*r/(r+1)
    )
simulation_test = MonteCarloSimulation(1, 3, 4, 4, rounds, 'Random')
magnetization_bf = 0
magnetization_bf = np.array([magnetization_bf for _ in range(rounds)])
magnetization_mc = [initial_lattice.get_magnetization()/(initial_lattice.row*initial_lattice.col)]
for r in range(1, rounds): # in a loop, r is the number of trials having been done
    simulation_test.Metropolis()
    magnetization_mc.append(
        simulation_test.lattice.get_magnetization()/(initial_lattice.row*initial_lattice.col)/(r+1) + magnetization_mc[r-1]*r/(r+1)
    )    
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))
plt.subplots_adjust(hspace=.4)
ax1.plot(np.arange(rounds), hamiltonian_bf, c='red', label='enumerated')
ax1.plot(np.arange(rounds), hamiltonian_mc, c='blue', label='simulated')
ax1.set_ylim(-1, .5)
ax1.set_xlabel('rounds', fontsize=22)
ax1.set_ylabel(r'$\frac{\langle H \rangle}{N}$ at $T=3$, $J=1$', fontsize=22)
ax1.set_title('average hamiltonian per spin', fontsize=22)
ax1.legend(fontsize=22)
ax1.tick_params(axis='both', which='major', labelsize=16)
ax2.plot(np.arange(rounds), magnetization_bf, c='red', label='enumerated')
ax2.plot(np.arange(rounds), magnetization_mc, c='blue', label='simulated')
ax2.set_ylim(-1.5, 1.5)
ax2.set_xlabel('rounds', fontsize=22)
ax2.set_ylabel(r'$\frac{\langle M \rangle}{N}$ at $T=3$, $J=1$', fontsize=22)
ax2.set_title('average magnetization per spin', fontsize=22)
ax2.legend(fontsize=22)
ax2.tick_params(axis='both', which='major', labelsize=16)
plt.savefig(r'cm1.png', dpi=300)
plt.show()
