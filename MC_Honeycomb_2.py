from Honeycomb_4 import IsingModelHoneycomb
from MC_Honeycomb_1 import MonteCarloSimulation
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

class MonteCarloSimulationwithBins(MonteCarloSimulation):
    def __init__(self, J, T, row, col, rounds, bin_size, bin_number, mode='Random'):
        super().__init__(J, T, row, col, rounds, mode)
        self.bin_size = bin_size
        self.bin_number = bin_number
        # assert self.rounds > self.bin_number * self.bin_size, 'insufficient rounds'

    def initialize(self, initial_rounds=None):
        if not initial_rounds:
            initial_rounds = self.bin_size*4
        for _ in range(initial_rounds):
            self.Metropolis()
    
    def bin_forward(self, observables):
        self.onum = len(observables)
        bin_mean = np.array([0 for _ in range(self.onum)])
        for _ in range(self.bin_size):
            self.Metropolis()
            h, m = self.lattice.get_hamiltonian(), self.lattice.get_magnetization()
            for i in range(self.onum):
                bin_mean[i] += observables[i](h, m)
        bin_mean = [m/self.bin_size for m in bin_mean]
        return bin_mean
    
    def sample(self, *observables): # observables are functions that receive h, m as arguments
        mean_series = []
        self.initialize()
        for _ in range(self.bin_number):
            mean_series.append(self.bin_forward(observables))
        self.series = mean_series
        return mean_series
    
    def analyze(self):
        mean, squared = np.array([0 for _ in range(self.onum)]), np.array([0 for _ in range(self.onum)])
        for s in self.series:
            for o in range(self.onum):
                mean[o] += s[o]
                squared[o] += s[o]**2
        mean = mean / self.bin_number
        squared = squared / self.bin_number
        stnadard_error = (squared - mean**2)**.5    
        return mean, stnadard_error
    

# rounds = 100000
# bin_number = 10000
# simulation_test = MonteCarloSimulationwithBins(1, 3, 4, 4, rounds, 250, bin_number, 'Random')
# sites = simulation_test.lattice.sites
# initial_lattice = simulation_test.lattice
# t1, _, t3, _ = initial_lattice.get_average()
# t3 *= initial_lattice.sites
# t2 = 0
# t1 = np.array([t1 for _ in range(bin_number)]) # average hamiltonian 
# t2 = np.array([t2 for _ in range(bin_number)]) # average magnetization
# t3 = np.array([t3 for _ in range(bin_number)]) # average magnetization squared 
# f1 = lambda h, m: h
# f2 = lambda h, m: m
# f3 = lambda h, m: m**2
# simulation_test.initialize()
# simulation_test.sample(f1, f2, f3)
# mean, standard_error = simulation_test.analyze()

# fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 12))
# plt.subplots_adjust(hspace=.4)
# ax1.plot(np.arange(bin_number), t1, c='red', label='enumerated')
# ax1.plot(np.arange(bin_number), [m[0]/sites for m in simulation_test.series], c='blue', label='simulated')
# ax1.scatter(np.arange(bin_number), [m[0]/sites for m in simulation_test.series], color='blue')
# ax1.set_ylim(-1, .5)
# ax1.set_xlabel('bins', fontsize=22)
# ax1.set_ylabel(r'$\frac{\langle H \rangle}{N}$ at $T=3$, $J=1$', fontsize=22)
# ax1.set_title('average hamiltonian per spin', fontsize=22)
# ax1.legend(fontsize=22)
# ax1.tick_params(axis='both', which='major', labelsize=16)
# ax2.plot(np.arange(bin_number), t2, c='red', label='enumerated')
# ax2.plot(np.arange(bin_number), [m[1]/sites for m in simulation_test.series], c='blue', label='simulated')
# ax2.scatter(np.arange(bin_number), [m[1]/sites for m in simulation_test.series], color='blue')
# ax2.set_ylim(-1.5, 1.5)
# ax2.set_xlabel('bins', fontsize=22)
# ax2.set_ylabel(r'$\frac{\langle M \rangle}{N}$ at $T=3$, $J=1$', fontsize=22)
# ax2.set_title('average magnetization per spin', fontsize=22)
# ax2.legend(fontsize=22)
# ax2.tick_params(axis='both', which='major', labelsize=16)
# ax3.plot(np.arange(bin_number), t3, c='red', label='enumerated')
# ax3.plot(np.arange(bin_number), [m[2]/sites for m in simulation_test.series], c='blue', label='simulated')
# ax3.scatter(np.arange(bin_number), [m[2]/sites for m in simulation_test.series], color='blue')
# ax3.set_ylim(-0, 10)
# ax3.set_xlabel('bins', fontsize=22)
# ax3.set_ylabel(r'$\frac{\langle M^2 \rangle}{N}$ at $T=3$, $J=1$', fontsize=22)
# ax3.set_title('average magnetization squared per spin', fontsize=22)
# ax3.legend(fontsize=22)
# ax3.tick_params(axis='both', which='major', labelsize=16)
# plt.show()

# hdata = [m[0]/sites for m in simulation_test.series] # hamiltonians fall into the range [-6.0, -4.0]
# step = -.02
# sorted_hdata = defaultdict(int)
# for h in hdata:
#     sorted_hdata[h//step] += 1
# fig, ax = plt.subplots(figsize=(16, 12))
# ax.bar(sorted_hdata.keys(), sorted_hdata.values())
# ax.set_xlabel(r'$\frac{\langle H \rangle}{N} \quad (\times100)$ ', fontsize=22)
# ax.set_ylabel('number of bins', fontsize=22)
# ax.set_title('distribution of average hamiltonian in bins', fontsize=22)
# ax.tick_params(axis='both', which='major', labelsize=16)
# plt.savefig('distribution.png', dpi=300)
# plt.show()