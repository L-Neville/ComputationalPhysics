import numpy as np

class IsingModel:
    def __init__(self, num, J):
        # __init__ method sets the coupling coefficient and number of spins for the ising system
        self.num = num
        self.J = J
        self.spins = np.zeros(num, dtype=int) # the default spins before being configured

    def configure(self, state):
        if not isinstance(state, int):
            raise ValueError("State representation not an integer")
        if not (0 <= state < 2*self.num):
            raise ValueError("State representation too large or too small")
        bin_state = format(state, f'0{self.num}b') 
        # parse the numerical representation of state, which yields a binary number with length identical to number of spins
        self.spins = np.array([1 if char=='1' else -1 for char in bin_state])
        list_state = [f"up_{i} " if v==1 else f"down_{i} " for i, v in enumerate(self.spins)]
        print('spins: '+''.join(list_state))

    def get_magnetization(self):
        return np.sum(self.spins)
    
    def get_energy(self):
        spins_shifted = np.append(self.spins[1:], self.spins[0])
        return -self.J * np.sum(self.spins * spins_shifted)
    
