import unittest
import numpy as np
import matplotlib.pyplot as plt
from random import random
from time import time
import json

class IsingModelHoneycombTest(unittest.TestCase):
    pass

class IsingModelHoneycomb():
    def __init__(self, J, T, row, col, spin_config=None):
        self.J, self.T = J, T
        assert row%2==0 and col%2==0 and row*col>0, "number of rows and columns must be even"
        self.row, self.col = row, col
        # self.sites = self.col*self.row/2+(self.col-1)*self.row/2
        self.sites = self.row*self.col
        # def is_long(row_number): # distinguish long and short rows
        #     if row_number%4==0 or row_number%4==1:
        #         return True
        #     else: 
        #         return False
        # def get_site_number(row_number): # get the number of sites on a row
        #     if is_long(row_number):
        #         return self.col
        #     else:
        #         return self.col-1
        """
        set the spins automatically or manually.
        to improve efficiency, spins configuration is represented as an integer (and its binary form), instead of containers
        """
        if not spin_config: # if spin_config is not passed, randomly set each spin
            self.spins = 0
            for _ in range(self.sites):
                self.spins <<= 1 # left shift, i.e. *2
                if random() > .5:
                    self.spins += 1
        else: 
            assert type(spin_config)==int and spin_config>=0 and spin_config<=2**self.sites-1, "spin_config too large or too small or not an integer"
            self.spins = spin_config

    def get_magnetization(self):
        ups, n = 0, self.spins
        while n: # count the number of '1' in binary m
            n &= n - 1
            ups += 1 
        self.M = 2*ups-self.sites
        return self.M
    
    def get_hamiltonion(self):
        # make the convention that number of rows and cols start from 1
        def get_spin(spins, site):
            mask = 1 << site
            return (spins & mask) >> site
        def get_site(x, y): # get the site number of a given row, col number
            return self.col*(x-1)+y-1
        def mol_0_internal_internal_adjoin(i, j):
            return [get_site(i-1,j),get_site(i-1,j+1),get_site(i+1,j)]
        def mol_1_internal_internal_adjoin(i, j):
            return [get_site(i-1,j),get_site(i+1,j),get_site(i+1,j+1)]
        def mol_2_internal_internal_adjoin(i, j):
            return [get_site(i+1,j),get_site(i-1,j-1),get_site(i-1,j)]
        def mol_3_internal_internal_adjoin(i, j):
            return [get_site(i-1,j),get_site(i+1,j-1),get_site(i+1,j)]
        def mol_0_left_internal_adjoin(i, j):
            return [get_site(i-1,j),get_site(i-1,j+1),get_site(i+1,j)]
        def mol_1_left_internal_adjoin(i, j):
            return [get_site(i-1,j),get_site(i+1,j),get_site(i+1,j+1)] 
        def mol_2_left_internal_adjoin(i, j):
            return [get_site(i+1,j),get_site(i-1,self.col),get_site(i-1,j)]    
        def mol_3_left_internal_adjoin(i, j):
            return [get_site(i-1,j),get_site(i+1,self.col),get_site(i+1,j)]
        def mol_0_right_internal_adjoin(i, j):
            return [get_site(i-1,j),get_site(i-1,1),get_site(i+1,j)]
        def mol_1_right_internal_adjoin(i, j):
            return [get_site(i-1,j),get_site(i+1,j),get_site(i+1,1)]
        def mol_2_right_internal_adjoin(i, j):
            return [get_site(i+1,j),get_site(i-1,j-1),get_site(i-1,j)]
        def mol_3_right_internal_adjoin(i, j):
            return [get_site(i-1,j),get_site(i+1,j-1),get_site(i+1,j)]
        def mol_0_internal_top_adjoin(i, j):
            return [get_site(self.row-1,j),get_site(self.row-1,j+1),get_site(1,j)] 
        def mol_1_internal_bottom_adjoin(i, j):
            return [get_site(self.row,j),get_site(2,j+1),get_site(2,j)]
        def mol_0_left_top_adjoin(i, j):
            return [get_site(i-1,1),get_site(i-1,2),get_site(1,1)]    
        def mol_0_right_top_adjoin(i, j):
            return [get_site(self.row-1,j),get_site(self.row-1,1),get_site(1,j)]
        def mol_1_left_bottom_adjoin(i, j):
            return [get_site(1,1),get_site(1,2),get_site(self.row,1)]
        def mol_1_right_bottom_adjoin(i, j):
            return [get_site(2,j),get_site(2,1),get_site(self.row,j)]   
        adjoin_map = {
                '0':{
                    '1':mol_0_left_internal_adjoin,str(self.col):mol_0_right_internal_adjoin
                },
                '1':{
                    '1':mol_1_left_internal_adjoin,str(self.col):mol_1_right_internal_adjoin
                },
                '2':{
                    '1':mol_2_left_internal_adjoin,str(self.col):mol_2_right_internal_adjoin
                },
                '3':{
                    '1':mol_3_left_internal_adjoin,str(self.col):mol_3_right_internal_adjoin
                },
                'bottom':{
                    '1':mol_1_left_bottom_adjoin,str(self.col):mol_1_right_bottom_adjoin
                },
                'top':{
                    '1':mol_0_left_top_adjoin,str(self.col):mol_0_right_top_adjoin
                }
        }     
        adjoin_default_map = {
            '0':mol_0_internal_internal_adjoin,
            '1':mol_1_internal_internal_adjoin,
            '2':mol_2_internal_internal_adjoin,
            '3':mol_3_internal_internal_adjoin,
            'bottom':mol_1_internal_bottom_adjoin,
            'top':mol_0_internal_top_adjoin
        }              
        key1_map={
            1:'bottom', self.row:'top'
        }        
        H = 0
        for i in range(1, 1+self.row):
            for j in range(1, 1+self.col):
                key1 = key1_map.get(i, str(i%4))
                func = adjoin_map.get(key1, None).get(str(j), adjoin_default_map.get(key1, None))
                for adjacent in func(i, j):
                    H += -.5 * self.J * get_spin(self.spins,get_site(i, j)) * get_spin(self.spins,adjacent)
        self.H = H
        return H
    
    def get_partition(self):
        Z = 0
        for s in range(2**(self.sites-1)):
            temp = IsingModelHoneycomb(self.J, self.T, 4, 4, spin_config=s)
            Z += np.exp(-temp.get_hamiltonion()/temp.T)
        self.Z = Z*2
        return self.Z
    
    def get_average(self):
        time_start = time()
        Z, HZ, H2Z, M2Z, M4Z = 0, 0, 0, 0, 0
        for s in range(2**(self.sites-1)): # both M squared and H are +/- symmetric
            temp = IsingModelHoneycomb(self.J, self.T, self.row, self.col, spin_config=s)
            h, m = temp.get_hamiltonion(), temp.get_magnetization()
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
        return HZ/Z/self.row/self.col, (H2Z/Z-(HZ/Z)**2)/self.T**2, M4Z*Z/M2Z**2


            
    
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

# with open('honeycomb4by4.txt', 'r') as file:
#     data = json.load(file)
#     E, C, U_M = data[0], data[1], data[2]

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

a = time()
E, C, U_M = [], [], []
J = 1
row, col = (4, 4)
def get_spin(spins, site):
    mask = 1 << site
    return (spins & mask) >> site
def get_site(x, y): 
    return col*(x-1)+y-1
def mol_0_internal_internal_adjoin(i, j):
    return [get_site(i-1,j),get_site(i-1,j+1),get_site(i+1,j)]
def mol_1_internal_internal_adjoin(i, j):
    return [get_site(i-1,j),get_site(i+1,j),get_site(i+1,j+1)]
def mol_2_internal_internal_adjoin(i, j):
    return [get_site(i+1,j),get_site(i-1,j-1),get_site(i-1,j)]
def mol_3_internal_internal_adjoin(i, j):
    return [get_site(i-1,j),get_site(i+1,j-1),get_site(i+1,j)]
def mol_0_left_internal_adjoin(i, j):
    return [get_site(i-1,j),get_site(i-1,j+1),get_site(i+1,j)]
def mol_1_left_internal_adjoin(i, j):
    return [get_site(i-1,j),get_site(i+1,j),get_site(i+1,j+1)] 
def mol_2_left_internal_adjoin(i, j):
    return [get_site(i+1,j),get_site(i-1,col),get_site(i-1,j)]    
def mol_3_left_internal_adjoin(i, j):
    return [get_site(i-1,j),get_site(i+1,col),get_site(i+1,j)]
def mol_0_right_internal_adjoin(i, j):
    return [get_site(i-1,j),get_site(i-1,1),get_site(i+1,j)]
def mol_1_right_internal_adjoin(i, j):
    return [get_site(i-1,j),get_site(i+1,j),get_site(i+1,1)]
def mol_2_right_internal_adjoin(i, j):
    return [get_site(i+1,j),get_site(i-1,j-1),get_site(i-1,j)]
def mol_3_right_internal_adjoin(i, j):
    return [get_site(i-1,j),get_site(i+1,j-1),get_site(i+1,j)]
def mol_0_internal_top_adjoin(i, j):
    return [get_site(row-1,j),get_site(row-1,j+1),get_site(1,j)] 
def mol_1_internal_bottom_adjoin(i, j):
    return [get_site(row,j),get_site(2,j+1),get_site(2,j)]
def mol_0_left_top_adjoin(i, j):
    return [get_site(i-1,1),get_site(i-1,2),get_site(1,1)]    
def mol_0_right_top_adjoin(i, j):
    return [get_site(row-1,j),get_site(row-1,1),get_site(1,j)]
def mol_1_left_bottom_adjoin(i, j):
    return [get_site(1,1),get_site(1,2),get_site(row,1)]
def mol_1_right_bottom_adjoin(i, j):
    return [get_site(2,j),get_site(2,1),get_site(row,j)]   
adjoin_map = {
        '0':{
            '1':mol_0_left_internal_adjoin,str(col):mol_0_right_internal_adjoin
        },
        '1':{
            '1':mol_1_left_internal_adjoin,str(col):mol_1_right_internal_adjoin
        },
        '2':{
            '1':mol_2_left_internal_adjoin,str(col):mol_2_right_internal_adjoin
        },
        '3':{
            '1':mol_3_left_internal_adjoin,str(col):mol_3_right_internal_adjoin
        },
        'bottom':{
            '1':mol_1_left_bottom_adjoin,str(col):mol_1_right_bottom_adjoin
        },
        'top':{
            '1':mol_0_left_top_adjoin,str(col):mol_0_right_top_adjoin
        }
}     
adjoin_default_map = {
    '0':mol_0_internal_internal_adjoin,
    '1':mol_1_internal_internal_adjoin,
    '2':mol_2_internal_internal_adjoin,
    '3':mol_3_internal_internal_adjoin,
    'bottom':mol_1_internal_bottom_adjoin,
    'top':mol_0_internal_top_adjoin
}              
key1_map={
    1:'bottom', row:'top'
}        
for t in np.arange(.05,4,.05):
    Z, HZ, H2Z, M2Z, M4Z = 0, 0, 0, 0, 0
    for spins in range(2**(row*col-1)):
        h = 0
        for i in range(1, 1+row):
            for j in range(1, 1+col):
                key1 = key1_map.get(i, str(i%4))
                func = adjoin_map.get(key1, None).get(str(j), adjoin_default_map.get(key1, None))
                for adjacent in func(i, j):
                    h += -.5 * J * get_spin(spins,get_site(i, j)) * get_spin(spins,adjacent)
        ups, n = 0, spins
        while n: # count the number of '1' in binary m
            n &= n - 1
            ups += 1 
        m = 2*ups-row*col
        z = np.exp(-h/t)
        Z += z
        HZ += z*h
        H2Z += z*h**2
        M2Z += z*m**2
        M4Z += z*m**4
    e, c, u_m = HZ/Z/row/col, (H2Z/Z-(HZ/Z)**2)/t**2, M4Z*Z/M2Z**2
    E.append(e)
    C.append(c)
    U_M.append(u_m)
b = time()
print(f'overall takes {b-a}s to calculate under all temperatures')

# data = [E, C, U_M]
# with open('honeycomb8by4.txt', 'w') as file:
#     json.dump(data, file)


"""
this is the first version of hoenycomb.
summary:
    1. calculating 4*4 lattice takes about 88s
    2. shape whose row number is not integer times of 4 can't be calculated without modification
    3. in next version, np.roll() and @njit will be applied to improve performance and extensitivity
"""