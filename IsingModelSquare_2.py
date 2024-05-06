import unittest
import numpy as np
import numpy.testing as npt
import random
import matplotlib
import matplotlib.pyplot as plt
import tkinter as tk

class IsingModelSquareTest(unittest.TestCase):
    def test_init_num_assertion(self):
        with self.assertRaises(AssertionError):
            IsingModelSquare(1, spins=[[1, 2, 3, 4, 5, 6]])

    def test_init_square_assertion(self):
        with self.assertRaises(AssertionError):
            IsingModelSquare(1, spins=[[1, -1, 1, 1, 1], [-1, 1, 1, -1]])

    def test_init_shape_assertion(self):
        with self.assertRaises(AssertionError):
            IsingModelSquare(1, shape=(2,1), spins=[[1, -1, -1, -1, -1]])

    def test_init_set_spin(self):
        s1 = IsingModelSquare(1, spins=[[1, -1, 1, 1, 1], [-1, -1, -1, -1, 1]])
        npt.assert_array_equal(s1.spins, np.array([[1, -1, 1, 1, 1], [-1, -1, -1, -1, 1]]))

    def test_init_random_spin(self):
        s2 = IsingModelSquare(1, shape=(3,6))
        for i in range(3):
            for j in range(6):
                self.assertEqual((s2.spins[i, j])**2, 1)

    def test_init_convert(self):
        s7 = IsingModelSquare(1, shape=(3,3))
        self.assertEqual(s7.convert(2, 2), 4)

    def test_init_anticonvert(self):
        s8 = IsingModelSquare(1, shape=(2, 2))
        self.assertEqual(s8.anticonvert(2), (1, 2))

    def test_flip_assertion(self):
        with self.assertRaises(AssertionError):
            s3 = IsingModelSquare(1, shape=(2,2))
            s3.flip(4, 3)

    def test_flip(self):
        s4 = IsingModelSquare(1, shape=(4,5))
        spinsbef = s4.spins
        s4.flip(3, 3)
        s4.flip(3, 3)
        spinsaft = s4.spins
        npt.assert_array_equal(spinsbef, spinsaft)

    def test_set_as_up(self):
        s5 = IsingModelSquare(1, shape=(4, 5))
        s5.set_as_up(2, 2)
        self.assertEqual(s5.spins[2, 2], 1)

    def test_set_as_down(self):
        s6 = IsingModelSquare(1, shape=(4, 5))
        s6.set_as_down(2, 2)
        self.assertEqual(s6.spins[2, 2], -1)

    def test_adjoin(self):
        s9 = IsingModelSquare(1, shape=(3, 3))
        self.assertEqual(s9.adjoin(2, 2), set([(1, 2), (2, 1), (3, 2), (2, 3)]))
        self.assertEqual(s9.adjoin(1, 1), set([(1, 2), (2, 1), (3, 1), (1, 3)]))

    def test_adjoin_new(self):
        s9 = IsingModelSquare(1, shape=(2, 2))
        self.assertEqual(s9.adjoin_new(4), set([2, 3]))

    def test_magnetization(self):
        s10 = IsingModelSquare(1, spins=[[1, -1, 1, 1, 1], [-1, -1, -1, -1, 1]])
        self.assertEqual(s10.get_magnetization(), 0)
    
    def test_hamiltonion(self):
        s11 = IsingModelSquare(1, spins=[[1, 1], [1, 1]])
        self.assertEqual(s11.get_hamiltonion(), -4)
        s12 = IsingModelSquare(1, spins=[[1, 1, 1], [1, 1, 1], [1, 1, 1]])
        self.assertEqual(s12.get_hamiltonion(), -18)

    def test_partition(self):
        s12 = IsingModelSquare(1, T=1, shape=(4, 4))
        self.assertAlmostEqual(s12.get_partition(), (158808692495248.5))
        # a 2*2 lattice has partition 121.23293134406595 at T=1
        # a 3*3 lattice has partition 131737262.01527607 at T=1
        # a 4*4 lattice has partition 158808692495248.5 at T=1

    # def test_weight(self):
    #     s13 = IsingModelSquare(1, T=1, spins=[[1, 1], [-1, -1]])
    #     self.assertAlmostEqual(s13.get_weight(121.23293134406595), 1/121.23293134406595)

    # def test_avg(self):
    #     s14 = IsingModelSquare(1, T=1, shape=(3, 3))
    #     self.assertAlmostEqual(s14.get_average(131737262.01527607, 'get_magnetization', lambda x:x), 0)

class IsingModelSquare():
    def __init__(self, J, T=1, shape=None, spins=None):
    # initialize spins and shape, manually oor by default
        self.J = J # coupling coefficient between different sites are set equal, and self-coupling effect is neglected
        self.T = T # temperature here is weighed by a Boltzmann constant
        if not (spins is None): # if spins are specified, verify that they are legal and comply with shape
            row_num = len(spins)
            col_num = len(spins[0])
            for row in spins:
                assert col_num == len(row), "spins do not form a square lattie" # ensure that parameters passed into spins form a square lattice
                for s in row:
                    assert s**2 == 1, "spin is not 1 or -1" # ensure the spin is either 1 or -1
            if shape:
                assert shape == (row_num, col_num), "spins do not comply with shape"
            else:
                self.shape = (row_num, col_num)
            self.spins = np.array(spins)
        else: # if spins are not specified, set them automatically
            if not shape:
                self.shape = (6, 6) # if shape is not specified, set it as 6 * 6
            else:
                self.shape = shape
            self.spins = np.zeros(self.shape)
            for x in range(self.shape[0]):
                for y in range(self.shape[1]):
                    self.spins[x, y] = random.randint(0, 1) * 2 - 1
        self.convert = lambda x, y: (x - 1) * self.shape[0] + y - 1 # a lambda function that converts coordinate in the lattice into site number
        def anticonverter(i):
            if i % shape[1] == 0:
                return (i//shape[1], shape[1])
            else:
                return (i//shape[1]+1, i%shape[1])
        self.anticonvert = anticonverter

    def flip(self, *coordinates):
    # flip the spin at the given coordinate
        assert (self.shape[0] >= coordinates[0] and self.shape[1] >= coordinates[1]), "flipped coordinate out of range" 
        self.spins[coordinates[0], coordinates[1]] *= -1

    def set_as_up(self, *coordinates):
    # set the spin at the given coordinate as up
        assert (self.shape[0] >= coordinates[0] and self.shape[1] >= coordinates[1]), "target coordinate out of range" 
        self.spins[coordinates[0], coordinates[1]] = 1

    def set_as_down(self, *coordinates):
    # set the spin at the given coordinate as down
        assert (self.shape[0] >= coordinates[0] and self.shape[1] >= coordinates[1]), "target coordinate out of range" 
        self.spins[coordinates[0], coordinates[1]] = -1
            
    def adjoin(self, *coordinates):
    # find out four adjacent points and return them as an iterable
        x, y = coordinates
        if x == 1:
            xp, xm = (x + 1, y), (self.shape[0], y)
        elif x == self.shape[0]:
            xp, xm = (1, y), (x - 1, y)
        else:
            xp, xm = (x + 1, y), (x - 1, y)
        if y == 1:
            yp, ym = (x, y + 1), (x, self.shape[1])
        elif y == self.shape[1]:
            yp, ym = (x, 1), (x, y - 1)
        else:
            yp, ym = (x, y + 1), (x, y - 1)
        # all conditions, including boundaries, are considered
        return set([xp, xm, yp, ym]) # in common cases there are 4 adjacent points, but for 2*2 square there's only 2

    def adjoin_new(self, site):
        num_row, num_col = self.shape[0], self.shape[1]
        i, j = self.anticonvert(site)
        if i > 1 and num_col > j > 1:
            return set([site+1, site-1, (site+num_col)%(num_col*num_row), site-num_col])
        elif i > 1 and j == num_col:
            return set([site+1-num_col, site-1, (site+num_col)%(num_col*num_row), site-num_col])
        elif i > 1 and j == 1:
            return set([site+1, site-1+num_col, (site+num_col)%(num_col*num_row), site-num_col])
        elif i == 1 and num_col > j > 1:
            return set([site+1, site-1, site+num_col, site-num_col+num_col*num_row])
        elif i == 1 and j == num_col:
            return set([site+1-num_col, site-1, site+num_col, site-num_col+num_col*num_row])
        elif i == 1 and j == 1:
            return set([site+1, site-1+num_col, site+num_col, site-num_col+num_col*num_row])
    
    def get_magnetization(self):
    # get total magnetization of the state
        self.magnetization = np.sum(self.spins)
        return self.magnetization
    
    def get_hamiltonion(self):
    # based on previous adjoin method, get hamiltonion of the state
        num_row = self.shape[0]
        num_col = self.shape[1]
        h = 0
        # iterate over each site
        for x in range(1, num_row+1):
            for y in range(1, num_col+1):
                # for each site, iterate over all its adjacent points
                adjacents = self.adjoin(x, y)
                for adjacent in adjacents:
                    h += -.5 * self.J * self.spins[x-1, y-1] * self.spins[adjacent[0]-1, adjacent[1]-1]
        self.hamiltonion = h
        return self.hamiltonion
    
    # def get_partition(self):
    # # calculate partition that is summed over all spins configuration
    #     Z = 0
    #     num_row, num_col = self.shape[0], self.shape[1]
    #     def generate_half_spins():
    #     # due to the positive-negative symmetry, calculate only half the states
    #         for n in range(2**(num_row*num_col-1)):
    #             # use binary statement, to efficiently iterate over every spin configuration
    #             spins_bin = format(n, f'0{num_row*num_col}b')
    #             spins_1D = [1 if b == '1' else -1 for b in spins_bin]
    #             spins = np.array(spins_1D).reshape(num_row, num_col)
    #             yield spins
    #     for config in generate_half_spins():
    #         isingmodeltemp = IsingModelSquare(self.J, T=self.T, spins=config)
    #         Z += np.exp(-isingmodeltemp.get_hamiltonion()/self.T)
    #     Z *= 2
    #     return Z

    def get_partition(self):
    # calculate partition summed over all spins configuration
        Z, updown = 0, {'0':-1, '1':1}
        num_row, num_col = self.shape
        for n in range(2**(num_row*num_col-1)):
        # due to the positive-negative symmetry, calculate only half the states
            sbin, H = format(n, f"0{num_row*num_col}b"), 0
            for i in range(1, num_row*num_col+1):
                adjacents = self.adjoin_new(i)
                for adjacent in adjacents:
                    H += -self.J*updown[sbin[i-1]]*updown[sbin[adjacent-1]]/2
            Z += np.exp(-H/self.T)
        Z *= 2
        return Z
    
    def get_average_magnetization_squared(self):
        updown, Z, M2 = {'0':-1, '1':1}, 0, 0
        num_row, num_col = self.shape
        for n in range(2**(num_row*num_col)):
            sbin, H = format(n, f"0{num_row*num_col}b"), 0
            ms = (sbin.count('1')*2 - num_row*num_col) ** 2
            for i in range(1, num_row*num_col+1):
                adjacents = self.adjoin_new(i)
                for adjacent in adjacents:
                    H += -self.J*updown[sbin[i-1]]*updown[sbin[adjacent-1]]/2
            Z += np.exp(-H/self.T)
            M2 += np.exp(-H/self.T) * ms
        return M2 / Z
    
    def get_average_hamiltonion(self):
        updown, Z, Htotal = {'0':-1, '1':1}, 0, 0
        num_row, num_col = self.shape
        for n in range(2**(num_row*num_col)):
            sbin, H = format(n, f"0{num_row*num_col}b"), 0
            for i in range(1, num_row*num_col+1):
                adjacents = self.adjoin_new(i)
                for adjacent in adjacents:
                    H += -self.J*updown[sbin[i-1]]*updown[sbin[adjacent-1]]/2
            Z += np.exp(-H/self.T)
            Htotal += np.exp(-H/self.T) * H
        return Htotal / Z
    
    def get_average_hamiltonion_squared(self):
        updown, Z, Hstotal = {'0':-1, '1':1}, 0, 0
        num_row, num_col = self.shape
        for n in range(2**(num_row*num_col)):
            sbin, H = format(n, f"0{num_row*num_col}b"), 0
            for i in range(1, num_row*num_col+1):
                adjacents = self.adjoin_new(i)
                for adjacent in adjacents:
                    H += -self.J*updown[sbin[i-1]]*updown[sbin[adjacent-1]]/2
            Z += np.exp(-H/self.T)
            Hstotal += np.exp(-H/self.T) * H**2
        return Hstotal / Z        
        
    # def get_weight(self, partition):
    # # to avoid repeated calculation, partition is calculated beforehand and pased in
    #     return np.exp(-self.get_hamiltonion()/self.T) / partition
    
    # def get_average(self, partition, meth_name, func):
    # # a comprehensive method that calculates the avg of func(meth)
    # # here, meth is either get_magnetization or get_hamiltonion, and func can be ^2
    #     num_row, num_col = self.shape[0], self.shape[1]
    #     avg = 0
    #     def generate_spins():
    #         for n in range(2**(num_row*num_col)):
    #             spins_bin = format(n, f'0{num_row*num_col}b')
    #             spins_1D = [1 if b == '1' else -1 for b in spins_bin]
    #             spins = np.array(spins_1D).reshape(num_row, num_col)
    #             yield spins
    #     for config in generate_spins():  
    #         isingmodeltemp = IsingModelSquare(self.J, T=self.T, spins=config)
    #         quantity = eval(f"isingmodeltemp.{meth_name}()")
    #         # use eval() to combine instance and passed method
    #         avg += isingmodeltemp.get_weight(partition) * func(quantity)
    #     return avg

    # def get_specific_heat(self, partition):
    #     return (self.get_average(
    #         partition, 'get_hamiltonion', lambda x: x*x 
    #     ) - self.get_average(
    #         partition, 'get_hamiltonion', lambda x: x
    #     )) / (self.T) ** 2

# print(matplotlib.get_backend())
matplotlib.use('TkAgg')            

# if __name__ == '__main__':
#     unittest.main()

size = [(2, 2), (3, 3)]
t_start, t_end, t_step = .05, 4, .05
temperature = np.linspace(t_start, t_end, int((t_end-t_start)/t_step+1))
fig, (ax1, ax2) = plt.subplots(1, 2)
marker = {(2, 2):'o', (3, 3):'^', (4, 4):'s', (5, 5):'D'}
for s in size:
    y1, y2 = [], []
    for t in temperature:
        isingmodeltemp = IsingModelSquare(1, T=t, shape=s)
        y1.append(isingmodeltemp.get_average_magnetization_squared() / (s[0]*s[1])**2)
        y2.append((isingmodeltemp.get_average_hamiltonion_squared() - isingmodeltemp.get_average_hamiltonion()**2) / (isingmodeltemp.T)**2 / (s[0]*s[1]))
    ax1.scatter(temperature, y1, marker=marker[s], s=10, label=f'size {s}')
    ax2.scatter(temperature, y2, marker=marker[s], s=10, label=f'size {s}')
ax1.set_xlabel('T')
ax1.set_ylabel('M^2')
ax1.legend()
ax1.set_title('magnetic susceptibility')
ax2.set_xlabel('T')
ax2.set_ylabel('C')
ax2.legend()
ax2.set_title('specific heat')
plt.savefig('square_lattice_1.jpg', dpi=300)

# fig, ax = plt.subplots()
# for s in size:
#     num_row, num_col = s
#     for t in temperature:
#         isingmodeltemp = IsingModelSquare(1, T=t, shape=s)
#         p = isingmodeltemp.get_partition()      
#         m = isingmodeltemp.get_average(p, 'get_magnetization', lambda x: x*x)
#         ax.scatter(t, m)
# plt.show()


        
            


 