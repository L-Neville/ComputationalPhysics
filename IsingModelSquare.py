import unittest
import numpy as np
import numpy.testing as npt
import random

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
        s8 = IsingModelSquare(1, shape=(3,3))
        self.assertEqual(s8.anticonvert(4), (2, 2))

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

    def test_magnetization(self):
        s10 = IsingModelSquare(1, spins=[[1, -1, 1, 1, 1], [-1, -1, -1, -1, 1]])
        self.assertEqual(s10.get_magentization(), 0)
    
    def test_hamiltonion(self):
        s11 = IsingModelSquare(1, spins=[[1, 1], [1, 1]])
        self.assertEqual(s11.get_hamiltonion(), -4)
        s12 = IsingModelSquare(1, spins=[[1, 1, 1], [1, 1, 1], [1, 1, 1]])
        self.assertEqual(s12.get_hamiltonion(), -18)

    
    


class IsingModelSquare():
    def __init__(self, J, T=1, shape=None, spins=None):
    # initialize spins and shape, manually oor by default
        self.J = J # coupling coefficient between different sites are set equal, and self-coupling effect is neglected
        self.T = T # temperature here is weighed by a Boltzmann constant
        if spins: # if spins are psecified, verify that they are legal and comply with shape
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
        self.anticonvert = lambda  i: (i // shape[0] + 1, i % shape[0] + 1) # a lambda function that converts site number into coordinate

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
    
    def get_magentization(self):
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

    def get_partition(self):
    # calculate partition that is summed all spins configuration
        Z = 0
        num_row, num_col = self.shape[0], self.shape[1]
        def generate_spins():
            for n in range(2**(num_row*num_col)):
                # use binary statement, to efficiently iterate over every spin configuration
                spins_bin = format(n, f'0{num_row*num_col}b')
                spins_1D = [1 if b == '1' else -1 for b in spins_bin]
                spins = np.array(spins_1D)
                yield spins
        for config in generate_spins():
            isingmodeltemp = IsingModelSquare(self.J, T=self.T, spins=config)
            Z += np.exp(-isingmodeltemp.get_hamiltonion()/self.T)
        return Z
    
    def get_weight(self, partition):
    # to avoid repeated calculation, partition is calculated beforehand and pased in
        return np.exp(-self.get_hamiltonion()/self.T) / partition
    
    def get_average(self, partition, meth_name, func):
    # a comprehensive method that calculates the avg of func(meth)
    # here, meth is either get_magnetization or get_hamiltonion, and func can be ^2
        num_row, num_col = self.shape[0], self.shape[1]
        mag_avg = 0
        def generate_spins():
            for n in range(2**(num_row*num_col)):
                spins_bin = format(n, f'0{num_row*num_col}b')
                spins_1D = [1 if b == '1' else -1 for b in spins_bin]
                spins = np.array(spins_1D)
                yield spins
        for config in generate_spins():  
            isingmodeltemp = IsingModelSquare(self.J, T=self.T, spins=config)
            quantity = eval('isingmodeltemp.' + meth_name)
            # use eval() to combine instance and passed method
            mag_avg += isingmodeltemp.get_weight(partition) * func(quantity)

    def get_specific_heat(self, partition):
        return (self.get_average(
            partition, 'get_hamiltonion', lambda x: x*x 
        ) - self.get_average(
            partition, 'get_hamiltonion', lambda x: x
        )) / (self.T) ** 2            

    


            
if __name__ == '__main__':
    unittest.main()

