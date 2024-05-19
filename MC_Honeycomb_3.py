import numpy as np
import matplotlib.pyplot as plt
from MC_Honeycomb_2 import MonteCarloSimulationwithBins
import unittest

class MonteCarloTest(unittest.TestCase):
    def test_95_percent_normal_distr(self):
        bin_number, bin_size = 10000, 200
        test = MonteCarloSimulationwithBins(1, 3, 4, 4, None, bin_size, bin_number, 'Random')
        sites = test.lattice.sites
        func1 = lambda h, m: h
        func2 = lambda h, m: m
        func3 = lambda h, m: m**2
        test.initialize()
        test.sample(func1, func2, func3)
        hdata = [m[0]/sites for m in test.series]
        mdata = [m[1]/sites for m in test.series]
        m2data = [m[2]/sites for m in test.series]
        mean, standard_error = test.analyze()
        mean /= sites
        standard_error /= sites
        interval = [-1.96, 1.96]
        interval1 = [mean[0] + standard_error[0] * i for i in interval]
        interval2 = [mean[1] + standard_error[1] * i for i in interval]
        interval3 = [mean[2] + standard_error[2] * i for i in interval]
        total, count1, count2, count3 = bin_number, 0, 0, 0
        for h in hdata:
            if interval1[0] < h < interval1[1]:
                count1 += 1
        for m in mdata:
            if interval2[0] < m < interval2[1]:
                count2 += 1
        for m2 in m2data:
            if interval3[0] < m2 < interval3[1]:
                count3 += 1
        message = f" hamiltonian: in total {total} trials, {count1} trials succeed; ratio of successive trials: {count1/total} \n magnetization: in total {total} trials, {count2} trials succeed; ratio of successive trials is {count2/total} \n magnetization squared: in total {total} trials, {count3} trials succeed; ratio of successive trials is {count3/total}"
        print(message)        
        self.assertGreater(min(count1/total, count2/total, count3/total), .95)

if __name__ == "__main__":
    unittest.main()