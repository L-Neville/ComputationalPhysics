import numpy as np
from IsingModel import IsingModel 

def test_initial_spins():
    model = IsingModel(5, 1.0)
    assert (model.spins == np.zeros(5)).all()

def test_configure():
    model = IsingModel(5, 1.0)
    model.configure(0b10101)
    assert (model.spins == np.array([1, -1, 1, -1, 1])).all()

def test_magnetization():
    model = IsingModel(5, 1.0)
    model.configure(0b11111)
    assert model.get_magnetization() == 5
    model.configure(0b00000)
    assert model.get_magnetization() == -5

def test_energy():
    model = IsingModel(5, 1.0)
    model.configure(0b10101)
    expected_energy = -1.0 * model.J * (1*(-1) + -1*1 + 1*(-1) + -1*1 + 1*1)
    assert model.get_energy() == expected_energy
