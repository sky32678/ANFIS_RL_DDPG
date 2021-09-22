

import sys
import itertools
import numpy as np

import torch
from torch.utils.data import TensorDataset, DataLoader

import anfis
from membership import TrapezoidalMembFunc, make_trap_mfs, make_bell_mfs, BellMembFunc
from experimental import train_anfis, test_anfis
dtype = torch.float


def eqn(x, y, z):
    '''
        The three input non-linear function used in Jang's example 2
    '''
    #output = torch.sin(x) * torch.cos(y) + torch.tan(z)
    output = 1 + torch.pow(x, 0.5) + torch.pow(y, -1) + torch.pow(z, -1.5)
    output = torch.pow(output, 2)
#    output = 1.3 + torch.pow(x, 0.8) + torch.pow(y, -2) + torch.pow(z, -2)
#    output = torch.pow(output, 3)
    return output


def _make_data_xyz(inp_range):
    '''
        Given a range, return a dataset with the product of these values.
        Assume we want triples returned - i.e. (x,y,z) points
    '''
    xyz_vals = itertools.product(inp_range, inp_range, inp_range)
    x = torch.tensor(list(xyz_vals), dtype=dtype)
    y = torch.tensor([[eqn(*p)] for p in x], dtype=dtype)
#    print(*TensorDataset(x,y))
    return TensorDataset(x, y)


def training_data(batch_size=1024):
    '''
        Jang's training data uses integer values between 1 and 6 inclusive
    '''
    inp_range = range(1, 7, 1)
    td = _make_data_xyz(inp_range)
    #print(len(td))
    return DataLoader(td, batch_size=batch_size, shuffle=True)


def testing_data():
    '''
        Jang's test data uses values 1.5, 2.5 etc.
    '''
    inp_range = np.arange(1.3,6.3, 1)
    td = _make_data_xyz(inp_range)
    print(len(td))
    return DataLoader(td)


def ex1_model():
    '''
        These are the original (untrained) MFS for Jang's example 1.
    '''
    invardefs = [
            ('x0', make_bell_mfs(2.5, 2, [1, 3,6])),
            ('x1', make_bell_mfs(2.5, 2, [1, 4,6])),
            ('x2', make_bell_mfs(2.5, 2, [1, 5,6])),
            ]
    outvars = ['y0']
    anf = anfis.AnfisNet('sin + cos + tan', invardefs, outvars,False)
    return anf

if __name__ == '__main__':
    show_plots = True
    if len(sys.argv) == 2:  # One arg: example
        example = sys.argv[1]
        show_plots = False

    model = ex1_model()

    ##initial membership parameters for xo and mf0
    print(model.layer['fuzzify'].varmfs['x0'].mfdefs['mf0'].pretty())
#    b1 = model.layer['fuzzify'].varmfs['x0'].mfdefs['mf0'].b
    ######################################

    train_data = training_data()
    #print(*train_data)
    train_anfis(model, train_data, 100, show_plots)

    ##tuned membership parameters for x0 and mf0
    print(model.layer['fuzzify'].varmfs['x0'].mfdefs['mf0'].pretty())

    ############################################sav and load
    torch.save(model, 'anfis_model.npy')
    x = torch.tensor([[5., 5., 1.]])
    print(model(x))
    pt_model = torch.load('anfis_model.npy')
    print(pt_model(x))
    ##############################################

#    print(model.layer['rules'].mf_indices)      ##print the rule bases you got
#    print(model.layer['rules'].num_rules())     ##print the number of rule bases
    test_data = testing_data()
    test_anfis(model, test_data, show_plots)
#    print(model.layer['consequent']._coeff)     ##print final coeffs in cnsequent layer.
