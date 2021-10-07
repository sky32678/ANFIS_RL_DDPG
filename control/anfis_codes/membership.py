#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
    ANFIS in torch: some fuzzy membership functions.
    @author: James Power <james.power@mu.ie> Apr 12 18:13:10 2019
"""
from abc import ABC, abstractmethod
from functools import partial
import torch

from anfis_codes.anfis import AnfisNet

class JointMamdaniMembership(torch.nn.Module, ABC):
    @abstractmethod
    def __getitem__(self, item):
        pass

    @abstractmethod
    def cache(self):
        pass

    @abstractmethod
    def release_cache(self):
        pass


class JointSymmetricTriangleMembership(JointMamdaniMembership):

    def __getitem__(self, item):
        return self.cache_output_values[item]

    def cache(self):
        self.abs_cache['center'] = self.center
        self.abs_cache['soft'] = torch.abs(self.soft)
        self.abs_cache['normal'] = torch.abs(self.normal)
        self.abs_cache['hard'] = torch.abs(self.hard)

        for key, val in self.output_function.items():
            self.cache_output_values[key] = val()

    def release_cache(self):
        self.abs_cache.clear()
        self.cache_output_values.clear()

    def get_center(self):
        return self.abs_cache['center']

    def get_soft(self, direction=1):
        return self.abs_cache['center'] + direction * self.abs_cache['soft']

    def get_normal(self, direction=1):
        return self.abs_cache['center'] + direction * (self.abs_cache['soft'] + self.abs_cache['normal'])

    def get_hard(self, direction=1):
        return self.center + direction * (self.abs_cache['soft'] + self.abs_cache['normal'] + self.abs_cache['hard'])

    def __init__(self, center, soft, normal, constant_center=True, dtype=torch.float) -> None :
        super().__init__()
        constant_center=True
        dtype=torch.float
        if constant_center:
            self.center = torch.tensor(center, dtype=dtype, requires_grad=False)
        else:
            self.register_parameter('center', _mk_param(center, dtype=dtype))

        self.register_parameter('soft', _mk_param(soft, dtype=dtype))
        self.register_parameter('normal', _mk_param(normal, dtype=dtype))
        self.register_parameter('hard', _mk_param(hard, dtype=dtype))

        self.abs_cache = dict()

        self.output_function = {
            0: partial(self.get_hard, direction=1),
            1: partial(self.get_normal, direction=1),
            2: partial(self.get_soft, direction=1),
            3: self.get_center,
            4: partial(self.get_soft, direction=-1),
            5: partial(self.get_normal, direction=-1),
            6: partial(self.get_hard, direction=-1),
        }

        self.names = {
            0: 'Hard Left',
            1: 'Left',
            2: 'Soft Left',
            3: 'Zero',
            4: 'Soft Right',
            5: 'Right',
            6: 'Hard Right',
        }

        self.cache_output_values = dict()

class JointSymmetric9TriangleMembership(JointMamdaniMembership):

    def __getitem__(self, item):
        return self.cache_output_values[item]

    def cache(self):
        self.abs_cache['center'] = self.center
        self.abs_cache['soft'] = torch.abs(self.soft)
        self.abs_cache['normal'] = torch.abs(self.normal)
        self.abs_cache['hard'] = torch.abs(self.hard)
        self.abs_cache['very_hard'] = torch.abs(self.very_hard)

        for key, val in self.output_function.items():
            self.cache_output_values[key] = val()

    def release_cache(self):
        self.abs_cache.clear()
        self.cache_output_values.clear()

    def get_center(self):
        return self.abs_cache['center']

    def get_soft(self, direction=1):
        return self.abs_cache['center'] + direction * self.abs_cache['soft']

    def get_normal(self, direction=1):
        return self.abs_cache['center'] + direction * (self.abs_cache['soft'] + self.abs_cache['normal'])

    def get_hard(self, direction=1):
        return self.center + direction * (self.abs_cache['soft'] + self.abs_cache['normal'] + self.abs_cache['hard'])

    def get_very_hard(self, direction=1):
        return self.center + direction * (
                self.abs_cache['soft'] + self.abs_cache['normal'] + self.abs_cache['hard'] +
                self.abs_cache['very_hard'])

    def __init__(self, center, soft, normal, hard, very_hard, constant_center=True, dtype=torch.float) -> None:
        super().__init__()

        if constant_center:
            self.center = torch.tensor(center, dtype=dtype, requires_grad=False)
        else:
            self.register_parameter('center', _mk_param(center))

        self.register_parameter('soft', _mk_param(soft))
        self.register_parameter('normal', _mk_param(normal))
        self.register_parameter('hard', _mk_param(hard))
        self.register_parameter('very_hard', _mk_param(very_hard))

        self.abs_cache = dict()

        self.output_function = {
            0: partial(self.get_very_hard, direction=1),
            1: partial(self.get_hard, direction=1),
            2: partial(self.get_normal, direction=1),
            3: partial(self.get_soft, direction=1),
            4: self.get_center,
            5: partial(self.get_soft, direction=-1),
            6: partial(self.get_normal, direction=-1),
            7: partial(self.get_hard, direction=-1),
            8: partial(self.get_very_hard, direction=-1),
        }

        self.names = {
            0: 'Very Hard Left',
            1: 'Hard Left',
            2: 'Left',
            3: 'Soft Left',
            4: 'Zero',
            5: 'Soft Right',
            6: 'Right',
            7: 'Hard Right',
            8: 'Very Hard Right',
        }

        self.cache_output_values = dict()

########################3

class Zero(torch.nn.Module):
    '''
        this is for NONE feature, it would be the last membership function
        that outputs 1 to use NONE feature for each rule base (And)
    '''
    def __init__(self):
        super(Zero, self).__init__()

    def forward(self, x):
        yvals = torch.ones_like(x)
        return yvals
def make_zero():
    return Zero()



def _mk_param(val):
    '''Make a torch parameter from a scalar value'''
    if isinstance(val, torch.Tensor):
        val = val.item()
    return torch.nn.Parameter(torch.tensor(val, dtype=torch.float))


class GaussMembFunc(torch.nn.Module):
    '''
        Gaussian membership functions, defined by two parameters:
            mu, the mean (center)
            sigma, the standard deviation.
    '''
    def __init__(self, mu, sigma):
        super(GaussMembFunc, self).__init__()
        self.register_parameter('mu', _mk_param(mu))
        self.register_parameter('sigma', _mk_param(sigma))

    def forward(self, x):
        val = torch.exp(-torch.pow(x - self.mu, 2) / (2 * self.sigma**2))
        return val

    def pretty(self):
        return 'GaussMembFunc {} {}'.format(self.mu, self.sigma)


def make_gauss_mfs(sigma, mu_list):
    '''Return a list of gaussian mfs, same sigma, list of means'''
    return [GaussMembFunc(mu, sigma) for mu in mu_list]


class BellMembFunc(torch.nn.Module):
    '''
        Generalised Bell membership function; defined by three parameters:
            a, the half-width (at the crossover point)
            b, controls the slope at the crossover point (which is -b/2a)
            c, the center point
    '''
    def __init__(self, a, b, c):
        super(BellMembFunc, self).__init__()
        self.register_parameter('a', _mk_param(a))
        self.register_parameter('b', _mk_param(b))
        self.register_parameter('c', _mk_param(c))
        self.b.register_hook(BellMembFunc.b_log_hook)

    @staticmethod
    def b_log_hook(grad):
        '''
            Possibility of a log(0) in the grad for b, giving a nan.
            Fix this by replacing any nan in the grad with ~0.
        '''
        grad[torch.isnan(grad)] = 1e-9
        return grad

    def forward(self, x):
        dist = torch.pow((x - self.c)/self.a, 2)
        return torch.reciprocal(1 + torch.pow(dist, self.b))

    def pretty(self):
        return 'BellMembFunc {} {} {}'.format(self.a, self.b, self.c)


def make_bell_mfs(a, b, clist):
    '''Return a list of bell mfs, same (a,b), list of centers'''
    temp = [BellMembFunc(a, b, c) for c in clist]
    temp.append(Zero())
    return temp
    #return [BellMembFunc(a, b, c) for c in clist]


class TriangularMembFunc(torch.nn.Module):
    '''
        Triangular membership function; defined by three parameters:
            a, left foot, mu(x) = 0
            b, midpoint, mu(x) = 1
            c, right foot, mu(x) = 0
    '''
    def __init__(self, a, b, c):
        super(TriangularMembFunc, self).__init__()
        assert a <= b and b <= c,\
            'Triangular parameters: must have a <= b <= c.'
        self.register_parameter('a', _mk_param(a))
        self.register_parameter('b', _mk_param(b))
        self.register_parameter('c', _mk_param(c))

    @staticmethod
    def isosceles(width, center):
        '''
            Construct a triangle MF with given width-of-base and center
        '''
        return TriangularMembFunc(center-width, center, center+width)

    def forward(self, x):
        return torch.where(
            torch.ByteTensor(self.a < x) & torch.ByteTensor(x <= self.b),
            (x - self.a) / (self.b - self.a),
            # else
            torch.where(
                torch.ByteTensor(self.b < x) & torch.ByteTensor(x <= self.c),
                (self.c - x) / (self.c - self.b),
                torch.zeros_like(x, requires_grad=True)))

    def pretty(self):
        return 'TriangularMembFunc {} {} {}'.format(self.a, self.b, self.c)


def make_tri_mfs(width, clist):
    '''Return a list of triangular mfs, same width, list of centers'''
    return [TriangularMembFunc(c-width/2, c, c+width/2) for c in clist]


class TrapezoidalMembFunc(torch.nn.Module):
    '''
        Trapezoidal membership function; defined by four parameters.
        Membership is defined as:
            to the left of a: always 0
            from a to b: slopes from 0 up to 1
            from b to c: always 1
            from c to d: slopes from 1 down to 0
            to the right of d: always 0
    '''
    def __init__(self, a, b, c, d, constraint=True, n_mfs = 5):
        super(TrapezoidalMembFunc, self).__init__()
        assert a <= b and b <= c and c <= d,\
            'Trapezoidal parameters: must have a <= b <= c <= d.'
        if n_mfs == 5:
            if constraint == 1:
                self.a = torch.tensor(a, requires_grad=False)
                self.b = torch.tensor(b, requires_grad=False)
                self.register_parameter('c', _mk_param(c))
                self.d = torch.tensor(d, requires_grad=False)
            elif constraint == 2:
                self.a = torch.tensor(a, requires_grad=False)
                self.register_parameter('b', _mk_param(b))
                self.register_parameter('c', _mk_param(c))
                self.d = torch.tensor(d, requires_grad=False)
            elif constraint == 3:
                self.a = torch.tensor(a, requires_grad=False)
                self.register_parameter('b', _mk_param(b))
                self.register_parameter('c', _mk_param(c))
                self.d = torch.tensor(d, requires_grad=False)
            elif constraint == 4:
                self.a = torch.tensor(a, requires_grad=False)
                self.register_parameter('b', _mk_param(b))
                self.register_parameter('c', _mk_param(c))
                self.d = torch.tensor(d, requires_grad=False)
            elif constraint == 5:
                self.a = torch.tensor(a, requires_grad=False)
                self.register_parameter('b', _mk_param(b))
                self.c = torch.tensor(c, requires_grad=False)
                self.d = torch.tensor(d, requires_grad=False)

        if n_mfs == 7:
            if constraint == 1:
                self.a = torch.tensor(a, requires_grad=False)
                self.b = torch.tensor(b, requires_grad=False)
                self.register_parameter('c', _mk_param(c))
                self.d = torch.tensor(d, requires_grad=False)
            elif constraint == 2:
                self.a = torch.tensor(a, requires_grad=False)
                self.register_parameter('b', _mk_param(b))
                self.register_parameter('c', _mk_param(c))
                self.d = torch.tensor(d, requires_grad=False)
            elif constraint == 3:
                self.a = torch.tensor(a, requires_grad=False)
                self.register_parameter('b', _mk_param(b))
                self.register_parameter('c', _mk_param(c))
                self.d = torch.tensor(d, requires_grad=False)
            elif constraint == 4:
                self.a = torch.tensor(a, requires_grad=False)
                self.register_parameter('b', _mk_param(b))
                self.register_parameter('c', _mk_param(c))
                self.d = torch.tensor(d, requires_grad=False)
            elif constraint == 5:
                self.a = torch.tensor(a, requires_grad=False)
                self.register_parameter('b', _mk_param(b))
                self.register_parameter('c', _mk_param(c))
                self.d = torch.tensor(d, requires_grad=False)
            elif constraint == 6:
                self.a = torch.tensor(a, requires_grad=False)
                self.register_parameter('b', _mk_param(b))
                self.register_parameter('c', _mk_param(c))
                self.d = torch.tensor(d, requires_grad=False)
            elif constraint == 7:
                self.a = torch.tensor(a, requires_grad=False)
                self.register_parameter('b', _mk_param(b))
                self.c = torch.tensor(c, requires_grad=False)
                self.d = torch.tensor(d, requires_grad=False)


        if n_mfs == 2:
            if constraint == 1:
                self.a = torch.tensor(a, requires_grad=False)
                self.b = torch.tensor(b, requires_grad=False)
                self.c = torch.tensor(c, requires_grad=False)
                self.register_parameter('d', _mk_param(d))
            elif constraint == 2:
                self.a = torch.tensor(a, requires_grad=False)
                self.register_parameter('b', _mk_param(b))
                self.c = torch.tensor(c, requires_grad=False)
                self.d = torch.tensor(d, requires_grad=False)


#        else:
#            self.register_parameter('a', _mk_param(a))
#            self.b = b
#            self.c = c
#            self.d = d


#    def __init__(self, a, b, c, d, constraint=True):
#        super(TrapezoidalMembFunc, self).__init__()
#        assert a <= b and b <= c and c <= d,\
#            'Trapezoidal parameters: must have a <= b <= c <= d.'
#        self.register_parameter('a', _mk_param(a))
#        if constraint:
#            self.register_parameter('b', _mk_param(b))
#            self.register_parameter('c', _mk_param(c))
#        else:
#            self.b = b
#            self.c = c
#        self.register_parameter('d', _mk_param(d))

    @staticmethod
    def symmetric(topwidth, slope, midpt):
        '''
            Make a (symmetric) trapezoid mf, given
                topwidth: length of top (when mu == 1)
                slope: extra length at either side for bottom
                midpt: center point of trapezoid
        '''
        b = midpt - topwidth / 2
        c = midpt + topwidth / 2
        return TrapezoidalMembFunc(b - slope, b, c, c + slope)

    @staticmethod
    def rectangle(left, right):
        '''
            Make a Trapezoidal MF with vertical sides (so a==b and c==d)
        '''
        return TrapezoidalMembFunc(left, left, right, right)

    @staticmethod
    def triangle(left, midpt, right):
        '''
            Make a triangle-shaped MF as a special case of a Trapezoidal MF.
            Note: this may revert to general trapezoid under learning.
        '''
        return TrapezoidalMembFunc(left, midpt, midpt, right)

    def forward(self, x):
        yvals = torch.zeros_like(x)
        if self.a < self.b:
        #    print((self.a < x).type(torch.ByteTensor) )
        #    print(torch.ByteTensor([True,1,0,0,1]) & torch.ByteTensor([True,1,1,1,1]))
            incr = (self.a < x) & (x <= self.b)
            yvals[incr] = (x[incr] - self.a) / (self.b - self.a)
        if self.b < self.c:
            decr = (self.b < x) & (x < self.c)
            yvals[decr] = 1
        if self.c < self.d:
            decr = (self.c <= x) & (x < self.d)
            yvals[decr] = (self.d - x[decr]) / (self.d - self.c)
#        if self.a < self.b:
#            print((self.a < x).type(torch.ByteTensor) )
#            print(torch.ByteTensor([True,1,0,0,1]) & torch.ByteTensor([True,1,1,1,1]))
#            incr = torch.ByteTensor((self.a < x)) & torch.ByteTensor((x <= self.b))
#            yvals[incr] = (x[incr] - self.a) / (self.b - self.a)
#        if self.b < self.c:
#            decr = torch.ByteTensor((self.b < x)) & torch.ByteTensor((x < self.c))
#            yvals[decr] = 1
#        if self.c < self.d:
#            decr = torch.ByteTensor((self.c <= x)) & torch.ByteTensor((x < self.d))
#            yvals[decr] = (self.d - x[decr]) / (self.d - self.c)
        return yvals

    def pretty(self):
        return 'TrapezoidalMembFunc {} {} {} {}'.\
            format(self.a, self.b, self.c, self.d)


def make_trap_mfs(width, slope, clist):
    '''Return a list of symmetric Trap mfs, same (w,s), list of centers'''
    temp = [TrapezoidalMembFunc.symmetric(width, slope, c) for c in clist]
    temp.append(Zero())
    return temp
#    return [TrapezoidalMembFunc.symmetric(width, slope, c) for c in clist]



# Make the classes available via (controlled) reflection:
get_class_for = {n: globals()[n]
                 for n in ['BellMembFunc',
                           'GaussMembFunc',
                           'TriangularMembFunc',
                           'TrapezoidalMembFunc',
                           ]}


def make_anfis(x, num_mfs=5, num_out=1, hybrid=True):
    '''
        Make an ANFIS model, auto-calculating the (Gaussian) MFs.
        I need the x-vals to calculate a range and spread for the MFs.
        Variables get named x0, x1, x2,... and y0, y1, y2 etc.
    '''
    num_invars = x.shape[1]
    minvals, _ = torch.min(x, dim=0)
    maxvals, _ = torch.max(x, dim=0)
    ranges = maxvals-minvals
    invars = []
    for i in range(num_invars):
        sigma = ranges[i] / num_mfs
        mulist = torch.linspace(minvals[i], maxvals[i], num_mfs).tolist()
        invars.append(('x{}'.format(i), make_gauss_mfs(sigma, mulist)))
    outvars = ['y{}'.format(i) for i in range(num_out)]
    model = AnfisNet('Simple classifier', invars, outvars, hybrid=hybrid)
    return model
