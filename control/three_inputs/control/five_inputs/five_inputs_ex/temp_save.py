#    input1 = (torch.mean(x[:,0])).item()
#    input2 = (torch.mean(x[:,1])).item()
#    input3 = (torch.mean(x[:,2])).item()

if t == 5:
    with torch.no_grad():
        print('hi')
        model.layer['fuzzify'].varmfs['theta_near'].mfdefs['mf2'].b.copy_((torch.tensor(-0.09,dtype=torch.float)))
        model.layer['fuzzify'].varmfs['theta_near'].mfdefs['mf2'].b.requires_grad = False

    invardefs = [
            ('distance_line', [TrapezoidalMembFunc(-20,-20,-1.425,-0.825,1),
                               TrapezoidalMembFunc(-1.425,-0.825,-0.675,-0.075,2),
                               TrapezoidalMembFunc(-0.675,-0.075,0.075,0.675,3),
                               TrapezoidalMembFunc(0.075, 0.675, 0.825, 1.425,4),
                               TrapezoidalMembFunc(0.825, 1.425, 20, 20,5),
                               Zero()]),
            ('theta_far', [TrapezoidalMembFunc(-3.15, -3.15, -2.375, -1.375,1),
                           TrapezoidalMembFunc(-2.375, -1.375, -1.125, -0.125,2),
                           TrapezoidalMembFunc(-1.125, -0.125, 0.125, 1.125,3),
                           TrapezoidalMembFunc(0.125, 1.125, 1.375, 2.375,4),
                           TrapezoidalMembFunc(1.375, 2.375, 3.15, 3.15,5),
                           Zero()]),
            ('theta_near', [TrapezoidalMembFunc(-3.15, -3.15, -1.71, -0.99,1),
                            TrapezoidalMembFunc(-1.71, -0.99, -0.81, -0.09,2),
                            TrapezoidalMembFunc(-0.81, -0.09, 0.09, 0.81,3),
                            TrapezoidalMembFunc(0.09, 0.81, 0.99, 1.71,4),
                            TrapezoidalMembFunc(0.99, 1.71, 3.15, 3.15,5),
                            Zero()])
                                                                ]
    def __init__(self, a, b, c, d, constraint=True):
        super(TrapezoidalMembFunc, self).__init__()
        assert a <= b and b <= c and c <= d,\
            'Trapezoidal parameters: must have a <= b <= c <= d.'
        if constraint == 1:
            self.a = a
            self.b = b
            self.c = c
            self.d = d
        elif constraint == 2:
            self.register_parameter('a', _mk_param(a))
            self.register_parameter('b', _mk_param(b))
            self.c = c
            self.d = d
        elif constraint == 3:
            self.register_parameter('a', _mk_param(a))
            self.register_parameter('b', _mk_param(b))
            self.register_parameter('c', _mk_param(c))
            self.register_parameter('d', _mk_param(d))
        elif constraint == 4:
            self.a = a
            self.b = b
            self.register_parameter('c', _mk_param(c))
            self.register_parameter('d', _mk_param(d))
        elif constraint == 5:
            self.a = a
            self.b = b
            self.c = c
            self.d = d

def mfs_constraint(model,x):
    #print(x)
    input1 = (torch.mean(x[:,0])).item()
    input2 = (torch.mean(x[:,1])).item()
    input3 = (torch.mean(x[:,2])).item()

    model.layer['fuzzify'].varmfs['distance_line'].mfdefs['mf0'].c = model.layer['fuzzify'].varmfs['distance_line'].mfdefs['mf1'].a.item()
    model.layer['fuzzify'].varmfs['distance_line'].mfdefs['mf0'].d = model.layer['fuzzify'].varmfs['distance_line'].mfdefs['mf1'].b.item()
    model.layer['fuzzify'].varmfs['distance_line'].mfdefs['mf1'].c = model.layer['fuzzify'].varmfs['distance_line'].mfdefs['mf2'].a.item()
    model.layer['fuzzify'].varmfs['distance_line'].mfdefs['mf1'].d = model.layer['fuzzify'].varmfs['distance_line'].mfdefs['mf2'].b.item()
    model.layer['fuzzify'].varmfs['distance_line'].mfdefs['mf3'].a = model.layer['fuzzify'].varmfs['distance_line'].mfdefs['mf2'].c.item()
    model.layer['fuzzify'].varmfs['distance_line'].mfdefs['mf3'].b = model.layer['fuzzify'].varmfs['distance_line'].mfdefs['mf2'].d.item()
    model.layer['fuzzify'].varmfs['distance_line'].mfdefs['mf4'].a = model.layer['fuzzify'].varmfs['distance_line'].mfdefs['mf3'].c.item()
    model.layer['fuzzify'].varmfs['distance_line'].mfdefs['mf4'].b = model.layer['fuzzify'].varmfs['distance_line'].mfdefs['mf3'].d.item()

    model.layer['fuzzify'].varmfs['theta_far'].mfdefs['mf0'].c = model.layer['fuzzify'].varmfs['theta_far'].mfdefs['mf1'].a.item()
    model.layer['fuzzify'].varmfs['theta_far'].mfdefs['mf0'].d = model.layer['fuzzify'].varmfs['theta_far'].mfdefs['mf1'].b.item()
    model.layer['fuzzify'].varmfs['theta_far'].mfdefs['mf1'].c = model.layer['fuzzify'].varmfs['theta_far'].mfdefs['mf2'].a.item()
    model.layer['fuzzify'].varmfs['theta_far'].mfdefs['mf1'].d = model.layer['fuzzify'].varmfs['theta_far'].mfdefs['mf2'].b.item()
    model.layer['fuzzify'].varmfs['theta_far'].mfdefs['mf3'].a = model.layer['fuzzify'].varmfs['theta_far'].mfdefs['mf2'].c.item()
    model.layer['fuzzify'].varmfs['theta_far'].mfdefs['mf3'].b = model.layer['fuzzify'].varmfs['theta_far'].mfdefs['mf2'].d.item()
    model.layer['fuzzify'].varmfs['theta_far'].mfdefs['mf4'].a = model.layer['fuzzify'].varmfs['theta_far'].mfdefs['mf3'].c.item()
    model.layer['fuzzify'].varmfs['theta_far'].mfdefs['mf4'].b = model.layer['fuzzify'].varmfs['theta_far'].mfdefs['mf3'].d.item()

    if model.layer['fuzzify'].varmfs['theta_near'].mfdefs['mf2'].b.item() > -0.01:
        model.layer['fuzzify'].varmfs['theta_near'].mfdefs['mf2'].b = torch.nn.Parameter(torch.tensor(-0.01))
    if model.layer['fuzzify'].varmfs['theta_near'].mfdefs['mf2'].c.item() < 0.01:
        model.layer['fuzzify'].varmfs['theta_near'].mfdefs['mf2'].c = torch.nn.Parameter(torch.tensor(0.01))
#    model.layer['fuzzify'].varmfs['theta_near'].mfdefs['mf0'].c = model.layer['fuzzify'].varmfs['theta_near'].mfdefs['mf1'].a.item()
    print(model.layer['fuzzify'].varmfs['theta_near'].mfdefs['mf0'].c)
    model.layer['fuzzify'].varmfs['theta_near'].mfdefs['mf0'].d = model.layer['fuzzify'].varmfs['theta_near'].mfdefs['mf1'].b.item()
    model.layer['fuzzify'].varmfs['theta_near'].mfdefs['mf1'].c = model.layer['fuzzify'].varmfs['theta_near'].mfdefs['mf2'].a.item()
    model.layer['fuzzify'].varmfs['theta_near'].mfdefs['mf1'].d = model.layer['fuzzify'].varmfs['theta_near'].mfdefs['mf2'].b.item()
    model.layer['fuzzify'].varmfs['theta_near'].mfdefs['mf3'].a = model.layer['fuzzify'].varmfs['theta_near'].mfdefs['mf2'].c.item()
    model.layer['fuzzify'].varmfs['theta_near'].mfdefs['mf3'].b = model.layer['fuzzify'].varmfs['theta_near'].mfdefs['mf2'].d.item()
    model.layer['fuzzify'].varmfs['theta_near'].mfdefs['mf4'].a = model.layer['fuzzify'].varmfs['theta_near'].mfdefs['mf3'].c.item()
    model.layer['fuzzify'].varmfs['theta_near'].mfdefs['mf4'].b = model.layer['fuzzify'].varmfs['theta_near'].mfdefs['mf3'].d.item()





def mfs_constraint(model,x):
    #print(x)
    input1 = (torch.mean(x[:,0])).item()
    input2 = (torch.mean(x[:,1])).item()
    input3 = (torch.mean(x[:,2])).item()
    if input1 < 0:
    #    print('input1<')
    #    print(model.layer['fuzzify'].varmfs['distance_line'].mfdefs['mf0'].c)
    #    print(model.layer['fuzzify'].varmfs['distance_line'].mfdefs['mf1'].a.item())
        model.layer['fuzzify'].varmfs['distance_line'].mfdefs['mf0'].c = torch.nn.Parameter(torch.tensor(model.layer['fuzzify'].varmfs['distance_line'].mfdefs['mf1'].a.item()))
        model.layer['fuzzify'].varmfs['distance_line'].mfdefs['mf0'].d = torch.nn.Parameter(torch.tensor(model.layer['fuzzify'].varmfs['distance_line'].mfdefs['mf1'].b.item()))
        model.layer['fuzzify'].varmfs['distance_line'].mfdefs['mf1'].c = torch.nn.Parameter(torch.tensor(model.layer['fuzzify'].varmfs['distance_line'].mfdefs['mf2'].a.item()))
        model.layer['fuzzify'].varmfs['distance_line'].mfdefs['mf1'].d = torch.nn.Parameter(torch.tensor(model.layer['fuzzify'].varmfs['distance_line'].mfdefs['mf2'].b.item()))
        model.layer['fuzzify'].varmfs['distance_line'].mfdefs['mf2'].c = torch.nn.Parameter(torch.tensor(-1. * model.layer['fuzzify'].varmfs['distance_line'].mfdefs['mf2'].a.item()))
        model.layer['fuzzify'].varmfs['distance_line'].mfdefs['mf2'].d = torch.nn.Parameter(torch.tensor(-1. * model.layer['fuzzify'].varmfs['distance_line'].mfdefs['mf2'].b.item()))
        model.layer['fuzzify'].varmfs['distance_line'].mfdefs['mf3'].a = torch.nn.Parameter(torch.tensor(-1. * model.layer['fuzzify'].varmfs['distance_line'].mfdefs['mf2'].b.item()))
        model.layer['fuzzify'].varmfs['distance_line'].mfdefs['mf3'].b = torch.nn.Parameter(torch.tensor(-1. * model.layer['fuzzify'].varmfs['distance_line'].mfdefs['mf2'].a.item()))
        model.layer['fuzzify'].varmfs['distance_line'].mfdefs['mf3'].c = torch.nn.Parameter(torch.tensor(-1. * model.layer['fuzzify'].varmfs['distance_line'].mfdefs['mf1'].b.item()))
        model.layer['fuzzify'].varmfs['distance_line'].mfdefs['mf3'].d = torch.nn.Parameter(torch.tensor(-1. * model.layer['fuzzify'].varmfs['distance_line'].mfdefs['mf1'].a.item()))
        model.layer['fuzzify'].varmfs['distance_line'].mfdefs['mf4'].a = torch.nn.Parameter(torch.tensor(-1. * model.layer['fuzzify'].varmfs['distance_line'].mfdefs['mf1'].b.item()))
        model.layer['fuzzify'].varmfs['distance_line'].mfdefs['mf4'].b = torch.nn.Parameter(torch.tensor(-1. * model.layer['fuzzify'].varmfs['distance_line'].mfdefs['mf1'].a.item()))

    elif input1 > 0:
    #    print('input1>')
    #    print(model.layer['fuzzify'].varmfs['distance_line'].mfdefs['mf0'].c)
    #    print(torch.nn.Parameter(torch.tensor(-0.01 * model.layer['fuzzify'].varmfs['distance_line'].mfdefs['mf3'].d.item())))
        model.layer['fuzzify'].varmfs['distance_line'].mfdefs['mf0'].c = torch.nn.Parameter(torch.tensor(-1. * model.layer['fuzzify'].varmfs['distance_line'].mfdefs['mf3'].d.item()))
        model.layer['fuzzify'].varmfs['distance_line'].mfdefs['mf0'].d = torch.nn.Parameter(torch.tensor(-1. * model.layer['fuzzify'].varmfs['distance_line'].mfdefs['mf3'].c.item()))
        model.layer['fuzzify'].varmfs['distance_line'].mfdefs['mf1'].a = torch.nn.Parameter(torch.tensor(-1. * model.layer['fuzzify'].varmfs['distance_line'].mfdefs['mf3'].d.item()))
        model.layer['fuzzify'].varmfs['distance_line'].mfdefs['mf1'].b = torch.nn.Parameter(torch.tensor(-1. * model.layer['fuzzify'].varmfs['distance_line'].mfdefs['mf3'].c.item()))
        model.layer['fuzzify'].varmfs['distance_line'].mfdefs['mf1'].c = torch.nn.Parameter(torch.tensor(-1. * model.layer['fuzzify'].varmfs['distance_line'].mfdefs['mf2'].d.item()))
        model.layer['fuzzify'].varmfs['distance_line'].mfdefs['mf1'].d = torch.nn.Parameter(torch.tensor(-1. * model.layer['fuzzify'].varmfs['distance_line'].mfdefs['mf2'].c.item()))
        model.layer['fuzzify'].varmfs['distance_line'].mfdefs['mf2'].a = torch.nn.Parameter(torch.tensor(-1. * model.layer['fuzzify'].varmfs['distance_line'].mfdefs['mf2'].d.item()))
        model.layer['fuzzify'].varmfs['distance_line'].mfdefs['mf2'].b = torch.nn.Parameter(torch.tensor(-1. * model.layer['fuzzify'].varmfs['distance_line'].mfdefs['mf2'].c.item()))
        model.layer['fuzzify'].varmfs['distance_line'].mfdefs['mf3'].a = torch.nn.Parameter(torch.tensor(model.layer['fuzzify'].varmfs['distance_line'].mfdefs['mf2'].c.item()))
        model.layer['fuzzify'].varmfs['distance_line'].mfdefs['mf3'].b = torch.nn.Parameter(torch.tensor(model.layer['fuzzify'].varmfs['distance_line'].mfdefs['mf2'].d.item()))
        model.layer['fuzzify'].varmfs['distance_line'].mfdefs['mf4'].a = torch.nn.Parameter(torch.tensor(model.layer['fuzzify'].varmfs['distance_line'].mfdefs['mf3'].c.item()))
        model.layer['fuzzify'].varmfs['distance_line'].mfdefs['mf4'].b = torch.nn.Parameter(torch.tensor(model.layer['fuzzify'].varmfs['distance_line'].mfdefs['mf3'].d.item()))


    if input2 < 0:
    #    print('input2<')
    #    print(model.layer['fuzzify'].varmfs['theta_far'].mfdefs['mf0'].c)
    #    print(model.layer['fuzzify'].varmfs['theta_far'].mfdefs['mf1'].a.item())
        model.layer['fuzzify'].varmfs['theta_far'].mfdefs['mf0'].c = torch.nn.Parameter(torch.tensor(model.layer['fuzzify'].varmfs['theta_far'].mfdefs['mf1'].a.item()))
        model.layer['fuzzify'].varmfs['theta_far'].mfdefs['mf0'].d = torch.nn.Parameter(torch.tensor(model.layer['fuzzify'].varmfs['theta_far'].mfdefs['mf1'].b.item()))
        model.layer['fuzzify'].varmfs['theta_far'].mfdefs['mf1'].c = torch.nn.Parameter(torch.tensor(model.layer['fuzzify'].varmfs['theta_far'].mfdefs['mf2'].a.item()))
        model.layer['fuzzify'].varmfs['theta_far'].mfdefs['mf1'].d = torch.nn.Parameter(torch.tensor(model.layer['fuzzify'].varmfs['theta_far'].mfdefs['mf2'].b.item()))
        model.layer['fuzzify'].varmfs['theta_far'].mfdefs['mf2'].c = torch.nn.Parameter(torch.tensor(-1. * model.layer['fuzzify'].varmfs['theta_far'].mfdefs['mf2'].a.item()))
        model.layer['fuzzify'].varmfs['theta_far'].mfdefs['mf2'].d = torch.nn.Parameter(torch.tensor(-1. * model.layer['fuzzify'].varmfs['theta_far'].mfdefs['mf2'].b.item()))
        model.layer['fuzzify'].varmfs['theta_far'].mfdefs['mf3'].a = torch.nn.Parameter(torch.tensor(-1. * model.layer['fuzzify'].varmfs['theta_far'].mfdefs['mf2'].b.item()))
        model.layer['fuzzify'].varmfs['theta_far'].mfdefs['mf3'].b = torch.nn.Parameter(torch.tensor(-1. * model.layer['fuzzify'].varmfs['theta_far'].mfdefs['mf2'].a.item()))
        model.layer['fuzzify'].varmfs['theta_far'].mfdefs['mf3'].c = torch.nn.Parameter(torch.tensor(-1. * model.layer['fuzzify'].varmfs['theta_far'].mfdefs['mf1'].b.item()))
        model.layer['fuzzify'].varmfs['theta_far'].mfdefs['mf3'].d = torch.nn.Parameter(torch.tensor(-1. * model.layer['fuzzify'].varmfs['theta_far'].mfdefs['mf1'].a.item()))
        model.layer['fuzzify'].varmfs['theta_far'].mfdefs['mf4'].a = torch.nn.Parameter(torch.tensor(-1. * model.layer['fuzzify'].varmfs['theta_far'].mfdefs['mf1'].b.item()))
        model.layer['fuzzify'].varmfs['theta_far'].mfdefs['mf4'].b = torch.nn.Parameter(torch.tensor(-1. * model.layer['fuzzify'].varmfs['theta_far'].mfdefs['mf1'].a.item()))

    elif input2 > 0:
    #    print('input2>')
    #    print(model.layer['fuzzify'].varmfs['theta_far'].mfdefs['mf0'].c)
    #    print(torch.nn.Parameter(torch.tensor(-0.01 * model.layer['fuzzify'].varmfs['theta_far'].mfdefs['mf3'].d.item())))
        model.layer['fuzzify'].varmfs['theta_far'].mfdefs['mf0'].c = torch.nn.Parameter(torch.tensor(-1. * model.layer['fuzzify'].varmfs['theta_far'].mfdefs['mf3'].d.item()))
        model.layer['fuzzify'].varmfs['theta_far'].mfdefs['mf0'].d = torch.nn.Parameter(torch.tensor(-1. * model.layer['fuzzify'].varmfs['theta_far'].mfdefs['mf3'].c.item()))
        model.layer['fuzzify'].varmfs['theta_far'].mfdefs['mf1'].a = torch.nn.Parameter(torch.tensor(-1. * model.layer['fuzzify'].varmfs['theta_far'].mfdefs['mf3'].d.item()))
        model.layer['fuzzify'].varmfs['theta_far'].mfdefs['mf1'].b = torch.nn.Parameter(torch.tensor(-1. * model.layer['fuzzify'].varmfs['theta_far'].mfdefs['mf3'].c.item()))
        model.layer['fuzzify'].varmfs['theta_far'].mfdefs['mf1'].c = torch.nn.Parameter(torch.tensor(-1. * model.layer['fuzzify'].varmfs['theta_far'].mfdefs['mf2'].d.item()))
        model.layer['fuzzify'].varmfs['theta_far'].mfdefs['mf1'].d = torch.nn.Parameter(torch.tensor(-1. * model.layer['fuzzify'].varmfs['theta_far'].mfdefs['mf2'].c.item()))
        model.layer['fuzzify'].varmfs['theta_far'].mfdefs['mf2'].a = torch.nn.Parameter(torch.tensor(-1. * model.layer['fuzzify'].varmfs['theta_far'].mfdefs['mf2'].d.item()))
        model.layer['fuzzify'].varmfs['theta_far'].mfdefs['mf2'].b = torch.nn.Parameter(torch.tensor(-1. * model.layer['fuzzify'].varmfs['theta_far'].mfdefs['mf2'].c.item()))
        model.layer['fuzzify'].varmfs['theta_far'].mfdefs['mf3'].a = torch.nn.Parameter(torch.tensor(model.layer['fuzzify'].varmfs['theta_far'].mfdefs['mf2'].c.item()))
        model.layer['fuzzify'].varmfs['theta_far'].mfdefs['mf3'].b = torch.nn.Parameter(torch.tensor(model.layer['fuzzify'].varmfs['theta_far'].mfdefs['mf2'].d.item()))
        model.layer['fuzzify'].varmfs['theta_far'].mfdefs['mf4'].a = torch.nn.Parameter(torch.tensor(model.layer['fuzzify'].varmfs['theta_far'].mfdefs['mf3'].c.item()))
        model.layer['fuzzify'].varmfs['theta_far'].mfdefs['mf4'].b = torch.nn.Parameter(torch.tensor(model.layer['fuzzify'].varmfs['theta_far'].mfdefs['mf3'].d.item()))


    if model.layer['fuzzify'].varmfs['theta_near'].mfdefs['mf2'].b.item() > -0.01:
        model.layer['fuzzify'].varmfs['theta_near'].mfdefs['mf2'].b = torch.nn.Parameter(torch.tensor(-0.01))
    if model.layer['fuzzify'].varmfs['theta_near'].mfdefs['mf2'].c.item() < 0.01:
        model.layer['fuzzify'].varmfs['theta_near'].mfdefs['mf2'].c = torch.nn.Parameter(torch.tensor(0.01))


    if input3 < 0:
    #    print('input3<')
    #    print(model.layer['fuzzify'].varmfs['theta_near'].mfdefs['mf0'].c)
    #    print(model.layer['fuzzify'].varmfs['theta_near'].mfdefs['mf1'].a.item())
        model.layer['fuzzify'].varmfs['theta_near'].mfdefs['mf0'].c = torch.nn.Parameter(torch.tensor(model.layer['fuzzify'].varmfs['theta_near'].mfdefs['mf1'].a.item()))
        model.layer['fuzzify'].varmfs['theta_near'].mfdefs['mf0'].d = torch.nn.Parameter(torch.tensor(model.layer['fuzzify'].varmfs['theta_near'].mfdefs['mf1'].b.item()))
        model.layer['fuzzify'].varmfs['theta_near'].mfdefs['mf1'].c = torch.nn.Parameter(torch.tensor(model.layer['fuzzify'].varmfs['theta_near'].mfdefs['mf2'].a.item()))
        model.layer['fuzzify'].varmfs['theta_near'].mfdefs['mf1'].d = torch.nn.Parameter(torch.tensor(model.layer['fuzzify'].varmfs['theta_near'].mfdefs['mf2'].b.item()))
        model.layer['fuzzify'].varmfs['theta_near'].mfdefs['mf2'].c = torch.nn.Parameter(torch.tensor(-1. * model.layer['fuzzify'].varmfs['theta_near'].mfdefs['mf2'].a.item()))
        model.layer['fuzzify'].varmfs['theta_near'].mfdefs['mf2'].d = torch.nn.Parameter(torch.tensor(-1. * model.layer['fuzzify'].varmfs['theta_near'].mfdefs['mf2'].b.item()))
        model.layer['fuzzify'].varmfs['theta_near'].mfdefs['mf3'].a = torch.nn.Parameter(torch.tensor(-1. * model.layer['fuzzify'].varmfs['theta_near'].mfdefs['mf2'].b.item()))
        model.layer['fuzzify'].varmfs['theta_near'].mfdefs['mf3'].b = torch.nn.Parameter(torch.tensor(-1. * model.layer['fuzzify'].varmfs['theta_near'].mfdefs['mf2'].a.item()))
        model.layer['fuzzify'].varmfs['theta_near'].mfdefs['mf3'].c = torch.nn.Parameter(torch.tensor(-1. * model.layer['fuzzify'].varmfs['theta_near'].mfdefs['mf1'].b.item()))
        model.layer['fuzzify'].varmfs['theta_near'].mfdefs['mf3'].d = torch.nn.Parameter(torch.tensor(-1. * model.layer['fuzzify'].varmfs['theta_near'].mfdefs['mf1'].a.item()))
        model.layer['fuzzify'].varmfs['theta_near'].mfdefs['mf4'].a = torch.nn.Parameter(torch.tensor(-1. * model.layer['fuzzify'].varmfs['theta_near'].mfdefs['mf1'].b.item()))
        model.layer['fuzzify'].varmfs['theta_near'].mfdefs['mf4'].b = torch.nn.Parameter(torch.tensor(-1. * model.layer['fuzzify'].varmfs['theta_near'].mfdefs['mf1'].a.item()))
    elif input3 > 0:
    #    print('input3>')
    #    print(model.layer['fuzzify'].varmfs['theta_near'].mfdefs['mf0'].c)
    #    print(torch.nn.Parameter(torch.tensor(-0.01 * model.layer['fuzzify'].varmfs['theta_near'].mfdefs['mf3'].d.item())))
        model.layer['fuzzify'].varmfs['theta_near'].mfdefs['mf0'].c = torch.nn.Parameter(torch.tensor(-1. * model.layer['fuzzify'].varmfs['theta_near'].mfdefs['mf3'].d.item()))
        model.layer['fuzzify'].varmfs['theta_near'].mfdefs['mf0'].d = torch.nn.Parameter(torch.tensor(-1. * model.layer['fuzzify'].varmfs['theta_near'].mfdefs['mf3'].c.item()))
        model.layer['fuzzify'].varmfs['theta_near'].mfdefs['mf1'].a = torch.nn.Parameter(torch.tensor(-1. * model.layer['fuzzify'].varmfs['theta_near'].mfdefs['mf3'].d.item()))
        model.layer['fuzzify'].varmfs['theta_near'].mfdefs['mf1'].b = torch.nn.Parameter(torch.tensor(-1. * model.layer['fuzzify'].varmfs['theta_near'].mfdefs['mf3'].c.item()))
        model.layer['fuzzify'].varmfs['theta_near'].mfdefs['mf1'].c = torch.nn.Parameter(torch.tensor(-1. * model.layer['fuzzify'].varmfs['theta_near'].mfdefs['mf2'].d.item()))
        model.layer['fuzzify'].varmfs['theta_near'].mfdefs['mf1'].d = torch.nn.Parameter(torch.tensor(-1. * model.layer['fuzzify'].varmfs['theta_near'].mfdefs['mf2'].c.item()))
        model.layer['fuzzify'].varmfs['theta_near'].mfdefs['mf2'].a = torch.nn.Parameter(torch.tensor(-1. * model.layer['fuzzify'].varmfs['theta_near'].mfdefs['mf2'].d.item()))
        model.layer['fuzzify'].varmfs['theta_near'].mfdefs['mf2'].b = torch.nn.Parameter(torch.tensor(-1. * model.layer['fuzzify'].varmfs['theta_near'].mfdefs['mf2'].c.item()))
        model.layer['fuzzify'].varmfs['theta_near'].mfdefs['mf3'].a = torch.nn.Parameter(torch.tensor(model.layer['fuzzify'].varmfs['theta_near'].mfdefs['mf2'].c.item()))
        model.layer['fuzzify'].varmfs['theta_near'].mfdefs['mf3'].b = torch.nn.Parameter(torch.tensor(model.layer['fuzzify'].varmfs['theta_near'].mfdefs['mf2'].d.item()))
        model.layer['fuzzify'].varmfs['theta_near'].mfdefs['mf4'].a = torch.nn.Parameter(torch.tensor(model.layer['fuzzify'].varmfs['theta_near'].mfdefs['mf3'].c.item()))
        model.layer['fuzzify'].varmfs['theta_near'].mfdefs['mf4'].b = torch.nn.Parameter(torch.tensor(model.layer['fuzzify'].varmfs['theta_near'].mfdefs['mf3'].d.item()))




    #model.layer['fuzzify'].varmfs['theta_near'].mfdefs['mf2'].b = -1. * model.layer['fuzzify'].varmfs['theta_near'].mfdefs['mf2'].a.item()
