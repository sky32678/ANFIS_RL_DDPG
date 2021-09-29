import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd
import anfis_codes.anfis as anfis
from anfis_codes.membership import make_tri_mfs, TriangularMembFunc, TrapezoidalMembFunc, make_bell_mfs, BellMembFunc, make_trap_mfs, Zero, make_zero
from torch.autograd import Variable
from anfis_codes.antecedent_layer import JointTrapMembershipV2, JointSingleConstrainedEdgeMembership, Joint7TrapMembership
from anfis_codes.joint_mamdani_membership import JointSymmetricTriangleMembership,JointSymmetric9TriangleMembership
import numpy as np

def dist_target_dist_per_theta_lookahead_theta_far_theta_near():
    """
    0 dist target
    1 dist line
    2 theta lookahead
    3 theta far
    4 theta near
    :return:
    """

    # FIXME for rules with uneven indices number i.e sometimes 2 or sometimes 3, pad the lesser ones with duplicates
    variable_rule_index = [
        (0, 2, 0),
        (0, 2, 0),
        (0, 2, 0),

        (0, 1, 3),
        (0, 1, 3),
        (0, 1, 3),
        (0, 1, 3),
        (0, 1, 3),

        (0, 1, 4),
        (0, 1, 4),
        (0, 1, 4),
        (0, 1, 4),
        (0, 1, 4),

        (0, 1, 4),
        (0, 1, 4),
        (0, 1, 4),
        (0, 1, 4),
        (0, 1, 4),

        (0, 1, 4),
        (0, 1, 4),
        (0, 1, 4),
        (0, 1, 4),
        (0, 1, 4),

        # Other side
        (0, 1, 4),
        (0, 1, 4),
        (0, 1, 4),
        (0, 1, 4),
        (0, 1, 4),

        (0, 1, 4),
        (0, 1, 4),
        (0, 1, 4),
        (0, 1, 4),
        (0, 1, 4),

        (0, 1, 3),
        (0, 1, 3),
        (0, 1, 3),
        (0, 1, 3),
        (0, 1, 3),

        (0, 2, 0),
        (0, 2, 0),
        (0, 2, 0),
    ]

    '''
    0 = near
    1 = far

    0 = far left
    1 = close left
    2 = zero
    3 = close right
    4 = far right

    0 = far left
    1 = near left
    1 = close left
    2 = zero
    3 = close right
    3 = near right
    4 = far right
    '''
    membership_indices = [
        (0, 0, 0),  # 1
        (0, 1, 0),  # 2
        (0, 2, 0),  # 3

        (1, 0, 0),  # 3
        (1, 0, 1),  # 3
        (1, 0, 2),  # 3
        (1, 0, 3),  # 3
        (1, 0, 4),  # 3

        (1, 1, 0),  # 3
        (1, 1, 1),  # 3
        (1, 1, 2),  # 3
        (1, 1, 3),  # 3
        (1, 1, 4),  # 3

        (1, 2, 0),  # 3
        (1, 2, 1),  # 3
        (1, 2, 2),  # 3
        (1, 2, 3),  # 3
        (1, 2, 4),  # 3

        (1, 3, 0),  # 3
        (1, 3, 1),  # 3
        (1, 3, 2),  # 3
        (1, 3, 3),  # 3
        (1, 3, 4),  # 3

        # OTHER SIDE

        (1, 4, 0),  # 3
        (1, 4, 1),  # 3
        (1, 4, 2),  # 3
        (1, 4, 3),  # 3
        (1, 4, 4),  # 3

        (1, 5, 0),  # 3
        (1, 5, 1),  # 3
        (1, 5, 2),  # 3
        (1, 5, 3),  # 3
        (1, 5, 4),  # 3

        (1, 6, 0),  # 3
        (1, 6, 1),  # 3
        (1, 6, 2),  # 3
        (1, 6, 3),  # 3
        (1, 6, 4),  # 3

        (0, 2, 0),
        (0, 3, 0),
        (0, 4, 0),
    ]

    '''
    0 = far left
    1 = left
    2 = close left
    3 = zero
    4 = close right
    5 = right
    6 = far right
    '''
    outputs_membership = [
        (8,),  # 1
        (6,),  # 1
        (4,),  # 1
        (8,),  # 1
        (5,),  # 1
        (4,),  # 1
        (3,),  # 1
        (0,),  # 1

        (8,),  # 1
        (7,),  # 2
        (6,),  # 3
        (5,),  # 4
        (2,),  # 5
        (8,),  # 6
        (6,),  # 7
        (5,),  # 8
        (4,),  # 9
        (1,),  # 10
        (7,),  # 11
        (5,),  # 12
        (4,),  # 13
        (3,),  # 14
        (1,),  # 15

        # OTHER SIDE

        (7,),  # 1
        (4,),  # 2
        (3,),  # 3
        (2,),  # 4
        (0,),  # 5
        (6,),  # 6
        (3,),  # 7
        (2,),  # 8
        (1,),  # 9
        (0,),  # 10
        (8,),  # 11
        (5,),  # 12
        (4,),  # 13
        (3,),  # 14
        (0,),  # 15
        (4,),  # 15
        (2,),  # 15
        (0,),  # 15
    ]

    mamdani_ruleset = {
        'variable_rule_index': variable_rule_index,
        'membership_indices': membership_indices,
        'outputs_membership': outputs_membership
    }

    return mamdani_ruleset

class Critic(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Critic, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, output_size)

    def forward(self, state, action):
        """
        Params state and actions are torch tensors
        """
        x = torch.cat([state, action], 1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)

        return x


class Anfis(nn.Module):
    def __init__(self):
        super(Anfis, self).__init__()
    def my_model(self):
        parameter_values = [
            [0, 1],

            [0, 2, .1, .2, .2],
            [0, 1, .6, 0.6],
            [0, 0.976657846180786, 0.27020001411438, 0.1281498670578],
            [0, 1.52320299803035, 0.081358410418034, 0.103709816932678],  #

            [0, 1, 1, 1, 1]
        ]

        x_joint_definitons = [
            ('distance_target', JointSingleConstrainedEdgeMembership(*parameter_values[0], constant_center=False)),

            ('distance_line', Joint7TrapMembership(*parameter_values[1], constant_center=True)),
            ('theta_lookahead', JointTrapMembershipV2(*parameter_values[2], constant_center=True)),
            ('theta_far', JointTrapMembershipV2(*parameter_values[3], constant_center=True)),
            ('theta_near', JointTrapMembershipV2(*parameter_values[4], constant_center=True)),
        ]
        invardefs = [
                ##three inputs
                # ('distance_line', [TrapezoidalMembFunc(-100, -100, -1.947, -1.2515,1),
                #                    TrapezoidalMembFunc(-1.947, -1.2515, -0.9112, -0.2513,2),
                #                    TrapezoidalMembFunc( -0.9112, -0.2513, 0.2513, 0.9112,3),
                #                    TrapezoidalMembFunc(0.2513, 0.9112, 1.2515, 1.947,4),
                #                    TrapezoidalMembFunc( 1.2515, 1.947, 100, 100,5),
                #                    Zero()]),


                # ('distance_line', [TrapezoidalMembFunc(-100, -100, -1.947, -1.2515,1,7),
                #                    TrapezoidalMembFunc(-1.947, -1.2515, -0.9112, -0.6,2,7),
                #                    TrapezoidalMembFunc(-0.9112, -0.6, -0.4, -0.2513,3,7),
                #                    TrapezoidalMembFunc( -0.4, -0.2513, 0.2513, 0.4,4,7),
                #                    TrapezoidalMembFunc(0.2513, 0.4, 0.6, 0.9112,5,7),
                #                    TrapezoidalMembFunc(0.6, 0.9112, 1.2515, 1.947,6,7),
                #                    TrapezoidalMembFunc( 1.2515, 1.947, 100, 100,7,7),
                #                    Zero()]),
                #
                # ('theta_far', [TrapezoidalMembFunc(-np.pi, -np.pi, -2.3541, -1.2865,1),
                #                TrapezoidalMembFunc( -2.3541, -1.2865, -1.159, -0.1351,2),
                #                TrapezoidalMembFunc(-1.159, -0.1351, 0.1351, 1.159,3),
                #                TrapezoidalMembFunc( 0.1351, 1.159, 1.2878, 2.3541,4),
                #                TrapezoidalMembFunc( 1.2878, 2.3541, np.pi, np.pi,5),
                #                Zero()]),
                #
                # ('theta_near', [TrapezoidalMembFunc(-np.pi, -np.pi, -1.726, -1.0347,1),
                #                 TrapezoidalMembFunc( -1.726, -1.0347, -0.9905, -0.025,2),
                #                 TrapezoidalMembFunc(-0.9905, -0.025, 0.025, 0.9905,3),
                #                 TrapezoidalMembFunc(0.025, 0.9905, 1.0347, 1.726,4),
                #                 TrapezoidalMembFunc(1.0347, 1.726, np.pi, np.pi,5),
                #                 Zero()]),
                #
                # ('theta_lookahead', [TrapezoidalMembFunc(-np.pi, -np.pi, -1.71, -0.99,1),
                #                 TrapezoidalMembFunc(-1.71, -0.99, -0.81, -0.09,2),
                #                 TrapezoidalMembFunc(-0.81, -0.09, 0.09, 0.81,3),
                #                 TrapezoidalMembFunc(0.09, 0.81, 0.99, 1.71,4),
                #                 TrapezoidalMembFunc(0.99, 1.71, np.pi, np.pi,5),
                #                 Zero()]),
                #
                # ('distance', [TrapezoidalMembFunc(0, 0, 0.55, 0.75,1,2),
                #                 TrapezoidalMembFunc(0.55, 0.75, 100, 100,2,2)
                #                 ])


                ##trained model using offline
                # ('distance_line', [TrapezoidalMembFunc(-100, -100, -1.5, -1.0,1,7),
                #                    TrapezoidalMembFunc(-1.5, -1.1, -1.07, -0.7794,2,7),
                #                    TrapezoidalMembFunc(-1.07, -0.7794, -0.7315, -0.4008,3,7),
                #                    TrapezoidalMembFunc( -0.7315, -0.4008, 0.4008, 0.7315,4,7),
                #                    TrapezoidalMembFunc(0.4008, 0.7315, 0.7794, 1.07,5,7),
                #                    TrapezoidalMembFunc(0.7794, 1.07, 1.1, 1.5,6,7),
                #                    TrapezoidalMembFunc( 1.0, 1.5, 100, 100,7,7),
                #                    Zero()]),
                #
                # ('theta_far', [TrapezoidalMembFunc(-np.pi, -np.pi, -1.71, -0.99,1),
                #                TrapezoidalMembFunc( -1.71, -0.99, -0.81, -0.09,2),
                #                TrapezoidalMembFunc(-0.81, -0.09, 0.09, 0.81,3),
                #                TrapezoidalMembFunc( 0.09, 0.81, 0.99, 1.71,4),
                #                TrapezoidalMembFunc( 0.99, 1.71, np.pi, np.pi,5),
                #                Zero()]),
                #
                # ('theta_near', [TrapezoidalMembFunc(-np.pi, -np.pi, -1.7086, -0.9813,1),
                #                 TrapezoidalMembFunc(-1.7086, -0.9813, -0.7599, -0.0617,2),
                #                 TrapezoidalMembFunc(-0.7599, -0.0617, 0.0617, 0.7599,3),
                #                 TrapezoidalMembFunc(0.0617, 0.7599, 0.9813, 1.7086,4),
                #                 TrapezoidalMembFunc(0.9813, 1.7086, np.pi, np.pi,5),
                #                 Zero()]),
                #
                # ('theta_lookahead', [TrapezoidalMembFunc(-np.pi, -np.pi, -1.835, -1.135,1),
                #                 TrapezoidalMembFunc( -1.835, -1.135, -0.4054, -0.01143,2),
                #                 TrapezoidalMembFunc(-0.4054, -0.01143, 0.01143, 0.4054,3),
                #                 TrapezoidalMembFunc(0.01143, 0.4054, 1.135, 1.835,4),
                #                 TrapezoidalMembFunc(1.135, 1.835, np.pi, np.pi,5),
                #                 Zero()]),
                #
                # ('distance', [TrapezoidalMembFunc(0, 0, 0.2882, 0.864,1,2),
                #                 TrapezoidalMembFunc(0.2882, 0.864, 100, 100,2,2)
                #                 ])
                # ('distance', [TrapezoidalMembFunc(0, 0, 0.55, 0.75,1,2),
                #                 TrapezoidalMembFunc(0.55, 0.75, 100, 100,2,2)
                #                 ])

                ##mar
                ('distance_line', [TrapezoidalMembFunc(-100, -100, -2, -1.6,1,7),
                                   TrapezoidalMembFunc(-2, -1.6, -1.4, -0.8,2,7),
                                   TrapezoidalMembFunc(-1.4, -0.8, -0.5, -0.1,3,7),
                                   TrapezoidalMembFunc( -0.5, -0.1, 0.1, 0.5,4,7),
                                   TrapezoidalMembFunc(0.1, 0.5, 0.8, 1.4,5,7),
                                   TrapezoidalMembFunc(0.8, 1.4, 1.6, 2,6,7),
                                   TrapezoidalMembFunc( 1.6, 2, 100, 100,7,7),
                                   Zero()]),

                ('theta_far', [TrapezoidalMembFunc(-np.pi, -np.pi, -2.4, -1.4,1),
                               TrapezoidalMembFunc( -2.5, -1.4, -1.2, -0.2,2),
                               TrapezoidalMembFunc(-1.2, -0.2, 0.2, 1.2,3),
                               TrapezoidalMembFunc( 0.2, 1.2, 1.4, 2.5,4),
                               TrapezoidalMembFunc( 1.4, 2.5, np.pi, np.pi,5),
                               Zero()]),

                ('theta_near', [TrapezoidalMembFunc(-np.pi, -np.pi, -1.5, -0.9,1),
                                TrapezoidalMembFunc(-1.5, -0.9, -0.8, -0.05,2),
                                TrapezoidalMembFunc(-0.8, -0.05, 0.05, 0.8,3),
                                TrapezoidalMembFunc(0.05, 0.8, 0.9, 1.5,4),
                                TrapezoidalMembFunc(0.9, 1.5, np.pi, np.pi,5),
                                Zero()]),
                # ('theta_lookahead', [TrapezoidalMembFunc(-np.pi, -np.pi, -1.71, -0.99,1),
                #                 TrapezoidalMembFunc(-1.71, -0.99, -0.81, -0.09,2),
                #                 TrapezoidalMembFunc(-0.81, -0.09, 0.09, 0.81,3),
                #                 TrapezoidalMembFunc(0.09, 0.81, 0.99, 1.71,4),
                #                 TrapezoidalMembFunc(0.99, 1.71, np.pi, np.pi,5),
                #                 Zero()]),

                # ('theta_lookahead', [TrapezoidalMembFunc(-np.pi, -np.pi, -2.9, -1.9,1),
                #                 TrapezoidalMembFunc( -2.9, -1.9, -1.7, -0.2,2),
                #                 TrapezoidalMembFunc(-1.7, -0.2, 0.2, 1.7,3),
                #                 TrapezoidalMembFunc(0.2, 1.7, 1.9, 2.9,4),
                #                 TrapezoidalMembFunc(1.9, 2.9, np.pi, np.pi,5),
                #                 Zero()]),
                ('theta_lookahead', [TrapezoidalMembFunc(-np.pi, -np.pi, -2.4, -1.4,1),
                                TrapezoidalMembFunc( -2.4, -1.4, -1.2, -0.2,2),
                                TrapezoidalMembFunc(-1.2, -0.2, 0.2, 1.2,3),
                                TrapezoidalMembFunc(0.2, 1.2, 1.4, 2.4,4),
                                TrapezoidalMembFunc(1.4, 2.4, np.pi, np.pi,5),
                                Zero()]),

                ('distance', [TrapezoidalMembFunc(0, 0, 0.5, 0.7,1,2),
                                TrapezoidalMembFunc(0.5, 0.7, 100, 100,2,2)
                                ])




    #    invardefs = [
    #            ('distance_line', [TrapezoidalMembFunc(-100, -100, -1.391729712486267, -0.80529651927948,1),
    #                               TrapezoidalMembFunc(-1.391729712486267, -0.80529651927948, -0.8044833025932312, -0.1535172462463379,2),
    #                               TrapezoidalMembFunc(-0.8044833025932312, -0.1535172462463379, 0.1535172462463379, 0.8084833025932312,3),
    #                               TrapezoidalMembFunc(0.1535172462463379, 0.8044833025932312, 0.80529651927948, 1.391729712486267,4),
    #                               TrapezoidalMembFunc(0.80529651927948, 1.391729712486267, 100, 100,5),
    #                               Zero()]),
    #            ('theta_far', [TrapezoidalMembFunc(-3.15, -3.15, -2.3750057220458984, -1.3739324808120728,1),
    #                           TrapezoidalMembFunc(-2.3750057220458984, -1.3739324808120728, -1.103203296661377, -0.1184181272983551,2),
    #                           TrapezoidalMembFunc(-1.103203296661377, -0.1184181272983551, 0.1184181272983551, 1.103203296661377,3),
    #                           TrapezoidalMembFunc(0.1184181272983551, 1.103203296661377, 1.3739324808120728, 2.3750057220458984,4),
    #                           TrapezoidalMembFunc(1.3739324808120728, 2.3750057220458984, 3.15, 3.15,5),
    #                           Zero()]),
    #            ('theta_near', [TrapezoidalMembFunc(-3.15, -3.15, -1.7530126571655273, -1.0022785663604736,1),
    #                            TrapezoidalMembFunc(-1.7530126571655273, -1.0022785663604736, -0.8224078416824341, -0.004677863791584969,2),
    #                            TrapezoidalMembFunc(-0.8224078416824341, -0.004677863791584969, 0.004677863791584969 ,0.8224078416824341,3),
    #                            TrapezoidalMembFunc(0.004677863791584969, 0.8224078416824341, 1.0022785663604736, 1.7530126571655273,4),
    #                            TrapezoidalMembFunc(1.0022785663604736, 1.7530126571655273, 3.15, 3.15,5),
    #                            Zero()])

    #    invardefs = [
    #            ('distance_line', [TrapezoidalMembFunc(-100,-100,-1.425,-0.825,1),
    #                               TrapezoidalMembFunc(-1.425,-0.825,-0.675,-0.075,2),
    #                               TrapezoidalMembFunc(-0.675,-0.075,0.075,0.675,3),
    #                               TrapezoidalMembFunc(0.075, 0.675, 0.825, 1.425,4),
    #                               TrapezoidalMembFunc(0.825, 1.425, 100, 100,5),
    #                               Zero()]),
    #            ('theta_far', [TrapezoidalMembFunc(-3.15, -3.15, -2.375, -1.375,1),
    #                           TrapezoidalMembFunc(-2.375, -1.375, -1.125, -0.125,2),
    #                           TrapezoidalMembFunc(-1.125, -0.125, 0.125, 1.125,3),
    #                           TrapezoidalMembFunc(0.125, 1.125, 1.375, 2.375,4),
    #                           TrapezoidalMembFunc(1.375, 2.375, 3.15, 3.15,5),
    #                           Zero()]),
    #            ('theta_near', [TrapezoidalMembFunc(-3.15, -3.15, -1.71, -0.99,1),
    #                            TrapezoidalMembFunc(-1.71, -0.99, -0.81, -0.09,2),
    #                            TrapezoidalMembFunc(-0.81, -0.09, 0.09, 0.81,3),
    #                            TrapezoidalMembFunc(0.09, 0.81, 0.99, 1.71,4),
    #                            TrapezoidalMembFunc(0.99, 1.71, 3.15, 3.15,5),
    #                            Zero()])
                                                                    ]

        outvars = ['control_law']

        # mamdani_out = [
        #     ('right3', [TriangularMembFunc(-1.25, -1, -0.75)]),
        #     ('right2', [TriangularMembFunc(-1, -0.75, -0.5)]),
        #     ('rigt1', [TriangularMembFunc(-0.75, -0.5, 0)]),
        #     ('zero', [TriangularMembFunc(-0.5, 0, 0.5)]),
        #     ('left1', [TriangularMembFunc(0, 0.5, 0.75)]),
        #     ('left2', [TriangularMembFunc(0.5, 0.75, 1)]),
        #     ('left3', [TriangularMembFunc(0.75, 1, 1.25)])
        #                                                         ]

        input_keywords = [] #list
        number_of_mfs = {} #dict
        for i in range(len(invardefs)):
            input_keywords.append(invardefs[i][0])
            number_of_mfs[invardefs[i][0]] = len(invardefs[i][1]) - 1
        #print(invardefs)
        # mamdani_out = JointSymmetricTriangleMembership(0, .9726, .007, .2036)
        # mamdani_out = JointSymmetric9TriangleMembership(0,-0.2287,1.0375,0.4749,0.4709)
        mamdani_out = JointSymmetric9TriangleMembership(0,1,1,1,1)
        # Marius
        # ruleset = dist_target_dist_per_theta_lookahead_theta_far_theta_near()
        # anf = anfis.AnfisNetV2('ANFIS', x_joint_definitons , outvars, mamdani_out, input_keywords, number_of_mfs, False, ruleset)
        anf = anfis.AnfisNet('ANFIS', invardefs, outvars, mamdani_out, input_keywords, number_of_mfs, False)
        return anf
