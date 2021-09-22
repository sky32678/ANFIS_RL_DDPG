In anfis.py

AntecedentLayer -- Modify AntecedentLayer classs to add or delete the rule bases
You can customize the rule bases instead of Full combinations

ConsequentLayer -- this is a class for ConsequentLayer, but only for hybrid method.

In RL approach, we are not able to use the hybrid method.

PlainConsequentLayer -- this is a class for PlainConsequentLayer, which is used
for backpropagation method only. there is nothing to change in it.

AnfisNet -- this is a class to construct anfisnet.
            you can set the number of rules at here using self.num_rules.
            If you want to use 20 rules, type self.num_rules = 20.
            this is going to automatically set the number of coefficients
            in consequent layer, but not antecedentlayer.
            The way to change the number of rules in antecedent layer
            is stated above.

How to build an initial modelled

def ex1_model():
    '''
        These are the original (untrained) MFS for Jang's example 1.
    '''
    invardefs = [
            ('x0', make_bell_mfs(2.5, 2, [1, 2,6])),
            ('x1', make_bell_mfs(2.5, 2, [1, 3,5])),
            ('x2', make_bell_mfs(2.5, 2, [1, 4,5])),
            ]
    outvars = ['y0']
    anf = anfis.AnfisNet('sin + cos + tan', invardefs, outvars,False)
    return anf

invardefs -- this is a default inputs and corresponding membership functions.
outvars -- you can set the number of outputs.
Building net -- anfis.AnfisNet('description', input, output, hybrid=True or False).
                If true left it blank
