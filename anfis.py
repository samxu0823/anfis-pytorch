#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
    ANFIS in torch: the ANFIS layers
    @author: James Power <james.power@mu.ie> Apr 12 18:13:10 2019
    Acknowledgement: twmeggs' implementation of ANFIS in Python was very
    useful in understanding how the ANFIS structures could be interpreted:
        https://github.com/twmeggs/anfis
'''

import itertools
from collections import OrderedDict

import numpy as np

import torch
import torch.nn.functional as F

dtype = torch.float64


class FuzzifyVariable(torch.nn.Module):
    '''
        Represents a single fuzzy variable, holds a list of its MFs.
        Forward pass will then fuzzify the input (value for each MF).
    '''

    def __init__(self, mfdefs):
        """
        Constructor of FuzzifyVariable.
        :param mfdefs: [list], list of all MF Classes for 1 input per iteration
        """
        super(FuzzifyVariable, self).__init__()
        if isinstance(mfdefs, list):  # No MF names supplied
            mfnames = ['mf{}'.format(i) for i in range(len(mfdefs))]  # set MF name
            mfdefs = OrderedDict(zip(mfnames, mfdefs))  # set dict of MF name and MF
        self.mfdefs = torch.nn.ModuleDict(mfdefs)
        self.padding = 0

    @property
    def num_mfs(self):
        '''Return the actual number of MFs (ignoring any padding)'''
        return len(self.mfdefs)

    def members(self):
        '''
            Return an iterator over this variables's membership functions.
            Yields tuples of the form (mf-name, MembFunc-object)
        '''
        return self.mfdefs.items()

    def pad_to(self, new_size):
        """
            Will pad result of forward-pass (with zeros) so it has new_size,
            i.e.  padding = max_size - no. of MF, as if it had new_size MFs.
            If inputs have different no. MF, return non zero.
        """
        self.padding = new_size - len(self.mfdefs)

    def fuzzify(self, x):
        '''
            Yield a list of (mf-name, fuzzy values) for these input values.
        '''
        for mfname, mfdef in self.mfdefs.items():
            yvals = mfdef(x)  # check foward() function in certain MF class
            yield (mfname, yvals)

    def forward(self, x):
        '''
            Return a tensor giving the membership value for each MF.
            x.shape: n_cases
            y.shape: n_cases * n_mfs
        '''
        y_pred = torch.cat([mf(x) for mf in self.mfdefs.values()], dim=1)
        if self.padding > 0:  # When inputs have different no. of MF
            y_pred = torch.cat([y_pred,
                                torch.zeros(x.shape[0], self.padding)], dim=1)
        return y_pred


class FuzzifyLayer(torch.nn.Module):
    '''
        Layer 1.
        A list of fuzzy variables, representing the inputs to the FIS.
        Forward pass will fuzzify each variable individually.
        We pad the variables so they all seem to have the same number of MFs,
        as this allows us to put all results in the same tensor.
    '''

    def __init__(self, varmfs, varnames=None):
        super(FuzzifyLayer, self).__init__()
        if not varnames:
            self.varnames = ['x{}'.format(i) for i in range(len(varmfs))]
        else:
            self.varnames = list(varnames)
        maxmfs = max([var.num_mfs for var in varmfs])  # max. no. of MF
        for var in varmfs:
            var.pad_to(maxmfs)
        self.varmfs = torch.nn.ModuleDict(zip(self.varnames, varmfs))

    @property
    def num_in(self):
        '''Return the number of input variables'''
        return len(self.varmfs)

    @property
    def max_mfs(self):
        ''' Return the max number of MFs in any variable'''
        return max([var.num_mfs for var in self.varmfs.values()])

    def __repr__(self):
        '''
            Print the variables, MFS and their parameters (for info only)
        '''
        r = ['Input variables']
        for varname, members in self.varmfs.items():
            r.append('Variable {}'.format(varname))
            for mfname, mfdef in members.mfdefs.items():
                r.append('- {}: {}({})'.format(mfname,
                                               mfdef.__class__.__name__,
                                               ', '.join(['{}={}'.format(n, p.item())
                                                          for n, p in mfdef.named_parameters()])))
        return '\n'.join(r)

    def forward(self, x):
        ''' Fuzzyify each variable's value using each of its corresponding mfs.
            x.shape = n_cases * n_in
            y.shape = n_cases * n_in * n_mfs
        '''
        assert x.shape[1] == self.num_in, \
            '{} is wrong no. of input values'.format(self.num_in)
        y_pred = torch.stack([var(x[:, i:i + 1])
                              for i, var in enumerate(self.varmfs.values())],
                             dim=1)  # MF(x), fuzzify the input!
        return y_pred


class AntecedentLayer(torch.nn.Module):
    '''
        Layer 2.
        Form the 'rules' by taking all possible combinations of the MFs
        for each variable. Forward pass then calculates the fire-strengths.
    '''

    def __init__(self, varlist, grid=True):
        super(AntecedentLayer, self).__init__()
        # Count the (actual) mfs for each variable:
        mf_count = [var.num_mfs for var in varlist]  # No. of MF of each input
        # Now make the MF indices for each rule:
        if grid:
            mf_indices = itertools.product(*[range(n) for n in mf_count])
        else:  # Set rules manually
            mf_indices = [[0, 0, 0], [0, 1, 1], [1, 1, 1], [1, 1, 0]]
        self.mf_indices = torch.tensor(list(mf_indices))  # Grid partition: [[0, 0], [0, 1]...]
        # mf_indices.shape is n_rules * n_in

    def num_rules(self):
        return len(self.mf_indices)

    def extra_repr(self, varlist=None):
        if not varlist:
            return None
        row_ants = []
        mf_count = [len(fv.mfdefs) for fv in varlist.values()]
        # for rule_idx in itertools.product(*[range(n) for n in mf_count]):
        for rule_idx in self.mf_indices:
            thisrule = []
            for (varname, fv), i in zip(varlist.items(), rule_idx):
                thisrule.append('{} is {}'
                                .format(varname, list(fv.mfdefs.keys())[i]))
            row_ants.append(' and '.join(thisrule))
        return '\n'.join(row_ants)

    def forward(self, x):
        ''' Calculate the fire-strength for (the antecedent of) each rule
            x.shape = n_cases * n_in * n_mfs
            y.shape = n_cases * n_rules
        '''
        # Expand (repeat) the rule indices to equal the batch size:
        a = self.mf_indices.size()
        batch_indices = self.mf_indices.expand((x.shape[0], -1, -1))
        # Then use these indices to populate the rule-antecedents
        ants = torch.gather(x.transpose(1, 2), 1, batch_indices)
        # ants.shape is n_cases * n_rules * n_in
        # Last, take the AND (= product) for each rule-antecedent
        rules = torch.prod(ants, dim=2)  # Calculate the fire-strength: A1 * B1, A1 * B2, ...
        return rules

    def set_rules(self, rules_index):
        """
        Set fuzzy rules manually.
        :param rules_index: [list], list of indices of rules. E.g. [[0, 0, 0], [0, 0, 1], ...]
        0: Low, 1: Medium, 2: High...; length: no. of rules
        :return:
        void
        """
        self.mf_indices = torch.tensor(list(rules_index))  # Grid partition: [[0, 0], [0, 1]...]


class ConsequentLayer(torch.nn.Module):
    '''
        A simple linear layer to represent the TSK consequents.
        Hybrid learning, so use MSE (not BP) to adjust coefficients.
        Hence, coeffs are no longer parameters for backprop.
    '''

    def __init__(self, d_in, d_rule, d_out):
        """
        Constructor of ConsequentLayer.
        :param d_in: [int], no. of input.
        :param d_rule: [int], no. of rule.
        :param d_out: [int], no. of output.
        """
        super(ConsequentLayer, self).__init__()
        c_shape = torch.Size([d_rule, d_out, d_in + 1])  # consider constant consequent parameter
        self._coeff = torch.zeros(c_shape, dtype=dtype, requires_grad=True)

    @property
    def coeff(self):
        '''
            Record the (current) coefficients for all the rules
            coeff.shape: n_rules * n_out * (n_in+1)
        '''
        return self._coeff

    @coeff.setter
    def coeff(self, new_coeff):
        '''
            Record new coefficients for all the rules
            coeff: for each rule, for each output variable:
                   a coefficient for each input variable, plus a constant
        '''
        assert new_coeff.shape == self.coeff.shape, \
            'Coeff shape should be {}, but is actually {}' \
                .format(self.coeff.shape, new_coeff.shape)
        self._coeff = new_coeff

    def fit_coeff(self, x, weights, y_actual):
        '''
            Use LSE to solve for coeff: y_actual = coeff * (weighted)x
                  x.shape: n_cases * n_in
            weights.shape: n_cases * n_rules
            [ coeff.shape: n_rules * n_out * (n_in+1) ]
                  y.shape: n_cases * n_out
        '''
        # Append 1 to each list of input vals, for the constant term:
        x_plus = torch.cat([x, torch.ones(x.shape[0], 1)], dim=1)
        # Shape of weighted_x is (n_cases * n_rules * 1) * (n_cases * 1 * n_in + 1) = n_cases * n_rules * (n_in+1)
        weighted_x = torch.einsum('bp, bq -> bpq', weights, x_plus)
        # weighted_x = torch.bmm(weights.unsqueeze(2), x_plus.unsqueeze(2).transpose(1, 2))  # alternative
        # Can't have value 0 for weights, or LSE won't work:
        weighted_x[weighted_x == 0] = 1e-12
        # Squash x and y down to 2D matrices for gels:
        weighted_x_2d = weighted_x.view(weighted_x.shape[0], -1)  # n_cases * n_features
        y_actual_2d = y_actual.view(y_actual.shape[0], -1)  # n_cases * n_output
        # Use gels to do LSE, then pick out the solution rows:
        try:
            coeff_2d, _ = torch.lstsq(y_actual_2d, weighted_x_2d)
        except RuntimeError as e:
            print('Internal error in gels', e)
            print('Weights are:', weighted_x)
            raise e
        coeff_2d = coeff_2d[0:weighted_x_2d.shape[1]]  # n_features * n_output, First n_features rows are the parameter
        # Reshape to 3D tensor: divide by rules, n_in+1, then swap last 2 dims
        self.coeff = coeff_2d.view(weights.shape[1], x.shape[1] + 1, -1) \
            .transpose(1, 2)
        # coeff dim is thus: n_rules * n_out * (n_in+1)

    def forward(self, x):
        '''
            Calculate: y = coeff * x + const   [NB: no weights yet]
                  x.shape: n_cases * n_in
              coeff.shape: n_rules * n_out * (n_in+1)
                  y.shape: n_cases * n_out * n_rules
        '''
        # Append 1 to each list of input vals, for the constant term:
        x_plus = torch.cat([x, torch.ones(x.shape[0], 1)], dim=1)  # shape: n_sample * n_input + 1
        # Need to switch dimension for the multiply, then switch back:
        y_pred = torch.matmul(self.coeff, x_plus.t())  # shape: n_rules * n_out * n_sample
        return y_pred.transpose(0, 2)  # swaps cases and rules: n_sample * n_out* n_rules


class PlainConsequentLayer(ConsequentLayer):
    '''
        A linear layer to represent the TSK consequents.
        Not hybrid learning, so coefficients are backprop-learnable parameters.
    '''

    def __init__(self, *params):
        super(PlainConsequentLayer, self).__init__(*params)
        self.register_parameter('coefficients',
                                torch.nn.Parameter(self._coeff))

    @property
    def coeff(self):
        '''
            Record the (current) coefficients for all the rules
            coeff.shape: n_rules * n_out * (n_in+1)
        '''
        return self.coefficients

    def fit_coeff(self, x, weights, y_actual):
        '''
        '''
        assert False, \
            'Not hybrid learning: I\'m using BP to learn coefficients'


class WeightedSumLayer(torch.nn.Module):
    '''
        Sum the TSK for each outvar over rules, weighted by fire strengths.
        This could/should be layer 5 of the Anfis net.
        I don't actually use this class, since it's just one line of code.
    '''

    def __init__(self):
        super(WeightedSumLayer, self).__init__()

    def forward(self, weights, tsk):
        '''
            weights.shape: n_cases * n_rules
                tsk.shape: n_cases * n_out * n_rules
             y_pred.shape: n_cases * n_out
        '''
        # Add a dimension to weights to get the bmm to work:
        y_pred = torch.bmm(tsk, weights.unsqueeze(2))
        return y_pred.squeeze(2)


class AnfisNet(torch.nn.Module):
    """
        This is a container for the 5 layers of the ANFIS net.
        The forward pass maps inputs to outputs based on current settings,
        and then fit_coeff will adjust the TSK coeff using LSE.
    """

    def __init__(self, description, invardefs, outvarnames, hybrid=True, grid=True):
        """
        Constructor of ANFIS.
        :param description: [Str], name of the model.
        :param invardefs: [list], list contain input info: (input name, [its MF]).
        :param outvarnames: [list], list contain name of output.
        :param hybrid: [boolean], determine using BP or LSE.
        """
        super(AnfisNet, self).__init__()
        self.description = description
        self.outvarnames = outvarnames
        self.hybrid = hybrid
        varnames = [v for v, _ in invardefs]
        mfdefs = [FuzzifyVariable(mfs) for _, mfs in invardefs]  # set MF for each input
        self.num_in = len(invardefs)
        if grid:
            self.num_rules = np.prod([len(mfs) for _, mfs in invardefs])  # No. of fuzzy rules: grid partition
        if not grid:  # set rules manually
            self.num_rules = 4
        if self.hybrid:
            cl = ConsequentLayer(self.num_in, self.num_rules, self.num_out)  # hybrid: LSE for consequent
        else:
            cl = PlainConsequentLayer(self.num_in, self.num_rules, self.num_out)  # no hybrid: BP for consequent
        self.layer = torch.nn.ModuleDict(OrderedDict([
            ('fuzzify', FuzzifyLayer(mfdefs, varnames)),  # Layer 1
            ('rules', AntecedentLayer(mfdefs, grid=grid)),  # Layer 2
            # Layer 3: normalisation layer is just implemented as a function.
            ('consequent', cl),  # Layer 4
            # Layer 5: weighted-sum layer is just implemented as a function.
        ]))

    @property
    def num_out(self):
        return len(self.outvarnames)

    @property
    def coeff(self):
        return self.layer['consequent'].coeff

    @coeff.setter
    def coeff(self, new_coeff):
        self.layer['consequent'].coeff = new_coeff

    def fit_coeff(self, x, y_actual):
        '''
            Do a forward pass (to get weights), then fit to y_actual.
            Does nothing for a non-hybrid ANFIS, so we have same interface.
        '''
        if self.hybrid:
            self(x)
            self.layer['consequent'].fit_coeff(x, self.weights, y_actual)

    def input_variables(self):
        '''
            Return an iterator over this system's input variables.
            Yields tuples of the form (var-name, FuzzifyVariable-object)
        '''
        return self.layer['fuzzify'].varmfs.items()

    def output_variables(self):
        '''
            Return an list of the names of the system's output variables.
        '''
        return self.outvarnames

    def extra_repr(self):
        """
        Override the built in extra_repr function,
        print linguistic label automatically.
        :return: Description of fuzzy rules.
        """
        rstr = []
        vardefs = self.layer['fuzzify'].varmfs
        rule_ants = self.layer['rules'].extra_repr(vardefs).split('\n')
        for i, crow in enumerate(self.layer['consequent'].coeff):
            rstr.append('Rule {:2d}: IF {}'.format(i, rule_ants[i]))
            rstr.append(' ' * 9 + 'THEN {}'.format(crow.tolist()))
        return '\n'.join(rstr)

    def forward(self, x):
        '''
            Forward pass: run x thru the five layers and return the y values.
            I save the outputs from each layer to an instance variable,
            as this might be useful for comprehension/debugging.
        '''
        self.fuzzified = self.layer['fuzzify'](x)  # layer 1 out: Fuzzified value
        self.raw_weights = self.layer['rules'](self.fuzzified)  # layer 2 out: Firing strength of rules
        self.weights = F.normalize(self.raw_weights, p=1, dim=1)  # layer 3 out: Normalized firing strength
        self.rule_tsk = self.layer['consequent'](x)  # layer 4 out: consequent f = px + qy + r,
        # y_pred = self.layer['weighted_sum'](self.weights, self.rule_tsk)
        y_pred = torch.bmm(self.rule_tsk, self.weights.unsqueeze(2))  # layer 5 out: weighted sum w * f, overall output
        self.y_pred = y_pred.squeeze(2)  # 3D -> 2D, due to single output system
        return self.y_pred

    def set_rules(self, rules_index, hybrid=True):
        """
        Set fuzzy rules manually.
        :param rules_index: [list], list of indices of rules. E.g. [[0, 0, 0], [0, 0, 1], ...]
        0: Low, 1: Medium, 2: High...; length: no. of rules
        :return:
        void
        """
        self.num_rules = len(rules_index)
        if hybrid:
            customized_c1 = ConsequentLayer(self.num_in, self.num_rules, self.num_out)
        else:
            customized_c1 = PlainConsequentLayer(self.num_in, self.num_rules, self.num_out)
        self.layer['rules'].set_rules(rules_index)
        self.layer['consequent'] = customized_c1


# These hooks are handy for debugging:

def module_hook(label):
    ''' Use this module hook like this:
        m = AnfisNet()
        m.layer.fuzzify.register_backward_hook(module_hook('fuzzify'))
        m.layer.consequent.register_backward_hook(modul_hook('consequent'))
    '''
    return (lambda module, grad_input, grad_output:
            print('BP for module', label,
                  'with out grad:', grad_output,
                  'and in grad:', grad_input))


def tensor_hook(label):
    '''
        If you want something more fine-graned, attach this to a tensor.
    '''
    return (lambda grad:
            print('BP for', label, 'with grad:', grad))
