import chainer
import chainer.functions as F
import chainer.links as L
import numpy
import six

import util


class MLP(chainer.Chain):
    def __init__(self, out_dim):
        super(MLP, self).__init__()
        hidden_dim = 128
        with self.init_scope():
            self.l1 = L.Linear(hidden_dim)
            self.l2 = L.Linear(hidden_dim)
            self.l3 = L.Linear(out_dim)

    def __call__(self, x):
        h = F.sigmoid(self.l1(x))
        h = F.sigmoid(self.l2(h))
        return self.l3(h)


class Predictor(chainer.Chain):

    def __init__(self, nfp, out_dim):
        super(Predictor, self).__init__()
        with self.init_scope():
            self.nfp = nfp
            self.mlp = MLP(out_dim)

    def __call__(self, adj, atom_types):
        x = self.nfp(adj, atom_types)
        x = self.mlp(x)
        return x


class NFP(chainer.Chain):

    def __init__(self, hidden_dim, out_dim, max_degree,
                 n_atom_types, radius, concat_hidden=False):
        """Neural Finger Print (NFP)

        Args:
            hidden_dim (int): dimension of feature vector
                associated to each atom
            out_dim (int): dimension of output feature vector
            max_degree (int): max degree of atoms
                when molecules are regarded as graphs
            n_atom_types (int): number of types of atoms
            radius (int): number of iteration
        """

        super(NFP, self).__init__()
        num_degree_type = max_degree+1
        with self.init_scope():
            self.embed = L.EmbedID(n_atom_types, hidden_dim)
            # If molculars do not have atoms whose degree are d,
            # self.inner_weight[d] does run forward propagation.
            # If its weight is uninitialized, it is never initialized.
            # This is problematic when optimizer.update is called.
            self.inner_weight = chainer.ChainList(
                *[L.Linear(hidden_dim, hidden_dim)
                  for _ in range(num_degree_type)])
            self.outer_weight = chainer.ChainList(
                *[L.Linear(hidden_dim, out_dim)
                  for _ in range(num_degree_type)])
        self.hidden_dim = hidden_dim
        self.max_degree = max_degree
        self.num_degree_type = num_degree_type
        self.radius = radius
        self.concat_hidden = concat_hidden

    def one_step(self, x, h, adj, deg_conds):
        # Take sum along adjacent atoms
        fv = F.batch_matmul(adj, x)
        ##print('fv', fv.shape)  # (minibatch, max_num_atoms, hidden_dim)
        s0, s1, s2 = fv.shape
        zero_array = self.xp.zeros_like(fv)
        fvds = [F.reshape(F.where(cond, fv, zero_array), (s0 * s1, s2)) for cond in deg_conds]

        out_x = 0
        dh = 0
        for wout, win, fvd in zip(self.outer_weight, self.inner_weight, fvds):
            out_x = out_x + win(fvd)
            dh = dh + wout(fvd)

        out_x = F.sigmoid(out_x)
        out_x = F.reshape(out_x, (s0, s1, s2))

        dh = F.softmax(dh)
        dh = F.sum(F.reshape(dh, (s0, s1, s2)), axis=1)  # sum along atom's axis

        out_h = h + dh
        return out_x, out_h

    def __call__(self, adj, atom_array):
        """Forward propagation

        Args:
            adj (list of list of tuple of ints): minibatch of adjancency matrix
            atom_types (list of list of ints): minibatch of molcular IDs

        adj[mol_id][atom_id] = (tuple of adjacent atoms)
        atom_types[mol_id][atom_id] = atom type ID (C, O, S, ...)

        Returns:
            ~chainer.Variable: minibatch of fingerprint
        """
        x = self.embed(atom_array)
        ##print('x', x.shape)  # (minibatch, max_num_atoms , hidden_dim)

        h = 0
        degree_mat = F.sum(adj, axis=1)
        ##print('degree_mat', degree_mat.shape)  # (minibatch, max_num_atoms)
        deg_conds = [self.xp.broadcast_to(((degree_mat - degree).data == 0)[:, :, None], x.shape)
                 for degree in range(1, self.num_degree_type + 1)]
        h_list = []
        for _ in range(self.radius):
            x, h = self.one_step(x, h, adj, deg_conds)
            if self.concat_hidden:
                h_list.append(h)

        if self.concat_hidden:
            return F.concat(h_list, axis=1)
        else:
            return h
