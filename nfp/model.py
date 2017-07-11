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
            #self.mlp = L.Linear(out_dim)
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
        num_degree_type = max_degree + 1
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

    def one_step(self, x, h, adj, id2pid, n_mols):
        # fvs[degree][i] = sum of features of adjacent atoms of
        # `i`-th atom whose degree is `degree`
        fvs = [[] for _ in range(self.num_degree_type)]
        # ids[degree] = list of ids whose `degree` is degree
        ids = [[] for _ in range(self.num_degree_type)]

        s_list = [F.sum(x[adj_atoms + [i]], axis=0) for i, adj_atoms in enumerate(adj)]
        for i, adj_atoms in enumerate(adj):
            # degree = 0 until max_degree (inclusive)
            d = len(adj_atoms)
            fvs[d].append(s_list[i])
            ids[d].append(i)
        ids = sum(ids, [])

        mol_ids = [id2pid(id_)[0] for id_ in ids]
        mol_ids_mat = self.xp.asarray(util.to_one_hot(mol_ids, n_mols),
                                      dtype=numpy.float32)

        out_x_shuffled = []
        dh_shuffled = []
        for wout, w, fv in six.moves.zip(self.outer_weight, self.inner_weight, fvs):
            # As, Variable.__nonzero__ is not implemented,
            # we use __len__ instead.
            if not len(fv):
                continue
            fv = F.stack(fv, axis=0)
            out_x_shuffled.append(F.sigmoid(w(fv)))
            dh_shuffled.append(F.softmax(wout(fv)))

        out_x_shuffled = F.concat(out_x_shuffled, axis=0)
        out_x = out_x_shuffled[ids]

        dh_shuffled = F.concat(dh_shuffled, axis=0)
        dh = F.matmul(mol_ids_mat.T, dh_shuffled)
        out_h = h + dh

        return out_x, out_h

    def __call__(self, adj, atom_types):
        """Forward propagation

        Args:
            adj (list of list of tuple of ints): minibatch of adjancency matrix
            atom_types (list of list of ints): minibatch of molcular IDs

        adj[mol_id][atom_id] = (tuple of adjacent atoms)
        atom_types[mol_id][atom_id] = atom type ID (C, O, S, ...)

        Returns:
            ~chainer.Variable: minibatch of fingerprint
        """

        n_atoms = [len(m) for m in atom_types]
        # pid = pair of ids (mol_id and atom_id)
        pid2id, id2pid = util.make_converter(n_atoms)
        n_mols = len(atom_types)

        atom_types_flatten = self.xp.asarray(sum(atom_types, []), numpy.int32)
        adj_flatten = []
        for i, adj_mol in enumerate(adj):
            # adj_mol = adj[i] is i-th molecule's adjancy list
            # [[1st atom's adj], [2nd atom's adj], ...]
            for atoms in adj_mol:
                adj_flatten.append([pid2id(i, a) for a in atoms])

        x = self.embed(atom_types_flatten)
        h = chainer.Variable(
            self.xp.zeros((n_mols, self.hidden_dim), numpy.float32))
        h_list = []
        for _ in range(self.radius):
            x, h = self.one_step(x, h, adj_flatten, id2pid, n_mols)
            if self.concat_hidden:
                h_list.append(h)

        if self.concat_hidden:
            return F.concat(h_list, axis=1)
        else:
            return h
