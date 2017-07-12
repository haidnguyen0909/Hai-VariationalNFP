import chainer
import chainer.functions as F
import chainer.links as L
import numpy
import six
from chainer import cuda
from chainer.dataset import concat_examples

import util


class MLP(chainer.Chain):
    def __init__(self, hidden_dim, out_dim):
        super(MLP, self).__init__()
        with self.init_scope():
            self.l1 = L.Linear(hidden_dim)
            self.l2 = L.Linear(hidden_dim)
            self.l3 = L.Linear(out_dim)

    def __call__(self, x):
        h = F.leaky_relu(self.l1(x))
        h = F.leaky_relu(self.l2(h))
        return self.l3(h)


class MLP2(chainer.Chain):
    def __init__(self, hidden_dim, out_dim):
        super(MLP2, self).__init__()
        with self.init_scope():
            self.l1 = L.Linear(hidden_dim)
            self.l2 = L.Linear(hidden_dim)
            self.l3 = L.Linear(out_dim)
            self.bn1 = L.BatchNormalization(hidden_dim)
            self.bn2 = L.BatchNormalization(hidden_dim)

    def __call__(self, x):
        h = F.relu(self.bn1(self.l1(x)))
        h = F.relu(self.bn2(self.l2(h)))
        return self.l3(h)


# activation leaky_relu
class MLP3(chainer.Chain):
    def __init__(self, hidden_dim, out_dim):
        super(MLP3, self).__init__()
        with self.init_scope():
            self.l1 = L.Linear(hidden_dim)
            self.l2 = L.Linear(hidden_dim)
            self.l3 = L.Linear(hidden_dim)
            self.l4 = L.Linear(out_dim)
            self.bn1 = L.BatchNormalization(hidden_dim)
            self.bn2 = L.BatchNormalization(hidden_dim)
            self.bn3 = L.BatchNormalization(hidden_dim)

    def __call__(self, x):
        h = F.leaky_relu(self.bn1(self.l1(x)), 0.05)
        h = F.leaky_relu(self.bn2(self.l2(h)), 0.05)
        h = F.leaky_relu(self.bn3(self.l3(h)), 0.05)
        return self.l4(h)


# activation sigmoid, deep
class MLP4(chainer.Chain):
    def __init__(self, hidden_dim, out_dim):
        super(MLP4, self).__init__()
        with self.init_scope():
            self.l1 = L.Linear(hidden_dim)
            self.l2 = L.Linear(hidden_dim)
            self.l3 = L.Linear(hidden_dim)
            self.l4 = L.Linear(hidden_dim)
            self.l5 = L.Linear(hidden_dim)
            self.l6 = L.Linear(out_dim)
            self.bn1 = L.BatchNormalization(hidden_dim)
            self.bn2 = L.BatchNormalization(hidden_dim)
            self.bn3 = L.BatchNormalization(hidden_dim)
            self.bn4 = L.BatchNormalization(hidden_dim)
            self.bn5 = L.BatchNormalization(hidden_dim)

    def __call__(self, x):
        h = F.sigmoid(self.bn1(self.l1(x)))
        h = F.sigmoid(self.bn2(self.l2(h)))
        h = F.sigmoid(self.bn3(self.l3(h)))
        h = F.sigmoid(self.bn4(self.l4(h)))
        h = F.sigmoid(self.bn5(self.l5(h)))
        return self.l6(h)


# activation leaky_relu, deep
class MLP5(chainer.Chain):
    def __init__(self, hidden_dim, out_dim):
        super(MLP5, self).__init__()
        with self.init_scope():
            self.l1 = L.Linear(hidden_dim)
            self.l2 = L.Linear(hidden_dim)
            self.l3 = L.Linear(hidden_dim)
            self.l4 = L.Linear(hidden_dim)
            self.l5 = L.Linear(hidden_dim)
            self.l6 = L.Linear(out_dim)
            self.bn1 = L.BatchNormalization(hidden_dim)
            self.bn2 = L.BatchNormalization(hidden_dim)
            self.bn3 = L.BatchNormalization(hidden_dim)
            self.bn4 = L.BatchNormalization(hidden_dim)
            self.bn5 = L.BatchNormalization(hidden_dim)

    def __call__(self, x):
        h = F.leaky_relu(self.bn1(self.l1(x)), slope=0.05)
        h = F.leaky_relu(self.bn2(self.l2(h)), slope=0.05)
        h = F.leaky_relu(self.bn3(self.l3(h)), slope=0.05)
        h = F.leaky_relu(self.bn4(self.l4(h)), slope=0.05)
        h = F.leaky_relu(self.bn5(self.l5(h)), slope=0.05)
        return self.l6(h)




class Predictor(chainer.Chain):

    def __init__(self, nfp, mlp):
        super(Predictor, self).__init__()
        with self.init_scope():
            self.nfp = nfp
            self.mlp = mlp

    def __call__(self, adj, atom_types):
        x = self.nfp(adj, atom_types)
        x = self.mlp(x)
        if len(x) == x.size:
            x = F.reshape(x, (-1,))
        return x

    def _predict(self, adj, atom_types):
        with chainer.no_backprop_mode():
            with chainer.using_config('train', False):
                x = self.__call__(adj, atom_types)
                return F.sigmoid(x)

    def predict(self, *args, batchsize=32, device=-1):
        if device >= 0:
            chainer.cuda.get_device(device).use()  # Make a specified GPU current
            self.to_gpu()  # Copy the model to the GPU

        data = args[0]
        y_list = []
        for i in range(0, len(data), batchsize):
            adj, atom_types = concat_examples(data[i:i + batchsize], device=device)
            y = self._predict(adj, atom_types)
            y_list.append(cuda.to_cpu(y.data))
        y_array = numpy.concatenate(y_list, axis=0)
        return y_array


class NFPSubModule(chainer.Chain):
    def __init__(self, max_degree, hidden_dim, out_dim):
        super(NFPSubModule, self).__init__()
        num_degree_type = max_degree + 1
        with self.init_scope():
            self.hidden_weights = chainer.ChainList(
                *[L.Linear(hidden_dim, hidden_dim)
                  for _ in range(num_degree_type)]
            )
            self.output_weight = L.Linear(hidden_dim, out_dim)
        self.max_degree = max_degree
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim

    def __call__(self, x, h, adj, deg_conds):
        # x is used for keep each atom's info for next step -> hidden_dim
        # h is used for output of this network              -> out_dim

        # (minibatch, max_num_atoms , hidden_dim)

        # --- Message part ---
        # Take sum along adjacent atoms
        fv = F.batch_matmul(adj, x)
        ##print('fv', fv.shape)  # (minibatch, max_num_atoms, hidden_dim)

        # --- Update part ---
        s0, s1, s2 = fv.shape
        zero_array = self.xp.zeros_like(fv)
        fvds = [F.reshape(F.where(cond, fv, zero_array), (s0 * s1, s2)) for cond in deg_conds]

        out_x = 0
        for hidden_weight, fvd in zip(self.hidden_weights, fvds):
            out_x = out_x + hidden_weight(fvd)

        # out_x shape (minibatch, max_num_atoms, hidden_dim)
        out_x = F.sigmoid(out_x)

        # --- Readout part ---
        dh = self.output_weight(out_x)
        # dh = dh + wout(fvd)
        # dh, h shape (minibatch, max_num_atoms, out_dim)
        dh = F.softmax(dh)
        dh = F.sum(F.reshape(dh, (s0, s1, self.out_dim)), axis=1)  # sum along atom's axis

        out_x = F.reshape(out_x, (s0, s1, s2))
        out_h = h + dh
        return out_x, out_h



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
            self.layers = chainer.ChainList(
                *[NFPSubModule(max_degree, hidden_dim, out_dim)
                  for _ in range(radius)])
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim
        self.max_degree = max_degree
        self.num_degree_type = num_degree_type
        self.radius = radius
        self.concat_hidden = concat_hidden

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
        for l in self.layers:
            x, h = l(x, h, adj, deg_conds)
            if self.concat_hidden:
                h_list.append(h)

        if self.concat_hidden:
            return F.concat(h_list, axis=1)
        else:
            return h
