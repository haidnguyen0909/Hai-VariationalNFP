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
            chainer.cuda.get_device(
                device).use()  # Make a specified GPU current
            self.to_gpu()  # Copy the model to the GPU

        data = args[0]
        y_list = []
        for i in range(0, len(data), batchsize):
            adj, atom_types = concat_examples(data[i:i + batchsize],
                                              device=device)
            y = self._predict(adj, atom_types)
            y_list.append(cuda.to_cpu(y.data))
        y_array = numpy.concatenate(y_list, axis=0)
        return y_array


class GGNN(chainer.Chain):
    """Gated Graph Neural Networks (GG-NN) Li et al."""
    NUM_EDGE_TYPE = 4

    def __init__(self, hidden_dim, out_dim,
                 n_atom_types, radius, concat_hidden=False, weight_tying=True):
        super(GGNN, self).__init__()
        num_layer = 1 if weight_tying else radius
        with self.init_scope():
            self.embed = L.EmbedID(n_atom_types, hidden_dim)
            self.edge_layer = L.Linear(hidden_dim, self.NUM_EDGE_TYPE * hidden_dim)
            self.update_layer = L.GRU(2 * hidden_dim, hidden_dim)
            self.i_layers = chainer.ChainList(
                *[L.Linear(2 * hidden_dim, out_dim) for _ in range(num_layer)]
            )
            self.j_layers = chainer.ChainList(
                *[L.Linear(hidden_dim, out_dim) for _ in range(num_layer)]
            )
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim
        self.radius = radius
        self.concat_hidden = concat_hidden
        self.weight_tying = weight_tying

    def one_step(self, x, h, adj, x0, step):
        # --- Message part ---
        # (minibatch, max_num_atoms , hidden_dim)
        s0, s1, s2 = x.shape
        #print('x', x.shape)  # (minibatch, max_num_atoms , hidden_dim)
        m = F.reshape(self.edge_layer(F.reshape(x, (s0 * s1, s2))), (s0, s1, s2, self.NUM_EDGE_TYPE))
        #print('m0', m.shape)  # (minibatch, max_num_atoms , hidden_dim, edge_type)
        # Transpose
        m = F.transpose(m, (0, 3, 1, 2))
        #print('m1', m.shape, 'adj', adj.shape)
        # (minibatch, edge_type, max_num_atoms , hidden_dim), (minibatch, edge_type, max_num_atoms, max_num_atoms)

        adj = F.reshape(adj, (s0 * self.NUM_EDGE_TYPE, s1, s1))
        m = F.reshape(m, (s0 * 4, s1, s2))
        #print('m15', m.shape, 'adj', adj.shape)
        # (minibatch*edge_type, max_num_atoms , hidden_dim), (minibatch*edge_type, max_num_atoms, max_num_atoms)

        m = F.batch_matmul(adj, m)

        #print('m2', m.shape)  # (minibatch*edge_type, max_num_atoms , hidden_dim)
        m = F.reshape(m, (s0, 4, s1, s2))
        #print('m25', m.shape)  # (minibatch, edge_type, max_num_atoms , hidden_dim)

        # Take sum
        # (minibatch, max_num_atoms , hidden_dim)
        m = F.sum(m, axis=1)
        #print('m3', m.shape)  # (minibatch, max_num_atoms, hidden_dim)

        # --- Update part ---
        x = F.reshape(x, (s0 * s1, s2))
        m = F.reshape(m, (s0 * s1, s2))
        #print('m4', m.shape, 'x', x.shape)  # (minibatch*max_num_atoms , hidden_dim)

        out_x = self.update_layer(F.concat((x, m), axis=1))
        ##print('fv', fv.shape)  # (minibatch, max_num_atoms, hidden_dim)

        # (minibatch*max_num_atoms , hidden_dim)
        #print('outx', out_x.shape)  # (minibatch*max_num_atoms , hidden_dim)

        # --- Readout part ---
        if self.weight_tying:
            step = 0
        #print('[DEBUG] step', step)
        dh = F.sigmoid(self.i_layers[step](F.concat((out_x, x0), axis=1))) * self.j_layers[step](out_x)
        # dh = dh + wout(fvd)
        # dh, h shape (minibatch, max_num_atoms, out_dim)
        # dh = F.softmax(dh)
        dh = F.sum(F.reshape(dh, (s0, s1, self.out_dim)),
                   axis=1)  # sum along atom's axis

        out_x = F.reshape(out_x, (s0, s1, s2))
        # print('outx', out_x.shape)  # (minibatch, max_num_atoms , hidden_dim)
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
        # reset state
        self.update_layer.reset_state()
        x = self.embed(atom_array)
        ##print('x', x.shape)  # (minibatch, max_num_atoms , hidden_dim)

        h = 0
        # degree_mat = F.sum(adj, axis=1)
        ##print('degree_mat', degree_mat.shape)  # (minibatch, max_num_atoms)
        # deg_conds = [self.xp.broadcast_to(((degree_mat - degree).data == 0)[:, :, None], x.shape)
        #         for degree in range(1, self.num_degree_type + 1)]
        h_list = []
        x0 = F.copy(x, cuda.get_device_from_array(x.data).id)
        s0, s1, s2 = x0.shape
        x0 = F.reshape(x0, (s0 * s1, s2))
        for step in range(self.radius):
            x, h = self.one_step(x, h, adj, x0, step)
            if self.concat_hidden:
                h_list.append(h)

        if self.concat_hidden:
            return F.concat(h_list, axis=1)
        else:
            return h
