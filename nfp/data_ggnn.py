import pickle
import os
from rdkit import Chem

import chainer.datasets as D
from chainer import dataset
import numpy
from rdkit.Chem import rdmolops

import util
import tox21


MAX_NUMBER_ATOM = 140  # should be more than 132 for tox21 dataset.
filepath_train = 'data/tox21_train_ggnn.pkl'
filepath_val = 'data/tox21_val_ggnn.pkl'
filepath_etc = 'data/tox21_dataset_ggnn.npz'


def construct_discrete_edge_matrix(mol: Chem.Mol):
    if mol is None:
        return None
    N = mol.GetNumAtoms()
    #adj = Chem.rdmolops.GetAdjacencyMatrix(mol)
    #size = adj.shape[0]
    size = MAX_NUMBER_ATOM
    adjs = numpy.zeros((4, size, size), dtype=numpy.float32)
    for i in range(N):
        for j in range(N):
            bond = mol.GetBondBetweenAtoms(i,j)  # type: Chem.Bond
            if bond is not None:
                bondType = str(bond.GetBondType())
                if bondType == 'SINGLE':
                    adjs[0, i, j] = 1.0
                elif bondType =='DOUBLE':
                    adjs[1, i, j] = 1.0
                elif bondType =='TRIPLE':
                    adjs[2, i, j] = 1.0
                elif bondType =='AROMATIC':
                    adjs[3, i, j] = 1.0
                else:
                    print("[ERROR] Unknown bond type", bondType)
                    assert False  # Should not come here
    return adjs


def preprocessor(mol_supplier, label_names):
    descriptors = []
    labels = []
    for mol in mol_supplier:
        if mol is None:
            continue
        if mol.GetNumAtoms() >= MAX_NUMBER_ATOM:
            print('skip this molecule, atom number exeeded.  {} >= {}'
                  .format(mol.GetNumAtoms(), MAX_NUMBER_ATOM))
            continue
        else:
            # print('debug', mol.GetNumAtoms())
            pass

        label = []
        for task in label_names:
            if mol.HasProp(task):
                label.append(int(mol.GetProp(task)))
            else:
                label.append(-1)

        adj = construct_discrete_edge_matrix(mol)
        atom_list = [a.GetSymbol() for a in mol.GetAtoms()]

        labels.append(label)
        descriptors.append((adj, atom_list))

    labels = numpy.array(labels, dtype=numpy.int32)
    return descriptors, labels


def _convert_dataset(dataset, atom2id):
    ret = []
    for d in dataset:
        (adj, atom_list), label = d

#        # 0 padding for adj matrix
#        s0, s1 = adj.shape
#        adj = adj + numpy.eye(s0)
#        adj_array = numpy.zeros((MAX_NUMBER_ATOM, MAX_NUMBER_ATOM),
#                                dtype=numpy.float32)
#
#        adj_array[:s0, :s1] = adj.astype(numpy.float32)
#        #print('adj_array', adj_array)

        # 0 padding for atom_list
        atom_list = [atom2id[a] for a in atom_list]
        n_atom = len(atom_list)
        atom_array = numpy.zeros((MAX_NUMBER_ATOM,), dtype=numpy.int32)
        atom_array[:n_atom] = numpy.array(atom_list)

        ret.append((adj, atom_array, label))
    return ret


def construct_dataset():
    train, val, _ = tox21.get_tox21(preprocessor=preprocessor)

    max_atom = 0
    max_degree = 0
    min_degree = numpy.inf
    for data in [train, val]:
        for d in data:
            adj = d[0][0]
            atom_list = d[0][1]
            max_atom = max(max_atom, len(atom_list))
            deg_array = numpy.sum(adj, axis=0)
            max_degree = max(max_degree, numpy.max(deg_array))
            min_degree = min(min_degree, numpy.min(deg_array))

    assert max_atom <= MAX_NUMBER_ATOM

    # Construct atom2id dictionary
    atom2id = {'empty': 0}  # atom id 0 is for empty padding
    atoms = [d[0][1] for d in train] + [d[0][1] for d in val]
    atoms = sum(atoms, [])  # flatten
    for a in atoms:
        if a not in atom2id:
            atom2id[a] = len(atom2id)

    train = _convert_dataset(train, atom2id)
    val = _convert_dataset(val, atom2id)

    C = 12  # class num
    print('max_atom', max_atom)
    print('max_degree', max_degree)
    print('min_degree', min_degree)

    # Save dataset
    with open(filepath_train, mode='wb') as f:
        pickle.dump(train, f)
    with open(filepath_val, mode='wb') as f:
        pickle.dump(train, f)
    d = {
        'max_degree': max_degree,
        'atom2id': atom2id,
        'C': C
    }
    numpy.savez(filepath_etc, **d)
    print('dataset saved to {}'.format(filepath_etc))
    return train, val, max_degree, atom2id, C


def load():
    if not os.path.exists(filepath_train):
        construct_dataset()

    with open(filepath_train, mode='rb') as f:
        train = pickle.load(f)
    with open(filepath_val, mode='rb') as f:
        val = pickle.load(f)
    d = numpy.load(filepath_etc)
    return train, val, d['max_degree'], d['atom2id'].item(), d['C']


if __name__ == '__main__':
    train, val, max_degree, atom2id, C = load()
    print('train', type(train), len(train))
    print('train0', train[0])
    adj, atom_array, label = train[0]
    print('adj: ', type(adj), adj.shape, adj)
    # [[], [2], [1, 3, 12], [2, 4, 9], [3, 5], [4, 6, 7], [5], [5, 8], [7, 9], [3, 8, 10], [9, 11], [10, 12, 17], [2, 11, 13], ...
    print('atom_list: ', type(atom_array), atom_array.shape, atom_array)
    print('label: ', label)

    print('train1')
    adj, atom_array, label = train[1]
    print('adj: ', type(adj), adj.shape, adj)
    # [[1, 14, 25], [0, 2, 22], [1, 3, 12], [2, 4, 11], [3, 5, 8], [4, 6, 7], [5], [5], [4, 9], [8, 10], [9, 11], [3, 10], [2, 13, 18], [12, 14, 15], [0, 13], [13, 16, 21], [15, 17, 20], [16, 18, 19], [12, 17], [17], [16], [15], [1, 23], [22, 24, 28], [23, 25, 27], [0, 24, 26], [25], [24], [23], [], []]
    print('atom_list: ', type(atom_array), atom_array.shape, atom_array)
    print('label: ', label)

    print('val', type(val), len(val))
    adj, atom_array, label = val[0]
    print('val0')
    print('adj: ', type(adj), adj.shape, adj)
    print('atom_list: ', type(atom_array), atom_array.shape, atom_array)
    print('label: ', label)

    # This is howmany type of atom is included.
    print('max degree', type(max_degree), max_degree)
    print('atom2id', type(atom2id), len(atom2id))
    print(atom2id)
    print('C', type(C), C)
