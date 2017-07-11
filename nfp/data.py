import chainer.datasets as D
from chainer import dataset
import numpy
from rdkit.Chem import rdmolops

import util
import tox21


def preprocessor(mol_supplier, label_names):
    descriptors = []
    labels = []
    for mol in mol_supplier:
        if mol is None:
            continue

        label = []
        for task in label_names:
            if mol.HasProp(task):
                label.append(int(mol.GetProp(task)))
            else:
                label.append(-1)

        adj = rdmolops.GetAdjacencyMatrix(mol)
        adj = util.adjmat2list(adj)

        atom_list = [a.GetSymbol() for a in mol.GetAtoms()]

        labels.append(label)
        descriptors.append((adj, atom_list))

    labels = numpy.array(labels, dtype=numpy.int32)
    return descriptors, labels


def concat_example(batch, device):
    if len(batch) == 0:
        raise ValueError('batch is empty')

    first_elem = batch[0]
    assert isinstance(first_elem, tuple)

    result = []
    for i in range(len(first_elem)):
        result.append([example[i] for example in batch])
    # labels should be ndarray
    result[-1] = dataset.to_device(
        device, numpy.array(result[-1], numpy.int32))
    return tuple(result)


def _convert_dataset(dataset, atom2id):
    ret = []
    for d in dataset:
        (adj, atom_list), label = d
        atom_list = [atom2id[a] for a in atom_list]
        ret.append((adj, atom_list, label))
    return ret


def load():
    train, val, _ = tox21.get_tox21(preprocessor=preprocessor)

    max_degree = 0
    for data in [train, val]:
        for d in data:
            adj = d[0][0]
            for adj_elem in adj:
                max_degree = max(max_degree, len(adj_elem))

    atom2id = {}
    atoms = [d[0][1] for d in train] + [d[0][1] for d in val]
    atoms = sum(atoms, [])  # flatten
    for a in atoms:
        if a not in atom2id:
            atom2id[a] = len(atom2id)

    train = _convert_dataset(train, atom2id)
    val = _convert_dataset(val, atom2id)

    C = 12  # class num
    return train, val, max_degree, atom2id, C



if __name__ == '__main__':
    train, val, max_degree, atom2id, C = load()
    print('train', type(train), len(train))
    print('train0', train[0])
    adj, atom_list, label = train[0]
    print('adj: ', adj)
    print('atom_list: ', atom_list)
    print('label: ', label)

    print('train1', train[1])
    adj, atom_list, label = train[1]
    print('adj: ', adj)
    print('atom_list: ', atom_list)
    print('label: ', label)

    print('val', type(val), len(val))
    print(val[0])

    # This is howmany type of atom is included.
    print('max degree', type(max_degree), max_degree)
    print('atom2id', type(atom2id), len(atom2id))
    print(atom2id)
    print(type(C), C)
