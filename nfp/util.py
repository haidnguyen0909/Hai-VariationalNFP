import numpy


def to_one_hot(ids, K):
    """convert IDs to 1-of-K one-hot vectors

    Args:
        ids (list of ints): list of ids
        K (int): the length of one-hot vector

    Returns: :class:`numpy.ndarray`
        whose shape is (len(ids), K)

    Each element in ids must be 0 <= id < K.
    """

    return (numpy.arange(K) == numpy.array(ids)[:, None]).astype(numpy.int32)


def make_converter(n_atoms):
    """make id converter

    Args:
        n_atoms (list of int): atoms

    Returns: pair of functions (f, g)
        f and g are inverse of each other meaning:
        f(mol_id, atom_id) = id
        g(id) = mol_id, atom_id

    mol_id, atom_id and id are 0-index.

    e.g. n_atoms = [3, 1, 2, 3] (xxx, x, xx, xxx)
    The 0-th atom of the second mols:
        f(2, 0) = 3 + 1 + 1 - 1 = 4
        g(4) = (2, 0)
    """

    n_atoms = numpy.array(n_atoms)
    if (n_atoms == 0).any():
        raise ValueError(
            'Invalid input. Each element should be positive')

    c_sum = numpy.cumsum(n_atoms)

    def pairids2id(mol_id, atom_id):
        if mol_id == 0:
            return int(atom_id)
        else:
            return int(c_sum[mol_id - 1] + atom_id)

    def id2pairids(id_):
        mol_id = numpy.searchsorted(c_sum, id_, side='right')
        if mol_id == 0:
            atom_id = id_
        else:
            atom_id = id_ - c_sum[mol_id - 1]
        return int(mol_id), int(atom_id)

    return pairids2id, id2pairids


def adjmat2list(mat):
    N = len(mat)
    return [list(numpy.arange(N)[numpy.array(m) == 1])
            for m in mat]
