import unittest

import numpy

import util


class TestMakeConverter(unittest.TestCase):

    def test_converter(self):
        arr = [3, 1, 2, 3]
        f, g = util.make_converter(arr)

        cnt = 0
        for i in range(len(arr)):
            for j in range(arr[i]):
                self.assertEqual(f(i, j), cnt)
                self.assertEqual(g(cnt), (i, j))
                cnt += 1


class TestToOneHot(unittest.TestCase):

    def test_one_hot(self):
        arr = [1, 2, 0, 1, 2]
        K = 4
        actual = util.to_one_hot(arr, K)
        expected = numpy.array(
            [[0, 1, 0, 0],
             [0, 0, 1, 0],
             [1, 0, 0, 0],
             [0, 1, 0, 0],
             [0, 0, 1, 0]], dtype=numpy.int32)
        numpy.testing.assert_equal(actual, expected)


class TestAdjMatList(unittest.TestCase):

    def test_convert(self):
        mat = [[0, 0, 0, 0],
               [1, 1, 1, 1],
               [1, 0, 0, 0],
               [1, 0, 1, 0]]

        actual = util.adjmat2list(mat)
        expected = [[],
                    [0, 1, 2, 3],
                    [0],
                    [0, 2]]
        self.assertEqual(actual, expected)
