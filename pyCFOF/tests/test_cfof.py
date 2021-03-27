# Authors: Lucas Foulon <lucas.foulon@gmail.com>
# License: BSD 3 clause
import unittest

from pyCFOF import ConcentrationFreeOutlierFactor as CFOF

from numpy.testing import assert_array_equal


class TestPyCFOF(unittest.TestCase):

    def test_cfof(self):
        X = [[3, -2], [1, -1], [-1, -1], [-1, 1], [1, 1], [0, 0], [1, 0], [-4, 1]]
        _rho_score_01 = [[0.625], [0.25], [0.375], [0.25], [0.375], [0.25], [0.25], [0.875]]

        cfof = CFOF(n_neighbors=len(X), rho=[0.1])
        assert_array_equal(cfof.fit_predict(X), 7 * [[1]] + [[-1]])
        assert_array_equal(cfof.outlier_factor_, _rho_score_01)

        cfof = CFOF(n_neighbors=len(X), rho=[0.1], contamination=0.2)

        assert_array_equal(cfof.fit_predict(X), [[-1]] + 6 * [[1]] + [[-1]])
        assert_array_equal(cfof.outlier_factor_, _rho_score_01)
        assert round(cfof.offset_, 3) == 0.525


if __name__ == '__main__':
    unittest.main()