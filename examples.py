#!/usr/bin/env python
# -*- coding: utf-8 -*-
from cython_functions import *
import numpy as np
import scipy.linalg as scl


def test_mm_product():
    """Test matrix multiplication.

    """
    amat = np.array([[1, 2, 3], [3, 4, 6]], float, order='F')
    bmat = np.array([[5, 6], [7, 8], [1, 2]], float, order='F')
    out1 = amat.dot(bmat)
    out2 = my_dot(amat, bmat)

    assert np.array_equal(out1, out2)


def test_mv_product():
    """Test matrix vector multiplication.

    """
    amat = np.array([[1, 2, 3], [3, 4, 6], [1, 2, 5]], float, order='F')
    bmat = np.array([5, 6, 7], float, order='F')
    out1 = np.sum(amat * bmat, 1)
    out2 = my_dgemv(amat, bmat)

    assert np.array_equal(out1, out2)


def test_vv_product():
    """Test vector vector product.

    """
    amat = np.array([1, 2, 3], float, order='F')
    out1 = np.triu(amat * amat[:, np.newaxis])
    out2 = my_dsyr(amat)

    assert np.array_equal(out1, out2)


def test_sym_m_product():
    """Test symmetric matrix product.

    """
    amat = np.array([[1, 2, 3], [3, 4, 6]], float, order='F')
    out1 = amat.T.dot(amat)
    out2 = my_dsyrk(amat)
    idx = np.triu_indices(amat.shape[1])

    assert np.allclose(out1[idx], out2[idx])

    amat = np.array([[1, 2, 3], [3, 4, 6]], float)
    amat = np.asfortranarray(amat.dot(amat.T))

    out1 = amat.T.dot(amat)
    out2 = my_dsyrk(amat)
    idx = np.triu_indices(amat.shape[1])

    assert np.allclose(out1[idx], out2[idx])


def test_vector_dot():
    """Test vector dot product.

    """
    amat = np.array([1, 2, 3], float)
    bmat = np.array([5, 6, 7], float)
    out1 = (amat * bmat).sum()
    out2 = my_ddot(amat, bmat)

    assert out1 == out2


def test_euclidean():
    """Test Euclidean norm.

    """
    amat = np.array([1, 2, 3], float)
    out1 = (amat ** 2).sum()
    out2 = my_dnorm(amat)

    assert out1 == out2


def test_cholesky2d():
    """Test Cholesky decomposition.

    """
    amat = np.array([[1, 2, 3], [3, 4, 6], [1, 2, 5]], float)
    amat = amat.dot(amat.T)
    out1 = scl.cholesky(amat)
    out2 = np.triu(my_cholesky2d(np.asfortranarray(amat)))

    assert np.allclose(out1, out2)


def test_cholesky3d():
    """Test Cholesky decomposition for 3d array.

    """
    amat = np.array([[1, 2, 3], [3, 4, 6], [1, 2, 5]], float)
    amat = amat.dot(amat.T)
    amat_big = np.array([amat, amat+1, amat+2])

    out1 = []
    for amat in amat_big:
        out1.append(scl.cholesky(amat, lower=True))
    out1 = np.array(out1)

    out2 = np.tril(my_cholesky3d(amat_big))

    assert np.allclose(out1, out2)


def test_cholesky_det():
    """Test Cholesky decomposition.

    """
    amat = np.array([[1, 2, 3], [3, 4, 6], [1, 2, 5]], float)
    amat = amat.dot(amat.T)
    lower = True
    amati, lower = scl.cho_factor(amat, lower=lower, check_finite=False)

    out1 = np.sum(np.diag(amati)**2)
    out2 = my_cholesky_det(np.asfortranarray(amat))

    assert out1 == out2


def test_cholesky_solve():
    """Test Cholesky solve.

    """
    amat = np.array([[1, 2, 3], [3, 4, 6], [1, 2, 5]], float)
    bvec = np.ones(3)
    amat = amat.dot(amat.T)
    lower = False
    amati, lower = scl.cho_factor(amat, lower=lower, check_finite=False)

    out1 = scl.cho_solve((amati, lower), bvec, check_finite=False)
    out2 = my_cholesky_solve(np.asfortranarray(amat), bvec)

    assert np.allclose(out1, out2)


if __name__ == "__main__":

    np.set_printoptions(precision=2, suppress=True)

    test_mm_product()
    test_mv_product()
    test_vv_product()
    test_sym_m_product()
    test_vector_dot()
    test_euclidean()
    test_cholesky2d()
    test_cholesky3d()
    test_cholesky_det()
    test_cholesky_solve()
