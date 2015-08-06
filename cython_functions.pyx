cimport cython
import numpy as np
cimport numpy as cnp
from scipy.linalg.cython_blas cimport *
from scipy.linalg.cython_lapack cimport *

# http://www.math.utah.edu/software/lapack/

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def my_dot(double[::1, :] amat, double[::1, :] bmat):

    cdef:
        int m = amat.shape[0]
        int lda = amat.shape[0]
        int ldc = amat.shape[0]
        int k = amat.shape[1]
        int ldb = amat.shape[1]
        int n = bmat.shape[1]
        double[::1, :] cmat = np.empty((m, n), float, order='F')
        double alpha, beta

    alpha = 1.0
    beta = 0.0

    # http://www.math.utah.edu/software/lapack/lapack-blas/dgemm.html
    dgemm('N', 'N', &m, &n, &k, &alpha, &amat[0, 0], &lda,
          &bmat[0, 0], &ldb, &beta, &cmat[0, 0], &ldc)

    return np.asarray(cmat)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def my_dgemv(double[::1, :] amat, double[:] bvec):

    cdef:
        int n = amat.shape[0]
        int inc = 1
        double[:] cmat = np.empty(n, float, order='F')
        double alpha = 1.0
        double beta = 0.0

    # http://www.math.utah.edu/software/lapack/lapack-blas/dgemv.html
    dgemv('N', &n, &n, &alpha, &amat[0, 0], &n, &bvec[0], &inc,
          &beta, &cmat[0], &inc)

    return np.asarray(cmat)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def my_ddot(double[:] avec, double[:] bvec):

    cdef:
        double out = 0.0
        int n = avec.shape[0]
        int inc = 1

    # http://www.mathkeisan.com/usersguide/man/ddot.html
    out = ddot(&n, &avec[0], &inc, &bvec[0], &inc)

    return np.asarray(out)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def my_dnorm(double[:] avec):

    cdef:
        double out = 0.0
        int n = avec.shape[0]
        int inc = 1

    # http://www.mathkeisan.com/usersguide/man/dnrm2.html
    out = dnrm2(&n, &avec[0], &inc)**2

    return out


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def my_dsyr(double[:] avec):

    cdef:
        double out = 0.0
        double alpha = 1.0
        int n = avec.shape[0]
        int inc = 1
        double[::1, :] cmat = np.zeros((n, n), float, order='F')

    # http://www.mathkeisan.com/usersguide/man/ddot.html
    dsyr('U', &n, &alpha, &avec[0], &inc, &cmat[0, 0], &n)

    return np.asarray(cmat)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def my_dsyrk(double[::1, :] amat):

    cdef:
        int n = amat.shape[1]
        int k = amat.shape[0]
        double[::1, :] cmat = np.empty((n, n), float, order='F')
        double alpha = 1.0
        double beta = 0.0

    # http://www.math.utah.edu/software/lapack/lapack-blas/dsyrk.html
    dsyrk('U', 'T', &n, &k, &alpha, &amat[0, 0], &k,
          &beta, &cmat[0, 0], &n)

    return np.asarray(cmat)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def my_cholesky2d(double[::1, :] amat):

    cdef:
        double alpha = 1.0
        double beta = 0.0
        int info = 0
        int n = amat.shape[0]

    # http://www.math.utah.edu/software/lapack/lapack-d/dpotrf.html
    dpotrf('U', &n, &amat[0, 0], &n, &info)

    return np.asarray(amat)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def my_cholesky3d(double[:, :, :] amat):

    cdef:
        double alpha = 1.0
        double beta = 0.0
        int info = 0
        int n = amat.shape[0]

    for i in range(n):
        # http://www.math.utah.edu/software/lapack/lapack-d/dpotrf.html
        dpotrf('U', &n, &amat[i, 0, 0], &n, &info)

    return np.asarray(amat)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def my_cholesky_det(double[::1, :] amat):

    cdef:
        double out = 0.0
        double alpha = 1.0
        double beta = 0.0
        int info = 0
        int n = amat.shape[0]

    # http://www.math.utah.edu/software/lapack/lapack-d/dpotrf.html
    dpotrf('L', &n, &amat[0, 0], &n, &info)

    for i in range(n):
        out += amat[i, i] ** 2

    return out


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def my_cholesky_solve(double[::1, :] amat, double[:] bvec):

    cdef:
        double alpha = 1.0
        double beta = 0.0
        int info = 0
        int nrhs = 1
        int n = amat.shape[0]

    # http://www.math.utah.edu/software/lapack/lapack-d/dpotrf.html
    dpotrf('L', &n, &amat[0, 0], &n, &info)

    # http://www.math.utah.edu/software/lapack/lapack-d/dpotrs.html
    dpotrs('L', &n, &nrhs, &amat[0, 0], &n, &bvec[0], &n, &info)

    return np.asarray(bvec)
