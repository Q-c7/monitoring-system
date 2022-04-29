import tensorflow as tf
import scipy
import numpy as np

from solver.utils.misc import COMPLEX


def same_matrix(matrix1: tf.Tensor, matrix2: tf.Tensor, eps: float = 1e-4) -> bool:
    if matrix1.shape != matrix2.shape:
        return False
    norm = tf.abs(tf.linalg.norm(matrix1 - matrix2))
    if norm >= eps:
        print(f"tf.linalg.norm between matrices {norm.numpy()} is too big for eps={eps}!")
        return False

    return True


MIXED_2Q = tf.eye(4, dtype=COMPLEX) / tf.constant(4, dtype=COMPLEX)
MIXED_1Q = tf.eye(2, dtype=COMPLEX) / tf.constant(2, dtype=COMPLEX)


def _pass_choi_sanity(matrix: tf.Tensor, eps: float = 1e-7) -> int:
    """
    Returns: 0 if everything is passed
    1 if one of eigs has imaginary part
    2 if one of eigs is negative
    3 if eigs do not sum to 1
    """
    eigs = scipy.linalg.eig(matrix)[0]

    for eig in eigs:
        if tf.abs(tf.math.imag(eig)) > 1e-6:  # weakened from 1e-10 to support float32
            print(f"Eig {eig} has too big imaginary part")
            return 1
        if tf.math.real(eig) < -1e-6:
            print(f"Eig {eig} is negative")
            return 2

    if tf.abs(eigs.sum() - 1) >= eps:
        print(f"Sum of eigs {eigs.sum()} is too far from 1: norm = {(tf.abs(eigs.sum() - 1)).numpy()}, eps={eps}")
        return 3

    return 0


def _is_choi_2q(matrix: tf.Tensor, eps: float = 1e-5) -> bool:
    if matrix.shape != (16, 16):
        raise NotImplementedError

    if _pass_choi_sanity(matrix, eps=eps) != 0:
        return False
    # assert _passes_choi_sanity(matrix) == 0 might be better for debugging & testing

    matrix = tf.reshape(matrix, (4, 4, 4, 4))
    part_trace = tf.einsum('ijik->jk', matrix)
    # 'jiki->jk' for 2nd subsystem
    return same_matrix(part_trace, MIXED_2Q, eps=eps)


def _is_choi_1q(matrix: tf.Tensor, eps: float = 1e-5) -> bool:
    if matrix.shape != (4, 4):
        raise NotImplementedError

    if _pass_choi_sanity(matrix, eps=eps) != 0:
        return False

    matrix = tf.reshape(matrix, (2, 2, 2, 2))
    part_trace = tf.einsum('ijik->jk', matrix)
    return same_matrix(part_trace, MIXED_1Q, eps=eps)


def is_choi(matrix: tf.Tensor, eps: float = 1e-5) -> bool:
    if matrix.shape == (4, 4):
        return _is_choi_1q(matrix, eps=eps)
    elif matrix.shape == (16, 16):
        return _is_choi_2q(matrix, eps=eps)
    raise NotImplementedError('Currently only shapes (4,4) and (16,16) are supported')


def is_dm(matrix: tf.Tensor, eps: float = 1e-5) -> bool:
    return _pass_choi_sanity(matrix, eps) == 0


def create_random_channel(qubits: int) -> tf.Tensor:
    """
    ONLY FOR TESTING PURPOSES
    """
    dim = 2 ** qubits
    dim_squared = dim ** 2
    kraus_rank = np.random.randint(1, dim_squared)
    kraus_ops: list[np.array] = []

    for i in range(kraus_rank):
        rnd_matrix = np.random.rand(dim, dim) * 1j / dim_squared
        rnd_matrix += np.random.rand(dim, dim) / dim_squared
        kraus_ops.append(rnd_matrix)

    sum_parts = np.eye(dim, dtype=np.complex128)
    for op in kraus_ops:
        sum_parts -= np.conj(op.T) @ op
    kraus_ops.append(scipy.linalg.sqrtm(sum_parts))

    sanity_check = np.eye(dim, dtype=np.complex128)
    for op in kraus_ops:
        sanity_check -= np.conj(op.T) @ op
    assert np.abs(scipy.linalg.norm(sanity_check)) < 1e-5

    ans = np.zeros((dim_squared, dim_squared), dtype=np.complex128)
    for op in kraus_ops:
        ans += np.reshape(np.einsum('ij,kl->ikjl', op, op.conj()), (dim_squared, dim_squared))

    return tf.convert_to_tensor(ans, dtype=COMPLEX)
