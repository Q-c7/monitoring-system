import tensorflow as tf
import numpy as np

import solver.utils.general_utils as util
import solver.utils.channel_utils as c_util
from solver.utils.misc import COMPLEX, TENSOR

sigmaX = tf.constant([[0. + 0.j, 1 + 0.j],
                      [1. + 0.j, 0. + 0.j]], dtype=COMPLEX)
sigmaY = tf.constant([[0. + 0.j, 0. - 1j],
                      [0. + 1j, 0. + 0.j]], dtype=COMPLEX)
sigmaZ = tf.constant([[1, 0],
                      [0, -1]], dtype=COMPLEX)
E = tf.eye(2, dtype=COMPLEX)
E_channel = c_util.convert_1qmatrix_to_channel(E)


# def make_1q_AD_channel(target: tf.Tensor, args_list: list[float]) -> tf.Tensor:
#     """
#     TODO: Write docstring
#     """
#     assert len(args_list) == 1  # TODO: make custom exception instead of asserts
#     gamma = args_list[0]
#
#     e0 = tf.convert_to_tensor([[1, 0], [0, tf.math.sqrt(1 - gamma)]], dtype=COMPLEX)
#     e1 = tf.convert_to_tensor([[0, tf.math.sqrt(gamma)], [0, 0]], dtype=COMPLEX)
#     e0_channel = c_util.convert_1qmatrix_to_channel(e0)
#     e1_channel = c_util.convert_1qmatrix_to_channel(e1)
#     output = (e0_channel + e1_channel) @ target
#     return output
#
#
# def make_1q_PD_channel(target: tf.Tensor, args_list: list[float]) -> tf.Tensor:
#     """
#     TODO: Write docstring
#     """
#     assert len(args_list) == 1
#     gamma = args_list[0]
#
#     e0 = tf.convert_to_tensor([[1, 0], [0, tf.math.sqrt(1 - gamma)]], dtype=COMPLEX)
#     e1 = tf.convert_to_tensor([[0, 0], [0, tf.math.sqrt(gamma)]], dtype=COMPLEX)
#     e0_channel = c_util.convert_1qmatrix_to_channel(e0)
#     e1_channel = c_util.convert_1qmatrix_to_channel(e1)
#     output = (e0_channel + e1_channel) @ target
#     return output


# def make_1q_APD_channel(target: tf.Tensor, args_list: tf.Tensor) -> tf.Tensor:
#     """
#     TODO: Write docstring
#     """
#     gamma = args_list[0]
#
#     e0 = tf.convert_to_tensor([[1, 0], [0, tf.math.sqrt(1 - gamma)]], dtype=COMPLEX)
#     e1_p = tf.convert_to_tensor([[0, 0], [0, tf.math.sqrt(gamma)]], dtype=COMPLEX)
#     e1_a = tf.convert_to_tensor([[0, tf.math.sqrt(gamma)], [0, 0]], dtype=COMPLEX)
#     e0_channel = c_util.convert_1qmatrix_to_channel(e0)
#     e1_p_channel = c_util.convert_1qmatrix_to_channel(e1_p)
#     e1_a_channel = c_util.convert_1qmatrix_to_channel(e1_a)
#
#     # TODO: check correctness
#     output = (e0_channel + e1_p_channel) @ target
#     output = (e0_channel + e1_a_channel) @ output
#
#     return output


@tf.function
def create_1q_depol_matrix(p: TENSOR) -> TENSOR:
    """
    Creates a Tensor(4, 4)[complex128] describing a 1-qubit depolarizing quantum channel
    """
    depol = (c_util.convert_1qmatrix_to_channel(sigmaX) * p * 0.25 +
             c_util.convert_1qmatrix_to_channel(sigmaY) * p * 0.25 +
             c_util.convert_1qmatrix_to_channel(sigmaZ) * p * 0.25 +
             E_channel * (1 - 0.75 * p))
    return depol


@tf.function
def create_2q_depol_matrix(p: TENSOR):
    """
    Creates a Tensor(4, 4)[complex128] describing a 2-qubit depolarizing quantum channel
    """
    big_e_channel = util.kron(E_channel, E_channel)
    depol = big_e_channel * (1 - p)
    for m1 in [sigmaX, sigmaY, sigmaZ, E]:
        for m2 in [sigmaX, sigmaY, sigmaZ, E]:
            m = np.kron(m1, m2)
            m = util.swap_legs(tf.reshape(m, (2, 2, 2, 2)))
            depol += c_util.convert_2qmatrix_to_channel(m) * p * 0.0625  # 1/16
    return depol


@tf.function
def create_AP_matrix(gamma1: TENSOR, gamma2: TENSOR):
    """
    Args:
        gamma1: Tensor()[float] - parameter for amplitude damping
        gamma2: Tensor()[float] - parameter for amplitude damping

    Returns:
        Tensor(4, 4)[complex128] describing a 1-qubit amplitude damping & phase damping quantum channel
    """
    e0_a = tf.convert_to_tensor([[1, 0], [0, tf.math.sqrt(1 - gamma1)]], dtype=COMPLEX)
    e0_p = tf.convert_to_tensor([[1, 0], [0, tf.math.sqrt(1 - gamma2)]], dtype=COMPLEX)
    e1_p = tf.convert_to_tensor([[0, 0], [0, tf.math.sqrt(gamma2)]], dtype=COMPLEX)
    e1_a = tf.convert_to_tensor([[0, tf.math.sqrt(gamma1)], [0, 0]], dtype=COMPLEX)
    e0_a_channel = c_util.convert_1qmatrix_to_channel(e0_a)
    e0_p_channel = c_util.convert_1qmatrix_to_channel(e0_p)
    e1_p_channel = c_util.convert_1qmatrix_to_channel(e1_p)
    e1_a_channel = c_util.convert_1qmatrix_to_channel(e1_a)

    # TODO: check correctness
    ap_channel = (e0_a_channel + e1_a_channel) @ (e0_p_channel + e1_p_channel)

    return ap_channel


@tf.function
def make_1q_hybrid_channel(target: TENSOR, args_list: TENSOR) -> TENSOR:
    """
    Args:
        target: a Tensor(4,4)[complex128] - a channel, which we are noising now will be applied
        args_list: a Tensor(3)[float] containing arguments for applying noise models.
        First arg is p for depolarization, and args 2 & 3 are for gamma1 & gamma2 - params for APD

    Returns:
        Tensor(4,4)[complex128] - new noised channel
    """
    p = tf.cast(args_list[0], COMPLEX)
    gamma1 = args_list[1] / 2
    gamma2 = args_list[2] / 2

    ap_channel = create_AP_matrix(gamma1, gamma2)

    output = (target * (1 - p) +
              c_util.convert_1qmatrix_to_channel(sigmaX) * p * 0.25 +
              E_channel * p * 0.25 +
              c_util.convert_1qmatrix_to_channel(sigmaY) * p * 0.25 +
              c_util.convert_1qmatrix_to_channel(sigmaZ) * p * 0.25)
    output = ap_channel @ output @ ap_channel

    return output


@tf.function
def make_2q_hybrid_channel(target: TENSOR, args_list: TENSOR) -> TENSOR:
    """
    Args:
        target: a Tensor(4,4)[complex128] - a channel, which we are noising now will be applied
        args_list: a Tensor(3)[float] containing arguments for applying noise models.
        First arg is p for depolarization, and args 2 & 3 are for gamma1 & gamma2 - params for APD
    Returns:
        Tensor(4,4)[complex128] - new noised channel
    """
    p = tf.cast(args_list[0], COMPLEX)
    gamma1 = args_list[1] / 2
    gamma2 = args_list[2] / 2

    ap_channel = create_AP_matrix(gamma1, gamma2)
    ap_channel_2q = util.kron(ap_channel, ap_channel)

    dp_channel = create_1q_depol_matrix(p)
    dp_channel_2q = util.kron(dp_channel, dp_channel)

    # TODO: check correctness
    reshaped_target = tf.reshape(target, (16, 16))
    output = ap_channel_2q @ dp_channel_2q @ reshaped_target @ ap_channel_2q
    output = tf.reshape(output, (4, 4, 4, 4))

    return output


@tf.function
def nearest_kron_product(A: TENSOR, n_qb: int) -> TENSOR:
    """
    Yields nearest Kronecker product to a matrix.

    Given a matrix A and a shape, solves the problem
    min || A - kron(B, C) ||_{Fro}^2
    where the minimization is over B with (the specified shape) and C.
    The size of the SVD computed in this implementation is the size of the input
    argument A, and so the complexity scales like O((N^2)^3) = O(N^6).
    Args:
        A: m x n matrix
        n_qb: number of qubits which define the dimension
    Returns:
        Approximating factor B (but calculates both B and C)
    """
    Bshape = [2 ** n_qb, 2 ** n_qb]
    # Cshape = A.shape[0] // Bshape[0], A.shape[1] // Bshape[1]

    blocks = map(lambda blockcol: tf.split(blockcol, Bshape[0], 0),
                 tf.split(A, Bshape[1], 1))
    Atilde = tf.stack([tf.reshape(block, (-1, )) for blockcol in blocks
                       for block in blockcol])

    s, U, V = tf.linalg.svd(Atilde)
    idx = tf.argmax(s)
    s = tf.cast(s, COMPLEX)
    U = tf.cast(U, COMPLEX)
    V = tf.cast(V, COMPLEX)

    B = tf.math.sqrt(s[idx]) * tf.transpose(tf.reshape(U[:, idx], Bshape))
    # C = np.sqrt(s[idx]) * V[idx, :].reshape(Cshape)

    return tf.convert_to_tensor(B, dtype=COMPLEX)


@tf.function
def _pseudokron_eigs(lambds: TENSOR) -> TENSOR:
    """
    TODO: Write docstring
    """
    return tf.reshape(tf.transpose(tf.stack([lambds] * len(lambds))) - lambds, (-1,))


@tf.function
def _dispersed_eigs(lambds: TENSOR, sigma: float) -> TENSOR:
    """
    TODO: Write docstring
    """
    sigma = tf.cast(sigma, COMPLEX)
    return tf.math.exp(1j * lambds - lambds ** 2 * sigma ** 2 / 2)


@tf.function
def create_1q_dispersed_channel(target: TENSOR, sigma: float) -> TENSOR:
    """
    TODO: Write docstring
    """
    # it seems that Choi form coincides with our channel in single-qubit case
    basic_gate = nearest_kron_product(target, 1)

    eigenvals, eigenvecs = tf.linalg.eig(basic_gate)

    lambds = tf.math.log(eigenvals) * -1j

    new_eigenvals = _pseudokron_eigs(lambds)
    middle_matrix = tf.linalg.diag(_dispersed_eigs(new_eigenvals, sigma))

    left_matrix = util.kron(eigenvecs, tf.math.conj(eigenvecs))

    right_matrix = util.kron(tf.linalg.adjoint(eigenvecs), tf.transpose(eigenvecs))

    return left_matrix @ middle_matrix @ right_matrix


@tf.function
def create_2q_dispersed_channel(target: TENSOR, sigma: float) -> TENSOR:
    """
    TODO: Write docstring
    """
    basic_gate = nearest_kron_product(util.convert_2q_to16x16(target), 2)

    eigenvals, eigenvecs = tf.linalg.eig(basic_gate)
    lambds = tf.math.log(eigenvals) * -1j
    new_eigenvals = _pseudokron_eigs(lambds)
    middle_matrix = tf.linalg.diag(_dispersed_eigs(new_eigenvals, sigma))

    left_matrix = util.kron(eigenvecs, tf.math.conj(eigenvecs))
    right_matrix = util.kron(tf.linalg.adjoint(eigenvecs), tf.transpose(eigenvecs))
    wrong_shaped_matrix = left_matrix @ middle_matrix @ right_matrix
    good_matrix = util.convert_2q_from16x16(wrong_shaped_matrix)

    return good_matrix


@tf.function
def make_1q_4pars_channel(target: TENSOR, args_list: list[float]) -> TENSOR:
    """
    TODO: Write docstring
    """
    # assert (len(args_list) == 4)
    # p_dep, gamma1, gamma2, sigma = args_list

    disp_channel = create_1q_dispersed_channel(target, args_list[0])
    output = make_1q_hybrid_channel(disp_channel, args_list[1:])

    return output


@tf.function
def make_2q_4pars_channel(target: TENSOR, args_list: list[float]) -> TENSOR:
    """
    TODO: Write docstring
    """
    # assert (len(args_list) == 4)
    # p_dep, gamma1, gamma2, sigma = args_list
    disp_channel = create_2q_dispersed_channel(target, args_list[0])
    output = make_2q_hybrid_channel(disp_channel, args_list[1:])

    return output

