import pytest
import tensorflow as tf
import QGOpt as qgo

import solver.utils.general_utils as util
import solver.utils.channel_utils as c_util
from solver.utils.testing import same_matrix
from solver.utils.misc import COMPLEX

MANIF = qgo.manifolds.StiefelManifold()

# TODO: more kron tests
KRON_TEST_LIST = [
    (tf.constant([[1, 2],
                  [3, 4]], dtype=COMPLEX),
     tf.constant([[5, 6],
                  [7, 8]], dtype=COMPLEX)),

]


@pytest.mark.parametrize(['matr_a', 'matr_b'], KRON_TEST_LIST)
def test_kron(matr_a: tf.Tensor, matr_b: tf.Tensor):
    big_dim = matr_a.shape[0] * matr_b.shape[0]
    ein_product = tf.einsum('ij,kl->ikjl', matr_a, matr_b)
    kronned_matrix = tf.reshape(ein_product, (big_dim, big_dim))
    assert same_matrix(kronned_matrix, util.kron(matr_a, matr_b))


def test_1q_fidelity_sanity_check():
    for _ in range(5):
        unitary1 = MANIF.random((2, 2), dtype=COMPLEX)
        unitary2 = MANIF.random((2, 2), dtype=COMPLEX)
        channel1 = c_util.convert_1qmatrix_to_channel(unitary1)
        channel2 = c_util.convert_1qmatrix_to_channel(unitary2)

        fid = util.fidel_calc_1q(channel1, channel2)

        assert tf.math.imag(fid) < 1e-3  # constants have been weakened from 1e-5 to support float32 precision
        assert tf.math.real(util.fidel_calc_1q(channel2, channel1) - fid) < 1e-3


def test_2q_fidelity_sanity_check():
    for _ in range(5):
        unitary1 = util.convert_44_to_2222(MANIF.random((4, 4), dtype=COMPLEX))
        unitary2 = util.convert_44_to_2222(MANIF.random((4, 4), dtype=COMPLEX))
        channel1 = c_util.convert_2qmatrix_to_channel(unitary1)
        channel2 = c_util.convert_2qmatrix_to_channel(unitary2)

        fid = util.fidel_calc_2q(channel1, channel2)

        assert tf.math.imag(fid) < 1e-3
        assert tf.math.real(util.fidel_calc_2q(channel2, channel1) - fid) < 1e-3


def test_qubit_swapper():
    unitary1 = MANIF.random((2, 2), dtype=COMPLEX)
    unitary2 = MANIF.random((2, 2), dtype=COMPLEX)
    channel1 = c_util.convert_1qmatrix_to_channel(unitary1)
    channel2 = c_util.convert_1qmatrix_to_channel(unitary2)

    assert same_matrix(tf.reshape(util.kron(channel1, channel2), (4, 4, 4, 4)),
                        util.swap_legs(tf.reshape(util.kron(channel2, channel1), (4, 4, 4, 4))))

    unitary_2q = MANIF.random((4, 4), dtype=COMPLEX)
    channel_2q = c_util.convert_2qmatrix_to_channel(util.convert_44_to_2222(unitary_2q))
    alt_channel_2q = c_util.convert_2qmatrix_to_channel(tf.reshape(unitary_2q, (2, 2, 2, 2)))

    assert same_matrix(channel_2q, util.swap_legs(alt_channel_2q))

    channel_2q_1616 = util.convert_2q_to16x16(channel_2q)
    alt_channel_1616 = util.convert_2q_to16x16(alt_channel_2q)

    assert same_matrix(util.swap_qubits_in_16x16(channel_2q_1616), alt_channel_1616)
    assert same_matrix(util.swap_qubits_in_16x16(alt_channel_1616), channel_2q_1616)

# TODO: test ravel / unravel
# TODO: test 1q 2q Choi swaps on known matrices NUMERICALLY! - this is almost done in a different way
# TODO: test that fidelity calculation is correct (but how?)
# TODO: test kron numerically?
