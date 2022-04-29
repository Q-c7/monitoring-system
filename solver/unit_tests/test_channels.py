import QGOpt as qgo
import tensorflow as tf

import solver.utils.general_utils as util
import solver.utils.channel_utils as c_util
from solver.utils.misc import COMPLEX
from solver.utils.testing import same_matrix, is_choi, create_random_channel


MANIF = qgo.manifolds.StiefelManifold()


def test_1q_channel_creation():
    for _ in range(5):
        unitary = MANIF.random((2, 2), dtype=COMPLEX)
        human_channel = util.kron(unitary, tf.math.conj(unitary))
        assert same_matrix(human_channel, c_util.convert_1qmatrix_to_channel(unitary))


def test_1q_channel_choi_sanity_check():
    """
    this one does not check the Choi conversion itself (it is in test_utils)
    but rather if the created channel satisfies some obvious conditions
    """
    for _ in range(5):
        unitary = MANIF.random((2, 2), dtype=COMPLEX)
        channel = c_util.convert_1qmatrix_to_channel(unitary)
        choi = util.choi_swap_1qchannel(channel) / tf.constant(2, dtype=COMPLEX)
        assert is_choi(choi)


def test_2q_channel_choi_sanity_check():
    for _ in range(5):
        unitary2222 = util.convert_44_to_2222(MANIF.random((4, 4), dtype=COMPLEX))
        channel = c_util.convert_2qmatrix_to_channel(unitary2222)
        choi = util.choi_swap_2qchannel(channel) / tf.constant(4, dtype=COMPLEX)
        assert is_choi(choi)


def test_2q_channel_conversion():
    for _ in range(5):
        unitary = MANIF.random((4, 4), dtype=COMPLEX)
        unitary_2222 = util.convert_44_to_2222(unitary)
        ncon_channel = c_util.convert_2qmatrix_to_channel(unitary_2222)

        human_channel = util.kron(unitary, tf.math.conj(unitary))

        assert same_matrix(human_channel, util.convert_2q_to16x16(ncon_channel))
        assert same_matrix(ncon_channel, util.convert_2q_from16x16(human_channel))

        assert same_matrix(human_channel, util.convert_2q_to16x16(util.convert_2q_from16x16(human_channel)))
        assert same_matrix(ncon_channel, util.convert_2q_from16x16(util.convert_2q_to16x16(ncon_channel)))


def test_qgo_conversion():
    for _ in range(5):
        ch1 = create_random_channel(1)
        pars1 = c_util.convert_channel_to_params(ch1)
        ch1_twin = c_util.convert_params_to_channel(pars1)
        assert same_matrix(ch1, ch1_twin)

        ch2 = util.convert_2q_from16x16(create_random_channel(2))
        pars2 = c_util.convert_channel_to_params(ch2)
        ch2_twin = c_util.convert_params_to_channel(pars2)
        assert same_matrix(ch2, ch2_twin)
