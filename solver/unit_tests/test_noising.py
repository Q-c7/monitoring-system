import pytest
import tensorflow as tf
import QGOpt as qgo
import typing as tp

import solver.utils.general_utils as util
import solver.noising_tools as ns
import solver.utils.channel_utils as c_util
from solver.utils.misc import COMPLEX
from solver.utils.testing import same_matrix, is_choi, create_random_channel, is_dm

MANIF = qgo.manifolds.StiefelManifold()


def test_zero_noise():
    channel = create_random_channel(1)
    assert same_matrix(ns.make_1q_hybrid_channel(channel, tf.convert_to_tensor([0., 0., 0.], dtype=COMPLEX)), channel)

    channel2 = util.convert_2q_from16x16(create_random_channel(2))
    assert same_matrix(ns.make_2q_hybrid_channel(channel2, [0, 0, 0]), channel2)

    unitary1 = MANIF.random((2, 2), dtype=COMPLEX)
    unit_channel = c_util.convert_1qmatrix_to_channel(unitary1)
    assert same_matrix(ns.make_1q_4pars_channel(unit_channel, [0, 0, 0, 0]), unit_channel)

    unitary2 = MANIF.random((4, 4), dtype=COMPLEX)
    unit_channel2 = c_util.convert_2qmatrix_to_channel(util.convert_44_to_2222(unitary2))
    assert same_matrix(ns.make_2q_4pars_channel(unit_channel2, [0, 0, 0, 0]), unit_channel2)


def test_nkp():
    for _ in range(5):
        unitary1 = MANIF.random((2, 2), dtype=COMPLEX)
        unit_channel = c_util.convert_1qmatrix_to_channel(unitary1)
        extracted_unitary = ns.nearest_kron_product(unit_channel, 1)
        # assert same_matrix(extracted_unitary, unitary1) will NOT work because of arbitrary phase
        assert same_matrix(c_util.convert_1qmatrix_to_channel(extracted_unitary), unit_channel)

        unitary2 = MANIF.random((4, 4), dtype=COMPLEX)
        unit_channel2 = c_util.convert_2qmatrix_to_channel(util.convert_44_to_2222(unitary2))
        channel_16x16 = util.convert_2q_to16x16(unit_channel2)
        extracted_unitary2 = util.convert_44_to_2222(ns.nearest_kron_product(channel_16x16, 2))
        assert same_matrix(c_util.convert_2qmatrix_to_channel(extracted_unitary2), unit_channel2)


NOISE_PARAMS_1Q = [
    [[0.3, 0, 0, 0]],
    [[0, 0.3, 0, 0]],
    [[0, 0, 0.3, 0]],
    [[0, 0, 0, 0.3]],
    [[0.15, 0.2, 0.2, 0.5]],
    [[0.05, 0.1, 0.05, 0.1]],
]

NAMES_1Q = [
    'depol-0.3',
    'gamma1-0.3',
    'gamma2-0.3',
    'gauss-0.3',
    'combined1',
    'combined2'
]


@pytest.mark.parametrize(['noise_list'], NOISE_PARAMS_1Q, ids=NAMES_1Q)
def test_1q_channel_noise_params(noise_list: list[int]):
    for _ in range(5):
        unitary = MANIF.random((2, 2), dtype=COMPLEX)
        channel = c_util.convert_1qmatrix_to_channel(unitary)
        noised_channel = ns.make_1q_4pars_channel(channel, noise_list)
        choi = util.choi_swap_1qchannel(noised_channel) / tf.constant(2, dtype=COMPLEX)
        assert is_choi(choi, eps=2e-6)
        rho_in = tf.transpose(tf.reshape(choi, (2, 2, 2, 2)), (0, 2, 1, 3))[:, :, 0, 0]
        assert is_dm(rho_in * 2, eps=2e-6)


NOISE_PARAMS_2Q = [
    [[0.3, 0, 0, 0]],
    [[0, 0.3, 0, 0]],
    [[0, 0, 0.3, 0]],
    [[0, 0, 0, 0.3]],
    [[0.15, 0.2, 0.2, 0.5]],
    [[0.05, 0.1, 0.05, 0.1]],
]


@pytest.mark.parametrize(['noise_list'], NOISE_PARAMS_2Q, ids=NAMES_1Q)
def test_2q_channel_noise_params(noise_list: list[int]):
    for _ in range(5):
        unitary = util.convert_44_to_2222(MANIF.random((4, 4), dtype=COMPLEX))
        channel = c_util.convert_2qmatrix_to_channel(unitary)
        noised_channel = ns.make_2q_4pars_channel(channel, noise_list)
        choi = util.choi_swap_2qchannel(noised_channel) / tf.constant(4, dtype=COMPLEX)
        assert is_choi(choi, eps=2e-6)


NOISE_GRAD_TESTS = [
    (ns.make_1q_hybrid_channel, ns.make_2q_hybrid_channel, tf.Variable([0.1, 0.1, 0.1])),
    (ns.make_1q_4pars_channel, ns.make_2q_4pars_channel, tf.Variable([0.0, 0.0, 0.0, 0.3])),
    (ns.make_1q_4pars_channel, ns.make_2q_4pars_channel, tf.Variable([0.1, 0.1, 0.1, 0.1]))
]
NAMES_GRAD = ['ad/pd/depol', 'gaussian-blur', 'combined']


@pytest.mark.parametrize(['func_1q', 'func_2q', 'params'], NOISE_GRAD_TESTS, ids=NAMES_GRAD)
def test_noise_does_not_kill_grad(func_1q: tp.Callable[[tf.Tensor, ...], tf.Tensor],
                                  func_2q: tp.Callable[[tf.Tensor, ...], tf.Tensor],
                                  params: tf.Variable):
    for _ in range(5):
        with tf.GradientTape() as tape:
            unitary = MANIF.random((2, 2), dtype=COMPLEX)
            channel = c_util.convert_1qmatrix_to_channel(unitary)
            noised_channel = func_1q(channel, params)
            loss = tf.math.abs(tf.linalg.norm(noised_channel - channel) ** 2) * 1000
            grad = tape.gradient(loss, params)

            for idx, elem in enumerate(grad):
                if tf.abs(params[idx]) > 1e-10:
                    assert elem > 1e-1

        with tf.GradientTape() as tape:
            unitary_2q = util.convert_44_to_2222(MANIF.random((4, 4), dtype=COMPLEX))
            channel_2q = c_util.convert_2qmatrix_to_channel(unitary_2q)
            noised_channel_2q = func_2q(channel_2q, params)
            loss = tf.math.abs(tf.linalg.norm(noised_channel_2q - channel_2q) ** 2) * 1000
            grad = tape.gradient(loss, params)

            for idx, elem in enumerate(grad):
                if tf.abs(params[idx]) > 1e-10:
                    assert elem > 1e-1

# TODO: more noised tests (maybe numeric assertions on create AD/PD matrix?)
