import numpy as np
import tensorflow as tf  # tf 2.x
import math
import scipy

from solver.utils.misc import COMPLEX, FLOAT
from qiskit.quantum_info import diamond_norm, Choi


@tf.function
def create_unitary_rotation_y(angle: float) -> tf.Tensor:
    """
    Creates a unitary operator describing rotations of quantum state via Y axis.
    """
    return tf.convert_to_tensor([[math.cos(angle/2), -math.sin(angle/2)],
                                [+math.sin(angle/2), math.cos(angle/2)]], dtype=COMPLEX)


@tf.function
def create_unitary_rotation_x(angle: float) -> tf.Tensor:
    """
    Creates a unitary operator describing rotations of quantum state via X axis.
    """
    return tf.convert_to_tensor([[math.cos(angle/2) + 0j, -1j * math.sin(angle/2)],
                                [-1j * math.sin(angle/2), math.cos(angle/2) + 0j]], dtype=COMPLEX)


@tf.function
def create_unitary_rotation_z(angle: float) -> tf.Tensor:
    """
    Creates a unitary operator describing rotations of quantum state via Z axis.
    """
    return tf.convert_to_tensor([[math.cos(angle/2) - 1j * math.sin(angle/2), 0],
                                [0, math.cos(angle/2) + 1j * math.sin(angle/2)]], dtype=COMPLEX)


@tf.function
def kron(a: tf.Tensor, b: tf.Tensor) -> tf.Tensor:
    """
    Returns Kronecker product of two square matrices. The resulting shape is (dim1 * dim2, dim1 * dim2).
    """
    dim1 = a.shape[-1]
    dim2 = b.shape[-1]
    ab = tf.transpose(tf.tensordot(a, b, axes=0), (0, 2, 1, 3))
    return tf.reshape(ab, (dim1 * dim2, dim1 * dim2))


def ravel_multi_index(multi_index: tf.Tensor, dims: tuple[..., int]) -> tf.Tensor:
    """
    Converts a batch of bitstrings into a batch of single int numbers.

    Is an inverse of tf.unravel_index function.

    Args:
        multi_index: tf.Tensor(batch_size, dims.shape[0])[int32]
        Is effectively a batch of 1D arrays, each array containing several digits to encode a bitstring.
        dims: a tuple representing dimensions (aka numeral system) for each digit in 1D array multi_index[k].

    Returns:
        Tensor(batch_size)[int32]
    """
    strides = tf.math.cumprod(dims, exclusive=True, reverse=True)
    return tf.reduce_sum(multi_index * strides, axis=-1)


@tf.function
def convert_44_to_2222(unitary44: tf.Tensor) -> tf.Tensor:
    """
    Converts a matrix represented by a Tensor(4,4) to a Tensor(2, 2, 2, 2).

    There is a leg swap describing transition from regular 4x4 unitary to a tensor with ncon structure.
    """
    unitary2222 = tf.reshape(unitary44, (2, 2, 2, 2))
    unitary2222 = tf.transpose(unitary2222, (1, 0, 3, 2))
    return unitary2222


@tf.function
def swap_legs(unitary2222: tf.Tensor) -> tf.Tensor:
    """
    Does the leg swap (ncon <-> default unitary) in the Tensor(2,2,2,2) representing a 2-qubit operator.

    Also might be used to swap controlled and controlling qubits in ncon channel.
    """
    return tf.transpose(unitary2222, (1, 0, 3, 2))


@tf.function
def convert_2q_to16x16(channel: tf.Tensor) -> tf.Tensor:
    """
    Converts a ncon Tensor(4,4,4,4) representing 2qubit channel in to Tensor(16,16) in default form.
    """
    channel = tf.reshape(channel, (2, 2, 2, 2, 2, 2, 2, 2))
    channel = tf.transpose(channel, (2, 0, 3, 1, 6, 4, 7, 5))
    channel = tf.reshape(channel, (16, 16))
    return channel


@tf.function
def convert_2q_from16x16(channel: tf.Tensor) -> tf.Tensor:
    """
    Converts a 2qubit channel Tensor(16,16) to ncon form with shape (4,4,4,4).
    """
    channel = tf.reshape(channel, (2, 2, 2, 2, 2, 2, 2, 2))
    channel = tf.transpose(channel, (1, 3, 0, 2, 5, 7, 4, 6))
    channel = tf.reshape(channel, (4, 4, 4, 4))
    return channel


@tf.function
def swap_qubits_in_16x16(channel: tf.Tensor) -> tf.Tensor:
    """
    Swaps controlling and controlled qubits in 16x16 matrix. Physical sense is same as swap_legs.
    """
    channel = tf.reshape(channel, (2, 2, 2, 2, 2, 2, 2, 2))
    channel = tf.transpose(channel, (1, 0, 3, 2, 5, 4, 7, 6))
    channel = tf.reshape(channel, (16, 16))
    return channel


@tf.function
def convert_1q1q_from16x16(kronned_channel: tf.Tensor) -> tf.Tensor:
    """
    Converts a Kronecker product of two 1qubit quantum channels as Tensor(16,16) to ncon form with shape (4,4,4,4)
    """
    kronned_channel = tf.reshape(kronned_channel, (2, 2, 2, 2, 2, 2, 2, 2))
    kronned_channel = tf.transpose(kronned_channel, (2, 3, 0, 1, 6, 7, 4, 5))
    kronned_channel = tf.reshape(kronned_channel, (4, 4, 4, 4))
    return kronned_channel


def choi_swap_2qchannel(channel: tf.Tensor) -> tf.Tensor:
    """
    Converts a ncon Tensor(4,4,4,4) representing 2qubit channel to Tensor(16,16) in Choi matrix representation.
    """
    channel = tf.reshape(channel, (2, 2, 2, 2, 2, 2, 2, 2))
    channel = tf.transpose(channel, (0, 2, 4, 6, 1, 3, 5, 7))
    channel = tf.reshape(channel, (16, 16))
    return channel


def choi_swap_1qchannel(channel: tf.Tensor) -> tf.Tensor:
    """
    Converts a ncon Tensor(4,4) representing 1qubit channel to Tensor(4,4) in Choi matrix representation.
    """
    channel = tf.reshape(channel, (2, 2, 2, 2))
    channel = tf.transpose(channel, (0, 2, 1, 3))
    channel = tf.reshape(channel, (4, 4))
    return channel


def fidel_calc_1q(channel1: tf.Tensor, channel2: tf.Tensor) -> tf.Tensor:
    """
    Calculates fidelity between two arbitrary ncon 1-qubit channels represented as Tensors(4,4)[complex128]
    """
    choi1 = choi_swap_1qchannel(channel1) / tf.constant(2, COMPLEX)
    sqrt_choi1 = scipy.linalg.sqrtm(choi1).astype(np.complex128)

    choi2 = choi_swap_1qchannel(channel2) / tf.constant(2, COMPLEX)

    sqrt_matrix = scipy.linalg.sqrtm(sqrt_choi1 @ choi2 @ sqrt_choi1).astype(np.complex128)
    trace = tf.linalg.trace(tf.convert_to_tensor(sqrt_matrix, dtype=COMPLEX))
    return trace ** 2


def fidel_calc_2q(channel1: tf.Tensor, channel2: tf.Tensor) -> tf.Tensor:
    """
    Calculates fidelity between two arbitrary ncon 2-qubit channels represented as Tensors(4,4,4,4)[complex128]
    """
    choi1 = choi_swap_2qchannel(channel1) / tf.constant(4, COMPLEX)
    sqrt_choi1 = scipy.linalg.sqrtm(choi1).astype(np.complex128)

    choi2 = choi_swap_2qchannel(channel2) / tf.constant(4, COMPLEX)

    sqrt_matrix = scipy.linalg.sqrtm(sqrt_choi1 @ choi2 @ sqrt_choi1).astype(np.complex128)
    trace = tf.linalg.trace(tf.convert_to_tensor(sqrt_matrix, dtype=COMPLEX))
    return trace ** 2


def get_l1_distances(channel1: tf.Tensor, channel2: tf.Tensor, v: bool = False) -> tf.Tensor:
    """
    Calculates l1 distances between two ncon 1-qubit channels represented as Tensors(4,4)[complex128]
    l1 distances are taken for two input states: |0><0| and |1><1| and are calculated separately.
    """
    in_states = tf.constant([[1, 0, 0, 0], [0, 0, 0, 1]], dtype=COMPLEX)
    out_states1 = tf.linalg.matvec(a=channel1, b=in_states)
    out_states2 = tf.linalg.matvec(a=channel2, b=in_states)

    if v:
        print('output states for gate 1', out_states1)
        print('output states for gate 2', out_states2)

    probs_1 = tf.linalg.matmul(a=out_states1, b=tf.transpose(in_states))
    probs_2 = tf.linalg.matmul(a=out_states2, b=tf.transpose(in_states))

    if v:
        print('probs for gate 1', probs_1)
        print('probs for gate 2', probs_2)

    assert tf.reduce_sum(tf.abs(tf.reduce_sum(probs_1, axis=-1) - 1)) < 1e-6, probs_1
    assert tf.reduce_sum(tf.abs(tf.reduce_sum(probs_2, axis=-1) - 1)) < 1e-6, probs_2

    l1_dists = tf.reduce_sum(tf.abs(probs_1 - probs_2), axis=-1)

    return l1_dists / 2


def probs_qubit_swap(channel: tf.Tensor) -> tf.Tensor:
    """
    Calculates probability of channel flipping qubit sign from 0 to 1 and from 1 to 0.
    """
    in_states = tf.constant([[1, 0, 0, 0], [0, 0, 0, 1]], dtype=COMPLEX)
    out_states = tf.linalg.matvec(a=channel, b=in_states)

    probs = tf.linalg.matmul(a=out_states, b=tf.transpose(in_states))

    # print(probs)
    assert tf.reduce_sum(tf.abs(tf.reduce_sum(probs, axis=-1) - 1)) < 1e-6, probs

    return tf.concat([probs[0][1], probs[1][0]], axis=0)


def get_povm_dist(channel1: tf.Tensor, channel2: tf.Tensor) -> tf.Tensor:
    """
    Calculates POVM distance between two single-qubit channels representing (sic!) measurement operators.
    """
    reshaped_ch1 = tf.reshape(channel1, (2, 2, 2, 2))  # two transposes cancel each other
    reshaped_ch2 = tf.reshape(channel2, (2, 2, 2, 2))

    m0_1 = reshaped_ch1[0, 0]
    m0_2 = reshaped_ch2[0, 0]
    eigs = scipy.linalg.eig(m0_1 - m0_2)[0]

    # TODO: tests on m0 + m1 == 1
    # TODO: maybe tests on eigenvals also

    return tf.reduce_max(tf.abs(eigs))


def get_prep_dist(channel1: tf.Tensor, channel2: tf.Tensor) -> tf.Tensor:
    """
    Calculates POVM distance between two single-qubit channels representing (sic!) state preparation operators.
    """
    reshaped_ch1 = tf.reshape(channel1, (2, 2, 2, 2))  # two transposes cancel each other
    reshaped_ch2 = tf.reshape(channel2, (2, 2, 2, 2))

    rho_in_1 = reshaped_ch1[:, :, 0, 0]
    rho_in_2 = reshaped_ch2[:, :, 0, 0]
    eigs = scipy.linalg.eig(rho_in_1 - rho_in_2)[0]

    return tf.reduce_sum(tf.abs(eigs)) / tf.constant(2, dtype=FLOAT)


def choi_1qchannel_forqiskit(channel: tf.Tensor) -> tf.Tensor:
    """
    Converts a ncon Tensor(4,4) representing 1qubit channel to Tensor(4,4) in Choi matrix representation.
    Meaningful partial trace is over 1st subsystem now.
    """
    channel = tf.reshape(channel, (2, 2, 2, 2))
    channel = tf.transpose(channel, (2, 0, 3, 1))
    channel = tf.reshape(channel, (4, 4))
    return channel


def choi_2qchannel_forqiskit(channel: tf.Tensor) -> tf.Tensor:
    """
    Converts a ncon Tensor(4,4,4,4) representing 2qubit channel to Tensor(16,16) in Choi matrix representation.
    Meaningful partial trace is over 1st subsystem now.
    """
    channel = tf.reshape(channel, (2, 2, 2, 2, 2, 2, 2, 2))
    channel = tf.transpose(channel, (4, 6, 0, 2, 5, 7, 1, 3))
    channel = tf.reshape(channel, (16, 16))
    return channel


def diamond_norm_1q(channel1: tf.Tensor, channel2: tf.Tensor) -> tf.Tensor:
    """
    Calculates POVM distance between two single-qubit quantum channels.
    """
    diff = choi_1qchannel_forqiskit(channel1) - choi_1qchannel_forqiskit(channel2)
    diff_qiskit = Choi(diff.numpy())
    return tf.convert_to_tensor(diamond_norm(diff_qiskit))


def diamond_norm_2q(channel1: tf.Tensor, channel2: tf.Tensor) -> tf.Tensor:
    """
    Calculates POVM distance between two single-qubit quantum channels.
    """
    diff = choi_2qchannel_forqiskit(channel1) - choi_2qchannel_forqiskit(channel2)
    diff_qiskit = Choi(diff.numpy())
    return tf.convert_to_tensor(diamond_norm(diff_qiskit))
