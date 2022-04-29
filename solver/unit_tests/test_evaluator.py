import pytest
import tensorflow as tf

import solver.utils.general_utils as util
import solver.utils.channel_utils as c_util
from solver.QCCalc import QCEvaluator
from solver.utils.testing import is_choi, create_random_channel
from solver.utils.misc import NconTemplate, COMPLEX

# def simple_qc(circ_map: list[tuple[tf.Tensor, list[int]]], length: int) -> tf.Tensor:
#     """
#     Executes a simple quantum circuit consisting of unitary matrices.
#     The circuit always acts upon quantum state |0> ^ n. Used for testing.
#     Args:
#         circ_map: 2D array of shape (batch_size, dims.shape[0])
#         Is effectively a batch of 1D arrays, each array containing several digits to encode a bitstring.
#         length: a tuple representing dimensions (aka numeral system) for each digit in 1D array multi_index[k].
#     Returns:
#         1D array of type int32 of shape (batch_size)
#     """
#     state = tf.constant([1] + (2 ** length - 1) * [0], dtype=COMPLEX)
#     state = tf.reshape(state, length * (2,))
#     for gate, sides in circ_map:
#         if sides[0] > sides[1]:
#             min_index = sides[1]
#             max_index = sides[0]
#             first_edge = length-1
#             second_edge = length-2
#         else:
#             min_index = sides[0]
#             max_index = sides[1]
#             first_edge = length-2
#             second_edge = length-1
#         new_ord = tuple(range(min_index)) + (first_edge,) + tuple(range(min_index, max_index-1)) + \
#                   (second_edge,) + tuple(range(max_index-1, length-2))  # noqa: E127
#         state = tf.tensordot(state, gate, axes=[sides, [2, 3]])  # sides
#         state = tf.transpose(state, new_ord)
#     return state


def test_random_channel_sanity():
    """
    A test for a test. Ironic!
    """
    for _ in range(10):
        ch1 = create_random_channel(1)
        ch2 = create_random_channel(2)
        ch2_ncon = util.convert_2q_from16x16(ch2)

        choi1 = util.choi_swap_1qchannel(ch1) / tf.constant(2, dtype=COMPLEX)
        assert is_choi(choi1)

        choi2 = util.choi_swap_2qchannel(ch2_ncon) / tf.constant(4, dtype=COMPLEX)
        assert is_choi(choi2)


def get_all_bistrings(qubit_num: int) -> tf.Tensor:
    return tf.transpose(tf.unravel_index(indices=tf.range(0, 2 ** qubit_num, delta=1), dims=[2] * qubit_num))


def test_schema_1():
    in_state = tf.constant([1, 0, 0, 0], dtype=COMPLEX)
    ch1 = create_random_channel(1)
    out_state = tf.linalg.matvec(a=ch1, b=in_state)

    evaluator = QCEvaluator(gates=ch1[tf.newaxis], n=1)

    tn_template_test = [[0],
                        [[-1, 1]],
                        [],
                        [-1]]

    evaluator.add_circuit(tn_template_test, 'test')

    bitstrings_probs = evaluator.evaluate(get_all_bistrings(1), name='test')

    for bit_1 in [0, 1]:
        assert tf.abs(out_state[bit_1 * 3] - bitstrings_probs[bit_1]) < 1e-6


# PATTERNS: 0, 1 - 2Q; 2, 3 - 1Q1Q
PATTERNS_2Q = [[0], [2], [0, 2], [2, 0], [2, 0, 3], [0, 2, 1]]

# NCON: 0, 1 - 2Q; 2, 3, 4, 5 - 1Q; 2 & 3 = PATTERN 2; 3 & 4 = PATTERN 3.
TEMPLATES_2Q = [
    # single 2Q gate
    [[0],
     [[-2, -1, 2, 1]],
     [],
     [-1, -2]],

    # two 1Q gates
    [[2, 3],
     [[-1, 1], [-2, 2]],
     [],
     [-1, -2]],

    # 2Q -> 1Q1Q
    [[0, 2, 3],
     [[4, 3, 2, 1], [-1, 3], [-2, 4]],
     [3, 4],
     [-1, -2]],

    # 1Q1Q -> 2Q
    [[0, 2, 3],
     [[-2, -1, 4, 3], [3, 1], [4, 2]],
     [3, 4],
     [-1, -2]],

    # 1Q1Q -> 2Q -> 1Q1Q
    [[0, 2, 3, 4, 5],
     [[6, 5, 4, 3], [3, 1], [4, 2], [-1, 5], [-2, 6]],
     [3, 4, 5, 6],
     [-1, -2]],

    # 2Q -> 1Q1Q -> 2Q
    [[0, 1, 2, 3],
     [[4, 3, 2, 1], [-2, -1, 6, 5], [5, 3], [6, 4]],
     [3, 4, 5, 6],
     [-1, -2]],
]

NAMES_2Q = [
    '2Q',
    '1Q1Q',
    '2Q->1Q1Q',
    '1Q1Q->2Q',
    '1Q1Q->2Q->1Q1Q',
    '2Q->1Q1Q->2Q'
]


@pytest.mark.parametrize(['lines_pattern', 'template'], zip(PATTERNS_2Q, TEMPLATES_2Q), ids=NAMES_2Q)
def test_schema_2(lines_pattern: list[int], template: NconTemplate):
    in_state = tf.constant([1] + 15 * [0], dtype=COMPLEX)
    gates = [create_random_channel(2),
             create_random_channel(2),
             create_random_channel(1),
             create_random_channel(1),
             create_random_channel(1),
             create_random_channel(1)]

    gates_ncon = gates.copy()
    for i in [0, 1]:
        gates_ncon[i] = util.convert_2q_from16x16(gates_ncon[i])
    lines = (gates[:2] +
             [(util.convert_2q_to16x16(util.kron(gates[3], gates[2])))] +
             [(util.convert_2q_to16x16(util.kron(gates[5], gates[4])))])

    out_state = in_state
    for matrix_id in lines_pattern:
        out_state = tf.linalg.matvec(a=lines[matrix_id], b=out_state)
    out_state = tf.reshape(out_state, (4, 4))

    evaluator = QCEvaluator(gates=gates_ncon, n=2)
    evaluator.add_circuit(template, 'test')
    bitstrings_probs = evaluator.evaluate(get_all_bistrings(2), name='test')

    # print(out_state)
    # print('AAA')
    # print(bitstrings_probs)

    # TODO (physics, not code): understand why answer is located on diag places

    for bit_1 in [0, 1]:
        for bit_2 in [0, 1]:
            idx = bit_1 * 2 + bit_2
            assert tf.abs(out_state[idx][idx] - bitstrings_probs[idx]) < 1e-6


def test_schema_3_simple():
    """This is a test made to make the next test work"""
    in_state = tf.constant([1] + 63 * [0], dtype=COMPLEX)
    gates: list[tf.Tensor] = []

    for _ in range(3):
        gates.append(create_random_channel(1))

    lines = []
    kronned_3 = util.kron(gates[0], util.kron(gates[1], gates[2]))
    kronned_3 = tf.reshape(kronned_3, (2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2))
    # (0, 3, 1, 4, 2, 5, 6, 9, 7, 10, 8, 11) is inverse
    kronned_3 = tf.transpose(kronned_3, (0, 2, 4, 1, 3, 5, 6, 8, 10, 7, 9, 11))
    lines.append(tf.reshape(kronned_3, (64, 64)))

    net_struc = [[-1, 1], [-2, 2], [-3, 3]]  # lines[0]
    template = [list(range(3)),
                net_struc,
                [],
                [-1, -2, -3]]

    out_state = tf.linalg.matvec(a=lines[0], b=in_state)
    out_state = tf.reshape(out_state, (8, 8))

    # print(out_state.numpy().round(2))
    # for row in out_state:
    #     print(' '.join(f'{x:.3f}' for x in row))

    evaluator = QCEvaluator(gates=gates, n=3)
    evaluator.add_circuit(template, 'test')
    bitstrings_probs = evaluator.evaluate(get_all_bistrings(3), name='test')

    # for i in range(8):
    #     print(bitstrings_probs[i].numpy().round(5), out_state[i][i].numpy().round(5))

    for bit_1 in [0, 1]:
        for bit_2 in [0, 1]:
            for bit_3 in [0, 1]:
                idx = bit_1 * 4 + bit_2 * 2 + bit_3
                assert tf.abs(out_state[idx][idx] - bitstrings_probs[idx]) < 1e-6


def test_schema_3():
    # pattern 1.1.1. - 2(0,1) - 1.1.1 - 2(1,2) - 1.1.1 - 2(0,2) - 1.1.1

    in_state = tf.constant([1] + 63 * [0], dtype=COMPLEX)
    # in_state = tf.convert_to_tensor(np.arange(64), dtype=COMPLEX)

    gates: list[tf.Tensor] = []

    for _ in range(12):
        gates.append(create_random_channel(1))
        # gates.append(tf.eye(4, dtype=COMPLEX))

    for _ in range(3):
        gates.append(create_random_channel(2))
        # gates.append(tf.eye(16, dtype=COMPLEX))

    gates_ncon = gates.copy()
    for i in range(12, 15):
        gates_ncon[i] = util.convert_2q_from16x16(gates_ncon[i])

    lines: list[tf.Tensor] = [tf.constant(0)] * 7
    for i in range(4):  # three single qubit channels kronned
        kronned_3 = util.kron(gates[3 * i], util.kron(gates[3 * i + 1], gates[3 * i + 2]))
        kronned_3 = tf.reshape(kronned_3, (2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2))
        kronned_3 = tf.transpose(kronned_3, (0, 2, 4, 1, 3, 5, 6, 8, 10, 7, 9, 11))
        lines[2 * i] = tf.reshape(kronned_3, (64, 64))

    first_line = util.kron(gates[-3], tf.eye(4, dtype=COMPLEX))
    first_line = tf.reshape(first_line, (2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2))
    first_line = tf.transpose(first_line, (0, 1, 4, 2, 3, 5, 6, 7, 10, 8, 9, 11))
    lines[1] = tf.reshape(first_line, (64, 64))

    third_line = util.kron(tf.eye(4, dtype=COMPLEX), gates[-2])
    third_line = tf.reshape(third_line, (2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2))
    third_line = tf.transpose(third_line, (0, 2, 3, 1, 4, 5, 6, 8, 9, 7, 10, 11))
    lines[3] = tf.reshape(third_line, (64, 64))

    fifth_line = util.kron(gates[-1], tf.eye(4, dtype=COMPLEX))
    fifth_line = tf.reshape(fifth_line, (2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2))
    fifth_line = tf.transpose(fifth_line, (0, 4, 1, 2, 5, 3, 6, 10, 7, 8, 11, 9))
    lines[5] = tf.reshape(fifth_line, (64, 64))

    net_struc = []
    net_struc += [[4, 1], [5, 2], [6, 3]]  # lines[0]
    net_struc += [[9, 7], [10, 8], [11, 6]]  # lines[2]
    net_struc += [[14, 9], [15, 12], [16, 13]]  # lines[4]
    net_struc += [[-1, 17], [-2, 15], [-3, 18]]  # lines[6]
    net_struc += [[8, 7, 5, 4]]  # lines[1]
    net_struc += [[13, 12, 11, 10]]  # lines[3]
    net_struc += [[18, 17, 16, 14]]  # lines[5]

    template = [list(range(15)),
                net_struc,
                list(range(4, 19)),
                [-1, -2, -3]]

    out_state = in_state
    for matrix in lines:
        out_state = tf.linalg.matvec(a=matrix, b=out_state)
    out_state = tf.reshape(out_state, (8, 8))

    evaluator = QCEvaluator(gates=gates_ncon, n=3)
    evaluator.add_circuit(template, 'test')
    bitstrings_probs = evaluator.evaluate(get_all_bistrings(3), name='test')

    # for i in range(8):
    #     print(bitstrings_probs[i].numpy().round(5), out_state[i][i].numpy().round(5))

    for i in range(8):
        assert tf.abs(out_state[i][i] - bitstrings_probs[i]) < 1e-6


def test_50_qubits():
    Hadamard = tf.constant([[1, 1],
                            [1, -1]], dtype=COMPLEX) / tf.math.sqrt(tf.constant(2, dtype=COMPLEX))

    CNOT_44 = tf.constant([[1, 0, 0, 0],
                           [0, 1, 0, 0],
                           [0, 0, 0, 1],
                           [0, 0, 1, 0]], dtype=COMPLEX)

    H_channel = c_util.convert_1qmatrix_to_channel(Hadamard)
    CNOT = util.convert_44_to_2222(CNOT_44)
    CNOT_channel = c_util.convert_2qmatrix_to_channel(CNOT)

    tn_template_GHZ_50 = [[0] + 49 * [1],
                          [[51, 1]] + [[50 + i + 1, -i, i + 1, 50 + i] for i in range(1, 49)] + [[-50, -49, 50, 99]],
                          [i for i in range(51, 100)],
                          [-i for i in range(1, 51)]]

    qc_50_c = QCEvaluator([H_channel, CNOT_channel], 50)
    qc_50_c.add_circuit(tn_template_GHZ_50, 'GHZ50')
    qc_50_c.make_full_samples('GHZ50', 10000)
    # no assert - test is VS timeout

# TODO: test samples generation
