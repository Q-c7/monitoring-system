# def avg_error_2q(channel1: tf.Tensor, channel2: tf.Tensor, v: bool = False) -> tf.Tensor:
#     """
#     Calculates average error between two ncon 2-qubit channels represented as Tensors(4,4,4,4)[complex128]
#     """
#     ch1_16x16 = convert_2q_to16x16(channel1)
#     ch2_16x16 = convert_2q_to16x16(channel2)
#
#     in_states = tf.constant([[1] + 15 * [0],  # |00>
#                              5 * [0] + [1] + 10 * [0],  # |01>
#                              10 * [0] + [1] + 5 * [0],  # |10>
#                              15 * [0] + [1]],  # |11>
#                             dtype=COMPLEX)
#
#     out_states1 = tf.linalg.matvec(a=ch1_16x16, b=in_states)
#     out_states2 = tf.linalg.matvec(a=ch2_16x16, b=in_states)
#
#     if v:
#         print('output states for gate 1', out_states1)
#         print('output states for gate 2', out_states2)
#
#     probs_1 = tf.linalg.matmul(a=out_states1, b=tf.transpose(in_states))
#     probs_2 = tf.linalg.matmul(a=out_states2, b=tf.transpose(in_states))
#
#     if v:
#         print('probs for gate 1', probs_1)
#         print('probs for gate 2', probs_2)
#
#     assert tf.reduce_sum(tf.abs(tf.reduce_sum(probs_1, axis=-1) - 1)) < 1e-6, probs_1
#     assert tf.reduce_sum(tf.abs(tf.reduce_sum(probs_2, axis=-1) - 1)) < 1e-6, probs_2
#
#     return tf.reduce_sum(tf.abs(probs_1 - probs_2)) / 8

# def avg_error_1q(channel1: tf.Tensor, channel2: tf.Tensor, v: bool = False) -> tf.Tensor:
#     """
#     Calculates average error between two ncon 1-qubit channels represented as Tensors(4,4)[complex128]
#     """
#     in_states = tf.constant([[1, 0, 0, 0], [0, 0, 0, 1]], dtype=COMPLEX)
#     out_states1 = tf.linalg.matvec(a=channel1, b=in_states)
#     out_states2 = tf.linalg.matvec(a=channel2, b=in_states)
#
#     if v:
#         print('output states for gate 1', out_states1)
#         print('output states for gate 2', out_states2)
#
#     probs_1 = tf.linalg.matmul(a=out_states1, b=tf.transpose(in_states))
#     probs_2 = tf.linalg.matmul(a=out_states2, b=tf.transpose(in_states))
#
#     if v:
#         print('probs for gate 1', probs_1)
#         print('probs for gate 2', probs_2)
#
#     assert tf.reduce_sum(tf.abs(tf.reduce_sum(probs_1, axis=-1) - 1)) < 1e-6, probs_1
#     assert tf.reduce_sum(tf.abs(tf.reduce_sum(probs_2, axis=-1) - 1)) < 1e-6, probs_2
#
#     return tf.reduce_sum(tf.abs(probs_1 - probs_2)) / 4


# def get_square_topology(circ_len_min: int = 5,
#                         circ_len_max: int = 10,
#                         qubit_num: int = 4,
#                         circ_num: int = 20,
#                         single_qubit_gates: tp.Optional[list[str]] = None,
#                         two_qubit_gates: tp.Optional[list[str]] = None,
#                         two_qubit_gate_prob: float = 0.25) -> list[list[str]]:
#     """
#     TODO: deal with code duplication. Delete this function?
#     """
#     circuits = []
#     for _ in range(circ_num):
#         circuit_cur = []
#         gates_num = randint(circ_len_min, circ_len_max)
#         for _ in range(gates_num):
#             # Single qubit case:
#             if random() > two_qubit_gate_prob:
#                 qubit_index = sample(list(range(qubit_num)), 1)[0]
#                 gate_name = sample(single_qubit_gates, 1)[0]
#                 gate_full = gate_name + '_' + str(qubit_index)
#             # Two qubit case:
#             else:
#                 qubit_indic1 = randint(0, 3)
#                 qubit_indic2 = (qubit_indic1 + 1) % qubit_num
#                 qubit_indices = [qubit_indic1, qubit_indic2]
#                 qubit_indices.sort()
#                 gate_name = sample(two_qubit_gates, 1)[0]
#                 gate_full = gate_name + '_' + str(qubit_indices[0]) + '_' + str(qubit_indices[1])
#             circuit_cur.append(gate_full)
#
#         circuits.append(circuit_cur)
#
#     return circuits