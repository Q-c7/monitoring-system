n_qubits = 3
single_qub_gates_names = {'RX'}
two_qub_gates_names = {'CX'}

def _convert_noises_to_tf(all_noises):
    return [tf.convert_to_tensor(noise_list, dtype=FLOAT) for noise_list in all_noises]

noises_RX = _convert_noises_to_tf([
    [0.0, 0.1, 0.2],  # 0
    [0.0, 0.1, 0.2],  # 1
    [0.0, 0.1, 0.2],  # 2
])

noises_CX = _convert_noises_to_tf([
    [0.0, 0.1, 0.2],  # 01
    [0.0, 0.1, 0.2],  # 02
    [0.0, 0.1, 0.2],  # 10
    [0.0, 0.1, 0.2],  # 12
    [0.0, 0.1, 0.2],  # 20
    [0.0, 0.1, 0.2],  # 21
])

NOISES_DICT = {
    'RX': noises_RX,
    'CX': noises_CX
}


def make_noised_checkpoint_gate_set(noises_dict,
                                    pure_channels_set,
                                    single_qub_gates_names,
                                    two_qub_gates_names):
    checkpoint_dict = {}

    for name in pure_channels_set:  # 'RX', '_E', 'CX'
        if name in single_qub_gates_names:
            all_gate_noises = noises_dict[name]
            assert len(all_gate_noises) == num_qubits
            all_channels = []

            for init_noise in all_gate_noises:
                noised_channel = ns.make_1q_hybrid_channel(pure_channels_set[name], init_noise)
                params = qgo.manifolds.complex_to_real(c_util.convert_channel_to_params(noised_channel))
                all_channels.append(params)

            checkpoint_dict[name] = tf.Variable(tf.concat(all_channels, axis=0))

        elif name in two_qub_gates_names:
            all_gate_noises = noises_dict[name]
            assert len(all_gate_noises) == num_qubits * (num_qubits - 1)
            all_channels = []

            for init_noise in all_gate_noises:
                noised_channel = ns.make_2q_hybrid_channel(pure_channels_set[name], init_noise)
                params = qgo.manifolds.complex_to_real(c_util.convert_channel_to_params(noised_channel))
                all_channels.append(params)

            checkpoint_dict[name] = tf.Variable(tf.concat(all_channels, axis=0))

        elif name == ID_GATE:  # '_E'
            params = qgo.manifolds.complex_to_real(c_util.convert_channel_to_params(pure_channels_set[ID_GATE]))
            self.estimated_gates_dict[ID_GATE] = tf.Variable(params[tf.newaxis])

        else:
            raise ValueError('Unknown gate')

    return checkpoint_dict
