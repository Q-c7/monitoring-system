import tensorflow as tf  # tf 2.x
import tensornetwork as tn
import numpy as np
import typing as tp
import QGOpt as qgo
import math

import solver.utils.general_utils as util
import solver.noising_tools as ns
import solver.utils.channel_utils as c_util
import solver.file_management as fm
import solver.circuits_generation as cg

from solver.QCCalc import QCEvaluator
from solver.experiments import ExperimentConductor, assert_psi2_eq_1
from solver.utils.misc import unwrap_dict, NconTemplate, INT, FLOAT, COMPLEX, ID_GATE
from solver.QCSolver import QGOptSolver, get_complex_channel_form, SimpleSolver

tn.set_default_backend("tensorflow")

UN_MANIF = qgo.manifolds.StiefelManifold()
E = tf.eye(2, dtype=COMPLEX)
E_channel = c_util.convert_1qmatrix_to_channel(E)


def test_grad_descent_qgo():
    unitary_1q1, unitary_1q2 = UN_MANIF.random((2, 2), dtype=COMPLEX), UN_MANIF.random((2, 2), dtype=COMPLEX)
    ch1q1, ch1q2 = c_util.convert_1qmatrix_to_channel(unitary_1q1), c_util.convert_1qmatrix_to_channel(unitary_1q2)
    unitary_2q = util.convert_44_to_2222(UN_MANIF.random((4, 4), dtype=COMPLEX))
    ch2q = c_util.convert_2qmatrix_to_channel(unitary_2q)

    random_pure_channels = {'1Q1': ch1q1, '1Q2': ch1q2, ID_GATE: E_channel, '2Q': ch2q}

    n_qubits = 3

    gen = cg.DataGenerator(qubits_num=n_qubits, gates_names=['1Q1', '1Q2', '2Q'],
                           single_qub_gates_num=2, two_qub_gates_num=1)

    ncon_tmpls = gen.generate_data(circ_len_min=8,
                                   circ_len_max=12,
                                   circ_num=20,
                                   two_qubit_gate_prob=0.15,
                                   custom_name='_')

    noise_cfg_test = [
        ('1Q1', 0, ns.make_1q_4pars_channel, 0.1, 0.1, 0.1, 0.05),
        ('1Q1', 2, ns.make_1q_4pars_channel, 0.03, 0.03, 0.03, 0.0),
        ('1Q2', 1, ns.make_1q_4pars_channel, 0.1, 0.05, 0.05, 0.05),
        ('2Q', 1, ns.make_2q_4pars_channel, 0.03, 0.06, 0.06, 0.05)
    ]
    exp_test = ExperimentConductor(pure_channels_set=random_pure_channels,
                                   noise_cfg=noise_cfg_test,
                                   exp_name='___',
                                   qubits_num=n_qubits,
                                   lr=0.01,
                                   lmbd1=1,
                                   lmbd2=1,
                                   iterations=10,
                                   sample_size=10000)

    QC_t = QGOptSolver(qubits_num=n_qubits, single_qub_gates_names={'1Q1', '1Q2'}, two_qub_gates_names={'2Q'},
                       pure_channels_set=random_pure_channels, compress_samples=False,
                       noise_params=exp_test.noise_params)

    for name, tmpl in ncon_tmpls.items():
        QC_t.add_circuit(tn_template=tmpl, name=name)

    QC_t.generate_all_samples(v=False, smpl_size=exp_test.sample_size)

    manif = qgo.manifolds.ChoiMatrix()
    opt_t = qgo.optimizers.RAdam(manif, exp_test.lr)

    true_loss = QC_t.true_loss_value(exp_test.lmbd1, exp_test.lmbd2)

    loss_dynamics_t = QC_t.train_optimizer(opt=opt_t,
                                           lmbd1=exp_test.lmbd1,
                                           lmbd2=exp_test.lmbd2,
                                           iters=exp_test.iters,
                                           v=2)

    print(np.array(loss_dynamics_t))

    assert all(earlier >= later for earlier, later in zip(loss_dynamics_t, loss_dynamics_t[1:]))
    assert loss_dynamics_t[-1] > true_loss
