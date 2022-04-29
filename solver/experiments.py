import tensorflow as tf
import typing as tp
import QGOpt as qgo
from collections import defaultdict

import solver.utils.general_utils as util
import solver.file_management as fm
import solver.circuits_generation as cg
from solver.QCSolver import BaseSolver, QGOptSolver, GateSet, get_complex_channel_form


def assert_psi2_eq_1(qc: BaseSolver, n_q: int, v: bool = False):
    """
    TODO: write docstring
    """
    dimdim = tf.constant([2] * n_q, dtype=tf.int32)
    all_bs = tf.transpose(tf.unravel_index(tf.range(2 ** n_q), dimdim))
    n_samp = None

    all_names = list(qc.samples_compressed.keys()) if qc.compress else list(qc.samples_vault.keys())

    for name in all_names:
        no_noise_psi2 = qc.eval_pure.evaluate(all_bs, name)
        psi2 = qc.eval_estimated.evaluate(all_bs, name)
        true_psi2 = qc.eval_hidden.evaluate(all_bs, name)
        if qc.compress:
            if n_samp is None:
                n_samp = qc.samples_compressed[name].numpy().sum()
            hist_ctr = qc.samples_compressed[name]
        else:
            if n_samp is None:
                n_samp = len(qc.samples_vault[name])
            counts = util.ravel_multi_index(qc.samples_vault[name], dimdim)
            counts = tf.concat([counts, tf.range(2 ** qc.n)], axis=0)
            hist_ctr = tf.math.bincount(counts)
            hist_ctr -= 1

        if v:
            print(name)
        if tf.abs(tf.reduce_sum(tf.abs(true_psi2)).numpy() - 1) > 1e-5:
            raise ValueError('Has a serious bug: sum of probs != 1 for HIDDEN SET')
        if tf.abs(tf.reduce_sum(tf.abs(psi2)).numpy() - 1) > 1e-5:
            raise ValueError('Sum of probs != 1 for ESTIMATED SET')

        if v:
            print(f'"naive" estimation {tf.abs(no_noise_psi2).numpy().round(5)}')
            print(f'"true" estimation {tf.abs(true_psi2).numpy().round(5)}')
            print(f'sampled distributions {(hist_ctr.numpy() / n_samp).round(5)}')
            print(f'solver estimation {tf.abs(psi2).numpy().round(5)}')


class ExperimentConductor:
    NoiseCfg = tp.Optional[list[tuple[str, int, tp.Callable[[tf.Tensor, ...], tf.Tensor], float, ...]]]

    def __init__(self,
                 pure_channels_set: dict[str, tf.Tensor],
                 noise_cfg: NoiseCfg = None,
                 exp_name: str = None,
                 qubits_num: int = 3,
                 lr: float = 0.03,
                 lmbd1: float = 50,
                 lmbd2: tp.Optional[float] = None,
                 iterations: int = 250,
                 sample_size: int = 1000):
        """
        A class which holds all the information about current experiment.

        Args:
            pure_channels_set: A set of pure channels for this experiment.
            noise_cfg: A list of tuples, which have a bit complicated form.
            tuple[0] (str) - name of the gate;
            tuple[1] (int) - id of the gate;
            tuple[2] (func) - function to use
            3, 4, ... - params for the function to use

            exp_name: Name of the experiment made for convenience; mainly used for dumping in files.
            qubits_num: Number of qubits.
            lr: Learning rate.
            lmbd1: Regularization coefficient for single-qubit gates.
            lmbd2: Regularization coefficient for two-qubit gates.
            iterations: Number of iterations for this experiment.
            sample_size: Number of outcomes in a sample.

        Extra attributes:
            noise_params: A dict of lists of tuples: key is name of the gate;
            tuples hold ID of the gate at idx 0 and a new Tensor to replace the default one at idx 1.
        """
        self.qubits_num = qubits_num
        self.lr = lr
        self.lmbd1 = lmbd1
        self.lmbd2 = lmbd2 if lmbd2 is not None else lmbd1
        self.iters = iterations
        self.noise_params: tp.Optional[dict[str, list[tuple[int, tf.Tensor]]]] = None
        self.exp_name = exp_name
        self.sample_size = sample_size

        noise_params = defaultdict(list)
        for tup in noise_cfg:
            name, index, func = tup[0:3]

            args = []
            for i in range(3, len(tup)):
                args.append(tup[i])
            new_tensor = func(pure_channels_set[name], args)

            noise_params[name].append((index, new_tensor))

        self.noise_params = dict(noise_params)


def conduct_full_experiment(exp_test: ExperimentConductor,
                            marks: tp.Optional[list[int]],
                            pure_channels_set: GateSet,
                            single_qubit_gates: set[str],
                            two_qubit_gates: set[str],
                            circ_num: int = 200,
                            noise_iter0=0.1,
                            save_l1_norms=True) -> None:
    # a function used only for running experiments for article
    # no need to be documented

    print(f"Conducting an experiment with name {exp_test.exp_name}")
    print(f"For circuits amount we are considering {marks}")

    list_to_dump = []
    all_l1_norms = []
    manif = qgo.manifolds.ChoiMatrix()

    QC_gen = BaseSolver(qubits_num=exp_test.qubits_num,
                        single_qub_gates_names=single_qubit_gates,
                        two_qub_gates_names=two_qubit_gates,
                        pure_channels_set=pure_channels_set,
                        compress_samples=True,
                        noise_params=exp_test.noise_params)

    try:
        ncon_tmpls = fm.generate_ncon_dict_from_file('', [exp_test.exp_name])
        QC_gen.samples_compressed = fm.import_bitstrings_from_file([exp_test.exp_name, 'zip'])

        for name, tmpl in ncon_tmpls.items():
            if save_l1_norms:
                # l1_norms are extracted by names via hard-coded way so names has to be either int or str(int)
                # pure(str) name will interfere with norms extraction so better to raise exception here
                will_raise = int(name)
            QC_gen.add_circuit(tn_template=tmpl, name=name)
        print("Samples and templates found in data folder")
    except FileNotFoundError:
        raise NotImplementedError('Generation on the spot is deprecated')
        # print("Generating everything on the spot with default parameters")
        # gen = cg.DataGenerator(qubits_num=exp_test.qubits_num,
        #                        gates_names=list(single_qubit_gates) + list(two_qubit_gates),
        #                        single_qub_gates_num=len(single_qubit_gates),
        #                        two_qub_gates_num=len(two_qubit_gates))
        #
        # strings = cg.get_square_topology(circ_len_min=8,
        #                                  circ_len_max=12,
        #                                  circ_num=circ_num,
        #                                  two_qubit_gate_prob=0.3,
        #                                  single_qubit_gates=list(single_qubit_gates),
        #                                  two_qubit_gates=list(two_qubit_gates),
        #                                  qubit_num=exp_test.qubits_num)
        #
        # ncon_tmpls = gen.get_tmpl_dict_from_human_circs(strings)
        # for name, tmpl in ncon_tmpls.items():
        #     QC_gen.add_circuit(tn_template=tmpl, name=name)
        # QC_gen.generate_all_samples(v=False, smpl_size=exp_test.sample_size)

    for i in reversed(marks):
        if isinstance(i, tuple):
            if save_l1_norms:
                raise NotImplementedError('Tuples are not supported when saving l1 norms')
            i1, i2 = i
        else:
            i1, i2 = 0, i
        QC_t = QGOptSolver(qubits_num=exp_test.qubits_num,
                           single_qub_gates_names=single_qubit_gates,
                           two_qub_gates_names=two_qubit_gates,
                           pure_channels_set=pure_channels_set,
                           compress_samples=True,
                           noise_params=exp_test.noise_params,
                           noise_iter0=noise_iter0)

        for j, key in enumerate(QC_gen.samples_compressed):
            if j < i1:
                continue
            if j > i2:
                break
            QC_t.samples_compressed[key] = QC_gen.samples_compressed[key]
            QC_t.add_circuit(tn_template=QC_gen.tn_templates[key], name=key)

        print(f'Current mark is {i}')
        print(f'Debug info: length of circuits dictionary is {len(QC_t.samples_compressed)}')

        opt_t = qgo.optimizers.RAdam(manif, exp_test.lr)
        _ = QC_t.train_optimizer(opt_t, exp_test.lmbd1, exp_test.lmbd2, exp_test.iters, v=1, timestamp=None)
        list_to_dump.append(get_complex_channel_form(QC_t.estimated_gates_dict))

        if save_l1_norms:
            for key in QC_gen.samples_compressed:
                QC_t.add_circuit(tn_template=QC_gen.tn_templates[key], name=key)

            norms_list = []
            for mark in marks:
                name = str(int(mark) - 1)
                l1_est_ideal, l1_est_true = QC_t.get_circ_l1_norms(name)
                norms_list.append((l1_est_ideal, l1_est_true))
            all_l1_norms.append(norms_list)

    fm.save_experiment_to_file(list_to_dump, [exp_test.exp_name])
    fm.save_norms_to_file(all_l1_norms, [exp_test.exp_name])

    print("Experiment is succesfully conducted and all the", len(list_to_dump), "tensors are saved!")


def conduct_time_experiment(exp_test: ExperimentConductor,
                            timestamps: tp.Optional[list[int]],
                            pure_channels_set: GateSet,
                            single_qubit_gates: set[str],
                            two_qubit_gates: set[str],
                            noise_iter0=0.1) -> None:
    # a function used only for running experiments for article
    # no need to be documented

    print(f"Conducting a TIMED experiment with name {exp_test.exp_name}")
    print(f"Key characteristics: LAM1 {exp_test.lmbd1} LAM2 {exp_test.lmbd2}")
    print(f"LR {exp_test.lr} ITERS {exp_test.iters} SIZE {exp_test.sample_size}")
    print("----------------------------------------------------------")
    print(f"For timestamps we are considering {timestamps}")

    list_to_dump = []
    manif = qgo.manifolds.ChoiMatrix()

    ncon_tmpls = fm.generate_ncon_dict_from_file('', [exp_test.exp_name])
    all_samples = fm.import_bitstrings_from_file([exp_test.exp_name, 'zip'])
    assert len(all_samples) == len(ncon_tmpls)
    assert all_samples.keys() == ncon_tmpls.keys()

    QC_t = QGOptSolver(qubits_num=exp_test.qubits_num,
                       single_qub_gates_names=single_qubit_gates,
                       two_qub_gates_names=two_qubit_gates,
                       pure_channels_set=pure_channels_set,
                       compress_samples=True,
                       noise_params=exp_test.noise_params,
                       noise_iter0=noise_iter0)

    opt_t = qgo.optimizers.RAdam(manif, exp_test.lr)

    for key in ncon_tmpls:
        QC_t.add_circuit(tn_template=ncon_tmpls[key], name=key)
        QC_t.samples_compressed[key] = all_samples[key]

    for t in timestamps:
        print(f'Current timestamp is at {t}')
        _ = QC_t.train_optimizer(opt_t, exp_test.lmbd1, exp_test.lmbd2, exp_test.iters, timestamp=t, v=1)
        list_to_dump.append(get_complex_channel_form(QC_t.estimated_gates_dict))

    fm.save_experiment_to_file(list_to_dump, [exp_test.exp_name])

    print("Experiment is succesfully conducted and all the", len(list_to_dump), "tensors are saved!")
