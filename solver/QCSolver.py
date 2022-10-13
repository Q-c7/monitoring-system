import tensorflow as tf  # tf 2.x
import tensornetwork as tn
import numpy as np
import typing as tp
import QGOpt as qgo

from abc import abstractmethod
from collections import defaultdict

import solver.utils.general_utils as util
import solver.utils.channel_utils as c_util
import solver.noising_tools as ns
import solver.utils.statistics as my_stats
from solver.QCCalc import QCEvaluator
from solver.utils.misc import unwrap_dict, NconTemplate, INT, FLOAT, COMPLEX, ID_GATE, TENSOR

tn.set_default_backend("tensorflow")

GateSet = dict[str, TENSOR]
NoiseParams = tp.Optional[dict[str, list[tuple[int, TENSOR]]]]
EXP_CONST = 50


def get_complex_channel_form(target: GateSet) -> GateSet:
    """
    Since functions qgo.manifolds.real_to_complex() and convert_params_to_channel() cannot be applied to a dict,
    this function breaks the dict into Tensors=arrays of gates and then converts each array one-by-one

    Args:
        target: a python dictionary containing some real-valued parameter representations
        of tensors that are in fact complex; shape is (..., 2)

    Returns:
        Complex-valued tensors in a dictionary with the same grouping
    """
    new_dict = {}
    for name in target:
        new_dict[name] = c_util.convert_params_to_channel(qgo.manifolds.real_to_complex(target[name]))
    return new_dict


STATS_FUNCS = {'ks': my_stats.ks_stat,
               'chi2': my_stats.chi2_stat,
               'prod': my_stats.scalar_product,
               'l1': my_stats.l1_stat}


class BaseSolver:
    """
    A base Solver class which contains all logic for 'pure' evaluator part and 'hidden' evaluator part.
    Methods include noise model selection, sample generation and gates initialization.
    TODO: Better docstring
    """
    def __init__(self,
                 qubits_num: int,
                 single_qub_gates_names: set[str],
                 two_qub_gates_names: set[str],
                 pure_channels_set: GateSet,
                 compress_samples: bool = False,
                 noise_params: NoiseParams = None):
        self.n = qubits_num
        self.single_qub_gates_names = single_qub_gates_names
        self.two_qub_gates_names = two_qub_gates_names
        assert len(pure_channels_set) == len(single_qub_gates_names) + len(two_qub_gates_names) + 1

        self.ideal_gates_list: list[TENSOR] = []
        for name in pure_channels_set:
            self.ideal_gates_list.append(pure_channels_set[name])

        self.hidden_gates_dict: GateSet = {}
        self._init_hidden_channels(pure_channels_set, noise_params)

        self.tn_templates: dict[str, NconTemplate] = {}  # label -> list(list1, list2, list3, list4)

        self.compress = compress_samples
        self.samples_vault: dict[str, tf.Tensor] = {}  # label -> tf.tensor(bs, n)
        if self.compress:
            self.samples_compressed: dict[str, tf.Tensor] = {}

        self.eval_pure = QCEvaluator(self.ideal_gates_list, self.n)
        self.eval_hidden = QCEvaluator(unwrap_dict(self.hidden_gates_dict), self.n)
        self.eval_estimated = None

    def _init_hidden_channels(self, pure_channels_set: GateSet, params: NoiseParams = None) -> None:
        """
        """
        for name in pure_channels_set:
            if name in self.single_qub_gates_names:
                self.hidden_gates_dict[name] = tf.concat([pure_channels_set[name][tf.newaxis]] * self.n, axis=0)
            elif name in self.two_qub_gates_names:
                self.hidden_gates_dict[name] = tf.concat([pure_channels_set[name][tf.newaxis]] *
                                                         (self.n * (self.n - 1)), axis=0)
            elif name == ID_GATE:
                self.hidden_gates_dict[ID_GATE] = pure_channels_set[ID_GATE][tf.newaxis]
            else:
                raise ValueError('Gate was not specified during __init__')  # TODO: custom exception

        if params is None:
            return

        for name in params:
            tmp_list = tf.unstack(self.hidden_gates_dict[name])
            for num, replacement in params[name]:
                tmp_list[num] = replacement
            self.hidden_gates_dict[name] = tf.stack(tmp_list)

    # ---------------------------------------------------------------------------------------------------

    def _simple_template(self, tmpl: NconTemplate) -> NconTemplate:
        """
        Converts a template made for hidden or estimator evaluators into a template suited for pure evaluator.
        This conversion changes only tensors IDs, leaving tensor network structure intact.
        """
        tensors, net_struc, con_order, out_order = tmpl
        new_tensors = tensors.copy()
        e_id = len(self.single_qub_gates_names) * self.n

        for idx, old_tensor_id in enumerate(tensors):
            if old_tensor_id == e_id:
                new_tensors[idx] = len(self.single_qub_gates_names)
            elif old_tensor_id < e_id:
                new_tensors[idx] = old_tensor_id // self.n
            else:
                shifted = old_tensor_id - e_id - 1
                new_tensors[idx] = shifted // (self.n * (self.n - 1))
                new_tensors[idx] += len(self.single_qub_gates_names) + 1

        new_template = [new_tensors, net_struc, con_order, out_order]
        return new_template

    def add_circuit(self, tn_template: NconTemplate, name: str) -> None:
        self.tn_templates[name] = tn_template
        self.eval_pure.add_circuit(self._simple_template(tn_template), name)
        self.eval_hidden.add_circuit(tn_template, name)

    def generate_sample(self, name: str, smpl_size=10000) -> None:
        """
        Creates a batch of all-qubit samples for a circuit with name 'name'
        using the 'hidden' evaluator.
        """
        sample = self.eval_hidden.make_full_samples(name, smpl_size)
        if self.compress:
            dimdim = tf.constant([2] * self.n, dtype=INT)
            ids = util.ravel_multi_index(sample, dimdim)
            compressor = np.bincount(ids, minlength=2 ** self.n)
            self.samples_compressed[name] = tf.convert_to_tensor(compressor, dtype=FLOAT)  # this float is important
        else:
            self.samples_vault[name] = sample

    def generate_all_samples(self, smpl_size=10000, v=False) -> None:
        print('generating samples...', end=' ')
        for idx, name in enumerate(self.tn_templates):
            if v:
                print(f'doing sample with idx {idx} and name {name}')
            self.generate_sample(name, smpl_size)
            if idx == 3 * (len(self.tn_templates) // 4):
                print('75%', end=' ')
            elif idx == len(self.tn_templates) // 2:
                print('50%', end=' ')
            elif idx == len(self.tn_templates) // 4:
                print('25%', end=' ')
        print('Done!')

    @tf.function
    def get_gaussian_reg(self, channels_dict: GateSet, lmbd1: float = 1, lmbd2: float = 1):
        total_reg = tf.constant(0, dtype=FLOAT)
        for idx, gate_type in enumerate(channels_dict):  # 'S', 'H', etc.
            if gate_type == ID_GATE:
                pass
            elif gate_type in self.single_qub_gates_names:
                gate_type_norm = tf.math.abs(tf.linalg.norm(channels_dict[gate_type] -
                                                            self.ideal_gates_list[idx]) ** 2)
                total_reg += gate_type_norm * lmbd1
            else:  # no need to check containment in two_qub_gates: it was done at __init__
                gate_type_norm = tf.math.abs(tf.linalg.norm(channels_dict[gate_type] -
                                                            self.ideal_gates_list[idx]) ** 2)
                total_reg += gate_type_norm * lmbd2
        return total_reg

    # TODO: understand why ncon interferes with tf.function
    def true_loss_value(self, lmbd1: float = 1, lmbd2: float = 1):
        channels_dict = self.hidden_gates_dict

        dimdim = tf.constant([2] * self.n, dtype=tf.int64)
        all_bitstrings = tf.transpose(tf.unravel_index(np.arange(2 ** self.n), dimdim))

        total_logp = tf.constant(0, dtype=FLOAT)
        for name in self.tn_templates:  # we iterate by each circuit, the circuit is defined by its name
            circuit_logp = tf.constant(0, dtype=FLOAT)  # noqa (it's not duplication, other evaluator is used)

            if self.compress:
                sample_compressed = self.samples_compressed[name]
                bitstring_logp = tf.math.log(tf.math.abs(self.eval_hidden.evaluate(all_bitstrings, name)) +
                                             tf.constant(1e-12, dtype=FLOAT))
                circuit_logp += tf.reduce_sum(bitstring_logp * sample_compressed)
            else:
                sample = self.samples_vault[name]
                probs = self.eval_hidden.evaluate(sample, name)
                circuit_logp += tf.reduce_sum(tf.math.log(tf.math.abs(probs) +
                                                          tf.constant(1e-12, dtype=FLOAT)))

            total_logp += circuit_logp

        # Loss from circuits is calculated; now time to calculate reg

        return -total_logp + self.get_gaussian_reg(channels_dict, lmbd1, lmbd2)

    def get_statistic_for_circ(self, names: list[str], channels_dict: tp.Optional[GateSet] = None,
                               statistic='chi2', replace_gates: bool = False, v=False) -> list[tf.Tensor]:
        if statistic not in STATS_FUNCS:
            raise NotImplementedError

        ans = []
        if replace_gates:
            assert channels_dict is not None
            self.eval_estimated.gates = unwrap_dict(channels_dict)

        dimdim = tf.constant([2] * self.n, dtype=tf.int64)
        all_bitstrings = tf.transpose(tf.unravel_index(np.arange(2 ** self.n), dimdim))

        for name in names:
            sampled_probs = self.samples_compressed[name] / tf.reduce_sum(self.samples_compressed[name])
            eval_probs = tf.abs(self.eval_estimated.evaluate(all_bitstrings, name))
            if v:
                print(sampled_probs)
                print(eval_probs)
            ans.append(STATS_FUNCS[statistic](sampled_probs, eval_probs))

        return ans

    @abstractmethod
    def _loss_and_grad(self, *args, **kwargs):
        pass

    @abstractmethod
    def train_optimizer(self, *args, **kwargs):
        pass


class QGOptSolver(BaseSolver):
    """
    A Solver which uses QGOpt as main optimizer for adjusting estimated gate set towards hitten gate set.
    TODO: better docstring
    """
    def __init__(self,
                 qubits_num: int,
                 single_qub_gates_names: set[str],
                 two_qub_gates_names: set[str],
                 pure_channels_set: GateSet,
                 compress_samples: bool = False,
                 noise_params: NoiseParams = None,
                 initial_estimated_gates_override: tp.Optional[dict[str, tf.Variable]] = None,
                 noise_iter0: float = 0.0):
        super().__init__(qubits_num, single_qub_gates_names, two_qub_gates_names, pure_channels_set,
                         compress_samples, noise_params)

        if initial_estimated_gates_override is not None:
            self._run_sanity_checks_for_checkpoint(pure_channels_set, initial_estimated_gates_override)
            self.estimated_gates_dict = initial_estimated_gates_override
            print("Sanity checks passed; estimated set is successfully loaded from provided checkpoint")
        else:
            self.estimated_gates_dict: dict[str, tf.Variable] = {}
            self._init_estimated(pure_channels_set, noise_iter0)

        self.eval_estimated = QCEvaluator(unwrap_dict(get_complex_channel_form(self.estimated_gates_dict)), self.n)
        self.timestamps: dict[str, float] = {}

    def _run_sanity_checks_for_checkpoint(self,
                                          pure_channels_set: GateSet,
                                          initial_estimated_gates_override: dict[str, tf.Variable]) -> None:
        # TODO: write tests on this checkpoint loading procedure
        if initial_estimated_gates_override.keys() != pure_channels_set.keys():
            raise ValueError("Wrong set of quantum gates")
        for one_qub_key in self.single_qub_gates_names:
            assert initial_estimated_gates_override[one_qub_key].shape[0] == self.n, \
                f"Wrong number of noised gates for single-qubit gate {one_qub_key}, must be {self.n}"
            assert initial_estimated_gates_override[one_qub_key].shape[1:] == (4, 4, 2), \
                f"Wrong shape of passed gates with label {one_qub_key}, must be {(4, 4, 2)}"
        for two_qub_key in self.two_qub_gates_names:
            assert initial_estimated_gates_override[two_qub_key].shape[0] == self.n * (self.n - 1), \
                f"Wrong number of noised gates for two-qubit gate {two_qub_key}," \
                f" must be {self.n * (self.n - 1)}"
            assert initial_estimated_gates_override[two_qub_key].shape[1:] == (16, 16, 2), \
                f"Wrong shape of passed gates with label {two_qub_key}, must be {(16, 16, 2)}"

    def _init_estimated(self, pure_channels_set: GateSet, noise_iter0: float = 0.0) -> None:
        init_noise = tf.convert_to_tensor([noise_iter0, 0.0, 0.0], dtype=FLOAT)
        for name in pure_channels_set:
            if name in self.single_qub_gates_names:
                noised_channel = ns.make_1q_hybrid_channel(pure_channels_set[name], init_noise)
                params = qgo.manifolds.complex_to_real(c_util.convert_channel_to_params(noised_channel))
                self.estimated_gates_dict[name] = tf.Variable(tf.concat([params[tf.newaxis]] * self.n, axis=0))
            elif name in self.two_qub_gates_names:
                noised_channel = ns.make_2q_hybrid_channel(pure_channels_set[name], init_noise)
                params = qgo.manifolds.complex_to_real(c_util.convert_channel_to_params(noised_channel))
                self.estimated_gates_dict[name] = tf.Variable(tf.concat([params[tf.newaxis]] *
                                                                        (self.n * (self.n - 1)), axis=0))
            elif name == ID_GATE:
                params = qgo.manifolds.complex_to_real(c_util.convert_channel_to_params(pure_channels_set[ID_GATE]))
                self.estimated_gates_dict[ID_GATE] = tf.Variable(params[tf.newaxis])
            else:
                raise ValueError('Gate was not specified during __init__')  # TODO: custom exception

        print(f"Estimated set is generated from pure channels by applying noise parameters {init_noise}")

    # ---------------------------------------------------------------------------------------------------

    def add_circuit(self, tn_template: NconTemplate, name: str, timestamp: tp.Optional[float] = None) -> None:
        self.timestamps[name] = float(len(self.tn_templates)) if timestamp is None else timestamp
        self.tn_templates[name] = tn_template
        self.eval_pure.add_circuit(self._simple_template(tn_template), name)
        self.eval_hidden.add_circuit(tn_template, name)
        self.eval_estimated.add_circuit(tn_template, name)

    # TODO: understand why ncon interferes with tf.function
    def _loss_and_grad(self, lmbd1: float, lmbd2: float, v: bool = False) -> [TENSOR, dict[TENSOR]]:
        with tf.GradientTape() as tape:
            channels_dict = get_complex_channel_form(self.estimated_gates_dict)
            # The estimated_set consists of several (default - four) tf.Variables. They get unwrapped in a 1D array
            # and then they get passed into 'eval_estimated'
            self.eval_estimated.gates = unwrap_dict(channels_dict)

            dimdim = tf.constant([2] * self.n, dtype=tf.int64)
            all_bitstrings = tf.transpose(tf.unravel_index(np.arange(2 ** self.n), dimdim))

            total_logp = tf.constant(0, dtype=FLOAT)
            for name in self.tn_templates:  # we iterate by each circuit, the circuit is defined by its name
                total_logp += self._get_circuit_logp(name, all_bitstrings)

            # Loss from circuits is calculated; now time to calculate reg

            loss = -total_logp + self.get_gaussian_reg(channels_dict, lmbd1, lmbd2)

            grad = tape.gradient(loss, self.estimated_gates_dict)

        return loss, grad

    def _get_circuit_logp(self, name: str, bitstrings: tp.Optional[tf.Tensor] = None) -> tf.Tensor:
        circuit_logp = tf.constant(0, dtype=FLOAT)
        if self.compress:
            sample_compressed = self.samples_compressed[name]
            bitstring_logp = tf.math.log(tf.math.abs(self.eval_estimated.evaluate(bitstrings, name)) +
                                         tf.constant(1e-12, dtype=FLOAT))
            circuit_logp += tf.reduce_sum(bitstring_logp * sample_compressed)
        else:
            sample = self.samples_vault[name]
            probs = self.eval_estimated.evaluate(sample, name)
            circuit_logp += tf.reduce_sum(tf.math.log(tf.math.abs(probs) +
                                                      tf.constant(1e-12, dtype=FLOAT)))

        return circuit_logp

    def _loss_and_grad_with_time(self, lmbd1: float, lmbd2: float,
                                 timestamp: float, v: bool = False) -> [tf.Tensor, dict[tf.Tensor]]:
        with tf.GradientTape() as tape:
            # same as loss and grad but with some time checks and exp
            channels_dict = get_complex_channel_form(self.estimated_gates_dict)
            self.eval_estimated.gates = unwrap_dict(channels_dict)

            dimdim = tf.constant([2] * self.n, dtype=tf.int64)
            all_bitstrings = tf.transpose(tf.unravel_index(np.arange(2 ** self.n), dimdim))

            total_logp = tf.constant(0, dtype=FLOAT)
            for name in self.tn_templates:
                if self.timestamps[name] > timestamp or self.timestamps[name] + 4 * EXP_CONST < timestamp:
                    continue
                passed_time = tf.constant(((timestamp - self.timestamps[name]) / EXP_CONST), dtype=FLOAT)
                total_logp += (self._get_circuit_logp(name, all_bitstrings) * tf.math.exp(-passed_time))

            # Loss from circuits is calculated; now time to calculate reg

            loss = -total_logp + self.get_gaussian_reg(channels_dict, lmbd1, lmbd2)

            grad = tape.gradient(loss, self.estimated_gates_dict)

        return loss, grad

    def get_circ_l1_norms(self, name: str) -> tuple[tf.Tensor, tf.Tensor]:
        dimdim = tf.constant([2] * self.n, dtype=tf.int64)
        all_bitstrings = tf.transpose(tf.unravel_index(np.arange(2 ** self.n), dimdim))

        probs_pure = self.eval_pure.evaluate(all_bitstrings, name)
        probs_est = self.eval_estimated.evaluate(all_bitstrings, name)
        probs_true = self.eval_hidden.evaluate(all_bitstrings, name)

        return (tf.reduce_sum(tf.abs(tf.subtract(probs_est, probs_pure))),
                tf.reduce_sum(tf.abs(tf.subtract(probs_est, probs_true))))

    def train_optimizer(self, opt: qgo.optimizers, lmbd1: float, lmbd2: float,
                        iters: int, v: int = 0, timestamp: tp.Optional[float] = None) -> list[float]:
        # this list will be filled by value of error per iteration
        loss_dynamics = []
        step_print = max(iters // (10 * v), 2) if v != 0 else 10000000

        # optimization loop
        for iteration in range(iters):
            if timestamp is None:
                loss, grad = self._loss_and_grad(lmbd1, lmbd2)
            else:
                loss, grad = self._loss_and_grad_with_time(lmbd1, lmbd2, timestamp)
            if v >= 2:
                print(iteration)
                print(loss)
            if v >= 3:
                print(grad)

            if v == 1:
                if (iteration % step_print) == 0:
                    print(f'{iteration} out of {iters} iterations passed')

            loss_dynamics.append(loss)

            opt.apply_gradients(zip(grad.values(), self.estimated_gates_dict.values()))

        return loss_dynamics


class SimpleSolver(BaseSolver):
    # this guy is not used in the experiments, has code duplication and possibly bugs
    pass
    # def __init__(self,
    #              qubits_num: int,
    #              single_qub_gates_names: set[str],
    #              two_qub_gates_names: set[str],
    #              pure_channels_set: GateSet,
    #              noise_funcs: dict[str, tp.Callable[[tf.Tensor, ...], tf.Tensor]],
    #              func_args_nums: dict[str, int],
    #              compress_samples: bool = False,
    #              noise_params: NoiseParams = None):
    #     super().__init__(qubits_num, single_qub_gates_names, two_qub_gates_names, pure_channels_set,
    #                      compress_samples, noise_params)
    #
    #     self.pure_channels_set = {}
    #     for name in pure_channels_set:
    #         self.pure_channels_set[name] = tf.constant(pure_channels_set[name], dtype=COMPLEX)
    #
    #     self.noise_parameters: dict[str, tf.Variable] = {}
    #     self.noise_funcs = noise_funcs
    #     self.funcs_args_nums = func_args_nums
    #
    #     self._init_estimated()
    #     channels_dict = self._reconstruct_set(pure_channels_set=self.pure_channels_set,
    #                                           n=self.n,
    #                                           single_qub_gates_names=self.single_qub_gates_names,
    #                                           two_qub_gates_names=self.two_qub_gates_names,
    #                                           noise_funcs=self.noise_funcs,
    #                                           noise_parameters=self.noise_parameters)
    #     self.eval_estimated = QCEvaluator(unwrap_dict(channels_dict), self.n)
    #
    # def _init_estimated(self) -> None:
    #     for name in self.pure_channels_set:
    #         params = tf.convert_to_tensor([0.0] * self.funcs_args_nums[name], dtype=FLOAT)
    #         if name in self.single_qub_gates_names:
    #             self.noise_parameters[name] = tf.Variable(tf.concat([params[tf.newaxis]] * self.n, axis=0))
    #         elif name in self.two_qub_gates_names:
    #             self.noise_parameters[name] = tf.Variable(tf.concat([params[tf.newaxis]] *
    #                                                                 round((self.n * (self.n - 1)) / 2), axis=0))
    #         elif name == ID_GATE:
    #             pass
    #         else:
    #             raise ValueError('Gate was not specified during __init__')  # TODO: custom exception
    #
    # @staticmethod
    # def _reconstruct_set(pure_channels_set: GateSet,
    #                      n: int,
    #                      single_qub_gates_names: set[str],
    #                      two_qub_gates_names: set[str],
    #                      noise_funcs: dict[str, tp.Callable[[tf.Tensor, ...], tf.Tensor]],
    #                      noise_parameters: dict[str, tf.Variable]
    #                      ) -> dict[str, tf.Tensor]:
    #     # method is fully functional because Tf's autograd works best like this
    #     ret = {}
    #     # TODO: make this faster, ideal is not a single Python for!
    #
    #     for name in pure_channels_set:
    #         sublist = []
    #         if name in single_qub_gates_names:
    #             for idx in range(n):
    #                 sublist.append(noise_funcs[name](pure_channels_set[name],
    #                                                  noise_parameters[name][idx])[tf.newaxis])
    #         elif name in two_qub_gates_names:
    #             for idx in range(round((n * (n - 1)) / 2)):
    #                 sublist.append(noise_funcs[name](pure_channels_set[name],
    #                                                  noise_parameters[name][idx])[tf.newaxis])
    #         elif name == ID_GATE:
    #             sublist.append(pure_channels_set[ID_GATE][tf.newaxis])
    #         # no exception here
    #         ret[name] = tf.concat(sublist, axis=0)
    #
    #     return ret
    #
    # # ---------------------------------------------------------------------------------------------------
    #
    # def add_circuit(self, tn_template: NconTemplate, name: str) -> None:
    #     self.tn_templates[name] = tn_template
    #     self.eval_pure.add_circuit(self._simple_template(tn_template), name)
    #     self.eval_hidden.add_circuit(tn_template, name)
    #     self.eval_estimated.add_circuit(tn_template, name)
    #
    # def _loss_and_grad(self, lmbd1: float, lmbd2: float, v: bool = False) -> [tf.Tensor, dict[str, tf.Tensor]]:
    #     # TODO: deal with code duplication
    #     with tf.GradientTape() as tape:
    #         channels_dict = self._reconstruct_set(pure_channels_set=self.pure_channels_set,
    #                                               n=self.n,
    #                                               single_qub_gates_names=self.single_qub_gates_names,
    #                                               two_qub_gates_names=self.two_qub_gates_names,
    #                                               noise_funcs=self.noise_funcs,
    #                                               noise_parameters=self.noise_parameters)
    #
    #         self.eval_estimated.gates = unwrap_dict(channels_dict)
    #
    #         dimdim = tf.constant([2] * self.n, dtype=INT)
    #         all_bitstrings = tf.transpose(tf.unravel_index(np.arange(2 ** self.n), dimdim))
    #
    #         total_logp = tf.constant(0, dtype=FLOAT)
    #         for name in self.tn_templates:  # we iterate by each circuit, the circuit is defined by its name
    #             if self.compress:
    #                 sample_compressed = self.samples_compressed[name]
    #             else:
    #                 sample = self.samples_vault[name]
    #
    #             circuit_logp = tf.constant(0, dtype=FLOAT)
    #
    #             if self.compress:
    #                 bitstring_logp = tf.math.log(tf.math.abs(self.eval_estimated.evaluate(all_bitstrings, name)) +
    #                                              tf.constant(1e-12, dtype=FLOAT))
    #                 circuit_logp += tf.reduce_sum(bitstring_logp * sample_compressed)
    #             else:
    #                 probs = self.eval_estimated.evaluate(sample, name)
    #                 circuit_logp += tf.reduce_sum(tf.math.log(tf.math.abs(probs) +
    #                                                           tf.constant(1e-12, dtype=FLOAT)))
    #
    #             total_logp += circuit_logp
    #
    #         # Loss from circuits is calculated; now time to calculate reg
    #         loss = -total_logp + self.get_gaussian_reg(channels_dict, lmbd1, lmbd2)
    #         grad = tape.gradient(loss, self.noise_parameters)
    #
    #     return loss, grad
    #
    # def train_optimizer(self, opt: tf.optimizers.Optimizer, lmbd1: float, lmbd2: float,
    #                     iters: int, v: int = 0) -> list[float]:
    #     # TODO: deal with code duplication
    #
    #     loss_dynamics = []
    #     step_print = max(iters // (10 * v), 2) if v != 0 else 10000000
    #
    #     # optimization loop
    #     for iteration in range(iters):
    #         loss, grad = self._loss_and_grad(lmbd1, lmbd2)
    #         if v >= 2:
    #             print(iteration)
    #             print(loss)
    #         if v >= 3:
    #             print(grad)
    #
    #         if v == 1:
    #             if (iteration % step_print) == 0:
    #                 print(f'{iteration} out of {iters} iterations passed')
    #
    #         loss_dynamics.append(loss)
    #
    #         opt.apply_gradients(zip(grad.values(), self.noise_parameters.values()))
    #
    #     return loss_dynamics


FUNCS = {'M': util.get_povm_dist,
         'P': util.get_prep_dist}


class QGOptSolverDebug(QGOptSolver):
    def __init__(self,
                 qubits_num: int,
                 single_qub_gates_names: set[str],
                 two_qub_gates_names: set[str],
                 pure_channels_set: GateSet,
                 compress_samples: bool = False,
                 noise_params: NoiseParams = None,
                 initial_estimated_gates_override: tp.Optional[dict[str, tf.Variable]] = None,
                 noise_iter0: float = 0.0):
        super().__init__(qubits_num=qubits_num,
                         single_qub_gates_names=single_qub_gates_names,
                         two_qub_gates_names=two_qub_gates_names,
                         pure_channels_set=pure_channels_set,
                         compress_samples=compress_samples,
                         noise_params=noise_params,
                         initial_estimated_gates_override=initial_estimated_gates_override,
                         noise_iter0=noise_iter0)

        self.pure_channels_set = pure_channels_set
        self.starting_pos_dict = get_complex_channel_form(self.estimated_gates_dict)

    def train_optimizer(self, opt: qgo.optimizers, lmbd1: float, lmbd2: float,
                        iters: int, v: int = 0,
                        fid_ctr: int = 1,
                        norm_ctr: int = -1,
                        timestamp: tp.Optional[float] = None) -> tuple[list[float],
                                                                       dict[tuple[str, int], list[tf.Tensor]],
                                                                       list[tf.Tensor]]:
        loss_dynamics = []
        l1_norms = []
        fids_dict = defaultdict(list)
        step_print = max(iters // (10 * v), 2) if v != 0 else 10000000

        # optimization loop
        for iteration in range(iters):
            if timestamp is None:
                loss, grad = self._loss_and_grad(lmbd1, lmbd2)
            else:
                loss, grad = self._loss_and_grad_with_time(lmbd1, lmbd2, timestamp)
            if v >= 2:
                print(iteration)
                print(loss)

            if v >= 3:
                print(grad)

            if v == 1:
                if (iteration % step_print) == 0:
                    print(f'{iteration} out of {iters} iterations passed')

            loss_dynamics.append(loss)

            if fid_ctr > 0 and iteration % fid_ctr == 0:
                channels_dict = get_complex_channel_form(self.estimated_gates_dict)
                for gate_name in self.single_qub_gates_names:
                    for gate_id in range(self.n):
                        func = FUNCS[gate_name] if gate_name in FUNCS else util.diamond_norm_1q
                        fid = func(channels_dict[gate_name][gate_id],
                                   self.hidden_gates_dict[gate_name][gate_id])
                        fids_dict[(gate_name, gate_id, 't')].append(fid)

                        fid = func(channels_dict[gate_name][gate_id],
                                   self.pure_channels_set[gate_name])
                        fids_dict[(gate_name, gate_id, 'i')].append(fid)

                for gate_name in self.two_qub_gates_names:
                    for gate_id in range(self.n * (self.n - 1)):
                        fid = util.diamond_norm_2q(channels_dict[gate_name][gate_id],
                                                   self.hidden_gates_dict[gate_name][gate_id])
                        fids_dict[(gate_name, gate_id, 't')].append(fid)

                        fid = util.diamond_norm_2q(channels_dict[gate_name][gate_id],
                                                   self.pure_channels_set[gate_name])
                        fids_dict[(gate_name, gate_id, 'i')].append(fid)

            opt.apply_gradients(zip(grad.values(), self.estimated_gates_dict.values()))

            if norm_ctr > 0 and iteration % norm_ctr == 0:
                l1_norms.append(self._get_current_l1_norm())

        return loss_dynamics, fids_dict, l1_norms

    def _get_current_l1_norm(self):
        # self.eval_estimated.gates = get_complex_channel_form(self.estimated_gates_dict) - SHOULD BE TRUE
        dimdim = tf.constant([2] * self.n, dtype=tf.int64)
        all_bitstrings = tf.transpose(tf.unravel_index(np.arange(2 ** self.n), dimdim))

        l1_norm = tf.constant(0, dtype=FLOAT)

        for name in self.tn_templates:
            probs_est = self.eval_estimated.evaluate(all_bitstrings, name)
            probs_true = self.eval_hidden.evaluate(all_bitstrings, name)
            l1_norm += tf.reduce_sum(tf.abs(tf.subtract(probs_est, probs_true)))

        return l1_norm

    def _negloglik(self, channels_dict: GateSet) -> tf.Tensor:
        self.eval_estimated.gates = unwrap_dict(channels_dict)

        dimdim = tf.constant([2] * self.n, dtype=tf.int64)
        all_bitstrings = tf.transpose(tf.unravel_index(np.arange(2 ** self.n), dimdim))

        total_logp = tf.constant(0, dtype=FLOAT)
        for name in self.tn_templates:  # we iterate by each circuit, the circuit is defined by its name
            total_logp += self._get_circuit_logp(name, all_bitstrings)

        return -total_logp

    def losses_between(self, lmbd1: float, lmbd2: float,
                       channels_dict_1: GateSet, channels_dict_2: GateSet,
                       points: int, v: bool = False) -> list[tf.Tensor]:
        losses = []
        step = max(points // 10, 1)

        new_dict = {}
        for idx, t in enumerate(tf.linspace(0, 1, points)):

            if idx % step == 0:
                print(f'{idx} losses out of {points} computed.')

            t_tf = tf.cast(t, COMPLEX)

            for name in channels_dict_1:
                new_dict[name] = channels_dict_1[name] * (1 - t_tf) + channels_dict_2[name] * t_tf

            losses.append(self._negloglik(new_dict) + self.get_gaussian_reg(new_dict, lmbd1, lmbd2))

        return losses

