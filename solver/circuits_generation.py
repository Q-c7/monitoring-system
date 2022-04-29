import typing as tp
from random import randint, random, sample

from numpy.random import choice
from solver.utils.misc import NconTemplate, ID_GATE


def random_circ_generator(circ_len_min: int = 5,
                          circ_len_max: int = 10,
                          qubit_num: int = 2,
                          circ_num: int = 20,
                          single_qubit_gates: tp.Optional[list[str]] = None,
                          two_qubit_gates: tp.Optional[list[str]] = None,
                          two_qubit_gate_prob: float = 0.25) -> list[list[str]]:
    """
    Generates random quantum circuits in the human form.

    Args:
        circ_len_min: minimum length of a circuit.
        circ_len_max: maximum length of a circuit.
        qubit_num: number of qubits in a circuit
        circ_num: number of circuits to be generated
        single_qubit_gates: names of single-qubit gates
        two_qubit_gates: names of two-qubit gates
        two_qubit_gate_prob: probability to pick a two-qubit gate as next random gate

    Returns:
        A sequence with strings containing human interpretation of quantum circuits.
        Each gate is written as GATE_TARGET(S) (example: 'CZ_3_5').
        For two-qubit gates, controlling qubit is always the smallest.
    """
    if single_qubit_gates is None:
        single_qubit_gates = ['H', 'S', 'T']
    if two_qubit_gates is None:
        two_qubit_gates = ['CZ']

    circuits = []
    for _ in range(circ_num):
        circuit_cur = []
        gates_num = randint(circ_len_min, circ_len_max)
        for _ in range(gates_num):
            # Single qubit case:
            if random() > two_qubit_gate_prob:
                qubit_index = sample(list(range(qubit_num)), 1)[0]
                gate_name = sample(single_qubit_gates, 1)[0]
                gate_full = gate_name + '_' + str(qubit_index)
            # Two qubit case:
            else:
                qubit_indices = sample(list(range(qubit_num)), 2)
                qubit_indices.sort()
                gate_name = sample(two_qubit_gates, 1)[0]
                gate_full = gate_name + '_' + str(qubit_indices[0]) + '_' + str(qubit_indices[1])
            circuit_cur.append(gate_full)

        circuits.append(circuit_cur)

    return circuits


def generate_t_shaped_5q(circ_num: int = 20,
                         min_lines: int = 0,
                         max_lines: int = 4,
                         enable_cx: int = 0b111,
                         enable_prep: bool = True,
                         enable_measurement: bool = True,
                         single_qubit_gates: tp.Optional[list[str]] = None) -> list[list[str]]:
    """
    Generates random quantum circuits in the human form specially for IBM T-shaped processor.
    This time the architecture of quantum processor is fixed, allowing for only a few two-qubit gate combinations.
    Also the circuit must consist of blocks (a. k. a. lines) composed of 5 random single-qubit gates
    or fixed combination of two-qubit gates (see article for details)
    It's worth noting that CX is hard-coded as sole two-qubit gate.

    Args:
        circ_num: number of circuits to be generated
        min_lines: minimum additional blocks (one block is always present, but it does not have two-qubit part)
        max_lines: maximum additional blocks
        enable_cx: a bit mask which defince whether blocks certain two-qubit lines or not. There are only three
        variants so it's encoded in ints between 0 and 7.
        enable_prep: whether or not state preparation operator is added in the beginning
        enable_measurement: whether or not measurement operator is added in the end
        single_qubit_gates: names of sinle-qubit gates. Default is ['ID', 'RZ', 'SX', 'X']

    Returns:
        A sequence with strings containing human interpretation of quantum circuits.
        Each gate is written as GATE_TARGET(S) (example: 'CZ_3_5').
    """
    qubit_num = 5

    def _get_line() -> list[str]:
        gates = list(choice(single_qubit_gates, qubit_num))
        for ij in range(qubit_num):
            gates[ij] += '_' + str(ij)
        return gates

    if single_qubit_gates is None:
        single_qubit_gates = ['ID', 'RZ', 'SX', 'X']

    case_a = ['CX_0_1', 'CX_3_4']
    case_b = ['CX_2_1', 'CX_3_4']
    case_c = ['CX_1_3']
    cases = []

    if enable_cx & 0b1:
        cases.append(case_a)
    if enable_cx & 0b10:
        cases.append(case_b)
    if enable_cx & 0b100:
        cases.append(case_c)

    circuits = []
    for circ_ctr in range(circ_num):
        circuit_cur = []
        if enable_prep:
            for i in range(qubit_num):
                circuit_cur += [('P_' + str(i))]  # state preparation
        circuit_cur += _get_line()

        lines_num = randint(min_lines, max_lines)

        for _ in range(lines_num):
            if enable_cx:
                circuit_cur += sample(cases, 1)[0]
            circuit_cur += _get_line()

        if enable_measurement:
            for i in range(qubit_num):
                circuit_cur += [('M_' + str(i))]  # measurement operator for read-out error

        circuits.append(circuit_cur)

    return circuits


def transform_struc_to_ncon(circ_array: list[list[str]], n_qubits: int) -> list[NconTemplate]:
    """
    Transforms a circuit in the format provided by random generator to ncon template format.

    Args:
        circ_array: List of circuits in human format: list[str]
        n_qubits: Number of qubits for these circuits.

    Returns:
        A list of half-ncon templates where tensors are still written in human form.
    """
    res = []
    for circ in circ_array:
        curr_legs = [i1 for i1 in range(1, n_qubits + 1)]
        net_struc = []
        counter = n_qubits + 1

        for gate in circ:
            gate_info = gate.split('_')

            # Single-qubit case
            if len(gate_info) == 2:
                qubit = int(gate_info[1])
                net_struc.append([counter, curr_legs[qubit]])
                curr_legs[qubit] = counter
                counter += 1

            # Two-qubit case
            elif len(gate_info) == 3:
                qubits = [int(gate_info[1]), int(gate_info[2])]
                net_struc.append([counter + 1, counter, curr_legs[qubits[1]], curr_legs[qubits[0]]])
                curr_legs[qubits[0]] = counter
                curr_legs[qubits[1]] = counter + 1
                counter += 2
        # The net_struc is formed, but it's not connected to the outputs
        circ_copy = circ.copy()

        for qubit in range(0, n_qubits):
            # no gates attached to leg - create an identity gate
            if curr_legs[qubit] < n_qubits + 1:
                curr_legs[qubit] = -(qubit + 1)
                net_struc.append([-(qubit + 1), qubit + 1])
                circ_copy.append(ID_GATE)  # no need to write a target

            # there were some gates - replace last leg id with -(qubit + 1)
            else:
                breaking = False
                for sublist in reversed(net_struc):
                    if breaking:
                        break

                    for idx, leg in enumerate(sublist):
                        if leg == curr_legs[qubit]:
                            sublist[idx] = -(qubit + 1)
                            curr_legs[qubit] = -(qubit + 1)
                            breaking = True
        # net_struc is modified properly and each leg is now connected to the outputs

        con_legs = set()
        for sublist in net_struc:
            con_legs.update([leg for leg in sublist if leg >= n_qubits + 1])

        out_legs = [-i1 for i1 in range(1, n_qubits + 1)]
        res.append([circ_copy, net_struc, list(con_legs), out_legs])

    return res


class DataGenerator:
    def __init__(self,
                 qubits_num: int,
                 gates_names: list[str],
                 single_qub_gates_num: int,
                 two_qub_gates_num: int):
        """
        A class which generates random circuits.

        Args:
            qubits_num: Number of qubits in a circuit.
            gates_names: All quantum gates present in a circuit. Single-qubit gates must go first.
            Identity gate is not counted.

            single_qub_gates_num: Number of single-qubit gates in list 'gates_names'.
            two_qub_gates_num: Number of two-qubit gates in list 'gates_names'.
        """

        assert single_qub_gates_num + two_qub_gates_num == len(gates_names)
        self.n = qubits_num
        self.single_qub_gates = {}
        for idx, gate in enumerate(gates_names[:single_qub_gates_num]):
            self.single_qub_gates[gate] = idx

        self.two_qub_gates = {}
        for idx, gate in enumerate(gates_names[-two_qub_gates_num:]):
            self.two_qub_gates[gate] = idx

    def _finish_ncon_transformation(self, humcon_circs: list[NconTemplate]) -> list[NconTemplate]:
        # This function replaces strings in ncon templates to tensor ID's.
        # Remember: first all single-qubit gates, then _E, then two-qubit gates.
        ans = []
        for circ in humcon_circs:
            tensors, net_struc, con_order, out_order = circ
            new_tensors = []
            for gate in tensors:
                gate_info = gate.split('_')
                # TODO: custom exceptions
                if gate_info[0] == ID_GATE.split('_')[0]:
                    new_tensor_id = len(self.single_qub_gates) * self.n

                elif gate_info[0] in self.single_qub_gates:
                    assert len(gate_info) == 2
                    shift = self.single_qub_gates[gate_info[0]]
                    new_tensor_id = shift * self.n + int(gate_info[1])

                elif gate_info[0] in self.two_qub_gates:
                    assert len(gate_info) == 3
                    shift = self.two_qub_gates[gate_info[0]]
                    new_tensor_id = len(self.single_qub_gates) * self.n
                    new_tensor_id += shift * self.n * (self.n - 1)
                    new_tensor_id += (self.n - 1) * int(gate_info[1])
                    new_tensor_id += int(gate_info[2])
                    if int(gate_info[2]) < int(gate_info[1]):
                        new_tensor_id += 1

                else:
                    raise ValueError("Wrong format of gate name", gate_info[0])

                new_tensors.append(new_tensor_id)

            ans.append([new_tensors, net_struc, con_order, out_order])
        return ans

    @staticmethod
    def _make_tmpl_dict(ncon_circs: list[NconTemplate], name: str = '') -> dict[str, NconTemplate]:
        new_dict = {}
        for i, template in enumerate(ncon_circs):
            new_dict[name + str(i)] = template
        return new_dict

    def generate_data(self,
                      circ_len_min: int = 5,
                      circ_len_max: int = 10,
                      circ_num: int = 20,
                      two_qubit_gate_prob: tp.Optional[float] = None,
                      custom_name: str = '_') -> dict[str, NconTemplate]:
        """
        Generates random quantum circuits in the ncon form. Always calls 'random_circ_generator' first, then converts.

        Args:
            circ_len_min: minimum length of a circuit.
            circ_len_max: maximum length of a circuit.
            circ_num: number of circuits to be generated.
            two_qubit_gate_prob: probability to pick a two-qubit gate as next random gate .
            custom_name: a prefix which determines circuit's name.

        Returns:
            Dict of ncon templates; names are chosen as custom_name + str(idx).
        """
        gate_prob = 0.25 if two_qubit_gate_prob is None else two_qubit_gate_prob
        human_circs = random_circ_generator(circ_len_min=circ_len_min,
                                            circ_len_max=circ_len_max,
                                            qubit_num=self.n,
                                            circ_num=circ_num,
                                            single_qubit_gates=list(self.single_qub_gates.keys()),
                                            two_qubit_gates=list(self.two_qub_gates.keys()),
                                            two_qubit_gate_prob=gate_prob)

        humcon_circs = transform_struc_to_ncon(human_circs, self.n)
        ncon_circs = self._finish_ncon_transformation(humcon_circs)
        return self._make_tmpl_dict(ncon_circs, name=custom_name)

    def get_tmpl_dict_from_human_circs(self, human_circs: list[list[str]]) -> dict[str, NconTemplate]:
        """
        This function converts a list of human templates to dict of ncon templates.

        Args:
            human_circs: List of circuits in human format: list[str]

        Returns:
            Dict of ncon templates; names are chosen as '_' + str(idx) by default.
        """
        humcon_circs = transform_struc_to_ncon(human_circs, self.n)
        ncon_circs = self._finish_ncon_transformation(humcon_circs)
        return self._make_tmpl_dict(ncon_circs)
