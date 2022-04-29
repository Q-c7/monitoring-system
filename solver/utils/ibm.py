import tensorflow as tf
from qiskit import QuantumCircuit
from math import pi

from solver.utils.misc import FLOAT


def _convert_circ_to_qiskit(n_q: int, circ: list[str]) -> QuantumCircuit:
    qc = QuantumCircuit(n_q, n_q)
    # print(circ)
    for gate in circ:
        info = gate.split('_')
        if info[0] == 'P' or info[0] == 'M':
            pass
        elif info[0] == 'ID':
            qc.id(int(info[1]))
        elif info[0] == 'RZ':
            qc.rz(pi / 4, int(info[1]))
        elif info[0] == 'X':
            qc.x(int(info[1]))
        elif info[0] == 'SX':
            qc.sx(int(info[1]))
        elif info[0] == 'CX':
            qc.cx(int(info[1]), int(info[2]))

    for i in range(n_q):
        qc.measure(i, i)

    return qc


def build_circuits(n_q: int, raw_dict: dict[str, list[str]]) -> dict[str, QuantumCircuit]:
    """
    Converts a dict of circuits specified in raw string format to dict of circuits in Qiskit format

    Args:
        n_q: number of qubits in quantum & classical registers
        raw_dict: dict of circuits specified in list[str] format ('GATE_Q1_Q2' or 'GATE_Q')

    Returns:
        dict of circuits, where each circuit is represented by Qiskit.QuantumCircuit class
    """
    circuits = dict()
    for nam, circ in raw_dict.items():
        circuits[nam] = _convert_circ_to_qiskit(n_q, circ)

    return circuits


ALL_BS = [bin(x)[2:].zfill(5)[::-1] for x in range(32)]


def convert_result_to_zipped_tf(result: dict[str, int]) -> tf.Tensor:
    """
    Converts IBMQ result obtained via job.result().get_counts(...) to tf.Tensor with shape (32,)
    """
    assert len(ALL_BS) == 32
    tmp_list = []
    for key in ALL_BS:
        try:
            tmp_list.append(result[key])
        except KeyError:
            tmp_list.append(0)

    return tf.convert_to_tensor(tmp_list, dtype=FLOAT)


