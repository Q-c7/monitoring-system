import pytest

from solver.utils.misc import NconTemplate, ID_GATE
import solver.circuits_generation as cg

TRANS_TEST_CASES = [
    (
        [['H_0', 'T_2', 'CZ_0_2', 'S_3', 'H_0', 'S_2', 'H_1', 'T_2'],
         ['S_3', 'T_1', 'T_2', 'S_0', 'S_2', 'S_2', 'CZ_0_2', 'CZ_0_3'],
         ['CZ_2_3', 'CZ_0_1', 'S_2', 'S_0', 'CZ_2_3', 'S_0', 'T_2', 'CZ_0_2', 'H_1', 'CZ_1_3'],
         ['T_2', 'S_2', 'CZ_1_3', 'H_2', 'T_1', 'H_2', 'CZ_0_3', 'T_2', 'CZ_2_3', 'CZ_0_1', 'T_3'],
         ['H_2', 'S_0', 'S_0', 'H_3', 'H_1', 'T_3', 'H_3', 'T_2', 'T_2', 'T_1', 'S_3']],
        5,
        [[['H_0', 'T_2', 'CZ_0_2', 'S_3',
           'H_0', 'S_2', 'H_1', 'T_2', ID_GATE],
          [[6, 1], [7, 3], [9, 8, 7, 6], [-4, 4],
           [-1, 8], [12, 9], [-2, 2], [-3, 12], [-5, 5]],
          [6, 7, 8, 9, 12],
          [-1, -2, -3, -4, -5]],

         [['S_3', 'T_1', 'T_2', 'S_0', 'S_2',
           'S_2', 'CZ_0_2', 'CZ_0_3', ID_GATE],
          [[6, 4], [-2, 2], [8, 3], [9, 1], [10, 8],
           [11, 10], [-3, 12, 11, 9], [-4, -1, 6, 12], [-5, 5]],
          [6, 8, 9, 10, 11, 12],
          [-1, -2, -3, -4, -5]],

         [['CZ_2_3', 'CZ_0_1', 'S_2', 'S_0', 'CZ_2_3',
           'S_0', 'T_2', 'CZ_0_2', 'H_1', 'CZ_1_3', ID_GATE],
          [[7, 6, 4, 3], [9, 8, 2, 1], [10, 6], [11, 8], [13, 12, 7, 10],
           [14, 11], [15, 12], [-3, -1, 15, 14], [18, 9], [-4, -2, 13, 18], [-5, 5]],
          [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 18],
          [-1, -2, -3, -4, -5]],

         [['T_2', 'S_2', 'CZ_1_3', 'H_2', 'T_1', 'H_2',
           'CZ_0_3', 'T_2', 'CZ_2_3', 'CZ_0_1', 'T_3', ID_GATE],
          [[6, 3], [7, 6], [9, 8, 4, 2], [10, 7], [11, 8],  [12, 10],
           [14, 13, 9, 1], [15, 12], [17, -3, 14, 15], [-2, -1, 11, 13], [-4, 17], [-5, 5]],
          [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17],
          [-1, -2, -3, -4, -5]],

         [['H_2', 'S_0', 'S_0', 'H_3', 'H_1', 'T_3',
           'H_3', 'T_2', 'T_2', 'T_1', 'S_3', ID_GATE],
          [[6, 3], [7, 1], [-1, 7], [9, 4], [10, 2], [11, 9],
           [12, 11], [13, 6], [-3, 13], [-2, 10], [-4, 12], [-5, 5]],
          [6, 7, 9, 10, 11, 12, 13],
          [-1, -2, -3, -4, -5]]]
    )

]

TRANS_TEST_IDS = ['big_random_test', ]
# TODO: more test cases


@pytest.mark.parametrize(['circs', 'qubits', 'etalon'], TRANS_TEST_CASES, ids=TRANS_TEST_IDS)
def test_trans(circs: list[list[str]], qubits: int, etalon: NconTemplate):
    assert cg.transform_struc_to_ncon(circs, qubits) == etalon


CONVERSION_TEST_CASES = [
    (
        ['H', 'S', 'T', 'CZ'],
        3,
        1,
        [['H_0', 'T_3', 'T_3', 'T_4', 'T_2', 'H_0', 'S_4'],
         ['T_3', 'S_4', 'S_4', 'H_2', 'H_1', 'CZ_0_3', 'T_0', 'CZ_2_4', 'CZ_0_1'],
         ['T_4', 'S_0', 'H_3', 'CZ_2_3', 'H_1', 'CZ_1_4', 'S_3', 'T_2', 'CZ_2_4'],
         ['CZ_1_4', 'CZ_3_4', 'T_4', 'S_1', 'S_3', 'T_3'],
         ['T_2', 'T_2', 'S_4', 'S_2', 'H_3', 'S_4', 'S_2', 'T_0', 'S_1', 'T_3']],
        5,
        [[0, 13, 13, 14, 12, 0, 9, 15],
         [13, 9, 9, 2, 1, 18, 10, 27, 16],
         [14, 5, 3, 26, 1, 23, 8, 12, 27],
         [23, 31, 14, 6, 8, 13, 15, 15],
         [12, 12, 9, 7, 3, 9, 7, 10, 6, 13]]
    ),

    (
        ['A', 'B', 'C'],
        1,
        2,
        [['A_' + str(i) for i in range(5)] +
         [f'B_{str(i)}_{str(j)}' for i in range(5) for j in range(5) if i != j] +
         [f'C_{str(i)}_{str(j)}' for i in range(5) for j in range(5) if i != j]],
        5,
        [[i for i in range(5)] + [i for i in range(6, 46)]]
    ),
]

CONVERSION_TEST_IDS = ['5q_random', '5q_allgates']
# TODO: more test cases


@pytest.mark.parametrize(['gates', 'q1', 'q2', 'circs', 'qubits', 'etalon'],
                         CONVERSION_TEST_CASES, ids=CONVERSION_TEST_IDS)
def test_tensor_ids(gates: list[str], q1: int, q2: int, circs: list[list[str]], qubits: int, etalon: list[list[int]]):
    gen = cg.DataGenerator(qubits, gates, q1, q2)
    output = gen.get_tmpl_dict_from_human_circs(circs)
    for ans, tmpl in zip(etalon, output.values()):
        assert tmpl[0] == ans
