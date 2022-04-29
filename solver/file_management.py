import tensorflow as tf
import pickle
import typing as tp
import os
import warnings

from solver.utils.misc import NconTemplate, INT, FLOAT, COMPLEX

# TODO: write tests on fm


def _parse_info(additional_info: list[tp.Any], prefix_args: int) -> tuple[str, str]:
    strred_info = [str(item) for item in additional_info]
    prefix = '' if prefix_args == 0 else '_'.join(strred_info[:prefix_args]) + '/'
    # suffix = '' if prefix_args == len(strred_info) else '_' + '_'.join(strred_info[prefix_args:]) does not copy
    suffix = '_' + '_'.join(strred_info)  # copies prefix here also
    suffix += ".pickle"
    return prefix, suffix


def _ensure_folder_existence(prefix: str) -> None:
    if prefix == '':
        return
    directory = "data/" + prefix
    if not os.path.exists(directory):
        os.makedirs(directory)


def save_ncon_list_to_file(ncon_list: list[NconTemplate],
                           additional_info: list[tp.Any],
                           prefix_args: int = 1) -> None:
    """
    TODO: Write docstring
    """
    prefix, suffix = _parse_info(additional_info, prefix_args)
    _ensure_folder_existence(prefix)
    with open("data/" + prefix + "nconlist" + suffix, "wb") as a_file:
        pickle.dump(ncon_list, a_file, protocol=4)


def generate_ncon_dict_from_file(circ_name: str,
                                 additional_info: list[tp.Any],
                                 prefix_args: int = 1) -> dict[str, NconTemplate]:
    """
    TODO: Write docstring
    """
    prefix, suffix = _parse_info(additional_info, prefix_args)
    with open("data/" + prefix + "nconlist" + suffix, "rb") as a_file:
        ncon_list = pickle.load(a_file)

    tmpl_dict = {}
    for i, template in enumerate(ncon_list):
        tmpl_dict[circ_name + str(i)] = template
    return tmpl_dict


def save_bitstrings_to_file(samples_vault: dict[str, tf.Tensor],
                            additional_info: list[tp.Any],
                            prefix_args: int = 1) -> None:
    """
    TODO: Write docstring
    """
    prefix, suffix = _parse_info(additional_info, prefix_args)
    _ensure_folder_existence(prefix)
    with open("data/" + prefix + "bitstrings" + suffix, "wb") as a_file:
        dict_to_export = {}
        for name in samples_vault:
            dict_to_export[name] = samples_vault[name].numpy().tolist()
        pickle.dump(dict_to_export, a_file, protocol=4)
    

def import_bitstrings_from_file(additional_info: list[tp.Any], prefix_args: int = 1) -> dict[str, tf.Tensor]:
    """
    TODO: Write docstring
    """
    prefix, suffix = _parse_info(additional_info, prefix_args)
    with open("data/" + prefix + "bitstrings" + suffix, "rb") as a_file:
        big_dict = pickle.load(a_file)
    try:
        for name in big_dict:
            big_dict[name] = tf.convert_to_tensor(big_dict[name], dtype=INT)
    except TypeError:  # compressed samples are stored as float
        for name in big_dict:
            big_dict[name] = tf.convert_to_tensor(big_dict[name], dtype=FLOAT)
    return big_dict

        
def save_templates_to_file(tn_templates: dict[str, NconTemplate],
                           additional_info: list[tp.Any],
                           prefix_args: int = 1) -> None:
    """
    deprecated
    """
    warnings.warn("Working with ncon templates as a dict is now deprecated since it contains hard-coded circuit names. "
                  "It is much safer to save ncon templates as a list, see fm.save_ncon_list_to_file. "
                  "To get templates from file as a dict with custom names, see fm.generate_ncon_dict_from_file.")
    prefix, suffix = _parse_info(additional_info, prefix_args)
    _ensure_folder_existence(prefix)
    with open("data/" + prefix + "templates" + suffix, "wb") as a_file:
        pickle.dump(tn_templates, a_file, protocol=4)
    

def import_templates_from_file(additional_info: list[tp.Any], prefix_args: int = 1) -> dict[str, NconTemplate]:
    """
    deprecated
    """
    warnings.warn("Working with ncon templates as a dict is now deprecated since it contains hard-coded circuit names. "
                  "It is much safer to save ncon templates as a list, see fm.save_ncon_list_to_file. "
                  "To get templates from file as a dict with custom names, see fm.generate_ncon_dict_from_file.")
    prefix, suffix = _parse_info(additional_info, prefix_args)
    with open("data/" + prefix + "templates" + suffix, "rb") as a_file:
        return pickle.load(a_file)

    
def save_tensors_to_file(workspace_set: dict[str, tf.Tensor],
                         additional_info: list[tp.Any],
                         prefix_args: int = 1) -> None:
    """
    TODO: Write docstring
    """
    prefix, suffix = _parse_info(additional_info, prefix_args)
    _ensure_folder_existence(prefix)
    with open("data/" + prefix + "tensors" + suffix, "wb") as a_file:
        dict_to_export = {}
        for name in workspace_set:
            dict_to_export[name] = workspace_set[name].numpy().tolist()
        pickle.dump(dict_to_export, a_file, protocol=4)
    

def import_tensors_from_file(additional_info: list[tp.Any], prefix_args: int = 1) -> dict[str, tf.Tensor]:
    """
    TODO: Write docstring
    """
    prefix, suffix = _parse_info(additional_info, prefix_args)
    with open("data/" + prefix + "tensors" + suffix, "rb") as a_file:
        big_dict = pickle.load(a_file)
    try:
        for name in big_dict:
            big_dict[name] = tf.convert_to_tensor(big_dict[name], dtype=FLOAT)
    except TypeError:
        for name in big_dict:
            big_dict[name] = tf.convert_to_tensor(big_dict[name], dtype=COMPLEX)
    return big_dict
    

def save_experiment_to_file(list_of_dicts: list[dict[str, tf.Tensor]],
                            additional_info: list[tp.Any],
                            prefix_args: int = 1) -> None:
    """
    TODO: Write docstring
    """
    prefix, suffix = _parse_info(additional_info, prefix_args)
    _ensure_folder_existence(prefix)
    with open("data/" + prefix + "experiment" + suffix, "wb") as a_file:
        for dict_to_export in list_of_dicts:
            for name in dict_to_export:
                dict_to_export[name] = dict_to_export[name].numpy().tolist()
        pickle.dump(list_of_dicts, a_file, protocol=4)


def import_experiment_from_file(additional_info: list[tp.Any], prefix_args: int = 1) -> list[dict[str, tf.Tensor]]:
    """
    TODO: Write docstring
    """
    prefix, suffix = _parse_info(additional_info, prefix_args)
    with open("data/" + prefix + "experiment" + suffix, "rb") as a_file:
        list_of_dicts = pickle.load(a_file)
        for dict_in_list in list_of_dicts:
            for name in dict_in_list:
                dict_in_list[name] = tf.convert_to_tensor(dict_in_list[name], dtype=COMPLEX)

    return list_of_dicts


def save_norms_to_file(l1_norms: list[list[tuple[tf.Tensor, tf.Tensor]]],
                       additional_info: list[tp.Any],
                       prefix_args: int = 1) -> None:
    """
    TODO: Write docstring
    """
    prefix, suffix = _parse_info(additional_info, prefix_args)
    _ensure_folder_existence(prefix)
    with open("data/" + prefix + "circ_norms" + suffix, "wb") as a_file:
        pickle.dump(l1_norms, a_file, protocol=4)


def import_norms_from_file(additional_info: list[tp.Any],
                           prefix_args: int = 1) -> None:
    """
    TODO: Write docstring
    """
    prefix, suffix = _parse_info(additional_info, prefix_args)
    with open("data/" + prefix + "circ_norms" + suffix, "rb") as a_file:
        l1_norms = pickle.load(a_file)

    return l1_norms
