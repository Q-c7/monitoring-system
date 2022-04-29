import typing as tp
import tensorflow as tf

# this file can't import any files!

NconTemplate = list[tp.Union[list[str], list[list[int]], list[int]]]
COMPLEX = tf.complex128  # tf.complex64 is an alternative, if memory is an issue
FLOAT = tf.float64  # tf.float32 must be used if tf.complex64 is used
INT = tf.int32
TENSOR = tp.Union[tf.Tensor, tf.Variable]

ID_GATE = '_E'

# TODO: Document the tensor network template somewhere


def unwrap_dict(target: dict[str, TENSOR]) -> list[TENSOR]:
    """
    Unwraps a dict of sublists in the form of list[Tensor] or Tensors(bs, ...) into a long list of Tensors
    """
    return [item for sublist in target.values() for item in tf.unstack(sublist, axis=0)]
