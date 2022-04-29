import tensorflow as tf
from solver.utils.misc import FLOAT


def chi2_stat(observed: tf.Tensor, expected: tf.Tensor, eps: float = 1e-10) -> tf.Tensor:
    return tf.reduce_sum((observed - expected) ** 2 / (expected + tf.convert_to_tensor(eps, dtype=FLOAT)))


def scalar_product(observed: tf.Tensor, expected: tf.Tensor, eps: float = 1e-10) -> tf.Tensor:
    eps_tensor = tf.convert_to_tensor(eps, dtype=FLOAT)
    return tf.tensordot(tf.math.sqrt(observed + eps_tensor), tf.math.sqrt(expected + eps_tensor), axes=1)


def ks_stat(observed: tf.Tensor, expected: tf.Tensor) -> tf.Tensor:
    return tf.reduce_max(tf.abs(tf.cumsum(observed) - tf.cumsum(expected)))


def l1_stat(observed: tf.Tensor, expected: tf.Tensor) -> tf.Tensor:
    return tf.reduce_sum(tf.abs(observed - expected))


def cm_stat(observed: tf.Tensor, expected: tf.Tensor) -> tf.Tensor:
    raise NotImplementedError
