import tensorflow as tf  # tf 2.x

from solver.utils.misc import INT


@tf.function
def convert_2qmatrix_to_channel(four_legged_unitary: tf.Tensor) -> tf.Tensor:
    """
    Converts 2-qubit unitary gate U into a quantum channel U x U*.

    Unitary must be first converted to four-legged Tensor(2,2,2,2) representing a channel in ncon form.
    """
    phi = tf.tensordot(four_legged_unitary, tf.math.conj(four_legged_unitary), axes=0)
    phi = tf.transpose(phi, perm=(0, 4, 1, 5, 2, 6, 3, 7))
    phi = tf.reshape(phi, (4, 4, 4, 4))
    return phi


@tf.function
def convert_1qmatrix_to_channel(unitary: tf.Tensor) -> tf.Tensor:
    """
    Converts 1-qubit unitary Tensor(2,2) into a ncon quantum channel Tensor(4,4).
    """
    phi = tf.tensordot(unitary, tf.math.conj(unitary), axes=0)
    phi = tf.transpose(phi, perm=(0, 2, 1, 3))
    phi = tf.reshape(phi, (4, 4))
    return phi


@tf.function
def convert_params_to_channel_4legs(params: tf.Tensor) -> tf.Tensor:
    """
    Converts a batch of two-qubit parameter matrices A into ncon quantum channels (transposed A * A^dagger).

    Args:
        params: Tensor(batch_size, dim^2, dim^2).

    Returns:
        A ncon channel - Tensor(batch_size, dim, dim, dim, dim). Meant to represent a two-qubit channel.
    """
    dim_squared = params.get_shape()[-2]
    bs_shape = params.get_shape()[:-2]
    dim = tf.cast(tf.math.sqrt(tf.cast(dim_squared, dtype=params.dtype)), dtype=INT)
    dim_rt = tf.cast(tf.math.sqrt(tf.cast(dim, dtype=params.dtype)), dtype=INT)

    chois = params @ tf.linalg.adjoint(params)
    # these chois from Ilya's manifold are not not quite the same as Chois we use for fidelity calculation
    # for example, bra & ket are inverted

    # source: QGOpt tutorial
    # out* in* out in -> out out* in in*
    # first transpose: (0, 2, 4, 1, 3)

    # now we have channel in form out12 out12* in12 in12* = out2 out1 out2* out1* in2 in1 in2* in1*
    # need to convert it to form out2 out2* out1 out1* in2 in2* in1 in1*
    # second transpose: (0, 1, 3, 2, 4, 5, 7, 6, 8)

    # the combination of these two transposes was simplified to one transpose below
    phis = tf.reshape(chois, (-1, dim_rt, dim_rt, dim_rt, dim_rt, dim_rt, dim_rt, dim_rt, dim_rt))
    phis = tf.transpose(phis, (0, 3, 7, 4, 8, 1, 5, 2, 6))

    phis = tf.reshape(phis, (*bs_shape, dim, dim, dim, dim))
    return phis


@tf.function
def convert_params_to_channel_2legs(params: tf.Tensor) -> tf.Tensor:
    """
    Converts a batch of single-qubit parameter matrices A into ncon quantum channels (transposed A * A^dagger).

    Args:
        params: Tensor(batch_size, dim^2, dim^2).

    Returns:
        A ncon channel - Tensor(batch_size, dim^2, dim^2). Meant to represent a one-qubit channel.
    """
    dim_squared = params.get_shape()[-2]
    dim = tf.cast(tf.math.sqrt(tf.cast(dim_squared, dtype=params.dtype)), dtype=INT)
    bs_shape = params.get_shape()[:-2]

    chois = params @ tf.linalg.adjoint(params)
    # these chois from Ilya's manifold are not not quite the same as Chois we use for fidelity calculation
    # for example, bra & ket are inverted

    # source: QGOpt tutorial
    # out1* in1* out1 in1 -> out1 out1* in1 in1*
    chois = tf.reshape(chois, (-1, dim, dim, dim, dim))
    phis = tf.transpose(chois, (0, 2, 4, 1, 3))

    phis = tf.reshape(phis, (*bs_shape, dim_squared, dim_squared))
    return phis


@tf.function
def convert_params_to_channel(params: tf.Tensor) -> tf.Tensor:
    """
    Converts a batch of parameter matrices A into quantum channel representations A * A^dagger.

    Is effectively a wrapper with check for shape to determine whether matrices are two-qubit or single-qubit.

    Args:
        params: Tensor(batch_size, dim^2, dim^2).

    Returns:
        A ncon channel - Tensor(batch_size, dim^2, dim^2) (single-qubit case)
        or Tensor(batch_size, dim, dim, dim, dim) (two-qubit case)
    """
    if params.shape[-2:] == [16, 16]:
        return convert_params_to_channel_4legs(params)
    elif params.shape[-2:] == [4, 4]:
        return convert_params_to_channel_2legs(params)
    else:
        raise NotImplementedError('Right now only conversion of shapes (4,4) and (16,16) is properly tested')


@tf.function
def convert_channel_to_params_2legs(phis: tf.Tensor) -> tf.Tensor:
    """
    Converts a batch of single-qubit ncon quantum channels (transposed A * A^dagger) into parameter matrices A.

    Args:
        phis: ncon quantum channels - Tensor(batch_size, dim^2, dim^2)

    Returns:
        Parameter matrices - Tensor(batch_size, dim^2, dim^2). Decomposition is NOT unique!
    """
    dim = phis.get_shape()[-2]
    dim = tf.cast(tf.math.sqrt(tf.cast(dim, dtype=phis.dtype)), dtype=INT)
    bs_shape = phis.get_shape()[:-2]

    chois = tf.reshape(phis, (-1, dim, dim, dim, dim))
    chois = tf.transpose(chois, (0, 3, 1, 4, 2))

    chois = tf.reshape(chois, (-1, dim ** 2, dim ** 2))
    lmbd, u, _ = tf.linalg.svd(chois)
    lmbd = tf.cast(lmbd, dtype=u.dtype)
    params = u * tf.math.sqrt(lmbd[:, tf.newaxis])

    params = tf.reshape(params, (*bs_shape, dim ** 2, dim ** 2))
    return params


@tf.function
def convert_channel_to_params_4legs(phis: tf.Tensor) -> tf.Tensor:
    """
    Converts a batch of two-qubit ncon quantum channels (reshaped & transposed A * A^dagger) into parameter matrices A.

    Args:
        phis: ncon quantum channels - Tensor(batch_size, dim, dim, dim, dim)

    Returns:
        Parameter matrices - Tensor(batch_size, dim^2, dim^2). Decomposition is NOT unique!
    """
    dim = phis.get_shape()[-4]
    dim_rt = tf.cast(tf.math.sqrt(tf.cast(dim, dtype=phis.dtype)), dtype=INT)
    dim_squared = dim ** 2
    bs_shape = phis.get_shape()[:-4]

    # INVERSED LEG SWAPPING (see "convert_params_to_channel_4legs")
    chois = tf.reshape(phis, (-1, dim_rt, dim_rt, dim_rt, dim_rt, dim_rt, dim_rt, dim_rt, dim_rt))
    chois = tf.transpose(chois, (0, 5, 7, 1, 3, 6, 8, 2, 4))

    chois = tf.reshape(chois, (-1, dim_squared, dim_squared))
    lmbd, u, _ = tf.linalg.svd(chois)
    lmbd = tf.cast(lmbd, dtype=u.dtype)
    params = u * tf.math.sqrt(lmbd[:, tf.newaxis])

    params = tf.reshape(params, (*bs_shape, dim_squared, dim_squared))
    return params


@tf.function
def convert_channel_to_params(phis: tf.Tensor) -> tf.Tensor:
    """
    Converts a batch of ncon quantum channels (reshaped & transposed A * A^dagger) into parameter matrices A.

    Is effectively a wrapper with check for shape to determine whether matrices are two-qubit or single-qubit.

    Args:
        phis: ncon quantum channels - Tensor(batch_size, dim^2, dim^2) (single-qubit case)
        OR Tensor(batch_size, dim, dim, dim, dim)  (two-qubit case)

    Returns:
        Parameter matrices - Tensor(batch_size, dim^2, dim^2). Decomposition is NOT unique!
    """
    if len(phis.shape) >= 4 and phis.shape[-4:] == [4, 4, 4, 4]:
        return convert_channel_to_params_4legs(phis)
    elif phis.shape[-2:] == [4, 4]:
        return convert_channel_to_params_2legs(phis)
    else:
        raise NotImplementedError('Right now only conversion of shapes (4,4) and (4,4,4,4) is properly tested')
