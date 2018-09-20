import tensorflow as tf
from tensorflow.python.ops import array_ops, math_ops
import matplotlib.pyplot as plt
import tfdeform
import convolve
from scipy import misc
import numpy as np


def image_gradients(image, mode='forward'):
    """Compute gradients of image."""
    if image.get_shape().ndims != 4:
        raise ValueError('image_gradients expects a 4D tensor '
                     '[batch_size, h, w, d], not %s.', image.get_shape())
    image_shape = array_ops.shape(image)
    batch_size, height, width, depth = array_ops.unstack(image_shape)
    dy = image[:, 1:, :, :] - image[:, :-1, :, :]
    dx = image[:, :, 1:, :] - image[:, :, :-1, :]

    if mode == 'forward':
        # Return tensors with same size as original image by concatenating
        # zeros. Place the gradient [I(x+1,y) - I(x,y)] on the base pixel (x, y).
        shape = array_ops.stack([batch_size, 1, width, depth])
        dy = array_ops.concat([dy, array_ops.zeros(shape, image.dtype)], 1)
        dy = array_ops.reshape(dy, image_shape)

        shape = array_ops.stack([batch_size, height, 1, depth])
        dx = array_ops.concat([dx, array_ops.zeros(shape, image.dtype)], 2)
        dx = array_ops.reshape(dx, image_shape)
    else:
        # Return tensors with same size as original image by concatenating
        # zeros. Place the gradient [I(x+1,y) - I(x,y)] on the base pixel (x, y).
        shape = array_ops.stack([batch_size, 1, width, depth])
        dy = array_ops.concat([array_ops.zeros(shape, image.dtype), dy], 1)
        dy = array_ops.reshape(dy, image_shape)

        shape = array_ops.stack([batch_size, height, 1, depth])
        dx = array_ops.concat([array_ops.zeros(shape, image.dtype), dx], 2)
        dx = array_ops.reshape(dx, image_shape)

    return dy, dx

def jacobian(vf):
    """Compute the jacobian of a vectorfield pointwise."""
    vf0_dy, vf0_dx = image_gradients(vf[..., 0:1])
    vf1_dy, vf1_dx = image_gradients(vf[..., 1:2])

    r1 = tf.concat([vf0_dy[..., None], vf0_dx[..., None]], axis=-1)
    r2 = tf.concat([vf1_dy[..., None], vf1_dx[..., None]], axis=-1)

    return tf.concat([r1, r2], axis=-2)

def matmul(mat, vec):
    """Compute matrix @ vec pointwise."""
    c11 = mat[..., 0, 0:1] * vec[..., 0:1]
    c12 = mat[..., 0, 1:2] * vec[..., 1:2]
    c21 = mat[..., 1, 0:1] * vec[..., 0:1]
    c22 = mat[..., 1, 1:2] * vec[..., 1:2]

    return tf.concat([c11 + c12, c21 + c22],
                     axis=-1)

def matmul_transposed(mat, vec):
    """Compute matrix.T @ vec pointwise."""
    c11 = mat[..., 0, 0:1] * vec[..., 0:1]
    c12 = mat[..., 1, 0:1] * vec[..., 1:2]
    c21 = mat[..., 0, 1:2] * vec[..., 0:1]
    c22 = mat[..., 1, 1:2] * vec[..., 1:2]

    return tf.concat([c11 + c12, c21 + c22],
                     axis=-1)

def div(vf):
    """Compute divergence of vector field."""
    dy, _ = image_gradients(vf[..., 0:1], mode='backward')
    _, dx = image_gradients(vf[..., 1:2], mode='backward')
    return dx + dy


def create_deformation_momentum(shape, std, distance, stepsize=0.1,
                                return_inverse=False):
    r"""Create a random diffeomorphic deformation.

    Parameters
    ----------
    shape : sequence of 3 ints
        Batch, height and width.
    std : float
        Correlation distance for the linear deformations.
    distance : float
        Expected total effective distance for the deformation.
    stepsize : float
        How large each step should be (as a propotion of ``std``).
    return_inverse : bool
        If true, also return an associated inverse.

    Notes
    -----
    ``distance`` should typically not be set to more than the sidelength of the
    image.

    The computational time is is propotional to

    .. math::
        \left(\frac{distance}{std * stepsize} \right)^2
    """
    grid_x, grid_y = array_ops.meshgrid(math_ops.range(shape[1]),
                                        math_ops.range(shape[2]))
    grid_x = tf.cast(grid_x[None, ..., None], 'float32')
    grid_y = tf.cast(grid_y[None, ..., None], 'float32')

    base_coordinates = tf.concat([grid_y, grid_x], axis=-1)
    coordinates = tf.identity(base_coordinates)

    # Create mask to stop movement at edges
    mask = (tf.cos((grid_x - shape[1] / 2 + 1) * np.pi / (shape[1] + 2)) *
            tf.cos((grid_y - shape[2] / 2 + 1) * np.pi / (shape[2] + 2))) ** (0.25)

    # Total distance is given by std * n_steps * dt, we use this
    # to work out the exact numbers.
    n_steps = tf.cast(tf.ceil(distance / (std * stepsize)), 'int32')
    dt = distance / (tf.cast(n_steps, 'float32') * std)

    # Scale to get std 1 after smoothing
    C = 2 * (std ** 2)

    # Multiply by dt here to keep values small-ish for numerical purposes
    momenta = dt * mask * convolve.gausssmooth(
            C * tf.random_normal(shape=[*shape, 2]), std)

    # Using a while loop, generate the deformation step-by-step.
    def cond(i, from_coordinates, momenta):
        return i < n_steps

    def body(i, from_coordinates, momenta):
        v = mask * convolve.gausssmooth(momenta, std)

        d1 = matmul(jacobian(momenta), v)
        d2 = matmul_transposed(jacobian(v), momenta)
        d3 = div(v) * momenta

        momenta = momenta - dt * (d1 + d2 + d3)
        v = tfdeform.dense_image_warp(v, from_coordinates - base_coordinates)
        from_coordinates = tfdeform.dense_image_warp(from_coordinates, v)

        return i + 1, from_coordinates, momenta

    i = tf.constant(0, dtype=tf.int32)
    i, from_coordinates, momenta = tf.while_loop(
        cond, body, [i, coordinates, momenta])

    from_total_offset = from_coordinates - base_coordinates

    return from_total_offset


if __name__ == '__main__':
    config = tf.ConfigProto(device_count = {'GPU': 1})
    session = tf.InteractiveSession(config=config)

    face = misc.face(gray=True)[None, 128:-128, 256:-256, None].astype('float32')
    face = tf.convert_to_tensor(face)

    from_total_offset = create_deformation_momentum(
        shape=[1, 512, 512], std=50.0, distance=100.0, stepsize=0.01)
    face_deform = tfdeform.dense_image_warp(face, from_total_offset)

    x0, y0 = np.meshgrid(np.arange(512), np.arange(512))

    result_face, result_deform = session.run(
        [face, face_deform])

    plt.figure()
    plt.imshow(result_face[0, ..., 0], cmap='gray')

    plt.figure()
    plt.imshow(result_deform[0, ..., 0], cmap='gray')

    plt.figure()
    plt.imshow(result_face[0, ..., 0] - result_deform[0, ..., 0], cmap='gray')
