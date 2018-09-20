import tensorflow as tf
from tensorflow.python.ops import array_ops, math_ops

def _centered(arr, newshape):
    # Return the center newshape portion of the array.
    currshape = tf.shape(arr)[-2:]
    startind = (currshape - newshape) // 2
    endind = startind + newshape
    return arr[..., startind[0]:endind[0], startind[1]:endind[1]]

def fftconv(in1, in2, mode="full"):
    """Reimplementation of ``scipy.signal.fftconv``."""
    # Reorder channels to come second (needed for fft)
    in1 = tf.transpose(in1, perm=[0, 3, 1, 2])
    in2 = tf.transpose(in2, perm=[0, 3, 1, 2])

    # Extract shapes
    s1 = tf.convert_to_tensor(tf.shape(in1)[-2:])
    s2 = tf.convert_to_tensor(tf.shape(in2)[-2:])
    shape = s1 + s2 - 1

    # Compute convolution in fourier space
    sp1 = tf.spectral.rfft2d(in1, shape)
    sp2 = tf.spectral.rfft2d(in2, shape)
    ret = tf.spectral.irfft2d(sp1 * sp2, shape)

    # Crop according to mode
    if mode == "full":
        cropped = ret
    elif mode == "same":
        cropped = _centered(ret, s1)
        cropped.set_shape(in1.shape)
    elif mode == "valid":
        cropped = _centered(ret, s1 - s2 + 1)
    else:
        raise ValueError("Acceptable mode flags are 'valid',"
                         " 'same', or 'full'.")

    # Reorder channels to last
    result = tf.transpose(cropped, perm=[0, 2, 3, 1])
    return result

def gausssmooth(img, std):
     # Create gaussian
    size = tf.cast(std * 5, 'int32')
    size = (size // 2) * 2 + 1
    size_f = tf.cast(size, 'float32')

    grid_x, grid_y = array_ops.meshgrid(math_ops.range(size),
                                        math_ops.range(size))
    grid_x = tf.cast(grid_x[None, ..., None], 'float32')
    grid_y = tf.cast(grid_y[None, ..., None], 'float32')

    gaussian = tf.exp(-((grid_x - size_f / 2 - 0.5) ** 2 + (grid_y - size_f / 2 + 0.5) ** 2) / std ** 2)
    gaussian = gaussian / tf.reduce_sum(gaussian)

    return fftconv(img, gaussian, 'same')

if __name__ == '__main__':
    from scipy import misc
    import matplotlib.pyplot as plt
    config = tf.ConfigProto(device_count = {'GPU': 0})
    session = tf.InteractiveSession(config=config)

    img = misc.face(gray=False)[None, ...].astype('float32')

    # Apply convolution
    result = gausssmooth(img, 20)
    result_r = session.run(result)

    # Show results
    plt.figure('face')
    plt.imshow(img[0, ...] / 256.0)

    plt.figure('convolved')
    plt.imshow(result_r[0, ...] / 256.0)