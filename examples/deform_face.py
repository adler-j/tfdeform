import tensorflow as tf
from scipy import misc
import matplotlib.pyplot as plt
import tfdeform

config = tf.ConfigProto(device_count = {'GPU': 1})
session = tf.InteractiveSession(config=config)

face = misc.face(gray=True)
shape = face.shape
face = tf.convert_to_tensor(face[None, ..., None].astype('float32'))

offset = tfdeform.create_deformation_momentum(
    shape=[1, *shape], std=50.0, distance=100.0, stepsize=0.05)
face_deform = tfdeform.dense_image_warp(face, offset)

result_face, result_deform = session.run([face, face_deform])

plt.figure()
plt.imshow(result_face[0, ..., 0], cmap='gray')

plt.figure()
plt.imshow(result_deform[0, ..., 0], cmap='gray')

plt.figure()
plt.imshow(result_face[0, ..., 0] - result_deform[0, ..., 0], cmap='gray')
