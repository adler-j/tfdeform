import tensorflow as tf
from scipy import misc
import matplotlib.pyplot as plt
import tfdeform

session = tf.InteractiveSession()

face = misc.face() / 256
shape = face.shape
face = tf.convert_to_tensor(face[None, ...].astype('float32'))

offset = tfdeform.random_deformation_momentum(
    shape=[1, *shape[:2]], std=100.0, distance=50.0, stepsize=0.01)
face_deform = tfdeform.dense_image_warp(face, offset)

result_face, result_deform = session.run([face, face_deform])

plt.figure()
plt.imshow(result_face[0, ...])

plt.figure()
plt.imshow(result_deform[0, ...])
