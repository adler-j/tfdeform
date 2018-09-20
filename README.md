tfdeform
========
This is a small utility library to create random _large_ deformations in tensorflow. It is mainly intended for data-augmentation.

Example

    # Generate a deformation field
    offset = tfdeform.create_deformation_momentum(
        shape=[1, *shape], std=100.0, distance=100.0, stepsize=0.01)
        
    # Apply the deformation field to an image
    face_deform = tfdeform.dense_image_warp(face, offset)

Input image:

![true](https://user-images.githubusercontent.com/2202312/45844547-134b3100-bd23-11e8-924e-fa09ccf6a3fe.png)

Deformed image:

![deformed](https://user-images.githubusercontent.com/2202312/45844555-19d9a880-bd23-11e8-9638-2074cdb96d3a.png)

Implementation details
----------------------
The implementation is done using the geodesic shooting method, which should guarantee that the generated deformation is a diffeomorphism.
Everything runs in tensorflow so it should be reasonably fast.
