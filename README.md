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

![true](https://user-images.githubusercontent.com/2202312/45844214-0548e080-bd22-11e8-80da-f84bbd0ccc8b.png)

Deformed image:

![deformed](https://user-images.githubusercontent.com/2202312/45844220-0974fe00-bd22-11e8-842c-38afd45da732.png)

Implementation details
----------------------
The implementation is done using the geodesic shooting method, which should guarantee that the generated deformation is a diffeomorphism.
Everything runs in tensorflow so it should be reasonably fast.
