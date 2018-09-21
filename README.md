tfdeform
========
This is a small utility library to create random _large_ deformations in tensorflow. It is mainly intended for data-augmentation.

Example
-------
Example of deforming the "face" image. 
The size (distance travelled) of the deformation is approximately 100 pixels with a correlation length of 100 pixels.

    # Generate a deformation field
    offset = tfdeform.create_deformation_momentum(
        shape=[1, *shape], std=100.0, distance=100.0)
        
    # Apply the deformation field to an image
    face_deform = tfdeform.dense_image_warp(face, offset)

Input image                |  Deformed image
:-------------------------:|:-------------------------:
![true](https://user-images.githubusercontent.com/2202312/45845052-4fcb5c80-bd24-11e8-969f-3ed6df9f07fa.png)  | ![deformed](https://user-images.githubusercontent.com/2202312/45845054-51952000-bd24-11e8-9dbf-c3bf4443623e.png)

Implementation details
----------------------
The implementation is done using the geodesic shooting method, which should guarantee that the generated deformation is a diffeomorphism.
Everything runs in tensorflow so it should be reasonably fast. Batch computation is supported.
