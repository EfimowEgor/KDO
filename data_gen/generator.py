import math
import os
import random

import bpy
import numpy as np
from mathutils import Euler
from mathutils import Vector

class Generator:
    def __init__(self: "Generator") -> None:
        scene = bpy.context.scene
        self.camera = bpy.data.cameras.new(name="Camera")
        self.camera_obj = bpy.data.objects.new(
            name="CameraObject", object_data=self.camera
        )
        bpy.context.scene.collection.objects.link(self.camera_obj)
        self.camera_obj.data.lens = 20.54
        self.camera_obj.location = Vector((0, 0, 24.435))
        bpy.context.scene.camera = self.camera_obj
        scene.render.pixel_aspect_x = 1.0
        scene.render.pixel_aspect_y = 1.6

    def rotate(self: "Generator") -> Euler:
        """Rotate selected object."""
        rotate_vector = Euler(
            (
                math.radians(random.randint(0, 360)),
                math.radians(random.randint(0, 360)),
                math.radians(random.randint(0, 360))
            )
        )
        return rotate_vector

    def __change_color(self: "Generator") -> None:
        """Change color of the selected object."""
        active_object = bpy.context.active_object
        mat = bpy.data.materials.new(name='RandomColor')
        active_object.data.materials.append(mat)
        bpy.context.object.active_material.diffuse_color = (
            random.random(),
            random.random(),
            random.random(),
            1
        )

    def resize(self: "Generator") -> Vector:
        """Change XYZ scale."""
        scale_vector = Vector(
            (
                random.uniform(0.4, 2),
                random.uniform(0.4, 2),
                random.uniform(0.4, 2)
            )
        )
        return scale_vector

    def create_geons(
            self: "Generator", min_y: int, max_y: int, min_x: int, max_x: int) -> None:
        """Generate meshes and apply transformations.

        Old objects are deleted before each generation.
        The position on the Oz is fixed and takes values in the range from 1 to 9.

        Args:
        ----
            min_y: Minimum position by Oy for generating a primitive.
            min_x: Minimum position by Ox for generating a primitive.
            max_y: Maximum position by Oy for generating a primitive.
            max_x: Maximum position by Ox for generating a primitive.

        """
        for obj in bpy.context.scene.objects:
            if obj.type == 'MESH':
                obj.select_set(True)
        # Deletes all selected objects in the scene.
        bpy.ops.object.delete()
        n = random.randint(1, 15)
        for i in range(n):
            proba = random.randint(1, 3)
            if proba == 1:
                bpy.ops.mesh.primitive_cube_add(
                    location=Vector(
                        (
                            np.random.uniform(min_x, max_x),
                            np.random.uniform(min_y, max_y),
                            np.random.uniform(1, 9)
                        )
                    ),
                    scale=self.resize(),
                    rotation=self.rotate()
                )
            elif proba == 2:
                bpy.ops.mesh.primitive_cone_add(
                    location=Vector(
                        (
                            np.random.uniform(min_x, max_x),
                            np.random.uniform(min_y, max_y),
                            np.random.uniform(1, 9)
                        )
                    ),
                    scale=self.resize(),
                    rotation=self.rotate()
                )
            else:
                bpy.ops.mesh.primitive_uv_sphere_add(
                    location=Vector(
                        (
                            np.random.uniform(min_x, max_x),
                            np.random.uniform(min_y, max_y),
                            np.random.uniform(1, 9)
                        )
                    ),
                    scale=self.resize(),
                    rotation=self.rotate()
                )
                bpy.context.object.data.polygons.foreach_set(
                    'use_smooth',  [True] * len(bpy.context.object.data.polygons)
                )
                bpy.context.object.data.update()
            self.__change_color()

    def render(self: "Generator", epochs: int) -> None:
        """Render a scene and generate a png file.

        Args:
        ----
            epochs: Number of generated images (number of renders)
        """
        # Path to save
        base_dir = 'C:/Users/egore/Desktop/KR_CV/data_gen/data/images'
        for i in range(epochs):
            self.create_geons(min_y=-10, min_x=-10, max_y=10, max_x=10)
            file_name = f"image{i}.png"
            path = os.path.join(base_dir, file_name)
            bpy.context.scene.render.filepath = path
            self.get_2d_bounding_box()
            bpy.ops.render.render(write_still=True)

if __name__ == "__main__":
    generator = Generator()
    generator.render(10)
