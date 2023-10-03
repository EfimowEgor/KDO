import json
import math
import os
import random

import bpy
import numpy as np
from mathutils import Euler
from mathutils import Vector

class Generator:
    def __init__(self: "Generator") -> None:
        self.labels = {"labels": {}}
        self.scene = bpy.context.scene
        self.camera = bpy.data.cameras.new(name="Camera")
        self.camera_obj = bpy.data.objects.new(
            name="CameraObject", object_data=self.camera
        )
        bpy.context.scene.collection.objects.link(self.camera_obj)
        self.camera_obj.data.lens = 20.54
        self.camera_obj.location = Vector((0, 0, 24.435))
        bpy.context.scene.camera = self.camera_obj
        self.scene.render.pixel_aspect_x = 1.0
        self.scene.render.pixel_aspect_y = 1.6

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

    def set_background(self: "Generator") -> None:
        """Понятия не имею как это реализовать."""
        ...

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

    def clamp(self: "Generator", x: int, minimum: int, maximum: int) -> int:
        """Select the maximum from the coordinates."""
        return max(minimum, min(x, maximum))

    def get_2d_bounding_box(self: "Generator", me_ob: bpy.types.Mesh) -> tuple:
        """Get the coordinates of the bounding box in 2d."""
        mat = self.camera_obj.matrix_world.normalized().inverted()
        depsgraph = bpy.context.evaluated_depsgraph_get()
        mesh_eval = me_ob.evaluated_get(depsgraph)
        me = mesh_eval.to_mesh()
        me.transform(me_ob.matrix_world)
        me.transform(mat)

        camera = self.camera_obj.data
        frame = [-v for v in camera.view_frame(scene=self.scene)[:3]]
        camera_persp = camera.type != 'ORTHO'

        lx = []
        ly = []

        for v in me.vertices:
            co_local = v.co
            z = -co_local.z

            if camera_persp:
                if z == 0.0:
                    lx.append(0.5)
                    ly.append(0.5)
                else:
                    frame = [(v / (v.z / z)) for v in frame]

            min_x, max_x = frame[1].x, frame[2].x
            min_y, max_y = frame[0].y, frame[1].y

            x = (co_local.x - min_x) / (max_x - min_x)
            y = (co_local.y - min_y) / (max_y - min_y)

            lx.append(x)
            ly.append(y)

        min_x = self.clamp(min(lx), 0.0, 1.0)
        max_x = self.clamp(max(lx), 0.0, 1.0)
        min_y = self.clamp(min(ly), 0.0, 1.0)
        max_y = self.clamp(max(ly), 0.0, 1.0)

        mesh_eval.to_mesh_clear()

        r = bpy.context.scene.render
        fac = r.resolution_percentage * 0.01
        dim_x = r.resolution_x * fac
        dim_y = r.resolution_y * fac

        if round((max_x - min_x) * dim_x) == 0 or round((max_y - min_y) * dim_y) == 0:
            return (0, 0, 0, 0)

        return (
            round(min_x * dim_x),
            round(dim_y - max_y * dim_y),
            round((max_x - min_x) * dim_x) + round(min_x * dim_x),
            round((max_y - min_y) * dim_y) + round(dim_y - max_y * dim_y)
        )

    def write_bounds_2d(self: "Generator",
                        meshes: list[bpy.types.Mesh],
                        frame_end: int, file_name: str) -> dict:
        """Record the coordinates of all image objects in the dictionary."""
        single_label = {file_name: []}
        for obj in meshes:
            print(obj.name)
            if obj.name.startswith("Cube"):
                cls = 0
            elif obj.name.startswith("Cone"):
                cls = 1
            else:
                cls = 2
            bpy.context.scene.frame_set(frame_end)
            coords = self.get_2d_bounding_box(obj)
            row = {
                "label": cls,
                "xmin": coords[0],
                "ymin": coords[1],
                "xmax": coords[2],
                "ymax": coords[3]
            }
            single_label[file_name].append(row)
        return single_label

    def render(self: "Generator", epochs: int) -> None:
        """Render a scene and generate a png file.

        Args:
        ----
            epochs: Number of generated images (number of renders)
        """
        # Path to save
        base_dir = 'C:/Users/egore/Desktop/KR_CV/data/train_images'
        for i in range(epochs):
            self.create_geons(min_y=-10, min_x=-10, max_y=10, max_x=10)
            file_name = f"image{i}.png"
            path = os.path.join(base_dir, file_name)
            bpy.context.scene.render.filepath = path
            # Получение координат bounding box
            meshes = []
            for obj in bpy.context.scene.objects:
                if obj.type == 'MESH':
                    obj.select_set(True)
                    meshes.append(obj)

            frame_current = bpy.context.scene.frame_current
            frame_end = bpy.context.scene.frame_end
            self.labels["labels"].update(
                self.write_bounds_2d(
                    meshes, frame_end, file_name
                )
            )
            bpy.context.scene.frame_set(frame_current)
            bpy.ops.render.render(write_still=True)
        with open("C:/Users/egore/Desktop/KR_CV/data/labels/labels.json", 'w') as file:
            json.dump(self.labels, file, indent=4)

if __name__ == "__main__":
    generator = Generator()
    generator.render(1000)
