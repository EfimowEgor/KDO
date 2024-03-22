import argparse
import json
import math
import os
import random

import bpy
import cv2
import numpy as np
import yaml
from mathutils import Euler
from mathutils import Vector

class Generator:
    def __init__(self: "Generator") -> None:
        with open("./data_gen/generator_config.yaml", "r") as stream:
            try:
                self.parameters = yaml.safe_load(stream)
            except yaml.YAMLError as e:
                print(f"Parameters not loaded: {e}")
        self.labels = {"labels": {}}
        self.scene = bpy.context.scene
        self.camera = bpy.data.cameras.new(name="Camera")
        self.camera_obj = bpy.data.objects.new(
            name="CameraObject", object_data=self.camera
        )
        bpy.context.scene.collection.objects.link(self.camera_obj)
        self.camera_obj.data.lens = 35
        self.camera_obj.location = Vector((0, 0, 30))
        # self.camera.type = "ORTHO"
        bpy.context.scene.camera = self.camera_obj
        self.scene.render.pixel_aspect_x = 1.0
        self.scene.render.pixel_aspect_y = 1.0

    def rotate(self: "Generator", proba:int, cls: int) -> Euler:
        """Rotate selected object."""
        # В зависимости от передаваемого подкласса

        rotate_vector = Euler(
            (
                math.radians(random.randint(0,
                                            self.parameters["parameters"][proba][cls]["max_rot_x"])),
                math.radians(random.randint(0,
                                            self.parameters["parameters"][proba][cls]["max_rot_y"])),
                math.radians(random.randint(0,
                                            self.parameters["parameters"][proba][cls]["max_rot_z"]))
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

    def resize(self: "Generator", it: int, num_epochs: int) -> Vector:
        """Change XYZ scale."""
        proba = it // (num_epochs // self.parameters["parameters"]["num_classes"]) + 1
        cls = (it * self.parameters["parameters"][proba]["num_subcls"]) // \
              (num_epochs // self.parameters["parameters"]["num_classes"]) % 5 + 1
        scale_vector = Vector(
            (
                random.uniform(self.parameters["parameters"][proba][cls]["min_scale_x"],
                               self.parameters["parameters"][proba][cls]["max_scale_x"]),
                random.uniform(self.parameters["parameters"][proba][cls]["min_scale_y"],
                               self.parameters["parameters"][proba][cls]["max_scale_y"]),
                random.uniform(self.parameters["parameters"][proba][cls]["min_scale_z"],
                               self.parameters["parameters"][proba][cls]["max_scale_z"])
            )
        )
        return scale_vector

    def create_geons(
            self: "Generator", it: int, num_epochs: int) -> None:
        """Generate meshes and apply transformations.

        Old objects are deleted before each generation.
        The position on the Oz is fixed and takes values in the range from 1 to 9.

        Args:
        ----
            it: Current epoch.
            num_epochs: Total number of epochs (images).
        """
        for obj in bpy.context.scene.objects:
            if obj.type == 'MESH':
                obj.select_set(True)
        # Deletes all selected objects in the scene.
        bpy.ops.object.delete()
        # Используем число объектов из файла конфига
        n = self.parameters["parameters"]["num_objects"]
        proba = it // (num_epochs // self.parameters["parameters"]["num_classes"]) + 1
        print(proba)
        cls = (it * self.parameters["parameters"][proba]["num_subcls"]) // \
              (num_epochs // self.parameters["parameters"]["num_classes"]) % 5 + 1
        for i in range(n):
            # proba = random.randint(1, 1)
            loc_vector = Vector(
                        (
                            np.random.uniform(self.parameters["parameters"][proba][cls]["min_loc_xy"],
                                              self.parameters["parameters"][proba][cls]["max_loc_xy"]),
                            np.random.uniform(self.parameters["parameters"][proba][cls]["min_loc_xy"],
                                              self.parameters["parameters"][proba][cls]["max_loc_xy"]),
                            np.random.uniform(self.parameters["parameters"][proba]["depth_min_z"],
                                              self.parameters["parameters"][proba]["depth_max_z"])
                        )
                    )
            if proba == 1:
                bpy.ops.mesh.primitive_cube_add(
                    location=loc_vector,
                    scale=self.resize(it, num_epochs),
                    rotation=self.rotate(proba, cls)
                )
            elif proba == 2:
                bpy.ops.mesh.primitive_cone_add(
                    location=loc_vector,
                    scale=self.resize(it, num_epochs),
                    rotation=self.rotate(proba, cls)
                )
            elif proba == 3:
                bpy.ops.mesh.primitive_uv_sphere_add(
                    location=loc_vector,
                    scale=self.resize(it, num_epochs),
                    rotation=self.rotate(proba, cls)
                )
                bpy.context.object.data.polygons.foreach_set(
                    'use_smooth',  [True] * len(bpy.context.object.data.polygons)
                )
                bpy.context.object.data.update()
            elif proba == 4:
                bpy.ops.mesh.primitive_cylinder_add(
                    location=loc_vector,
                    scale=self.resize(it, num_epochs),
                    rotation=self.rotate(proba, cls)
                )
                bpy.context.object.data.polygons.foreach_set(
                    'use_smooth',  [True] * len(bpy.context.object.data.polygons)
                )
                bpy.context.object.data.update()
            self.__change_color()

    def clamp(self: "Generator", x: int, minimum: int, maximum: int) -> int:
        """Select the maximum from the coordinates."""
        return max(minimum, min(x, maximum))

    def get_2d_bounding_box(self: "Generator", me_ob: bpy.types.Mesh, img_size: int) -> tuple:
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
            ((round(min_x * dim_x) / 1920) * img_size),
            ((round(dim_y - max_y * dim_y) / 1080) * img_size),
            ((round((max_x - min_x) * dim_x) + round(min_x * dim_x)) / 1920) * img_size,
            ((round((max_y - min_y) * dim_y) + round(dim_y - max_y * dim_y)) / 1080) * img_size
        )

    def write_bounds_2d(self: "Generator",
                        meshes: list[bpy.types.Mesh],
                        frame_end: int, file_name: str) -> dict:
        """Record the coordinates of all image objects in the dictionary."""
        single_label = {file_name: []}
        for obj in meshes:
            print(obj.name)
            if obj.name.startswith("Cube"):
                cls = 1
            elif obj.name.startswith("Cone"):
                cls = 2
            elif obj.name.startswith("Sphere"):
                cls = 3
            elif obj.name.startswith("Cylinder"):
                cls = 4
            # else:
            #     cls = 5
            bpy.context.scene.frame_set(frame_end)
            coords = self.get_2d_bounding_box(obj, 640)
            row = {
                "label": cls,
                "xmin": coords[0],
                "ymin": coords[1],
                "xmax": coords[2],
                "ymax": coords[3]
            }
            if row["xmin"] == 0 and row["ymin"] == 0 and \
               row["xmax"] == 640 and row["ymax"] == 640:
                continue
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
        # base_dir = 'C:/Users/egore/Desktop/KR_CV/models/data_finetune/val/cube'
        for i in range(epochs):
            self.create_geons(i, epochs)
            bpy.ops.mesh.primitive_cube_add(
                    location=Vector(
                        (
                            np.random.uniform(0, 0),
                            np.random.uniform(0, 0),
                            np.random.uniform(-10, -10)
                        )
                    ),
                    scale=Vector(
                            (
                                # Уменьшить число объектов, но увеличить их
                                random.uniform(100, 100),
                                random.uniform(100, 100),
                                random.uniform(1, 1)
                            )
                    ),
                )
            self.__change_color()
            # Удаление существующего света и создание его в новой позиции

            for obj in bpy.context.scene.objects:
                if obj.type == 'MESH':
                    obj.select_set(False)

            for obj in self.scene.objects:
                if obj.type == "LIGHT":
                    obj.select_set(True)
            bpy.ops.object.delete()

            light_data = bpy.data.lights.new(name='LightZ', type="POINT")
            light_data.energy = 6000
            light = bpy.data.objects.new(name="Light2", object_data=light_data)
            bpy.context.collection.objects.link(light)
            light.location = Vector((random.randint(-12, 12),
                                          random.randint(-12, 12),
                                          random.randint(35, 35)))
            bpy.context.view_layer.objects.active = light

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
            # resize image after render
            src = cv2.imread(path)
            src = cv2.resize(src, (640, 640))
            cv2.imwrite(path, src)
        with open("C:/Users/egore/Desktop/KR_CV/data/labels/labels.json", 'w') as file:
            json.dump(self.labels, file, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--epochs')

    args = parser.parse_args()
    epochs = int(args.epochs)

    generator = Generator()
    generator.render(epochs)
