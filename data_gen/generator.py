import bpy
from math import radians

class Generator(bpy.types.Operator):
    def __init__(self) -> None:
        super().__init__()
