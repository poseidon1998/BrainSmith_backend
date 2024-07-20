import cupy as cp 
import numpy as np
from uuid import uuid4
import warnings
import zarr
import matplotlib.pyplot as plt 
from tqdm import tqdm, trange 
from joblib import Parallel, delayed  
from itertools import product
from shapely.geometry import Polygon


class InvalidPatchSize(Exception):
    """Raised when the patch size dimension is invalid"""
    pass


class Store:
    def __init__(self, path):
        self.path = path
        self.store = zarr.open(path)


class Patch:
    def __init__(self, x, y, brain_id, section_id, patch_size=1024, store=None):
        self.min_x = x
        self.min_y = y
        self.patch_size = patch_size
        self.max_x = x + self.patch_size
        self.max_y = y + self.patch_size
        self.is_bg = None
        self.section_id = section_id
        self.brain_id = brain_id
        self.labels = []
        self.id = f"{brain_id}_{section_id}_{x}_{y}"
        self.store = store
        self.store_path = (
            f"/storage/BrainSAM/zarr_n5/optimum_1024/{brain_id}/{section_id}.n5"
        )
        self.corners = self.get_corners()
        self.polygon_coords = [
            [self.min_x, -self.max_y],
            [self.max_x, -self.max_y],
            [self.max_x, -self.min_y],
            [self.min_x, -self.min_y],
            [self.min_x, -self.max_y],
        ]
        self.polygon = Polygon(self.polygon_coords)
        self.z = None
        self.region_polygons = {}
        self.region_areas = {}
        self.has_annotation = True
        self.is_background = False

    def __hash__(self):
        return hash((self.min_x, self.min_y))

    def __eq__(self, other):
        if self.id != other.id:
            if self.brain_id == other.brain_id and self.section_id == other.section_id:
                if self.min_x == other.min_x and self.min_y == other.min_y:
                    return True
        else:
            return True
        return False

    def __str__(self):
        return f"{self.brain_id}:{self.section_id} : ({self.min_x}, {self.min_y})"

    def __repr__(self):
        return self.__str__()

    def set_store(self, store):
        self.store = store

    def load_store(self):
        if self.store is None:
            self.store = zarr.open(self.store_path)

    def get_corners(self) -> list[tuple[int, int]]:
        xs = [self.min_x, self.max_x]
        ys = [self.min_y, self.max_y]
        return list(product(xs, ys))

    def array(self):
        try:
            if self.store is None:
                warnings.warn(
                    "Store not loaded, Please load store before calling array(). This behaviour is deprecated and will be removed in future versions."
                )
                self.load_store()
                if self.store is not None:
                    if self.z is None:
                        return self.store[self.min_y : self.max_y, self.min_x : self.max_x, :]
                    else:
                        return self.store[self.min_y : self.max_y, self.min_x : self.max_x, self.z]

            else:
                if self.z is None:
                    return self.store[self.min_y : self.max_y, self.min_x : self.max_x, :]
                else:
                    return self.store[self.min_y : self.max_y, self.min_x : self.max_x, self.z]
        except IndexError:
            if self.z is not None:
                raise InvalidPatchSize(
                    f"Patch size {self.patch_size} x {self.patch_size} is invalid for the given image size"
                )
            else:
                raise InvalidPatchSize(
                    f"Patch size {self.patch_size}x {self.patch_size} x {self.z} is invalid for the given image size"
                )

    def cupy(self):
        return cp.array(self.array())

    def adjacent(self, other):
        if self.brain_id == other.brain_id and self.section_id == other.section_id:
            curr_corners = self.corners
            other_corners = other.corners
            for corner in curr_corners:
                if corner in other_corners:
                    return True
        return False

    def display(self):
        plt.imshow(self.array())  # type: ignore
        plt.title(self.__str__())
        # plt.axis('off')
        plt.show()

    # Given a patch find the 3x3 grid of patches
    def find_adjacent(self, patches):
        adjacent = []
        for p in patches:
            if p != self:
                if self.adjacent(p):
                    adjacent.append(p)
        return adjacent
