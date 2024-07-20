from .polygon_graph import PolygonGraph, InvalidGeojsonError
from .patch import Patch
from .path_config import PathConfig
import zarr as za
import json
import pandas as pd
from tqdm import tqdm, trange
from joblib import Parallel, delayed 
import numpy as np
import shapely
from shapely import GEOSException 
from functools import partial


class ImageNotFoundError(Exception):
    """Image is not stored in zarr format due to multiple reasons."""
    pass


def get_shape(patch):
    shape = patch.array().shape
    if shape != (1024, 1024, 3):
        return shape, False
    return shape, True


def get_nomenclature(nomenclature_path="/storage/BrainSAM/models/nomenclature.json"):
    with open(nomenclature_path, "r") as f:
        nomenclature = json.load(f)

    fields = [
        "text",
        "id",
        "name",
        "definition (description)",
        "acronym",
        "color_hex_triplet",
        "children",
        "type (gray matter/fiber tract/CNS cavity/developmental/other)",
        "parent_structure_id",
    ]
    rename_fields = {}
    for field in fields:
        if " " in field:
            rename_fields[field] = field.split(" ")[0]
        else:
            rename_fields[field] = field
    nomenclature = nomenclature["tree"][0]
    nomenclature_df = pd.DataFrame(nomenclature["children"])
    nomenclature_df = nomenclature_df.drop(
        columns=["children", "text", "parent_structure_id"]
    )
    nomenclature_df = nomenclature_df.rename(columns=rename_fields)
    nomenclature_df = nomenclature_df.set_index("id")
    return nomenclature_df


class PatchDB(object):
    def __init__(self, brain_id, section_id, patch_size, stride) -> None:
        self.brain_id = brain_id
        self.section_id = section_id
        self.path_config = PathConfig(brain_id, section_id)
        self.store = za.open(self.path_config.zarr_store_path())
        try:
            self.h = self.store.shape[0]
            self.w = self.store.shape[1]
        except AttributeError:
            raise ImageNotFoundError(
                "Image has not been annotated yet, so it has not been converted at the moment."
                " If you feel this is incorrect submit a request to get the image added anyway."
            )
        self.geo_json_origin = (self.w / 2, -self.h / 2)  # type: ignore
        self.no_geojson = False
        try:
            self.p_graph = PolygonGraph(self.path_config.gjson_path())
            self.geo_rotation = self.p_graph.json["rotation"]
            self.geoms = self.p_graph.geodf["geometry"]
            self.rotation_map = {270: -90, 90: 90, 180: 180, 0: 0}
            self.geoms = self.geoms.rotate(
                self.rotation_map[self.geo_rotation], origin=self.geo_json_origin
            )
            self.p_graph.geodf["geometry"] = self.geoms
            self.region_polygons = self.p_graph.geodf["geometry"].values.tolist()
        except InvalidGeojsonError:
            self.p_graph = None
            self.region_polygons = []
            self.no_geojson = True
        self.stride = stride
        self.patch_size = patch_size
        self.nomenclature_df = get_nomenclature()
        self.patches = self.make_patches()
        self.patch_polygons = [patch.polygon for patch in self.patches]
        self.tree = shapely.strtree.STRtree(self.region_polygons + self.patch_polygons)

    def make_patches(self):
        patches = []
        for x in range(0, self.w, self.stride):  # type: ignore
            for y in range(0, self.h, self.stride):  # type: ignore
                patch = Patch(
                    x, y, self.brain_id, self.section_id, self.patch_size, self.store
                )
                patches.append(patch)
        return patches

    def load_patches(self):
        _ = Parallel(n_jobs=16, backend="loky")(
            delayed(get_shape)(patch) for patch in tqdm(self.patches)
        )

    def create_db(self):
        label_columns = [str(n) for n in self.nomenclature_df.index.values.tolist()]
        patch_db = pd.DataFrame(
            columns=[
                "patch_index",
                "id",
                "min_x",
                "min_y",
                "brain_id",
                "section_id",
                "check_bg",
            ]
            + label_columns
        )
        patch_db[label_columns] = patch_db[label_columns].astype(np.float64)
        patch_db["id"] = [patch.id for patch in self.patches]
        patch_db["min_x"] = [patch.min_x for patch in self.patches]
        patch_db["min_y"] = [patch.min_y for patch in self.patches]
        patch_db["brain_id"] = [patch.brain_id for patch in self.patches]
        patch_db["section_id"] = [patch.section_id for patch in self.patches]
        patch_db["patch_index"] = np.arange(len(self.patches))
        patch_db["no_geojson"] = int(self.no_geojson)
        patch_db = patch_db.fillna(0)
        return patch_db

    def get_region_id(self, region_poly_tree_idx) -> str | None:
        if self.p_graph is not None:
            if region_poly_tree_idx < len(self.region_polygons):
                return self.p_graph.geodf.iloc[region_poly_tree_idx]["region_id"]
        return None

    def get_region_weightage(self, region_poly_tree_idx, patch_poly_idx):
        # Get the region polygon and the patch polygon
        region_poly = self.region_polygons[region_poly_tree_idx]
        patch_poly = self.patch_polygons[patch_poly_idx]
        # Get the region ID
        region_id = self.get_region_id(region_poly_tree_idx)
        if region_id is not None:
            region_id = int(region_id)
        try:
            # Calculate the intersection region
            patch_region = region_poly.intersection(patch_poly)
            # Discretize the patch region to avoid floating point errors
            discretize = lambda x: np.round(x)
            # Transform the patch region to a discrete grid
            patch_region = shapely.transform(patch_region, discretize)
            # Calculate the intersection area
            intersection_ratio = patch_region.area / (patch_poly.area)
            # Get the Patch object
            patch = self.patches[patch_poly_idx]
            if patch_region.area > 0:
                # Translate the intersection region to the origin of the patch
                patch_region = shapely.affinity.translate(
                    patch_region,
                    xoff=-self.patches[patch_poly_idx].min_x,
                    yoff=self.patches[patch_poly_idx].min_y,
                )
                if region_id not in patch.labels:
                    # Store the region ID as label in the patch object
                    patch.labels.append(region_id)
                    # Store the region polygon GeoJSON in the patch object
                    patch.region_polygons[region_id] = json.loads(
                        shapely.to_geojson(patch_region)
                    )
                    patch.region_areas[region_id] = intersection_ratio
        except GEOSException:
            intersection_ratio = 0
        return intersection_ratio

    @staticmethod
    def get_labels(
        patch_poly_idx,
        tree,
        region_polygons,
        patch_polygons,
        get_region_id,
        get_region_weightage,
    ):
        if len(region_polygons) != 0:
            region_poly_tree_idx = [
                idx
                for idx in tree.query(patch_polygons[patch_poly_idx])
                if idx < len(region_polygons)
            ]
            if len(region_poly_tree_idx) == 0:
                return {}
            region_ids = [
                get_region_id(region_tree_idx)
                for region_tree_idx in region_poly_tree_idx
            ]
            labels = {k: 0 for k in region_ids}
            for region_id, region_tree_idx in zip(region_ids, region_poly_tree_idx):
                labels[region_id] = get_region_weightage(
                    region_tree_idx, patch_poly_idx
                )
            return labels
        else:
            return {}

    def worker(self):
        return partial(
            self.get_labels,
            tree=self.tree,
            region_polygons=self.region_polygons,
            patch_polygons=self.patch_polygons,
            get_region_id=self.get_region_id,
            get_region_weightage=self.get_region_weightage,
        )

    @staticmethod
    def populate_labels(row, get_labels):
        labels = get_labels(row["patch_index"])
        for k, v in labels.items():
            row[str(k)] = v
        return row

    @staticmethod
    def normalize_patch(patch):
        total_region_area_ratio = sum(patch.region_areas.values())
        for region_id in patch.region_areas.keys():
            patch.region_areas[region_id] = (
                patch.region_areas[region_id] / total_region_area_ratio
            )
        return patch

    def remove_patch(self, patch_idx_list):
        for patch_idx in patch_idx_list:
            del self.patches[patch_idx]
        self.patch_db.drop(patch_idx_list, inplace=True)

    def qc_check(self, tol=0.05):
        """
        Quality control check for the patches. Removes all patches that have a 
        label percentage greater than tol. Normalizes the patches that have a
        label percentage greater than 1 and less than 1 + tol.
        """
        # Extract Label Columns
        labels = list(self.patch_db.columns)
        not_labels = [
            "id",
            "min_x",
            "min_y",
            "brain_id",
            "section_id",
            "check_bg",
            "is_bg",
            "has_geojson",
            "no_geojson",
        ]
        labels = [l for l in labels if l not in not_labels]
        # Calculate Label Area Percentage
        pct = self.patch_db[labels].sum(axis=1) # type: ignore
        # Calculate Rejection Tolerance
        rejection_tol = 1 + tol
        # Rejected Patch Index Extraction
        rejected_patches = self.patch_db[pct > rejection_tol].index.tolist()
        # Abnormal Patch Index Extraction
        abnormal_patches = self.patch_db[pct > 1][pct < rejection_tol].index.tolist()
        for patch_idx in abnormal_patches:
            self.patches[patch_idx] = self.normalize_patch(self.patches[patch_idx])
        self.remove_patch(rejected_patches)

    def populate_db(self):
        get_labels = self.worker()
        patch_db = self.create_db()
        if self.p_graph is not None:
            feeder = partial(self.populate_labels, get_labels=get_labels)
            patch_db = patch_db.apply(feeder, axis=1)
        patch_db = patch_db.drop(columns=["patch_index"])
        self.patch_db = patch_db


# if __name__ == "__main__":
#     brain_id=141
#     section_id = 349
#     patch_size = 1024
#     stride=256
#     pdb = PatchDB(brain_id, section_id, patch_size, stride)
#     pdb.populate_db()
#     pdb.qc_check()
