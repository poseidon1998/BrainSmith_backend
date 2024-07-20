from ...lib.patch import Patch
import zarr as za
import dask.array as da
from dask.array import from_zarr as da_zarr  # type: ignore

# import dask.delayed as da_delayed
from tqdm import tqdm
import torch
import safetensors as st
from safetensors.torch import save as st_save
from .dependencies import get_dask_client
from .models import PostPatchRecordsSchema, PostPatchRecordSchema


def get_patch_array(patch: PostPatchRecordSchema):
    return Patch(
        brain_id=patch.brain_id,
        section_id=patch.section_id,
        x=patch.x,
        y=patch.y,
        store=da_zarr(patch.store_path),
    ).array()


# @dask.delayed
def convert_to_tensor(patch):
    patch = get_patch_array(patch).compute()  # type: ignore
    torch_tensor = torch.tensor(patch)
    return torch_tensor


def process_patches(patch_records):
    client = get_dask_client()
    all_patches = [
        client.submit(convert_to_tensor, patch=patch_records[idx])
        for idx in patch_records
    ]
    all_patches = client.compute(all_patches, sync=True)
    patch_tensors = {
        f"{idx}": patch for idx, patch in enumerate(all_patches)  # type: ignore
    }
    data = st_save(patch_tensors)
    return data
