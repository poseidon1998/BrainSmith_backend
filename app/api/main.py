from contextlib import asynccontextmanager
from fastapi import Depends, FastAPI, Header, HTTPException, Response
from dask.distributed import LocalCluster
from .dependencies import get_session
from typing_extensions import Annotated
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import StreamingResponse
from ...lib.query.fetch import QFetch
from .models import PostPatchRecordsSchema
from fastapi.middleware.cors import CORSMiddleware
from .functions import process_patches


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the ML model
    import sys
    sys.path.append('/storage/')
    cluster = LocalCluster(n_workers=16,processes=True,threads_per_worker=1, scheduler_port=8786)
    yield
    # Clean up the ML models and release the resources
    cluster.close()

app = FastAPI(lifespan=lifespan)
app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get('/status')
async def status():
    return {"message": "OK"}


@app.post("/get_patches/")
async def get_patches(patch_records: PostPatchRecordsSchema):
    data = process_patches(patch_records.patches)
    return Response(data, media_type="application/octet-stream")
    


