from fastapi import Query

from typing import Annotated, Tuple, Optional

from src.domain.models.base_model import BaseEnum
from pydantic import BaseModel


class DatasetToolParameters(BaseModel):
    source: Annotated[str, Query(description="Directory or archive name for input dataset")]
    dest: Annotated[str, Query(description="Output directory or archive name for output dataset")]
    max_images: Annotated[Optional[int], Query(description="Output only up to `max-images` images")] = None
    force_channels: Annotated[Optional[str], Query(description="Force the number of channels in the image (1: grayscale, 3: RGB, 4: RGBA). Options: ['1', '3', '4']")] = None
    subfolders_as_labels: Annotated[Optional[bool], Query(description="Use the folder names as the labels, to avoid setting up `dataset.json`. Defaults to False.")] = None
    transform: Annotated[Optional[str], Query(description="Input crop/resize mode. Options: ['center-crop', 'center-crop-wide', 'center-crop-tall']")] = None
    resolution: Annotated[Optional[Tuple[int, int]], Query(description="Output resolution (e.g., '512x512')")] = None