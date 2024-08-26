from fastapi import Query

from typing import Annotated, Tuple, Optional

from src.domain.models.base_model import BaseEnum
from pydantic import BaseModel


class TrainParameters(BaseModel):
    cfg: Annotated[str, Query(description='Base configuration, Options: [stylegan3-t, stylegan3-r, stylegan2, stylegan2-ext]')] = "stylegan-t"
    data: Annotated[str, Query(description='Training data')]
    outdir: Annotated[str, Query(description='Where to save the results. Give the path as input')]
    gpus: Annotated[int, Query(description='Number of GPUs to use')] = 1
    batch: Annotated[int, Query(description='Total batch size')] = 32

    cond: Annotated[bool, Query(description='Train conditional model. Defaults to False')] = False
    mirror: Annotated[bool, Query(description='Enable dataset x-flips. Defaults to False')] = False
    mirror_y: Annotated[bool, Query(description='Enable dataset y-flips. Defaults to False')] = False

    aug: Annotated[str, Query(description='Augmentation mode. Options: [noaug, ada, fixed]. Defaults to ada')] = 'ada'
    augpipe: Annotated[str, Query(description='Augmentation pipeline. Options: [blit, geom, color, filter, noise, cutout, bg, bgc, bgcf, bgcfn, bgcfnc]. Defaults to bgc')] = 'bgc'

    resume: Annotated[str, Query(description='Resume from given network pickle. Give the path of the network as input')] = None
    initstrength: Annotated[float, Query(description='Override ADA augment strength at the beginning. Give a float value as input')] = None
    freezeD: Annotated[int, Query(description='Freeze first layers of D. Give an integer value as input')] = 0
    freezeM: Annotated[int, Query(description='Freeze first layers of the Mapping Network Gm. Give an integer value as input')] = 0
    freezeE: Annotated[bool, Query(description='Freeze the embedding layer for conditional models. Defaults to False')] = False

    blur_percent: Annotated[float, Query(description='Blur all images for the first % of training (before D). Give a float value as input')] = 0.0
    gamma: Annotated[float, Query(description='R1 regularization weight. Give a float value as input')] = None
    p: Annotated[float, Query(description='Probability for --aug=fixed. Give a float value as input')] = 0.2
    target: Annotated[float, Query(description='Target value for --aug=ada. Give a float value as input')] = 0.6
    batch_gpu: Annotated[int, Query(description='Limit batch size per GPU. Give an integer value as input')] = 8
    cbase: Annotated[int, Query(description='Capacity multiplier. Give an integer value as input')] = 32768
    cmax: Annotated[int, Query(description='Max. feature maps. Give an integer value as input')] = 512
    glr: Annotated[float, Query(description='G learning rate. Give a float value as input')] = None
    dlr: Annotated[float, Query(description='D learning rate. Give a float value as input')] = 0.002
    map_depth: Annotated[int, Query(description='Mapping network depth. Give an integer value as input')] = None
    mbstd_group: Annotated[int, Query(description='Minibatch std group size. Give an integer value as input')] = 4

    desc: Annotated[str, Query(description='Description string included in result subdir name. Give a string as input')] = None
    metrics: Annotated[str, Query(description='Comma-separated list of metrics to compute. Give a string as input')] = 'none' # fid50k_full,kid50k_full,pr50k3_full,ppl2_wend,eqt50k_int,eqt50k_frac,eqr50k,fid50k,kid50k,pr50k3,is50k
    kimg: Annotated[int, Query(description='Training duration in thousands of images. Give an integer value as input')] = 25000
    resume_kimg: Annotated[int, Query(description='Resume training at this point. Give an integer value as input')] = 0
    tick: Annotated[int, Query(description='Print tick every k images. Give an integer value as input')] = 4
    snap: Annotated[int, Query(description='Snapshot interval in ticks. Give an integer value as input')] = 50
    img_snap: Annotated[int, Query(description='Image snapshot interval in ticks. Give an integer value as input')] = 50
    snap_res: Annotated[str, Query(description='Snapshot image resolution. Options: [1080p, 4k, 8k]. Give a string as input')] = '4k'
    seed: Annotated[int, Query(description='Random seed. Give an integer value as input')] = 0
    fp32: Annotated[bool, Query(description='Disable mixed precision. Give a boolean value as input')] = False
    nobench: Annotated[bool, Query(description='Disable cuDNN benchmarking. Give a boolean value as input')] = False
    workers: Annotated[int, Query(description='Number of DataLoader workers. Give an integer value as input')] = 3
    dry_run: Annotated[bool, Query(description='Print training options and exit. Give a boolean value as input')] = False
