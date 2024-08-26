# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Train a GAN using the techniques described in the paper
"Alias-Free Generative Adversarial Networks"."""

# TODO: Change arguments to parameters for API

import os
import traceback

import click
import json
import tempfile
import torch

from src.domain.models.training.train_models import TrainParameters
from src.infrastructure import dnnlib
from src.application.training.models import training_loop
from application.metrics.models import metric_main
from src.infrastructure.torch_utils import training_stats, custom_ops, gen_utils


# ----------------------------------------------------------------------------


def subprocess_fn(rank, c, temp_dir):
    dnnlib.util.Logger(file_name=os.path.join(c.run_dir, 'log.txt'), file_mode='a', should_flush=True)

    # Init torch.distributed.
    if c.num_gpus > 1:
        init_file = os.path.abspath(os.path.join(temp_dir, '.torch_distributed_init'))
        if os.name == 'nt':
            init_method = 'file:///' + init_file.replace('\\', '/')
            torch.distributed.init_process_group(backend='gloo', init_method=init_method, rank=rank, world_size=c.num_gpus)
        else:
            init_method = f'file://{init_file}'
            torch.distributed.init_process_group(backend='nccl', init_method=init_method, rank=rank, world_size=c.num_gpus)

    # Init torch_utils.
    sync_device = torch.device('cuda', rank) if c.num_gpus > 1 else None
    training_stats.init_multiprocessing(rank=rank, sync_device=sync_device)
    if rank != 0:
        custom_ops.verbosity = 'none'

    # Execute training loop.
    training_loop.training_loop(rank=rank, **c)


# ----------------------------------------------------------------------------


def launch_training(c, desc, outdir, dry_run):
    dnnlib.util.Logger(should_flush=True)

    # Pick output directory.
    c.run_dir = gen_utils.make_run_dir(outdir=outdir, desc=desc, dry_run=dry_run)

    # Print options.
    print()
    print('Training options:')
    print(json.dumps(c, indent=2))
    print()
    print(f'Output directory:    {c.run_dir}')
    print(f'Number of GPUs:      {c.num_gpus}')
    print(f'Batch size:          {c.batch_size} images')
    print(f'Training duration:   {c.total_kimg} kimg')
    print(f'Dataset path:        {c.training_set_kwargs.path}')
    print(f'Dataset size:        {c.training_set_kwargs.max_size} images')
    print(f'Dataset resolution:  {c.training_set_kwargs.resolution}')
    print(f'Dataset labels:      {c.training_set_kwargs.use_labels}')
    print(f'Dataset x-flips:     {c.training_set_kwargs.xflip}')
    print(f'Dataset y-flips:     {c.training_set_kwargs.yflip}')
    print()

    # Dry run?
    if dry_run:
        print('Dry run; exiting.')
        return

    with open(os.path.join(c.run_dir, 'training_options.json'), 'wt') as f:
        json.dump(c, f, indent=2)

    # Launch processes.
    print('Launching processes...')
    torch.multiprocessing.set_start_method('spawn')
    with tempfile.TemporaryDirectory() as temp_dir:
        if c.num_gpus == 1:
            subprocess_fn(rank=0, c=c, temp_dir=temp_dir)
        else:
            torch.multiprocessing.spawn(fn=subprocess_fn, args=(c, temp_dir), nprocs=c.num_gpus)


# ----------------------------------------------------------------------------


def init_dataset_kwargs(data):
    try:
        dataset_kwargs = dnnlib.EasyDict(class_name='src.application.training.models.dataset.ImageFolderDataset', path=data, use_labels=True, max_size=None, xflip=False, yflip=False)
        dataset_obj = dnnlib.util.construct_class_by_name(**dataset_kwargs) # Subclass of training.dataset.Dataset.
        dataset_kwargs.resolution = dataset_obj.resolution # Be explicit about resolution.
        dataset_kwargs.use_labels = dataset_obj.has_labels # Be explicit about labels.
        dataset_kwargs.max_size = len(dataset_obj) # Be explicit about dataset size.
        return dataset_kwargs, dataset_obj.name
    except IOError as err:
        raise click.ClickException(f'--data: {err}')


# ----------------------------------------------------------------------------


def parse_comma_separated_list(s):
    if isinstance(s, list):
        return s
    if s is None or s.lower() == 'none' or s == '':
        return []
    return s.split(',')


# ----------------------------------------------------------------------------

def train(parameters: TrainParameters):
    # Initialize config.
    opts = dnnlib.EasyDict(parameters)  # Command line arguments.
    c = dnnlib.EasyDict()  # Main config dict.
    c.G_kwargs = dnnlib.EasyDict(class_name=None, z_dim=512, w_dim=512, mapping_kwargs=dnnlib.EasyDict())
    c.D_kwargs = dnnlib.EasyDict(class_name='src.application.training.models.networks_stylegan2.Discriminator', block_kwargs=dnnlib.EasyDict(), mapping_kwargs=dnnlib.EasyDict(), epilogue_kwargs=dnnlib.EasyDict())
    c.G_opt_kwargs = dnnlib.EasyDict(class_name='torch.optim.Adam', betas=[0,0.99], eps=1e-8)
    c.D_opt_kwargs = dnnlib.EasyDict(class_name='torch.optim.Adam', betas=[0,0.99], eps=1e-8)  # TODO: Use ComplexSGD: https://arxiv.org/abs/2102.08431
    c.loss_kwargs = dnnlib.EasyDict(class_name='src.application.training.models.loss.StyleGAN2Loss')
    c.data_loader_kwargs = dnnlib.EasyDict(pin_memory=True, prefetch_factor=2)
    # c.data_loader_kwargs = dnnlib.EasyDict(pin_memory=True, prefetch_factor=None)  # 2

    # Training set.
    c.training_set_kwargs, dataset_name = init_dataset_kwargs(data=opts.data)
    if opts.cond and not c.training_set_kwargs.use_labels:
        raise click.ClickException('--cond=True requires labels specified in dataset.json')
    c.training_set_kwargs.use_labels = opts.cond
    c.training_set_kwargs.xflip = opts.mirror
    c.training_set_kwargs.yflip = opts.mirror_y

    # Hyperparameters & settings.
    c.num_gpus = opts.gpus
    c.batch_size = opts.batch
    c.batch_gpu = opts.batch_gpu or opts.batch // opts.gpus
    c.G_kwargs.channel_base = c.D_kwargs.channel_base = opts.cbase
    c.G_kwargs.channel_max = c.D_kwargs.channel_max = opts.cmax
    c.G_kwargs.mapping_kwargs.num_layers = (8 if opts.cfg == 'stylegan2' else 2) if opts.map_depth is None else opts.map_depth
    c.G_kwargs.mapping_kwargs.freeze_layers = opts.freezeM
    c.G_kwargs.mapping_kwargs.freeze_embed = opts.freezeE
    c.D_kwargs.block_kwargs.freeze_layers = opts.freezeD
    c.D_kwargs.epilogue_kwargs.mbstd_group_size = opts.mbstd_group
    if opts.gamma is not None:
        c.loss_kwargs.r1_gamma = float(opts.gamma)
    else:
        # Use heuristic from StyleGAN2-ADA
        c.loss_kwargs.r1_gamma = 0.0002 * c.training_set_kwargs.resolution ** 2 / c.batch_size
        print(f'Using heuristic, R1 gamma set at: {c.loss_kwargs.r1_gamma:.4f}')
    c.G_opt_kwargs.lr = (0.002 if opts.cfg == 'stylegan2' else 0.0025) if opts.glr is None else opts.glr
    c.D_opt_kwargs.lr = opts.dlr


    c.metrics = opts.metrics
    c.metrics = parse_comma_separated_list(c.metrics)

    c.total_kimg = opts.kimg
    c.resume_kimg = opts.resume_kimg
    c.kimg_per_tick = opts.tick
    c.network_snapshot_ticks = opts.snap
    c.image_snapshot_ticks = opts.img_snap
    c.snap_res = opts.snap_res
    c.random_seed = c.training_set_kwargs.random_seed = opts.seed
    c.data_loader_kwargs.num_workers = opts.workers

    # Sanity checks.
    if c.batch_size % c.num_gpus != 0:
        raise click.ClickException('--batch must be a multiple of --gpus')
    if c.batch_size % (c.num_gpus * c.batch_gpu) != 0:
        raise click.ClickException('--batch must be a multiple of --gpus times --batch-gpu')
    if c.batch_gpu < c.D_kwargs.epilogue_kwargs.mbstd_group_size:
        raise click.ClickException('--batch-gpu cannot be smaller than --mbstd')
    if any(not metric_main.is_valid_metric(metric) for metric in c.metrics):
        raise click.ClickException('\n'.join(['--metrics can only contain the following values:'] + metric_main.list_valid_metrics()))

    # Base configuration.
    c.ema_kimg = c.batch_size * 10 / 32
    if opts.cfg == 'stylegan2':
        c.G_kwargs.class_name = 'src.application.training.models.networks_stylegan2.Generator'
        c.loss_kwargs.style_mixing_prob = 0.9 # Enable style mixing regularization.
        c.loss_kwargs.pl_weight = 2 # Enable path length regularization.
        c.G_reg_interval = 4 # Enable lazy regularization for G.
        c.G_kwargs.fused_modconv_default = 'inference_only' # Speed up training by using regular convolutions instead of grouped convolutions.
        c.loss_kwargs.pl_no_weight_grad = True # Speed up path length regularization by skipping gradient computation wrt. conv2d weights.
    elif opts.cfg == 'stylegan2-cifar':
        # CIFAR's config
        c.G_kwargs.class_name = 'src.application.training.models.networks_stylegan2.Generator'
        c.loss_kwargs.pl_weight = 0  # Disable path length regularization (default)
        c.loss_kwargs.style_mixing_prob = 0  # Disable style mixing regularization (default)
        c.D_kwargs.architecture = 'orig'  # Disable residual skip connections in D
        c.G_kwargs.mapping_kwargs.num_layers = 2  # Lower the number of layers in the mapping network
        c.ema_kimg = 500
    elif opts.cfg == 'stylegan2-ext':
        # Aydao's config (to be tested later)
        c.G_kwargs.class_name = 'src.application.training.models.networks_stylegan2.Generator'
        c.loss_kwargs.pl_weight = 0 # Disable path length regularization (default)
        c.loss_kwargs.style_mixing_prob = 0 # Disable style mixing regularization (default)

        # Double Generator capacity
        c.G_kwargs.extended_sgan2 = True
        c.G_kwargs.channel_base = 32 << 10  # (default already)
        c.G_kwargs.extended_sgan2 = True  # Double the number of feature maps
        c.G_kwargs.channel_max = 1024
        c.D_kwargs.epilogue_kwargs.mbstd_num_channels = 4

        # Mapping layer
        c.G_kwargs.z_dim = 1024
        c.G_kwargs.w_dim = 1024
        c.G_kwargs.mapping_kwargs.num_layers = 4  # TODO: test with a higher number later on
        c.G_kwargs.mapping_kwargs.layer_features = 1024  # TODO: test with a wider mapping network later on

        # TODO: Enable top-k training
        # TODO: try different values of c.ema_kimg
        # TODO: Reduce in-memory size (lower batch size, more layers with fp16, etc)

    else:
        c.G_kwargs.class_name = 'src.application.training.models.networks_stylegan3.Generator'
        c.G_kwargs.magnitude_ema_beta = 0.5 ** (c.batch_size / (20 * 1e3))
        if opts.cfg == 'stylegan3-r':
            c.G_kwargs.conv_kernel = 1 # Use 1x1 convolutions.
            c.G_kwargs.channel_base *= 2 # Double the number of feature maps.
            c.G_kwargs.channel_max *= 2
            c.G_kwargs.use_radial_filters = True # Use radially symmetric downsampling filters.
            c.loss_kwargs.blur_init_sigma = 10 # Blur the images seen by the discriminator.
            c.loss_kwargs.blur_fade_kimg = c.batch_size * 200 / 32 # Fade out the blur during the first N kimg.

    # Curriculum learning: start blurring all images before passing to the Discriminator, and fade it out
    if opts.blur_percent > 0:
        c.loss_kwargs.blur_init_sigma = 10
        total_kimg = opts.kimg - opts.resume_kimg
        c.loss_kwargs.blur_fade_kimg = (opts.blur_percent / 100.0) * total_kimg

    # Augmentation.
    augpipe_specs = {
        'blit': dict(xflip=1, rotate90=1, xint=1),
        'geom': dict(scale=1, rotate=1, aniso=1, xfrac=1),
        'color': dict(brightness=1, contrast=1, lumaflip=1, hue=1, saturation=1),
        'filter': dict(imgfilter=1),
        'noise': dict(noise=1),
        'cutout': dict(cutout=1),
        'bg': dict(xflip=1, rotate90=1, xint=1, scale=1, rotate=1, aniso=1, xfrac=1),
        'bgc': dict(xflip=1, rotate90=1, xint=1, scale=1, rotate=1, aniso=1, xfrac=1, brightness=1, contrast=1, lumaflip=1, hue=1, saturation=1),
        'bgcf': dict(xflip=1, rotate90=1, xint=1, scale=1, rotate=1, aniso=1, xfrac=1, brightness=1, contrast=1, lumaflip=1, hue=1, saturation=1, imgfilter=1),
        'bgcfn': dict(xflip=1, rotate90=1, xint=1, scale=1, rotate=1, aniso=1, xfrac=1, brightness=1, contrast=1, lumaflip=1, hue=1, saturation=1, imgfilter=1, noise=1),
        'bgcfnc': dict(xflip=1, rotate90=1, xint=1, scale=1, rotate=1, aniso=1, xfrac=1, brightness=1, contrast=1, lumaflip=1, hue=1, saturation=1, imgfilter=1, noise=1, cutout=1)
    }

    if opts.aug != 'noaug':
        c.augment_kwargs = dnnlib.EasyDict(class_name='src.application.training.models.augment.AugmentPipe', **augpipe_specs[opts.augpipe])
        if opts.aug == 'ada':
            c.ada_target = opts.target
        if opts.aug == 'fixed':
            c.augment_p = opts.p

    # Initial augmentation strength for opts.aug == 'ada'; thanks to @dvschultz
    if opts.initstrength is not None:
         c.augment_p = opts.initstrength

    # Resume.

    if opts.resume is None:
        resume_desc = 'no_resume'
    else:
        if opts.resume in gen_utils.resume_specs[opts.cfg]:
            c.resume_pkl = gen_utils.resume_specs[opts.cfg][opts.resume]
            resume_desc = f'resume_{opts.resume}'
        else:  # A local file
            c.resume_pkl = opts.resume
            resume_desc = f'resume_custom'
        c.ada_kimg = 100 # Make ADA react faster at the beginning.
        c.ema_rampup = None # Disable EMA rampup.
        c.loss_kwargs.blur_init_sigma = 0 # Disable blur rampup.

    # Performance-related toggles.
    if opts.fp32:
        c.G_kwargs.num_fp16_res = c.D_kwargs.num_fp16_res = 0
        c.G_kwargs.conv_clamp = c.D_kwargs.conv_clamp = None
    if opts.nobench:
        c.cudnn_benchmark = False

    # Description string.
    desc = f'{opts.cfg:s}-{dataset_name:s}-gpus{c.num_gpus:d}-batch{c.batch_size:d}-gamma{c.loss_kwargs.r1_gamma:g}-{resume_desc}'
    if opts.desc is not None:
        desc += f'-{opts.desc}'

    # Launch.
    try:
        launch_training(c=c, desc=desc, outdir=opts.outdir, dry_run=opts.dry_run)
    except:
        traceback.print_exc()

    return {"Message": "Training completed successfully."}


# ----------------------------------------------------------------------------
