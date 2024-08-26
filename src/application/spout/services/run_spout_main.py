# load library
import SpoutGL
import random
from random import randint
import time
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GL.framebufferobjects import *

import os
import re
from typing import List, Optional, Tuple, Union

import click
import numpy as np
import torch

from src.infrastructure import dnnlib
from application.network.services import legacy_service

from src.application.visualize.models.renderer import Renderer
from src.infrastructure.torch_utils import gen_utils

def parse_range(s: Union[str, List]) -> List[int]:
    '''Parse a comma separated list of numbers or ranges and return a list of ints.

    Example: '1,2,5-10' returns [1, 2, 5, 6, 7]
    '''
    if isinstance(s, list): return s
    ranges = []
    range_re = re.compile(r'^(\d+)-(\d+)$')
    for p in s.split(','):
        m = range_re.match(p)
        if m:
            ranges.extend(range(int(m.group(1)), int(m.group(2))+1))
        else:
            ranges.append(int(p))
    return ranges

#----------------------------------------------------------------------------

def parse_vec2(s: Union[str, Tuple[float, float]]) -> Tuple[float, float]:
    '''Parse a floating point 2-vector of syntax 'a,b'.

    Example:
        '0,1' returns (0,1)
    '''
    if isinstance(s, tuple): return s
    parts = s.split(',')
    if len(parts) == 2:
        return (float(parts[0]), float(parts[1]))
    raise ValueError(f'cannot parse 2-vector {s}')

#----------------------------------------------------------------------------

def make_transform(translate: Tuple[float,float], angle: float):
    m = np.eye(3)
    s = np.sin(angle/360.0*np.pi*2)
    c = np.cos(angle/360.0*np.pi*2)
    m[0][0] = c
    m[0][1] = s
    m[0][2] = translate[0]
    m[1][0] = -s
    m[1][1] = c
    m[1][2] = translate[1]
    return m

#----------------------------------------------------------------------------

def randcolor():
    return randint(0, 255)

@click.command()
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--seeds', type=parse_range, help='List of random seeds (e.g., \'0,1,4-6\')', required=True)
@click.option('--trunc', 'truncation_psi', type=float, help='Truncation psi', default=1, show_default=True)
@click.option('--class', 'class_idx', type=int, help='Class label (unconditional if not specified)')
@click.option('--noise-mode', help='Noise mode', type=click.Choice(['const', 'random', 'none']), default='const', show_default=True)
@click.option('--translate', help='Translate XY-coordinate (e.g. \'0.3,1\')', type=parse_vec2, default='0,0', show_default=True, metavar='VEC2')
@click.option('--rotate', help='Rotation angle in degrees', type=float, default=0, show_default=True, metavar='ANGLE')
@click.option('--outdir', help='Where to save the output images', type=str, required=True, metavar='DIR')
def main(
    network_pkl: str,
    seeds: List[int],
    truncation_psi: float,
    noise_mode: str,
    outdir: str,
    translate: Tuple[float,float],
    rotate: float,
    class_idx: Optional[int]
):

    #window setup
    pygame.init() 
    pygame.display.set_caption( 'Spout For Python' )
    pygame.display.set_mode( (512, 512), DOUBLEBUF|OPENGL )

    glMatrixMode( GL_PROJECTION )
    glLoadIdentity()
    glOrtho(0, 512, 512, 0, 1, -1 )
    glMatrixMode( GL_MODELVIEW )
    glDisable( GL_DEPTH_TEST )
    glClearColor( 0.0, 0.0, 0.0, 0.0)
    glEnable( GL_TEXTURE_2D )

    # create receiver
    sender = SpoutGL.SpoutSender()
    sender.setSenderName("Instance3")
    # sender.setSenderName("Instance2")

    print('Loading networks from "%s"...' % network_pkl)
    device = torch.device('cuda')
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy_service.load_network_pkl(f)['G_ema'].to(device) # type: ignore

    outdir = 'out'
    os.makedirs(outdir, exist_ok=True)

    # Labels.
    label = torch.zeros([1, G.c_dim], device=device)
    if G.c_dim != 0:
        if class_idx is None:
            raise click.ClickException('Must specify class label with --class when using a conditional network')
        label[:, class_idx] = 1
    else:
        if class_idx is not None:
            print ('warn: --class=lbl ignored when running on an unconditional network')

    # -------------------------------------------------------------------------------

    save_grayscale = False
    save_rgb = True
    save_rgba = False

    starting_channel = 0
    img_scale_db = 0
    img_normalize = False
    
    # layer_name = "L13_512_128"
    # layer_name = "L14_256_3"
    # layer_name = "output"
    layer_name = ""

    # -------------------------------------------------------------------------------
    
    while True:



        # -------------------------------------------------------------------------------
        w_avg = G.mapping.w_avg
        seeds = random.randrange(0,99999999)
        sel_channels = 3
        dlatent = gen_utils.get_w_from_seed(G, device, seeds, truncation_psi=1.0)
        # Do truncation trick with center (new or global)
        w = w_avg + (dlatent - w_avg) * truncation_psi


        # Sanity check (meh, could be done better)
        submodule_names = {name: mod for name, mod in G.synthesis.named_modules()}
        assert layer_name in submodule_names, f'Layer "{layer_name}" not found in the network! Available layers: {", ".join(submodule_names)}'
        assert True in (save_grayscale, save_rgb, save_rgba), 'You must select to save the image in at least one of the three possible formats! (L, RGB, RGBA)'

        sel_channels = 3 if save_rgb else (1 if save_grayscale else 4)
        res = Renderer().render(G=G, layer_name=layer_name, dlatent=w, sel_channels=sel_channels,
                                base_channel=starting_channel, img_scale_db=img_scale_db, img_normalize=img_normalize)
        img = res.image
        image = img
        # -------------------------------------------------------------------------------



        # Generate images.
        # -------------------------------------------------------------------------------
        # z = torch.from_numpy(np.random.RandomState(seeds).randn(1, G.z_dim)).to(device)

        # # Construct an inverse rotation/translation matrix and pass to the generator.  The
        # # generator expects this matrix as an inverse to avoid potentially failing numerical
        # # operations in the network.
        # if hasattr(G.synthesis, 'input'):
        #     m = make_transform(translate, rotate)
        #     m = np.linalg.inv(m)
        #     G.synthesis.input.transform.copy_(torch.from_numpy(m))

        # img = G(z, label, truncation_psi=truncation_psi, noise_mode=noise_mode)
        # img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        # image = img[0].cpu().numpy()

        # -------------------------------------------------------------------------------


        # save_image = PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB')
        # save_image.save(f'{outdir}/seed{seed:04d}.png')

        # ---------------------------------------------------------------------------------------------

        textureSenderID = glGenTextures(1)

        glBindTexture( GL_TEXTURE_2D, textureSenderID )
        glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE )
        glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE )
        glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST )
        glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST )

        # copy data into texture
        glTexImage2D( GL_TEXTURE_2D, 0, GL_RGB, 512, 512, 0, GL_RGB, GL_UNSIGNED_BYTE, image ) 
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
        glEnable(GL_TEXTURE_2D)
        glActiveTexture(GL_TEXTURE0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()
        glBegin(GL_QUADS)
        glTexCoord(0, 0)
        glVertex2f(0, 0)
        glTexCoord(1, 0)
        glVertex2f(512, 0)
        glTexCoord(1, 1)
        glVertex2f(512, 512)
        glTexCoord(0, 1)
        glVertex2f(0, 512)
        glEnd()
        # glBindTexture(GL_TEXTURE_2D, 0)

        # send data
        sender.sendTexture(textureSenderID.item(), GL_TEXTURE_2D, 512, 512, False, 0)
        sender.setFrameSync("Instance3")
        # sender.setFrameSync("Instance2")
        
        # Wait for next send attempt
        time.sleep(1./1)

if __name__ == "__main__":
    main()