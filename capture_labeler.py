import argparse
import FileSystemTools as fst
import traceback as tb
import paho.mqtt.client as mqtt
import time
import json
from PIL import ImageFile,Image,ImageDraw
import numpy as np
import os
import subprocess as sp

def flipY(vec, size_y):
    out_vec = vec
    out_vec[1] = size_y - out_vec[1]
    return(out_vec)




parser = argparse.ArgumentParser()
parser.add_argument('path')
parser.add_argument('--gif', action='store_true', default=False)
args = parser.parse_args()
print('passed args: ', args)


base_dir = args.path
labeled_dir = fst.combineDirAndFile(base_dir, 'labeled')
if os.path.exists(labeled_dir):
    print('{} exists, deleting'.format(labeled_dir))
    sp.check_call(['rm', '-rf', labeled_dir])
fst.makeDir(labeled_dir)

json_file = fst.combineDirAndFile(base_dir, 'iteration_info.json')
with open(json_file) as f:
    iter_info = json.load(f)

test_img = Image.open(fst.combineDirAndFile(base_dir, '0.jpg'))
wall_length = 1.25
wall_norm = wall_length/2
center_wall_dist_px = test_img.size[1]/2
center_pos_px = np.array(test_img.size)/2# - [20, 5]

arena_lims = np.array([-wall_length/2, wall_length/2])

target_positions = np.array([
[.19, 0],
[.55, 0],
[wall_length, .21],
[wall_length, .65],
[.97, wall_length],
[.58, wall_length],
[0, 1.02],
[0, .60]])

N_targets = len(target_positions)
target_positions = target_positions + arena_lims[0]

arrow = Image.open('arrow.png')
arrow = arrow.resize((0.16*np.array(test_img.size)).astype('int'))
box_half_width = (np.array(arrow.size)/2).astype('int')

#draw.rectangle([*(center_pos_px - 5), *(center_pos_px + 5)], outline='red', fill='red')
#print('pos', pos)
#print('ang', ang)
#print('pos_frac', pos_frac)
#print('pos_img_px', pos_img_px.astype('int'))

for iter in iter_info.keys():

    img = Image.open(fst.combineDirAndFile(base_dir, f'{iter}.jpg'))
    draw = ImageDraw.Draw(img)

    pos = np.array(iter_info[iter]['position'])
    ang = np.array(iter_info[iter]['angle'])*(180/3.14)
    tar_pos = target_positions[iter_info[iter]['current_target']]
    tar = np.array(tar_pos)

    pos_frac = pos/wall_norm
    tar_frac = tar/wall_norm

    pos_img_px = flipY(center_pos_px + pos_frac*center_wall_dist_px, img.size[1]).astype(int)
    tar_img_px = flipY(center_pos_px + tar_frac*center_wall_dist_px, img.size[1]).astype(int)
    draw.rectangle([*(tar_img_px - 10), *(tar_img_px + 10)], outline='blue', fill='blue')

    arrow_rot = arrow.rotate(ang)
    arrow_lower_left = pos_img_px - box_half_width
    img.paste(arrow_rot, box=tuple(arrow_lower_left), mask=arrow_rot)

    draw.rectangle([*(pos_img_px - 3), *(pos_img_px + 3)], outline='green', fill='green')

    img.save(fst.combineDirAndFile(labeled_dir, f'{iter}.jpg'))















#
