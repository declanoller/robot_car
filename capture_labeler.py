import argparse
import FileSystemTools as fst
import traceback as tb
import time
import json
from PIL import ImageFile,Image,ImageDraw
import numpy as np
import os
import subprocess as sp
from math import sin, cos, pi
import movieMaker

def flipY(vec, size_y):
    out_vec = vec
    out_vec[1] = size_y - out_vec[1]
    return(out_vec)




parser = argparse.ArgumentParser()
parser.add_argument('path')
parser.add_argument('--vid', action='store_true', default=False)
parser.add_argument('--show_first', action='store_true', default=False)
parser.add_argument('--save_first', action='store_true', default=False)
parser.add_argument('--distances', action='store_true', default=False)
parser.add_argument('--framerate', default='10')
args = parser.parse_args()
print('passed args: ', args)

############################ Make dir for labeled imgs
base_dir = args.path
dir_name = base_dir.split('/')[-1]
print('dir_name:', dir_name)
labeled_dir = fst.combineDirAndFile(base_dir, 'labeled')
if os.path.exists(labeled_dir):
    print('{} exists, deleting'.format(labeled_dir))
    sp.check_call(['rm', '-rf', labeled_dir])
fst.makeDir(labeled_dir)
input_img_ext = '.jpg'
output_img_ext = '.jpg'

############################# Open .json with info
json_file = fst.combineDirAndFile(base_dir, 'iteration_info.json')
with open(json_file) as f:
    iter_info = json.load(f)

##################### Open first image, to get some size info
test_img = Image.open(fst.combineDirAndFile(base_dir, f'0{input_img_ext}'))
img_height_px = test_img.size[1]
center_wall_dist_px = img_height_px/2 # Half the y distance
### Assumes the center of the arena is the center of the image
center_pos_px = 0.5*np.array(test_img.size).astype('int')# - [20, 5]
bottom_left_corner = (center_pos_px - center_wall_dist_px).astype('int')
top_right_corner = (center_pos_px + center_wall_dist_px).astype('int')

###################### Drawing the center and box, for reference
draw = ImageDraw.Draw(test_img)
draw.rectangle([*(center_pos_px - 5), *(center_pos_px + 5)], outline='red', fill='red')
draw.rectangle([*bottom_left_corner, *top_right_corner], outline='blue')

#test_img.show()

########################## Stuff to do with actual arena
wall_length = 1.25
wall_norm = wall_length/2.0
arena_lims = np.array([-wall_length/2, wall_length/2])
y_fudge = 0.05
target_positions = np.array([
[.19, y_fudge],
[.55, y_fudge],
[wall_length, .21],
[wall_length, .65],
[.97, wall_length-y_fudge],
[.58, wall_length-y_fudge],
[0, 1.02],
[0, .60]])

N_targets = len(target_positions)
target_positions = target_positions + arena_lims[0]



#################### Arrow
arrow = Image.open('other/arrow.png')
arrow = arrow.resize((0.26*np.array(test_img.size)).astype('int'))
box_half_width = (np.array(arrow.size)/2).astype('int')


print('\nLabeling images...')
dist_to_px_factor = (img_height_px/wall_length)
output_fname_list = []

for iter in iter_info.keys():

    img = Image.open(fst.combineDirAndFile(base_dir, f'{iter}{input_img_ext}'))
    draw = ImageDraw.Draw(img)

    # Pos/etc info about current iteration
    pos = np.array(iter_info[iter]['position'])
    ang = np.array(iter_info[iter]['angle'])*(180/pi)
    ang_rad = ang*pi/180.0
    tar_pos = target_positions[iter_info[iter]['current_target']]
    tar = np.array(tar_pos)

    pos_frac = pos/wall_norm
    tar_frac = tar/wall_norm


    pos_img_px = flipY(center_pos_px + pos_frac*center_wall_dist_px, img_height_px).astype(int)
    # Draw small dot on car pos (arrow first)
    if args.distances:
        d1 = iter_info[iter]['d1']
        d2 = iter_info[iter]['d2']
        d3 = iter_info[iter]['d3']

        d1_endpt = pos + np.array([cos(ang_rad), sin(ang_rad)])*d1
        d2_endpt = pos + np.array([cos(ang_rad + pi/2), sin(ang_rad + pi/2)])*d2
        d3_endpt = pos + np.array([cos(ang_rad - pi/2), sin(ang_rad - pi/2)])*d3
        d1_px = flipY(center_pos_px + d1_endpt*center_wall_dist_px/wall_norm, img_height_px).astype(int)
        d2_px = flipY(center_pos_px + d2_endpt*center_wall_dist_px/wall_norm, img_height_px).astype(int)
        d3_px = flipY(center_pos_px + d3_endpt*center_wall_dist_px/wall_norm, img_height_px).astype(int)

        draw.line([*pos_img_px, *d1_px], fill='red', width=4)
        draw.line([*pos_img_px, *d2_px], fill='blue', width=4)
        draw.line([*pos_img_px, *d3_px], fill='lime', width=4)
    else:
        arrow_rot = arrow.rotate(ang)
        arrow_lower_left = pos_img_px - box_half_width
        img.paste(arrow_rot, box=tuple(arrow_lower_left), mask=arrow_rot)

    # Dot on center of car
    car_rect_size = 10
    draw.ellipse([*(pos_img_px - car_rect_size), *(pos_img_px + car_rect_size)], fill='white')



    # Draw target
    tar_img_px = flipY(center_pos_px + tar_frac*center_wall_dist_px, img_height_px).astype(int)
    draw.rectangle([*(tar_img_px - 10), *(tar_img_px + 10)], outline='blue', fill='blue')

    if args.show_first:
        if args.save_first:
            img.save(fst.combineDirAndFile(labeled_dir, f'{iter}{output_img_ext}'), optimize=True)
        img.show()
        break

    outfile = fst.combineDirAndFile(labeled_dir, f'{iter}{output_img_ext}')
    img.save(outfile)
    output_fname_list.append(outfile)

print('\nDone labeling!\n')


######################### Creating a vid:

if args.vid:

    vid_fname = os.path.join(base_dir, f'{dir_name}.mp4')

    movieMaker.imgsToVid(labeled_dir, vid_fname, img_ext='jpg', framerate=args.framerate)

    print('\ndone!\n')












#
