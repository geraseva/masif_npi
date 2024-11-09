# Pablo Gainza Cirauqui 2016 LPDI IBI STI EPFL

from pymol import cmd, stored
import sys
import os, math, re
from pymol.cgo import *
import os.path
import numpy as np

colorDict = {
    "sky": [COLOR, 0.0, 0.76, 1.0],
    "sea": [COLOR, 0.0, 0.90, 0.5],
    "yellowtint": [COLOR, 0.88, 0.97, 0.02],
    "hotpink": [COLOR, 0.90, 0.40, 0.70],
    "greentint": [COLOR, 0.50, 0.90, 0.40],
    "blue": [COLOR, 0.0, 0.0, 1.0],
    "green": [COLOR, 0.0, 1.0, 0.0],
    "yellow": [COLOR, 1.0, 1.0, 0.0],
    "orange": [COLOR, 1.0, 0.5, 0.0],
    "red": [COLOR, 1.0, 0.0, 0.0],
    "black": [COLOR, 0.0, 0.0, 0.0],
    "white": [COLOR, 1.0, 1.0, 1.0],
    "gray": [COLOR, 0.9, 0.9, 0.9],
}

# Create a gradient color from color 1 to color 2. val goes from 0 (color1) to 1 (color2).
def color_gradient(vals, c1=[1.0, 0.0, 0.0], c2=[0.0, 0.0, 1.0]):
    c1 = np.array(c1)[None,:]
    c2 = np.array(c2)[None,:]
    mycolor=c1+vals[:,None]*(c2-c1)
    mycolor=[[COLOR, myc[0], myc[1], myc[2]] for myc in mycolor]

    return mycolor


def iface_color(iface, colormap=[colorDict['white'],colorDict['green'],colorDict['black'],colorDict['blue'],colorDict['red']]):
    # max value is 1, min values is 0
    hp = iface.copy()
    ncols=int(np.max(hp)-np.min(hp))
    if ncols<=1:
        hp = hp * 2 - 1
        mycolor = charge_color(-hp)
    elif ncols>1:
        b=np.array(colormap)
        mycolor=b[hp.astype(int)]
    return mycolor


# Returns the color of each vertex according to the charge.
# The most purple colors are the most hydrophilic values, and the most
# white colors are the most positive colors.
def hphob_color(hphob):
    # max value is 4.5, min values is -4.5
    hp = hphob.copy()
    # normalize
    hp = hp + 4.5
    hp = hp / 9.0
    # mycolor = [ [COLOR, 1.0, hp[i], 1.0]  for i in range(len(hp)) ]
    mycolor = [[COLOR, 1.0, 1.0 - hp[i], 1.0] for i in range(len(hp))]
    return mycolor


# Returns the color of each vertex according to the charge.
# The most red colors are the most negative values, and the most
# blue colors are the most positive colors.
def charge_color(charges):
    # Assume a std deviation equal for all proteins....
    max_val = 1.0
    min_val = -1.0
    charges = (charges-min(charges))/(max(charges)-min(charges))*2-1
    blue_charges = np.array(charges)
    red_charges = np.array(charges)
    blue_charges[blue_charges < 0] = 0
    red_charges[red_charges > 0] = 0
    red_charges = abs(red_charges)
    mycolor = [
        [
            COLOR,
            0.9999 - blue_charges[i],
            0.9999 - (blue_charges[i] + red_charges[i]),
            0.9999 - red_charges[i],
        ]
        for i in range(len(charges))
    ]
    return mycolor


def load_vector(
    filename, emb_num=1, color="white", dotSize=0.2, in_channels=16, emb_dims=8, thr=0.5):

    data=np.load(filename)

    pdbid=filename.split('/')[-1].split('.')[0]    
    
    group_names = ""
    in_channels=data['inputs'].shape[1]
    emb_dims=data['embeddings'].shape[1]
    n_outputs=data['predictions'].shape[1]
    
    nmap=['0','A','G','C','T/U']

    assert (len(nmap)==n_outputs) or (n_outputs==1)

    for feat in range(0,n_outputs):
        name = "pred_"+nmap[feat]
        obj = []
        color_array = charge_color(data['predictions'][:,feat])
        for v_ix, vert in enumerate(data['coords']):
            obj.extend(color_array[v_ix])
            obj.extend([SPHERE, vert[0], vert[1], vert[2], dotSize])
        cmd.load_cgo(obj, name, 1.0)
        group_names = group_names + " " + name

    for feat in range(0,n_outputs):
        name = "real_"+nmap[feat]
        obj = []
        color_array = charge_color(data['labels'][:,feat])
        for v_ix, vert in enumerate(data['coords']):
            obj.extend(color_array[v_ix])
            obj.extend([SPHERE, vert[0], vert[1], vert[2], dotSize])
        cmd.load_cgo(obj, name, 1.0)
        group_names = group_names + " " + name

    if n_outputs>1:
        name = "pred"
        obj = []
        color_array = iface_color(np.argmax(data['predictions'], axis=1))
        for v_ix, vert in enumerate(data['coords']):
            obj.extend(color_array[v_ix])
            obj.extend([SPHERE, vert[0], vert[1], vert[2], dotSize])
        cmd.load_cgo(obj, name, 1.0)
        group_names = group_names + " " + name

        name = "real"
        obj = []
        color_array = iface_color(np.argmax(data['labels'], axis=1))
        for v_ix, vert in enumerate(data['coords']):
            obj.extend(color_array[v_ix])
            obj.extend([SPHERE, vert[0], vert[1], vert[2], dotSize])
        cmd.load_cgo(obj, name, 1.0) 
        group_names = group_names + " " + name

    subgroup_names=''
    for feat in range(0,in_channels):
        name = "input_"+str(feat)
        obj = []
        color_array = charge_color(data['inputs'][:,feat])
        for v_ix, vert in enumerate(data['coords']):
            obj.extend(color_array[v_ix])
            obj.extend([SPHERE, vert[0], vert[1], vert[2], dotSize])
        cmd.load_cgo(obj, name, 1.0)
        subgroup_names = subgroup_names + " " + name

    for feat in range(0,emb_dims):
        name = "emb_"+str(feat)
        obj = []
        color_array = charge_color(data['embeddings'][:,feat])
        for v_ix, vert in enumerate(data['coords']):
            obj.extend(color_array[v_ix])
            obj.extend([SPHERE, vert[0], vert[1], vert[2], dotSize])
        cmd.load_cgo(obj, name, 1.0)
        subgroup_names = subgroup_names + " " + name

    cmd.group('features', subgroup_names)
    group_names=group_names+' features'

    cmd.group(pdbid, group_names)

    return 0
    # Draw surface charges.

def load_npy(filename, dotSize=0.2):

    data=np.load(filename)

    name = "predictions"
    obj = []
    color_array = charge_color(data["preds"])
    for v_ix, vert in enumerate(data['coords']):
        obj.extend(color_array[v_ix])
        obj.extend([SPHERE, vert[0], vert[1], vert[2], dotSize])
    cmd.load_cgo(obj, name, 1.0)

    return 0
    
cmd.extend("loadnpy", load_npy)
cmd.extend("load_vector", load_vector)


