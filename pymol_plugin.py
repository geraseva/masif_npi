# Pablo Gainza Cirauqui 2016 LPDI IBI STI EPFL
# This pymol plugin for Masif just enables the load ply functions.


from pymol import cmd, stored
import sys
import os, math, re
from pymol.cgo import *
import os.path
import numpy as np

"""
   loadPLY.py: This pymol function loads ply files into pymol.
    Pablo Gainza - LPDI STI EPFL 2016-2019
    This file is part of MaSIF.
    Released under an Apache License 2.0
"""
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


def load_npy(
    filename, emb_num=1, color="white", dotSize=0.2, in_channels=16, emb_dims=8, thr=0.5):

    verts=np.load(filename+"_predcoords.npy")
    feats=np.load(f'{filename}_predfeatures_emb{emb_num}.npy')


    pdbid=filename.split('/')[-1]    
    
    group_names = ""
    n_outputs=(feats.shape[1]-in_channels-emb_dims)//2
    
    nmap=['0','A','G','C','T/U']

    assert (len(nmap)==n_outputs) or (n_outputs==1)

    for feat in range(0,n_outputs):
        name = "pred_"+nmap[feat]
        obj = []
        color_array = charge_color(feats[:,in_channels+emb_dims+feat])
        for v_ix in range(len(verts)):
            vert = verts[v_ix]
            obj.extend(color_array[v_ix])
            obj.extend([SPHERE, vert[0], vert[1], vert[2], dotSize])

        cmd.load_cgo(obj, name, 1.0)
        group_names = group_names + " " + name

    for feat in range(0,n_outputs):
        name = "real_"+nmap[feat]
        obj = []
        color_array = charge_color(feats[:,in_channels+emb_dims+n_outputs+feat])
        for v_ix in range(len(verts)):
            vert = verts[v_ix]
            obj.extend(color_array[v_ix])
            obj.extend([SPHERE, vert[0], vert[1], vert[2], dotSize])

        cmd.load_cgo(obj, name, 1.0)
        group_names = group_names + " " + name

    if n_outputs>1:
        name = "pred"
        obj = []
        color_array = iface_color(np.argmax(feats[:,-2*n_outputs:-n_outputs], axis=1))
        for v_ix in range(len(verts)):
            vert = verts[v_ix]
            obj.extend(color_array[v_ix])
            obj.extend([SPHERE, vert[0], vert[1], vert[2], dotSize])

        cmd.load_cgo(obj, name, 1.0)
        group_names = group_names + " " + name

        name = "real"
        obj = []
        color_array = iface_color(np.argmax(feats[:,-n_outputs:], axis=1))
        for v_ix in range(len(verts)):
            vert = verts[v_ix]
            obj.extend(color_array[v_ix])
            obj.extend([SPHERE, vert[0], vert[1], vert[2], dotSize])
    
        cmd.load_cgo(obj, name, 1.0) 
        group_names = group_names + " " + name

    subgroup_names=''
    for feat in range(0,in_channels+emb_dims):
        if feat<in_channels:
            name = "input_"+str(feat)
        else:
            name = "emb_"+str(feat)
        obj = []
        color_array = charge_color(feats[:,feat])
        for v_ix in range(len(verts)):
            vert = verts[v_ix]
            obj.extend(color_array[v_ix])
            obj.extend([SPHERE, vert[0], vert[1], vert[2], dotSize])

        cmd.load_cgo(obj, name, 1.0)
        subgroup_names = subgroup_names + " " + name

    cmd.group('features', subgroup_names)
    group_names=group_names+' features'

  
    cmd.group(pdbid, group_names)

    return 0
    # Draw surface charges.
    
def load_ply(filename, colormap=['white','green','black','blue','red'], dotSize=0.2):

    colormap=[colorDict[i] for i in colormap]

    import meshio
    pdbid=filename.split('/')[-1].split('.')[0]
    path='/'.join(filename.split('/')[:-1])
    try:
        mesh=meshio.read(path+'/'+pdbid+'.ply')
    except: 
        mesh=save_npy_as_ply(path+'/'+pdbid+'.ply')

    verts = mesh.points
    faces = mesh.cells_dict['triangle']
    group_names=''

    name = pdbid+"_dots"
    obj = []
    for v_ix in range(len(verts)):
        vert = verts[v_ix]
        obj.extend([SPHERE, vert[0], vert[1], vert[2], dotSize])
    cmd.load_cgo(obj, name, 1.0)
    group_names = group_names + " " + name

    for name in mesh.point_data:
        obj = []   
        if mesh.point_data[name].dtype==int:
            color_array_surf=iface_color(mesh.point_data[name], colormap)
        else:
            color_array_surf=charge_color(mesh.point_data[name])
        for tri in faces:
            vert1 = verts[int(tri[0])]
            vert2 = verts[int(tri[1])]
            vert3 = verts[int(tri[2])]
            obj.extend([BEGIN, TRIANGLES])
            obj.extend(color_array_surf[int(tri[0])])
            obj.extend([VERTEX, (vert1[0]), (vert1[1]), (vert1[2])])
            obj.extend(color_array_surf[int(tri[1])])
            obj.extend([VERTEX, (vert2[0]), (vert2[1]), (vert2[2])])
            obj.extend(color_array_surf[int(tri[2])])
            obj.extend([VERTEX, (vert3[0]), (vert3[1]), (vert3[2])])
            obj.append(END)
        cmd.load_cgo(obj, name, 1.0)
        obj = []
        group_names = group_names + " " + name

    
    cmd.group(pdbid, group_names)

def save_npy_as_ply(filename, in_channels=16, emb_dims=8, labels=['0','A','G','C','T/U'], thr=2):

    import meshio

    verts=np.load(filename[:-4]+"_predcoords.npy")
    feats=np.load(filename[:-4]+'_predfeatures_emb1.npy')
    print('Compute triangles')
    '''
    import open3d as o3d

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(verts)
    pcd.estimate_normals()
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd, 
        o3d.utility.DoubleVector(radii))
    cells = [("triangle", np.asarray(mesh.triangles))]
    del mesh, pcd

    '''
    from scipy.spatial import Delaunay
    cells=Delaunay(verts).simplices
    cells=np.concatenate((
        cells[:,[0,1,2]],
        cells[:,[0,1,3]],
        cells[:,[0,2,3]],
        cells[:,[1,2,3]],
        ))
    filt=np.max(np.array(
         [np.sum((verts[cells[:,0],:]-verts[cells[:,1],:])**2, axis=-1),
          np.sum((verts[cells[:,1],:]-verts[cells[:,2],:])**2, axis=-1),
          np.sum((verts[cells[:,0],:]-verts[cells[:,2],:])**2, axis=-1)]),
          axis=0)
    filt=(filt<=thr**2)
    cells = [("triangle", cells[filt,:])]
    
    point_data={}
    
    for feat in range(0,in_channels):
        point_data["input_feature_"+str(feat)]= feats[:,feat]
    for feat in range(0,emb_dims):
        point_data["emb_feature_"+str(feat)]=feats[:,in_channels+feat]

    n_outputs=(feats.shape[1]-in_channels-emb_dims)//2

    if n_outputs>1:
        assert n_outputs==len(labels)
        point_data["pred"]=np.argmax(feats[:,-2*n_outputs:-n_outputs],axis=1)
        point_data["real"]=np.argmax(feats[:,-n_outputs:],axis=1)

    for feat in range(0,n_outputs):
        point_data["pred_"+labels[feat]]=feats[:,in_channels+emb_dims+n_outputs+feat]
    for feat in range(0,n_outputs):
        point_data["real_"+labels[feat]]=feats[:,in_channels+emb_dims+n_outputs+feat]

    mesh = meshio.Mesh(verts, cells, point_data=point_data)
    mesh.write(filename, file_format="ply")

    return mesh

cmd.extend("loadply", load_ply)
cmd.extend("loadnpy", load_npy)
