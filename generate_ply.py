import numpy as np
import os
import argparse
from scipy.io import loadmat, savemat
import trimesh

def get_args():
    parser = argparse.ArgumentParser(description='Generate 3D stl file', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-p', '--path', dest='path', type=str, default=False, help='The path of image folder')
    parser.add_argument('-i', '--id', dest='id', type=str, default=False, help='The id vector')
    parser.add_argument('-e', '--exp', dest='exp', type=str, default=False, help='The expression vector')
    parser.add_argument('-t', '--tex', dest='tex', type=str, default=None, help='The texture vector')
    return parser.parse_args()

args = get_args()
img_folder = args.path
shape_vector = np.load(os.path.join(img_folder, args.id))
exp_vector = np.load(os.path.join(img_folder, args.exp))

model = loadmat('./BFM/BFM_model_front.mat')
mean_shape = model['meanshape'].astype(np.float32)
id_base = model['idBase'].astype(np.float32)
exp_base = model['exBase'].astype(np.float32)
face = model['tri'].astype(np.int64) - 1
# Recenter
mean_shape = mean_shape.reshape([-1, 3])
mean_shape = mean_shape - np.mean(mean_shape, axis=0, keepdims=True)
mean_shape = mean_shape.reshape([-1, 1])
# Compute face
vertices = id_base.dot(shape_vector.reshape((-1, 1))) + mean_shape
vertices = vertices.reshape((-1, 3))
# Generate stl
if args.tex is not None:
    mean_tex = model['meantex'].astype(np.float32)
    tex_base = model['texBase'].astype(np.float32)
    tex_vector = np.load(os.path.join(img_folder, args.tex))
    texture = tex_base.dot(tex_vector.T) + mean_tex.T
    texture =  texture.reshape((-1, 3))
    mesh = trimesh.Trimesh(vertices=vertices, faces=face, vertex_colors=texture)
    filename = args.id.split('.')[-2] + '_' + args.exp.split('.')[-2] + '_' + args.tex.split('.')[-2] + '.ply'
else:
    mesh = trimesh.Trimesh(vertices=vertices, faces=face)
    filename = args.id.split('.')[-2] + '_' + args.exp.split('.')[-2] + '.ply'

mesh.export(os.path.join(img_folder, filename))
print(os.path.join(img_folder, filename), 'done.')
# Save mesh
#mdic = {"faces": face, "vertices": vertices}
#savemat(os.path.join(img_folder, "mesh.mat"), mdic)
#print("mesh.mat saved")
