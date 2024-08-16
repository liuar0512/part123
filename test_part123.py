import argparse

import imageio
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path

import trimesh
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, Callback
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import Trainer
from skimage.io import imsave
from tqdm import tqdm

import mcubes

from ldm.base_utils import read_pickle, output_points
from renderer.renderer import NeuSRenderer, DEFAULT_SIDE_LENGTH
from ldm.util import instantiate_from_config

from sklearn.cluster import KMeans
from sklearn import metrics
import os
import copy

from partnum_predict import estimate_partnum


CLUSTER_COLOR_MAP_40 = {-1: (0., 0., 0.), 0: (174., 199., 232.), 1: (152., 223., 138.), 2: (31., 119., 180.), 3: (255., 187., 120.), 4: (188., 189., 34.), 5: (140., 86., 75.),
                        6: (255., 152., 150.), 7: (214., 39., 40.), 8: (197., 176., 213.), 9: (148., 103., 189.), 10: (196., 156., 148.), 11: (23., 190., 207.), 12: (247., 182., 210.), 
                        13: (219., 219., 141.), 14: (255., 127., 14.), 15: (158., 218., 229.), 16: (44., 160., 44.), 17: (112., 128., 144.), 18: (227., 119., 194.), 19: (82., 84., 163.), 
                        20: (232., 199., 174.), 21: (138., 223., 152.), 22: (180., 119., 31.), 23: (120., 187., 255.), 24: (34., 189., 188.), 25: (75., 86., 140.),
                        26: (150., 152., 255.), 27: (40., 39., 214.), 28: (213., 176., 197.), 29: (189., 103., 148.), 30: (148., 156., 196.), 31: (207., 190., 23.), 32: (210., 182., 247.), 
                        33: (141., 219., 219.), 34: (14., 127., 255.), 35: (229., 218., 158.), 36: (44., 44., 160.), 37: (144., 128., 112.), 38: (194., 119., 227.), 39: (163., 84., 82.)}

def region_growing(mesh, vertex_labels):
    num_vertices = len(mesh.vertices)
    visited = np.zeros(num_vertices, dtype=bool)
    regions = []
    region_labels = np.zeros(num_vertices, dtype=int)
    count = 0
    
    for vertex_idx in range(num_vertices):
        if visited[vertex_idx]:
            continue
        
        label = vertex_labels[vertex_idx]
        stack = [vertex_idx]
        region = []

        while stack:
            current_vertex = stack.pop()
            
            if visited[current_vertex]:
                continue
            
            visited[current_vertex] = True
            region.append(current_vertex)
            
            neighbors = mesh.vertex_neighbors[current_vertex]
            for neighbor in neighbors:
                if not visited[neighbor] and vertex_labels[neighbor] == label:
                    stack.append(neighbor)
        
        if region:
            regions.append(region)
            region_labels[region] = count
            count += 1
    
    return region_labels

def num_to_natural(group_ids):
    '''
    Change the group number to natural number arrangement
    '''
    if np.all(group_ids == -1):
        return group_ids
    array = copy.deepcopy(group_ids)
    unique_values = np.unique(array[array != -1])
    mapping = np.full(np.max(unique_values) + 2, -1)
    mapping[unique_values + 1] = np.arange(len(unique_values))
    array = mapping[array + 1]
    return array


class ResumeCallBacks(Callback):
    def __init__(self):
        pass

    def on_train_start(self, trainer, pl_module):
        pl_module.optimizers().param_groups = pl_module.optimizers()._optimizer.param_groups


def extract_fields(bound_min, bound_max, resolution, query_func, batch_size=64, outside_val=1.0):
    N = batch_size
    X = torch.linspace(bound_min[0], bound_max[0], resolution).split(N)
    Y = torch.linspace(bound_min[1], bound_max[1], resolution).split(N)
    Z = torch.linspace(bound_min[2], bound_max[2], resolution).split(N)

    u = np.zeros([resolution, resolution, resolution], dtype=np.float32)
    with torch.no_grad():
        for xi, xs in enumerate(X):
            for yi, ys in enumerate(Y):
                for zi, zs in enumerate(Z):
                    xx, yy, zz = torch.meshgrid(xs, ys, zs)
                    pts = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1).cuda()
                    val = query_func(pts).detach()
                    outside_mask = torch.norm(pts,dim=-1)>=1.0
                    val[outside_mask]=outside_val
                    val = val.reshape(len(xs), len(ys), len(zs)).cpu().numpy()
                    u[xi * N: xi * N + len(xs), yi * N: yi * N + len(ys), zi * N: zi * N + len(zs)] = val
    return u

def extract_geometry(bound_min, bound_max, resolution, threshold, query_func, color_func, partseg_func, output_dir, outside_val=1.0):
    u = extract_fields(bound_min, bound_max, resolution, query_func, outside_val=outside_val)
    vertices, triangles = mcubes.marching_cubes(u, threshold)
    # change surface normal direction
    triangles_ = np.copy(triangles)
    for idt in range(triangles.shape[0]):
        triangles_[idt][1] = triangles[idt][2]
        triangles_[idt][2] = triangles[idt][1]
    triangles = triangles_
    b_max_np = bound_max.detach().cpu().numpy()
    b_min_np = bound_min.detach().cpu().numpy()

    vertices = vertices / (resolution - 1.0) * (b_max_np - b_min_np)[None, :] + b_min_np[None, :]
    vertex_colors = color_func(vertices.copy())

    # visualize part segmentation
    vertex_partseg_feats = partseg_func(vertices.copy())
    sam_cents_lists = np.load(f'{output_dir}/sam_cents.npz')
    overlap_lists = {'x': 0.1, 'y': 0.15, 'a': 0.2, 'b': 0.25, 'c': 0.3, 'd': 0.35, 'e': 0.4, 'f': 0.45, 'g': 0.5, 'h': 0.55, 'i': 0.6, 'j': 0.7}

    sse_lists = []
    ol_lists = []
    for ol_code in overlap_lists.keys():
        if ol_code in sam_cents_lists:
            ol_lists.append(ol_code)
            sam_cents = sam_cents_lists[ol_code]
            sam_cents_mesh = np.copy(sam_cents)
            # find nearest center points among mesh vertices
            cluster_nums = sam_cents.shape[0]
            for idx in range(cluster_nums):
                dist_mat = np.linalg.norm(vertices - sam_cents[idx].reshape(-1,3), axis=1)  # n,
                indice = np.argmin(dist_mat)
                sam_cents_mesh[idx] = vertices[indice]
            init_cluscents = partseg_func(sam_cents_mesh.copy())
            
            y_tmp = KMeans(n_clusters=cluster_nums, init=init_cluscents, random_state=9, n_init=1).fit_predict(vertex_partseg_feats.copy())
            sse_tmp = metrics.davies_bouldin_score(vertex_partseg_feats.copy(), y_tmp)
            
            sse_lists.append(sse_tmp)
    
    cid = np.argmin(np.array(sse_lists))
    cid = ol_lists[cid]
    

    # finetune with isolated unsolid parts
    num_solid_parts = int(sam_cents_lists[cid + '_sol'][0])
    sam_cents = sam_cents_lists[cid]
    if num_solid_parts < sam_cents.shape[0]:
        sam_cents_mesh = np.copy(sam_cents)
        for idx in range(sam_cents.shape[0]):
            dist_mat = np.linalg.norm(vertices - sam_cents[idx].reshape(-1,3), axis=1)  # n,
            indice = np.argmin(dist_mat)
            sam_cents_mesh[idx] = vertices[indice]
        init_cluscents = partseg_func(sam_cents_mesh.copy())
        sse_ft_lists = []
        for idx in range(num_solid_parts, sam_cents.shape[0]+1):
            y_tmp = KMeans(n_clusters=idx, init=init_cluscents[0:idx], random_state=9, n_init=1).fit_predict(vertex_partseg_feats.copy())
            sse_tmp = metrics.davies_bouldin_score(vertex_partseg_feats.copy(), y_tmp)
            sse_ft_lists.append(sse_tmp)
        cid_ft = np.argmin(np.array(sse_ft_lists))
        ft_ind = num_solid_parts + cid_ft
    else:
        ft_ind = sam_cents.shape[0]


    # use the optimal part number to generate final outputs
    sam_cents = sam_cents_lists[cid][0:ft_ind]
    
    cluster_nums = ft_ind
    for idx in range(cluster_nums):
        dist_mat = np.linalg.norm(vertices - sam_cents[idx].reshape(-1,3), axis=1)  # n,
        indice = np.argmin(dist_mat)
        sam_cents[idx] = vertices[indice]
    init_cluscents = partseg_func(sam_cents.copy())

    
    y_pred = KMeans(n_clusters=cluster_nums, init=init_cluscents, random_state=9, n_init=1).fit_predict(vertex_partseg_feats.copy())
    
    partseg_colors = np.array([CLUSTER_COLOR_MAP_40[label % 40] for label in y_pred.copy()])
    partseg_colors = partseg_colors / 255

    return vertices, triangles, vertex_colors, partseg_colors, y_pred

def extract_mesh(model, output, obj_name, resolution=512):
    if not isinstance(model.renderer, NeuSRenderer): return
    
    bbox_min = -torch.ones(3)*DEFAULT_SIDE_LENGTH
    bbox_max = torch.ones(3)*DEFAULT_SIDE_LENGTH
    with torch.no_grad():
        vertices, triangles, vertex_colors, partseg_colors, label_pred = extract_geometry(bbox_min, bbox_max, resolution, 0, lambda x: model.renderer.sdf_network.sdf(x), lambda x: model.renderer.get_vertex_colors(x), lambda x: model.renderer.get_vertex_partseg_feats(x), output)

    # # output geometry
    # mesh = trimesh.Trimesh(vertices, triangles, vertex_colors=vertex_colors)
    # mesh.export(str(f'{output}/test_mesh.ply'))

    if True:
        mesh = trimesh.Trimesh(vertices.copy(), triangles.copy(), process=False)

        label_pred = region_growing(mesh, label_pred)
        label_pred = num_to_natural(label_pred)
        partseg_colors = np.array([CLUSTER_COLOR_MAP_40[label % 40] for label in label_pred.copy()])

        mesh_vis = trimesh.Trimesh(vertices, triangles, vertex_colors=partseg_colors.astype(np.uint8), process=False)
        mesh_vis.export(str(f'{output}/samauto_vis.ply'))
        
    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--image_path', type=str, required=True)
    parser.add_argument('-n', '--name', type=str, required=True)
    parser.add_argument('-b', '--base', type=str, default='configs/neus.yaml')
    parser.add_argument('-l', '--log', type=str, default='output/renderer')
    parser.add_argument('-s', '--seed', type=int, default=6033)
    parser.add_argument('-g', '--gpus', type=str, default='0,')
    parser.add_argument('-r', '--resume', action='store_true', default=False, dest='resume')
    parser.add_argument('--fp16', action='store_true', default=False, dest='fp16')
    opt = parser.parse_args()
    # seed_everything(opt.seed)

    estimate_partnum(opt.name)

    # configs
    cfg = OmegaConf.load(opt.base)
    name = opt.name
    log_dir, ckpt_dir = Path(opt.log) / name, Path(opt.log) / name / 'ckpt'
    cfg.model.params['image_path'] = opt.image_path
    cfg.model.params['log_dir'] = log_dir

    # setup
    log_dir.mkdir(exist_ok=True, parents=True)
    ckpt_dir.mkdir(exist_ok=True, parents=True)
    trainer_config = cfg.trainer
    callback_config = cfg.callbacks
    model_config = cfg.model
    data_config = cfg.data

    data_config.params.seed = opt.seed
    data = instantiate_from_config(data_config)
    data.prepare_data()
    data.setup('fit')

    model = instantiate_from_config(model_config,)
    model.cpu()
    model.learning_rate = model_config.base_lr

    # logger
    logger = TensorBoardLogger(save_dir=log_dir, name='tensorboard_logs')
    callbacks=[]
    callbacks.append(LearningRateMonitor(logging_interval='step'))
    callbacks.append(ModelCheckpoint(dirpath=ckpt_dir, filename="{epoch:06}", verbose=True, save_last=True, every_n_train_steps=callback_config.save_interval))

    # trainer
    trainer_config.update({
        "accelerator": "cuda", "check_val_every_n_epoch": None,
        "benchmark": True, "num_sanity_val_steps": 0,
        "devices": 1, "gpus": opt.gpus,
    })
    if opt.fp16:
        trainer_config['precision']=16

    if opt.resume:
        callbacks.append(ResumeCallBacks())
        trainer_config['resume_from_checkpoint'] = str(ckpt_dir / 'last.ckpt')
    else:
        if (ckpt_dir / 'last.ckpt').exists():
            raise RuntimeError(f"checkpoint {ckpt_dir / 'last.ckpt'} existing ...")
    trainer = Trainer.from_argparse_args(args=argparse.Namespace(), **trainer_config, logger=logger, callbacks=callbacks)

    trainer.fit(model, data)

    model = model.cuda().eval()

    extract_mesh(model, log_dir, name, resolution=256)

if __name__=="__main__":
    main()