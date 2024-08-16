import abc
import os
from pathlib import Path

import cv2
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf

from skimage.io import imread, imsave
from PIL import Image
from torch.optim.lr_scheduler import LambdaLR

from ldm.base_utils import read_pickle, concat_images_list
from renderer.neus_networks import SDFNetwork, RenderingNetwork, SingleVarianceNetwork, SDFHashGridNetwork, RenderingFFNetwork, PartSegmentNetwork
from renderer.ngp_renderer import NGPNetwork
from ldm.util import instantiate_from_config
from info_nce import InfoNCE
import copy
import cc3d


DEFAULT_RADIUS = np.sqrt(3)/2
DEFAULT_SIDE_LENGTH = 0.6

COLOR_MAP_20 = {-1: (0., 0., 0.), 0: (174., 199., 232.), 1: (152., 223., 138.), 2: (31., 119., 180.), 3: (255., 187., 120.), 4: (188., 189., 34.), 5: (140., 86., 75.),
                        6: (255., 152., 150.), 7: (214., 39., 40.), 8: (197., 176., 213.), 9: (148., 103., 189.), 10: (196., 156., 148.), 11: (23., 190., 207.), 12: (247., 182., 210.), 
                        13: (219., 219., 141.), 14: (255., 127., 14.), 15: (158., 218., 229.), 16: (44., 160., 44.), 17: (112., 128., 144.), 18: (227., 119., 194.), 19: (82., 84., 163.)}

# borrowed from SAM3D
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

def remove_small_group(group_ids, th):
    unique_elements, counts = np.unique(group_ids, return_counts=True)
    result = group_ids.copy()
    for i, count in enumerate(counts):
        if count < th:
            result[group_ids == unique_elements[i]] = -1
    
    return result


def sample_pdf(bins, weights, n_samples, det=True):
    device = bins.device
    dtype = bins.dtype
    # This implementation is from NeRF
    # Get pdf
    weights = weights + 1e-5  # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)
    # Take uniform samples
    if det:
        u = torch.linspace(0. + 0.5 / n_samples, 1. - 0.5 / n_samples, steps=n_samples, dtype=dtype, device=device)
        u = u.expand(list(cdf.shape[:-1]) + [n_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [n_samples], dtype=dtype, device=device)

    # Invert CDF
    u = u.contiguous()
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds - 1), inds - 1)
    above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[..., 1] - cdf_g[..., 0])
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

    return samples

def near_far_from_sphere(rays_o, rays_d, radius=DEFAULT_RADIUS):
    a = torch.sum(rays_d ** 2, dim=-1, keepdim=True)
    b = torch.sum(rays_o * rays_d, dim=-1, keepdim=True)
    mid = -b / a
    near = mid - radius
    far = mid + radius
    return near, far



class BackgroundRemoval:
    def __init__(self, device='cuda'):
        from carvekit.api.high import HiInterface
        self.interface = HiInterface(
            object_type="object",  # Can be "object" or "hairs-like".
            batch_size_seg=5,
            batch_size_matting=1,
            device=device,
            seg_mask_size=640,  # Use 640 for Tracer B7 and 320 for U2Net
            matting_mask_size=2048,
            trimap_prob_threshold=231,
            trimap_dilation=30,
            trimap_erosion_iters=5,
            fp16=True,
        )

    @torch.no_grad()
    def __call__(self, image):
        # image: [H, W, 3] array in [0, 255].
        image = Image.fromarray(image)
        image = self.interface([image])[0]
        image = np.array(image)
        return image


class BaseRenderer(nn.Module):
    def __init__(self, train_batch_num, test_batch_num):
        super().__init__()
        self.train_batch_num = train_batch_num
        self.test_batch_num = test_batch_num

    @abc.abstractmethod
    def render_impl(self, ray_batch, is_train, step):
        pass

    @abc.abstractmethod
    def render_with_loss(self, ray_batch, is_train, step):
        pass

    def render(self, ray_batch, is_train, step):
        batch_num = self.train_batch_num if is_train else self.test_batch_num
        ray_num = ray_batch['rays_o'].shape[0]
        outputs = {}
        for ri in range(0, ray_num, batch_num):
            cur_ray_batch = {}
            for k, v in ray_batch.items():
                cur_ray_batch[k] = v[ri:ri + batch_num]
            cur_outputs = self.render_impl(cur_ray_batch, is_train, step)
            for k, v in cur_outputs.items():
                if k not in outputs: outputs[k] = []
                outputs[k].append(v)

        for k, v in outputs.items():
            outputs[k] = torch.cat(v, 0)
        return outputs


class NeuSRenderer(BaseRenderer):
    def __init__(self, train_batch_num, test_batch_num, lambda_eikonal_loss=0.1, use_mask=True,
                 lambda_rgb_loss=1.0, lambda_mask_loss=0.0, lambda_contra_loss=0.02, mvgen_backbone='syncdreamer', rgb_loss='soft_l1', coarse_sn=64, fine_sn=64):
        super().__init__(train_batch_num, test_batch_num)
        self.n_samples = coarse_sn
        self.n_importance = fine_sn
        self.up_sample_steps = 4
        self.anneal_end = 200
        self.use_mask = use_mask
        self.lambda_eikonal_loss = lambda_eikonal_loss
        self.lambda_rgb_loss = lambda_rgb_loss
        self.lambda_mask_loss = lambda_mask_loss
        self.lambda_contra_loss = lambda_contra_loss
        self.mvgen_backbone = mvgen_backbone
        self.rgb_loss = rgb_loss

        print(f"Set contrastive loss weight to : {self.lambda_contra_loss}")

        self.sdf_network = SDFNetwork(d_out=257, d_in=3, d_hidden=256, n_layers=8, skip_in=[4], multires=6, bias=0.5, scale=1.0, geometric_init=True, weight_norm=True)
        self.color_network = RenderingNetwork(d_feature=256, d_in=9, d_out=3, d_hidden=256, n_layers=4, weight_norm=True, multires_view=4, squeeze_out=True)
        self.default_dtype = torch.float32
        self.deviation_network = SingleVarianceNetwork(0.3)

        # part-segmentation net
        self.dim_partseg_feat = 16
        self.partseg_network = PartSegmentNetwork(d_feature=256, d_in=3, d_out=self.dim_partseg_feat)
        self.infonce_loss = InfoNCE(negative_mode='paired')

    @torch.no_grad()
    def get_vertex_colors(self, vertices):
        """
        @param vertices:  n,3
        @return:
        """
        V = vertices.shape[0]
        bn = 20480
        verts_colors = []
        with torch.no_grad():
            for vi in range(0, V, bn):
                verts = torch.from_numpy(vertices[vi:vi+bn].astype(np.float32)).cuda()
                feats = self.sdf_network(verts)[..., 1:]
                gradients = self.sdf_network.gradient(verts)  # ...,3
                gradients = F.normalize(gradients, dim=-1)
                colors = self.color_network(verts, gradients, gradients, feats)
                colors = torch.clamp(colors,min=0,max=1).cpu().numpy()
                verts_colors.append(colors)

        verts_colors = (np.concatenate(verts_colors, 0)*255).astype(np.uint8)
        return verts_colors
    
    @torch.no_grad()
    def get_vertex_partseg_feats(self, vertices):
        """
        @param vertices:  n,3
        @return:
        """
        V = vertices.shape[0]
        bn = 20480
        verts_colors = []
        with torch.no_grad():
            for vi in range(0, V, bn):
                verts = torch.from_numpy(vertices[vi:vi+bn].astype(np.float32)).cuda()
                # feats = self.sdf_network(verts)[..., 1:]
                feats, midf = self.sdf_network(verts, need_midf=True)
                partseg_feats = self.partseg_network(verts, midf)
                partseg_feats = F.normalize(partseg_feats, dim=-1)
                # partseg_feats = feats
                partseg_feats = partseg_feats.cpu().numpy()
                verts_colors.append(partseg_feats)

        verts_colors = np.concatenate(verts_colors, 0)
        return verts_colors

    def upsample(self, rays_o, rays_d, z_vals, sdf, n_importance, inv_s):
        """
        Up sampling give a fixed inv_s
        """
        device = rays_o.device
        batch_size, n_samples = z_vals.shape
        pts = rays_o[:, None, :] + rays_d[:, None, :] * z_vals[..., :, None]  # n_rays, n_samples, 3
        inner_mask = self.get_inner_mask(pts)
        # radius = torch.linalg.norm(pts, ord=2, dim=-1, keepdim=False)
        inside_sphere = inner_mask[:, :-1] | inner_mask[:, 1:]
        sdf = sdf.reshape(batch_size, n_samples)
        prev_sdf, next_sdf = sdf[:, :-1], sdf[:, 1:]
        prev_z_vals, next_z_vals = z_vals[:, :-1], z_vals[:, 1:]
        mid_sdf = (prev_sdf + next_sdf) * 0.5
        cos_val = (next_sdf - prev_sdf) / (next_z_vals - prev_z_vals + 1e-5)

        prev_cos_val = torch.cat([torch.zeros([batch_size, 1], dtype=self.default_dtype, device=device), cos_val[:, :-1]], dim=-1)
        cos_val = torch.stack([prev_cos_val, cos_val], dim=-1)
        cos_val, _ = torch.min(cos_val, dim=-1, keepdim=False)
        cos_val = cos_val.clip(-1e3, 0.0) * inside_sphere

        dist = (next_z_vals - prev_z_vals)
        prev_esti_sdf = mid_sdf - cos_val * dist * 0.5
        next_esti_sdf = mid_sdf + cos_val * dist * 0.5
        prev_cdf = torch.sigmoid(prev_esti_sdf * inv_s)
        next_cdf = torch.sigmoid(next_esti_sdf * inv_s)
        alpha = (prev_cdf - next_cdf + 1e-5) / (prev_cdf + 1e-5)
        weights = alpha * torch.cumprod(
            torch.cat([torch.ones([batch_size, 1], dtype=self.default_dtype, device=device), 1. - alpha + 1e-7], -1), -1)[:, :-1]

        z_samples = sample_pdf(z_vals, weights, n_importance, det=True).detach()
        return z_samples

    def cat_z_vals(self, rays_o, rays_d, z_vals, new_z_vals, sdf, last=False):
        batch_size, n_samples = z_vals.shape
        _, n_importance = new_z_vals.shape
        pts = rays_o[:, None, :] + rays_d[:, None, :] * new_z_vals[..., :, None]
        z_vals = torch.cat([z_vals, new_z_vals], dim=-1)
        z_vals, index = torch.sort(z_vals, dim=-1)

        if not last:
            device = pts.device
            new_sdf = self.sdf_network.sdf(pts.reshape(-1, 3)).reshape(batch_size, n_importance)
            sdf = torch.cat([sdf, new_sdf], dim=-1)
            xx = torch.arange(batch_size)[:, None].expand(batch_size, n_samples + n_importance).reshape(-1).to(device)
            index = index.reshape(-1)
            sdf = sdf[(xx, index)].reshape(batch_size, n_samples + n_importance)

        return z_vals, sdf

    def sample_depth(self, rays_o, rays_d, near, far, perturb):
        n_samples = self.n_samples
        n_importance = self.n_importance
        up_sample_steps = self.up_sample_steps
        device = rays_o.device

        # sample points
        batch_size = len(rays_o)
        z_vals = torch.linspace(0.0, 1.0, n_samples, dtype=self.default_dtype, device=device)   # sn
        z_vals = near + (far - near) * z_vals[None, :]            # rn,sn

        if perturb > 0:
            t_rand = (torch.rand([batch_size, 1]).to(device) - 0.5)
            z_vals = z_vals + t_rand * 2.0 / n_samples

        # Up sample
        with torch.no_grad():
            pts = rays_o[:, None, :] + rays_d[:, None, :] * z_vals[..., :, None]
            sdf = self.sdf_network.sdf(pts).reshape(batch_size, n_samples)

            for i in range(up_sample_steps):
                rn, sn = z_vals.shape
                inv_s = torch.ones(rn, sn - 1, dtype=self.default_dtype, device=device) * 64 * 2 ** i
                new_z_vals = self.upsample(rays_o, rays_d, z_vals, sdf, n_importance // up_sample_steps, inv_s)
                z_vals, sdf = self.cat_z_vals(rays_o, rays_d, z_vals, new_z_vals, sdf, last=(i + 1 == up_sample_steps))

        return z_vals

    def compute_sdf_alpha(self, points, dists, dirs, cos_anneal_ratio, step):
        # points [...,3] dists [...] dirs[...,3]
        # sdf_nn_output = self.sdf_network(points)
        sdf_nn_output, sdf_midf = self.sdf_network(points, need_midf=True)
        sdf = sdf_nn_output[..., 0]
        feature_vector = sdf_nn_output[..., 1:]

        gradients = self.sdf_network.gradient(points)  # ...,3
        inv_s = self.deviation_network(points).clip(1e-6, 1e6)  # ...,1
        inv_s = inv_s[..., 0]

        true_cos = (dirs * gradients).sum(-1)  # [...]
        iter_cos = -(F.relu(-true_cos * 0.5 + 0.5) * (1.0 - cos_anneal_ratio) +
                     F.relu(-true_cos) * cos_anneal_ratio)  # always non-positive

        # Estimate signed distances at section points
        estimated_next_sdf = sdf + iter_cos * dists * 0.5
        estimated_prev_sdf = sdf - iter_cos * dists * 0.5

        prev_cdf = torch.sigmoid(estimated_prev_sdf * inv_s)
        next_cdf = torch.sigmoid(estimated_next_sdf * inv_s)

        p = prev_cdf - next_cdf
        c = prev_cdf

        alpha = ((p + 1e-5) / (c + 1e-5)).clip(0.0, 1.0)  # [...]
        return alpha, gradients, feature_vector, inv_s, sdf, sdf_midf

    def get_anneal_val(self, step):
        if self.anneal_end < 0:
            return 1.0
        else:
            return np.min([1.0, step / self.anneal_end])

    def get_inner_mask(self, points):
        return torch.sum(torch.abs(points)<=DEFAULT_SIDE_LENGTH,-1)==3
    

    def render_impl(self, ray_batch, is_train, step):
        near, far = near_far_from_sphere(ray_batch['rays_o'], ray_batch['rays_d'])
        rays_o, rays_d = ray_batch['rays_o'], ray_batch['rays_d']
        z_vals = self.sample_depth(rays_o, rays_d, near, far, is_train)

        batch_size, n_samples = z_vals.shape

        # section length in original space
        dists = z_vals[..., 1:] - z_vals[..., :-1]  # rn,sn-1
        dists = torch.cat([dists, dists[..., -1:]], -1)  # rn,sn
        mid_z_vals = z_vals + dists * 0.5

        points = rays_o.unsqueeze(-2) + rays_d.unsqueeze(-2) * mid_z_vals.unsqueeze(-1) # rn, sn, 3
        inner_mask = self.get_inner_mask(points)

        dirs = rays_d.unsqueeze(-2).expand(batch_size, n_samples, 3)
        dirs = F.normalize(dirs, dim=-1)
        device = rays_o.device
        alpha, sampled_color, gradient_error, normal = torch.zeros(batch_size, n_samples, dtype=self.default_dtype, device=device), \
            torch.zeros(batch_size, n_samples, 3, dtype=self.default_dtype, device=device), \
            torch.zeros([batch_size, n_samples], dtype=self.default_dtype, device=device), \
            torch.zeros([batch_size, n_samples, 3], dtype=self.default_dtype, device=device)
        # part-segment 
        sampled_partseg = torch.zeros([batch_size, n_samples, self.dim_partseg_feat], dtype=self.default_dtype, device=device)
        if torch.sum(inner_mask) > 0:
            cos_anneal_ratio = self.get_anneal_val(step) if is_train else 1.0
            alpha[inner_mask], gradients, feature_vector, inv_s, sdf, sdf_midf = self.compute_sdf_alpha(points[inner_mask], dists[inner_mask], dirs[inner_mask], cos_anneal_ratio, step)
            sampled_color[inner_mask] = self.color_network(points[inner_mask], gradients, -dirs[inner_mask], feature_vector)
            # Eikonal loss
            gradient_error[inner_mask] = (torch.linalg.norm(gradients, ord=2, dim=-1) - 1.0) ** 2 # rn,sn
            normal[inner_mask] = F.normalize(gradients, dim=-1)

            # part-segment 
            # sampled_partseg[inner_mask] = self.partseg_network(points[inner_mask], feature_vector.clone().detach())
            # sampled_partseg[inner_mask] = self.partseg_network(points[inner_mask], feature_vector)
            sampled_partseg[inner_mask] = self.partseg_network(points[inner_mask], sdf_midf)

        weights = alpha * torch.cumprod(torch.cat([torch.ones([batch_size, 1], dtype=self.default_dtype, device=device), 1. - alpha + 1e-7], -1), -1)[..., :-1]  # rn,sn
        mask = torch.sum(weights,dim=1).unsqueeze(-1) # rn,1
        color = (sampled_color * weights[..., None]).sum(dim=1) + (1 - mask) # add white background
        normal = (normal * weights[..., None]).sum(dim=1)

        # generate depth maps for SAM centers prediction
        if 'rays_d_unnorm' in ray_batch:
            rays_d_unnorm = ray_batch['rays_d_unnorm']
            mid_z_vals_ = mid_z_vals * torch.norm(rays_d[...,None,:], dim=-1) / torch.norm(rays_d_unnorm[...,None,:], dim=-1)
            depth_map = torch.sum(weights * mid_z_vals_, -1) # [N_rays, 1]
        else:
            depth_map = mask * 0.0   # use 0-map for placeholder

        # part-segment rendering
        weights_partseg = weights.clone().detach()
        partseg_feature = (sampled_partseg * weights_partseg[..., None]).sum(dim=1)

        outputs = {
            'rgb': color,  # rn,3
            'gradient_error': gradient_error,  # rn,sn
            'inner_mask': inner_mask,  # rn,sn
            'normal': normal,  # rn,3
            'mask': mask,  # rn,1
            'partseg': partseg_feature,  # rn,dim_feat
            'depth': depth_map, # rn,1
        }

        return outputs

    def render_with_loss(self, ray_batch, is_train, step, num_anc, fg_start, neg_pos_ratio):
        render_outputs = self.render(ray_batch, is_train, step)

        rgb_gt = ray_batch['rgb']
        rgb_pr = render_outputs['rgb']
        if self.rgb_loss == 'soft_l1':
            epsilon = 0.001
            rgb_loss = torch.sqrt(torch.sum((rgb_gt - rgb_pr) ** 2, dim=-1) + epsilon)
        elif self.rgb_loss =='mse':
            rgb_loss = F.mse_loss(rgb_pr, rgb_gt, reduction='none')
        else:
            raise NotImplementedError
        rgb_loss = torch.mean(rgb_loss)

        # contrastive learning loss, InfoNCE
        part_feat = render_outputs['partseg']
        neg_samples = part_feat[fg_start+2*num_anc : fg_start+(neg_pos_ratio+2)*num_anc].reshape(num_anc, neg_pos_ratio, -1)
        contra_loss = self.infonce_loss(part_feat[fg_start : fg_start+num_anc], part_feat[fg_start+num_anc : fg_start+2*num_anc], neg_samples.contiguous())
        

        eikonal_loss = torch.sum(render_outputs['gradient_error'] * render_outputs['inner_mask']) / torch.sum(render_outputs['inner_mask'] + 1e-5)
        
        if step < 250:
            loss = rgb_loss * self.lambda_rgb_loss + eikonal_loss * self.lambda_eikonal_loss
        else:
            loss = rgb_loss * self.lambda_rgb_loss + eikonal_loss * self.lambda_eikonal_loss + contra_loss * self.lambda_contra_loss
                
        loss_batch = {
            'eikonal': eikonal_loss,
            'rendering': rgb_loss,
            # 'mask': mask_loss,
            'contrastive': contra_loss,
        }
        if self.lambda_mask_loss>0 and self.use_mask:
            mask_loss = F.mse_loss(render_outputs['mask'], ray_batch['mask'], reduction='none').mean()
            loss += mask_loss * self.lambda_mask_loss
            loss_batch['mask'] = mask_loss
        return loss, loss_batch


class NeRFRenderer(BaseRenderer):
    def __init__(self, train_batch_num, test_batch_num, bound=0.5, use_mask=False, lambda_rgb_loss=1.0, lambda_mask_loss=0.0):
        super().__init__(train_batch_num, test_batch_num)
        self.train_batch_num = train_batch_num
        self.test_batch_num = test_batch_num
        self.use_mask = use_mask
        self.field = NGPNetwork(bound=bound)

        self.update_interval = 16
        self.fp16 = True
        self.lambda_rgb_loss = lambda_rgb_loss
        self.lambda_mask_loss = lambda_mask_loss

    def render_impl(self, ray_batch, is_train, step):
        rays_o, rays_d = ray_batch['rays_o'], ray_batch['rays_d']
        with torch.cuda.amp.autocast(enabled=self.fp16):
            if step % self.update_interval==0:
                self.field.update_extra_state()

            outputs = self.field.render(rays_o, rays_d,)

        renderings={
            'rgb': outputs['image'],
            'depth': outputs['depth'],
            'mask': outputs['weights_sum'].unsqueeze(-1),
        }
        return renderings

    def render_with_loss(self, ray_batch, is_train, step):
        render_outputs = self.render(ray_batch, is_train, step)

        rgb_gt = ray_batch['rgb']
        rgb_pr = render_outputs['rgb']
        epsilon = 0.001
        rgb_loss = torch.sqrt(torch.sum((rgb_gt - rgb_pr) ** 2, dim=-1) + epsilon)
        rgb_loss = torch.mean(rgb_loss)
        loss = rgb_loss * self.lambda_rgb_loss
        loss_batch = {'rendering': rgb_loss}

        if self.use_mask:
            mask_loss = F.mse_loss(render_outputs['mask'], ray_batch['mask'], reduction='none')
            mask_loss = torch.mean(mask_loss)
            loss = loss + mask_loss * self.lambda_mask_loss
            loss_batch['mask'] = mask_loss
        return loss, loss_batch



class RendererTrainer(pl.LightningModule):
    def __init__(self, image_path, total_steps, warm_up_steps, log_dir, train_batch_fg_num=0,
                 use_cube_feats=False, cube_ckpt=None, cube_cfg=None, cube_bound=0.5,
                 train_batch_num=4096, test_batch_num=8192, use_warm_up=True, use_mask=True,
                 lambda_rgb_loss=1.0, lambda_mask_loss=0.0, renderer='neus', lambda_contra_loss=0.02,
                 # used for different backbones
                 mvbackbone='syncdreamer', num_mvimgs=16,  
                 # used in neus
                 lambda_eikonal_loss=0.1,
                 coarse_sn=64, fine_sn=64):
        super().__init__()
        # self.num_images = 16
        self.num_images = num_mvimgs
        self.mvbackbone = mvbackbone
        self.image_size = 256
        self.log_dir = log_dir
        (Path(log_dir)/'images').mkdir(exist_ok=True, parents=True)
        self.train_batch_num = train_batch_num
        self.train_batch_fg_num = train_batch_fg_num
        self.test_batch_num = test_batch_num
        self.image_path = image_path
        self.total_steps = total_steps
        self.warm_up_steps = warm_up_steps
        self.use_mask = use_mask
        self.lambda_eikonal_loss = lambda_eikonal_loss
        self.lambda_rgb_loss = lambda_rgb_loss
        self.lambda_mask_loss = lambda_mask_loss
        self.lambda_contra_loss = lambda_contra_loss
        self.use_warm_up = use_warm_up
        

        self.use_cube_feats, self.cube_cfg, self.cube_ckpt = use_cube_feats, cube_cfg, cube_ckpt

        self._init_dataset()
        if renderer=='neus':
            self.renderer = NeuSRenderer(train_batch_num, test_batch_num,
                                         lambda_rgb_loss=lambda_rgb_loss,
                                         lambda_eikonal_loss=lambda_eikonal_loss,
                                         lambda_mask_loss=lambda_mask_loss,
                                         lambda_contra_loss=lambda_contra_loss,
                                         mvgen_backbone=mvbackbone,
                                         coarse_sn=coarse_sn, fine_sn=fine_sn)
        elif renderer=='ngp':
            self.renderer = NeRFRenderer(train_batch_num, test_batch_num, bound=cube_bound, use_mask=use_mask, lambda_mask_loss=lambda_mask_loss, lambda_rgb_loss=lambda_rgb_loss,)
        else:
            raise NotImplementedError
        self.validation_index = 0


    def _construct_ray_batch(self, images_info):
        image_num = images_info['images'].shape[0]
        _, h, w, _ = images_info['images'].shape
        coords = torch.stack(torch.meshgrid(torch.arange(h), torch.arange(w)), -1)[:, :, (1, 0)]  # h,w,2
        coords = coords.float()[None, :, :, :].repeat(image_num, 1, 1, 1)  # imn,h,w,2
        coords = coords.reshape(image_num, h * w, 2)
        coords = torch.cat([coords, torch.ones(image_num, h * w, 1, dtype=torch.float32)], 2)  # imn,h*w,3

        # imn,h*w,3 @ imn,3,3 => imn,h*w,3
        rays_d = coords @ torch.inverse(images_info['Ks']).permute(0, 2, 1)
        poses = images_info['poses']  # imn,3,4
        R, t = poses[:, :, :3], poses[:, :, 3:]
        rays_d = rays_d @ R
        rays_d = F.normalize(rays_d, dim=-1)
        rays_o = -R.permute(0,2,1) @ t # imn,3,3 @ imn,3,1
        rays_o = rays_o.permute(0, 2, 1).repeat(1, h*w, 1) # imn,h*w,3

        ray_batch = {
            'rgb': images_info['images'].reshape(image_num*h*w,3),
            'mask': images_info['masks'].reshape(image_num*h*w,1),
            'rays_o': rays_o.reshape(image_num*h*w,3).float(),
            'rays_d': rays_d.reshape(image_num*h*w,3).float(),
            # part-seg mask load
            'sam_mask': images_info['sam_masks'].reshape(image_num*h*w,1),
        }
        return ray_batch

    @staticmethod
    def load_model(cfg, ckpt):
        config = OmegaConf.load(cfg)
        model = instantiate_from_config(config.model)
        print(f'loading model from {ckpt} ...')
        ckpt = torch.load(ckpt)
        model.load_state_dict(ckpt['state_dict'])
        model = model.cuda().eval()
        return model

    def _init_dataset(self):
        mask_predictor = BackgroundRemoval()
        self.K, self.azs, self.els, self.dists, self.poses = read_pickle(f'meta_info/camera-{self.num_images}.pkl')

        self.images_info = {'images': [] ,'masks': [], 'Ks': [], 'poses':[], 'sam_masks': []}  # add part-segment 2D mask

        img = imread(self.image_path)

        # part-seg, sam mask load
        sam_mask = np.load(self.image_path.replace('.png', '.npy'))  
        self.img_part_pairs = []  # ni*np,
        self.img_part_info = []  # ni,
        self.img_pairs = []  # ni,
        vis_sam_masks = []

        for index in range(self.num_images):
            rgb = np.copy(img[:,index*self.image_size:(index+1)*self.image_size,:])
            # predict mask
            if self.use_mask:
                imsave(f'{self.log_dir}/input-{index}.png', rgb)
                masked_image = mask_predictor(rgb)
                imsave(f'{self.log_dir}/masked-{index}.png', masked_image)
                mask = masked_image[:,:,3].astype(np.float32)/255
            else:
                h, w, _ = rgb.shape
                mask = np.zeros([h,w], np.float32)
            
            # part-seg, sam mask load
            sam_singl_mask = np.copy(sam_mask[:,index*self.image_size:(index+1)*self.image_size])  
            sam_singl_mask = num_to_natural(sam_singl_mask)  # remove index with no exact pixels, caused by sam mask overlapping
            # detect disconnected parts to remove noisy small pixel groups
            labs_connected = cc3d.connected_components(sam_singl_mask + 1)
            sam_singl_mask_new = -1 * np.ones_like(sam_singl_mask)
            extra_ind = np.max(sam_singl_mask)+1
            for idp in range(np.min(sam_singl_mask), np.max(sam_singl_mask)+1):
                cur_map = labs_connected[sam_singl_mask==idp]
                unique_values = np.unique(cur_map)
                unique_nums = np.bincount(cur_map)
                if len(unique_values) == 1:
                    sam_singl_mask_new[sam_singl_mask==idp] = idp
                elif len(unique_values) > 1 and np.max(unique_nums) > 19:   # preserve all disconnected large enough subregions
                    valid_maskid = np.argmax(unique_nums)
                    sam_singl_mask_new[labs_connected==valid_maskid] = idp
            bg_maskid = np.unique(sam_singl_mask_new[mask < 0.5])
            
            for idm in range(bg_maskid.shape[0]):
                if 0.9*np.sum(sam_singl_mask_new==bg_maskid[idm]) < np.sum(sam_singl_mask_new[mask < 0.5]==bg_maskid[idm]):
                    sam_singl_mask_new[sam_singl_mask_new==bg_maskid[idm]] = -1
            sam_singl_mask_new[mask < 0.5] = -1  # set background as invalid mask
            sam_singl_mask_new = remove_small_group(sam_singl_mask_new, 20)
            sam_singl_mask = num_to_natural(sam_singl_mask_new)
            self.img_part_info.append([np.max([np.min(sam_singl_mask), 0]), np.max(sam_singl_mask)])
            
            # SAM mask visualization (for checking bg removal)
            img_color = np.zeros((sam_singl_mask.shape[0], sam_singl_mask.shape[1], 3))
            for idp in range(0, np.max(sam_singl_mask)+1):
                color_mask = np.array(COLOR_MAP_20[idp % 20])
                img_color[sam_singl_mask==idp] = color_mask
            vis_sam_masks.append(img_color.astype(np.uint8))
            ##################################################

            rgb = rgb.astype(np.float32)/255
            K, pose = np.copy(self.K), self.poses[index]
            self.images_info['images'].append(torch.from_numpy(rgb.astype(np.float32))) # h,w,3
            self.images_info['masks'].append(torch.from_numpy(mask.astype(np.float32))) # h,w
            self.images_info['Ks'].append(torch.from_numpy(K.astype(np.float32)))
            self.images_info['poses'].append(torch.from_numpy(pose.astype(np.float32)))
            # part-seg, sam mask load
            self.images_info['sam_masks'].append(torch.from_numpy(sam_singl_mask.astype(np.float32))) # h,w

        
        # SAM mask visualization: saving to log dir
        vis_sam_masks = np.concatenate(vis_sam_masks, axis=1)
        cv2.imwrite(f'{self.log_dir}/vis_sam_mask.png', vis_sam_masks[...,[2,1,0]])

        for k, v in self.images_info.items(): self.images_info[k] = torch.stack(v, 0) # stack all values

        self.pose_all = torch.from_numpy(self.poses.astype(np.float32)).cpu()
        self.train_batch = self._construct_ray_batch(self.images_info)
        self.train_ray_num = self.num_images * self.image_size ** 2


        # group foreground rays according to part-segs predicted by sam 2D masks
        grouped_ray_batch = {'rgb': [], 'mask': [], 'rays_o': [], 'rays_d': [], 'part_id': []}
        
        img_h, img_w, _ = rgb.shape
        accum_head, accum_part_head = 0, 0
        part_count = 0
        for index in range(self.num_images):
            cur_start, cur_tail = index*img_h*img_w, (index+1)*img_h*img_w
            if self.img_part_info[index][0] < self.img_part_info[index][1]:   # ignore views that only have one fg part
                for idp in range(self.img_part_info[index][0], self.img_part_info[index][1]+1):
                    cur_sam_mask = torch.sum(self.train_batch['sam_mask'][cur_start:cur_tail,:],1)==idp
                    self.img_part_pairs.append([index, idp, accum_part_head, accum_part_head + torch.sum(cur_sam_mask).item()])
                    # print(self.img_part_pairs[-1])
                    accum_part_head = accum_part_head + torch.sum(cur_sam_mask).item()
                    for k, v in self.train_batch.items():
                        if 'sam' not in k:
                            grouped_ray_batch[k].append(v[cur_start:cur_tail][cur_sam_mask])
                    # record part id of each ray
                    cur_partid_mask = torch.ones((torch.sum(cur_sam_mask).item(),1), dtype=torch.int16) * part_count
                    grouped_ray_batch['part_id'].append(cur_partid_mask)
                    part_count += 1
            else:   # mark all pixels into bg set if only one part available
                self.train_batch['sam_mask'][cur_start:cur_tail,:] = -1
            self.img_pairs.append([accum_head, accum_part_head])
            accum_head = accum_part_head
        
        self.parted_ray_batch = {}   # for paired sampling
        for k, v in grouped_ray_batch.items():
            self.parted_ray_batch[k] = torch.cat(v, dim=0)
        
        self.train_batch_pseudo_fg = copy.deepcopy(self.parted_ray_batch)  # for random sampling of anchor samples
        self.train_ray_fg_num = self.train_batch_pseudo_fg['part_id'].shape[0]
        self._shuffle_train_fg_batch()
        ##########################################################################

        
        # add bg=-1 mask into training batch, at the end of the list
        bg_ray_batch = {'rgb': [], 'mask': [], 'rays_o': [], 'rays_d': []}
        
        for index in range(self.num_images):
            cur_start, cur_tail = index*img_h*img_w, (index+1)*img_h*img_w
            cur_sam_mask = torch.sum(self.train_batch['sam_mask'][cur_start:cur_tail,:],1)==-1
            for k, v in self.train_batch.items():
                if 'sam' not in k:
                    bg_ray_batch[k].append(v[cur_start:cur_tail][cur_sam_mask])

        self.train_bg_ray_batch = {}   # for background sampling
        for k, v in bg_ray_batch.items():
            self.train_bg_ray_batch[k] = torch.cat(v, dim=0)
        self.train_ray_bg_num = self.train_bg_ray_batch['mask'].shape[0]
        self._shuffle_train_bg_batch()
        ############################################################
        
        train_batch_size_tmp = self.train_batch['mask'].shape[0]
        assert (self.train_ray_fg_num + self.train_ray_bg_num == self.train_ray_num), f'The total number of rays is wrong! {train_batch_size_tmp}, {self.train_ray_fg_num} + {self.train_ray_bg_num} != {self.train_ray_num}'
        fgpix_ratio = self.train_ray_fg_num / (self.train_ray_fg_num + self.train_ray_bg_num)

        self.neg_pos_ratio = 2
        self.train_batch_num_anchors = (8 * int(self.train_batch_num * fgpix_ratio) // 8 + self.train_batch_fg_num) // (self.neg_pos_ratio + 2)
        self.train_batch_bg_num = self.train_batch_num - (self.neg_pos_ratio + 2) * self.train_batch_num_anchors

    
    def _shuffle_train_bg_batch(self):
        self.train_batch_bg_i = 0
        shuffle_idxs = torch.randperm(self.train_ray_bg_num, device='cpu') # shuffle
        for k, v in self.train_bg_ray_batch.items():
            self.train_bg_ray_batch[k] = v[shuffle_idxs]

    def _shuffle_train_batch(self):
        self.train_batch_i = 0
        shuffle_idxs = torch.randperm(self.train_ray_num, device='cpu') # shuffle
        for k, v in self.train_batch.items():
            self.train_batch[k] = v[shuffle_idxs]

    def _shuffle_train_fg_batch(self):
        self.train_batch_fg_i = 0
        shuffle_idxs = torch.randperm(self.train_ray_fg_num, device='cpu') # shuffle
        for k, v in self.train_batch_pseudo_fg.items():
            self.train_batch_pseudo_fg[k] = v[shuffle_idxs]


    def training_step(self, batch, batch_idx):
        # select anchor samples 
        train_ray_anc = {k: v[self.train_batch_fg_i:self.train_batch_fg_i + self.train_batch_num_anchors] for k, v in self.train_batch_pseudo_fg.items()}
        self.train_batch_fg_i += self.train_batch_num_anchors
        if self.train_batch_fg_i + self.train_batch_num_anchors >= self.train_ray_fg_num: self._shuffle_train_fg_batch()

        batch_part_id = torch.clone(train_ray_anc['part_id']).reshape(-1,)
        batch_part = np.unique(batch_part_id.numpy())
        batch_nums_part = np.bincount(batch_part_id.numpy())
        _, indices = torch.sort(batch_part_id, 0)
        for k, v in train_ray_anc.items():
            train_ray_anc[k] = v[indices]

        # extract positive rays + negative rays
        train_ray_pos = {k: [] for k in self.parted_ray_batch.keys()}
        train_ray_neg = {k: [] for k in self.parted_ray_batch.keys()}
        fg_headid = 0
        for idb in range(len(batch_part)):
            index = batch_part[idb]
            img_ind = self.img_part_pairs[index][0]
            head_cur, tail_cur = self.img_part_pairs[index][2], self.img_part_pairs[index][3]
            # print(self.img_part_pairs[index])
            inds_pos = np.random.choice(tail_cur-head_cur, size=[batch_nums_part[index]], replace=True)
            for k, v in self.parted_ray_batch.items():
                train_ray_pos[k].append(v[head_cur:tail_cur][inds_pos])
            
            neg_sample_sets = [idx for idx in range(self.img_pairs[img_ind][0], head_cur-1)]
            neg_sample_sets += [idx for idx in range(tail_cur, self.img_pairs[img_ind][1])]
            inds_neg = np.random.choice(neg_sample_sets, size=[batch_nums_part[index]*self.neg_pos_ratio], replace=True)
            for k, v in self.parted_ray_batch.items():
                train_ray_neg[k].append(torch.reshape(v[inds_neg], (batch_nums_part[index], self.neg_pos_ratio, -1)))
        
        for k in self.parted_ray_batch.keys():
            train_ray_pos[k] = torch.cat(train_ray_pos[k], dim=0)
            train_ray_neg[k] = torch.reshape(torch.cat(train_ray_neg[k], dim=0), (self.train_batch_num_anchors*self.neg_pos_ratio, -1))  # rn,6,-1

        train_ray_batch = {k: torch.cat([train_ray_anc[k], train_ray_pos[k], train_ray_neg[k]], dim=0).cuda() for k in self.parted_ray_batch.keys()}
        # print(train_ray_anc['rgb'].shape, train_ray_pos['rgb'].shape, train_ray_neg['rgb'].shape, train_ray_batch['rgb'].shape)

        
        # background sampling for reconstruction quality ensurance
        train_ray_batch_bg = {k: v[self.train_batch_bg_i:self.train_batch_bg_i+self.train_batch_bg_num].cuda() for k, v in self.train_bg_ray_batch.items()}
        self.train_batch_bg_i += self.train_batch_bg_num
        if self.train_batch_bg_i + self.train_batch_bg_num >= self.train_ray_bg_num: self._shuffle_train_bg_batch()
        for k, v in train_ray_batch_bg.items():
            if 'sam' not in k:
                train_ray_batch[k] = torch.cat([train_ray_batch[k], v], 0)

        loss, loss_batch = self.renderer.render_with_loss(train_ray_batch, is_train=True, step=self.global_step, num_anc=self.train_batch_num_anchors-fg_headid, fg_start=fg_headid, neg_pos_ratio=self.neg_pos_ratio)
        self.log_dict(loss_batch, prog_bar=True, logger=True, on_step=True, on_epoch=False, rank_zero_only=True)

        self.log('step', self.global_step, prog_bar=True, on_step=True, on_epoch=False, logger=False, rank_zero_only=True)
        lr = self.optimizers().param_groups[0]['lr']
        self.log('lr', lr, prog_bar=True, logger=True, on_step=True, on_epoch=False, rank_zero_only=True)
        return loss

    def _slice_images_info(self, index):
        return {k:v[index:index+1] for k, v in self.images_info.items()}

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            if self.global_rank==0:
                # we output an rendering image
                images_info = self._slice_images_info(self.validation_index)
                self.validation_index += 1
                self.validation_index %= self.num_images

                test_ray_batch = self._construct_ray_batch(images_info)
                test_ray_batch = {k: v.cuda() for k,v in test_ray_batch.items()}
                
                test_ray_batch['near'], test_ray_batch['far'] = near_far_from_sphere(test_ray_batch['rays_o'], test_ray_batch['rays_d'])
                
                render_outputs = self.renderer.render(test_ray_batch, False, self.global_step)

                process = lambda x: (x.cpu().numpy() * 255).astype(np.uint8)
                h, w = self.image_size, self.image_size
                rgb = torch.clamp(render_outputs['rgb'].reshape(h, w, 3), max=1.0, min=0.0)
                mask = torch.clamp(render_outputs['mask'].reshape(h, w, 1), max=1.0, min=0.0)
                mask_ = torch.repeat_interleave(mask, 3, dim=-1)
                output_image = concat_images_list(process(rgb), process(mask_))
                if 'normal' in render_outputs:
                    normal = torch.clamp((render_outputs['normal'].reshape(h, w, 3) + 1) / 2, max=1.0, min=0.0)
                    normal = normal * mask # we only show foregound normal
                    output_image = concat_images_list(output_image, process(normal))

                # save images
                imsave(f'{self.log_dir}/images/{self.global_step}.jpg', output_image)

    def configure_optimizers(self):
        lr = self.learning_rate
        opt = torch.optim.AdamW([{"params": self.renderer.parameters(), "lr": lr},], lr=lr)

        def schedule_fn(step):
            total_step = self.total_steps
            warm_up_step = self.warm_up_steps
            warm_up_init = 0.02
            warm_up_end = 1.0
            final_lr = 0.02
            interval = 1000
            times = total_step // interval
            ratio = np.power(final_lr, 1/times)
            if step<warm_up_step:
                learning_rate = (step / warm_up_step) * (warm_up_end - warm_up_init) + warm_up_init
            else:
                learning_rate = ratio ** (step // interval) * warm_up_end
            return learning_rate

        if self.use_warm_up:
            scheduler = [{
                    'scheduler': LambdaLR(opt, lr_lambda=schedule_fn),
                    'interval': 'step',
                    'frequency': 1
                }]
        else:
            scheduler = []
        return [opt], scheduler

