import numpy as np
import os
import argparse
import pickle
import torch
import copy
import cc3d
import cv2
from skimage.morphology import skeletonize, remove_small_holes

def read_pickle(pkl_path):
    with open(pkl_path, 'rb') as f:
        return pickle.load(f)


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
    fg_areas = np.sum(group_ids > -1)
    unique_elements, counts = np.unique(group_ids, return_counts=True)
    result = group_ids.copy()
    for i, count in enumerate(counts):
        # if count <= th:
        if count / fg_areas <= th:
            result[group_ids == unique_elements[i]] = -1
    
    return result


def get_3d_points(K_, pose, pixs, deps):
    # partly borrowed from Syncdreamer
    # 1,h*w,3 @ 1,3,3 => 1,h*w,3
    points = pixs @ torch.inverse(K_).permute(0, 2, 1)
    # 1,h*w,3 @ 1,hw,1 => 1,h*w,3
    points = points * deps
    hw = points.shape[1]
    points = torch.cat([points, torch.ones(1, hw, 1, dtype=torch.float32)], 2)  # 1,h*w,4
    # 1,h*w,4 @ 1,4,4 => 1,h*w,4
    pose_ = pose.unsqueeze(0).permute(0, 2, 1)
    points = points @ pose_
    return points[...,:3]

def get_2d_pixels(K_, pose, pts):
    # 1,h*w,4 @ 1,4,4 => 1,h*w,4
    pose_ = torch.inverse(pose).unsqueeze(0).permute(0, 2, 1)
    hw = pts.shape[1]
    pts_ = torch.cat([pts, torch.ones(1, hw, 1, dtype=torch.float32)], 2)
    points = pts_ @ pose_
    # 1,h*w,3 @ 1,3,3 => 1,h*w,3
    pixs = points[...,:3] @ K_.permute(0, 2, 1)
    depth_ = pixs[0, :, 2]
    pixs[..., :2] = pixs[..., :2] / pixs[..., 2:]
    return pixs[..., :2], depth_   # 1,h*w,2,   h*w,


def sam_mask_preprocess(sam_, alpha_map, th):
    sam_ = num_to_natural(sam_)  # remove index with no exact pixels, caused by sam mask overlapping
    # detect disconnected parts to remove noisy small pixel groups
    labs_connected = cc3d.connected_components(sam_ + 1)
    sam_new = -1 * np.ones_like(sam_)
    extra_ind = np.max(sam_)+1
    for idp in range(np.min(sam_), np.max(sam_)+1):
        cur_map = labs_connected[sam_==idp]
        unique_values = np.unique(cur_map)
        unique_nums = np.bincount(cur_map)
        if len(unique_values) == 1:
            sam_new[sam_==idp] = idp
        elif len(unique_values) > 1 and np.max(unique_nums) > 19:
            for ide in range(len(unique_values)):
                if unique_nums[unique_values[ide]] > 19:
                    sam_new[labs_connected==unique_values[ide]] = extra_ind
                    extra_ind += 1
    bg_maskid = np.unique(sam_new[alpha_map < 0.95])
    for idm in range(bg_maskid.shape[0]):
        if 0.9*np.sum(sam_new==bg_maskid[idm]) < np.sum(sam_new[alpha_map < 0.95]==bg_maskid[idm]):
            sam_new[sam_new==bg_maskid[idm]] = -1
    sam_new[alpha_map < 0.95] = -1  # set background as invalid mask
    sam_new = remove_small_group(sam_new, th)
    sam_new = num_to_natural(sam_new)
    
    return sam_new


def find_new_cent(coords_, sam_, index):
    skel = skeletonize(remove_small_holes(sam_==index, area_threshold=4))
    skel_x, skel_y = np.where(skel)
    branch_pt = False
    for i in range(skel_x.shape[0]):
        if np.sum(skel[skel_x[i]-1:skel_x[i]+2, skel_y[i]-1:skel_y[i]+2]) > 3:
            med_y, med_x = skel_x[i], skel_y[i]
            branch_pt = True
    if not branch_pt:
        med_y = int(np.median(skel_x))
        med_x = int(np.median(skel_y))
    return med_y, med_x


def find_neighb_pixs(pixs_, img_size):
    pixs_r = np.round(pixs_).astype(np.int64)   # nc,2
    pixs_r = np.clip(pixs_r, 0, img_size-1)
    
    pixs_f, pixs_u = np.floor(pixs_), np.ceil(pixs_)
    pixs_f1, pixs_u1 = pixs_f.copy(), pixs_f.copy()
    pixs_f1[...,0] = pixs_f1[...,0] + 1
    pixs_u1[...,1] = pixs_u1[...,1] + 1
    pixs = np.concatenate([pixs_f, pixs_u, pixs_f1, pixs_u1], axis=0).astype(np.int64)
    pixs = np.clip(pixs, 0, img_size-1)
    return pixs_r, pixs

def unique_2d(x, img_size, ox):
    x_ = x[...,1] * img_size + x[...,0]
    x_, indices = np.unique(x_, return_index=True)
    x_new = np.zeros((len(x_), 2))
    x_new[...,0] = x_ % img_size
    x_new[...,1] = x_ // img_size
    return x_new.astype(np.int64), ox[indices]



def component_search(vertices, visited, root=0, mask_cents=None, largest=None):
    if not visited[root]:
        visited[root] = True
        for edge in vertices[root]:
            if not visited[edge]:
                if mask_cents[edge][4] > largest[1]:
                    largest[0] = edge
                    largest[1] = mask_cents[edge][4]
                component_search(vertices, visited, edge, mask_cents, largest)

def find_connected_parts(vertices, visited, parts=None, mask_cents=None, largest=None):
    part_count = 0
    for k in vertices:
        if not visited[k]:
            # if len(vertices[k]) > 0:
                parts[part_count] = k
                largest[0] = k
                largest[1] = mask_cents[k][4]
                component_search(vertices, visited, k, mask_cents, largest)
                parts[part_count] = largest[0]
                part_count += 1
            


def estimate_partnum(name):
    samdir = "output/mvimgs"
    renderdir = "renderer"
    isoproc = True
    neighbs = 1

    depth_maps = np.load(os.path.join('output', renderdir, name, 'save_depth.npy'))
    alpha_maps = np.load(os.path.join('output', renderdir, name, 'save_alpha.npy'))
    objn = name.split('_')[0]
    sam_mask_dir = os.path.join(samdir, objn)
    sam_mask_path = [os.path.join(sam_mask_dir, f) for f in os.listdir(sam_mask_dir) if ('.npy' in f)]
    sam_masks = np.load(sam_mask_path[0])
    
    

    ### load pre-defined K and poses, borrowed from Syncdreamer
    K, _, _, _, poses = read_pickle(f'meta_info/camera-16.pkl')
    
    h, w = 256, 256
    default_size = 256
    K = np.diag([w/default_size,h/default_size,1.0]) @ K
    K_ = torch.from_numpy(K.astype(np.float32)).unsqueeze(0) # [1,3,3]

    coords = torch.stack(torch.meshgrid(torch.arange(h), torch.arange(w)), -1)[:, :, (1, 0)]  # h,w,2
    coords_np = coords.clone().numpy()
    coords = coords.float().reshape(h * w, 2)  # h*w,2
    coords = torch.cat([coords, torch.ones(h * w, 1, dtype=torch.float32)], 1)  # h*w,3

    # preprocess SAM masks
    for idx in range(16):
        alp_ = np.copy(alpha_maps[:, idx*w:(idx+1)*w])
        sam_ = np.copy(sam_masks[:, idx*w:(idx+1)*w])
        sam_new = sam_mask_preprocess(sam_, alp_, 0.02)
        sam_masks[:, idx*w:(idx+1)*w] = sam_new
    
    # common information of each sam mask
    part_dicts = {}  # key: (imgid, image_partid), value: vertex id
    count_part = 0
    parts_upbd, parts_lobd = 0, 100
    for idx in range(16):
        sam_new = np.copy(sam_masks[:, idx*w:(idx+1)*w])
        parts_upbd = max(parts_upbd, np.max(sam_new)+1)
        parts_lobd = min(parts_lobd, np.max(sam_new)+1)
        for idi in range(max(0, np.min(sam_new)), np.max(sam_new)+1):
            part_dicts[(idx, idi)] = count_part
            count_part += 1
    parts_upbd *= 2
    
    mask_cents = {}  # key: vertex id, value: [center, img_id, img_x, img_y]
    imgs_3dpts = {}  # key: img id, value: 3d points
    for idx in range(16):
        # data loading for current frame
        dep_ = np.copy(depth_maps[:, idx*w:(idx+1)*w])
        alp_ = np.copy(alpha_maps[:, idx*w:(idx+1)*w])
        sam_ = np.copy(sam_masks[:, idx*w:(idx+1)*w])
        val_pixs = np.where(alp_.reshape(-1,) >= 0.95)[0]

        pose_ = np.concatenate([poses[idx], np.array([0,0,0,1]).reshape(1,4)], axis=0)
        pose_ = np.linalg.inv(pose_)
        pose_ = torch.from_numpy(pose_).float()
        
        ## visualize selected centers in 2D image
        vis_map_ = np.zeros((h,w))
        
        ## project pixels in the current frame to 3D points
        pixs_ = coords.clone()
        deps_ = torch.from_numpy(dep_.reshape(-1,1)).float()
        points_all = get_3d_points(K_, pose_, pixs_.unsqueeze(0), deps_.unsqueeze(0)).squeeze(0)
        points = points_all[val_pixs]
        pts_sam = sam_.reshape(-1,)[val_pixs]
        points_all = points_all.reshape(h,w,3).numpy()

        imgs_3dpts[idx] = [points, pts_sam]
        
        ## assign mask centers to each part vertex
        for idi in range(max(0, np.min(sam_)), np.max(sam_)+1):
            xt, yt = find_new_cent(coords_np, sam_, idi)
            partid = part_dicts[(idx, idi)]
            mask_cents[partid] = [points_all[xt, yt], idx, xt, yt, np.sum(sam_==idi)]
            vis_map_[xt, yt] = 255



    overlap_lists = {'a': 0.2, 'b': 0.25, 'c': 0.3, 'd': 0.35, 'e': 0.4, 'f': 0.45, 'g': 0.5, 'h': 0.55, 'i': 0.6, 'j': 0.7}
    cents_lists, solid_cents_num = [], []

    for ol_code, ol_rate in overlap_lists.items():
        # vis_maps = []
        
        # build initial graph
        vertices = {}   # key: vertex id, value: connected parts (edge)
        count_part = 0
        for pk, pv in part_dicts.items():
            vertices[count_part] = []
            count_part += 1
        
        for idx in range(16):
            points = imgs_3dpts[idx][0]
            pts_sam = imgs_3dpts[idx][1]
            
            ## search for edges by warping to neighboring frames
            neigbs = [j % 16 for j in range(idx-neighbs, idx+neighbs+1) if j!=idx]
            for j, idn in enumerate(neigbs):
                dep_ = np.copy(depth_maps[:, idn*w:(idn+1)*w])
                sam_ = np.copy(sam_masks[:, idn*w:(idn+1)*w])
                
                ## load pose for neighboring frame
                pose_ = np.concatenate([poses[idn], np.array([0,0,0,1]).reshape(1,4)], axis=0)
                pose_ = np.linalg.inv(pose_)
                pose_ = torch.from_numpy(pose_).float()
                
                ### propogate to next frames
                pixs, projdep_ = get_2d_pixels(K_, pose_, points.unsqueeze(0))
                pixs = pixs.squeeze(0).numpy()
                pixs_r, pixs = find_neighb_pixs(pixs, default_size)
                renddep_ = dep_[pixs_r[...,1], pixs_r[...,0]]
                visible_mask_ = (projdep_.numpy() < 1.05 * renddep_)
                visible_mask_ = np.concatenate([visible_mask_ for idt in range(4)], axis=0)
                pts_sam_ = np.concatenate([pts_sam for idt in range(4)], axis=0)
                pixs, pts_sam_ = unique_2d(pixs[visible_mask_], default_size, pts_sam_[visible_mask_])
                
                # if idnx==0:
                #     vis_map_[pixs[...,1], pixs[...,0]] = 255.0

                overlap_masks = sam_[pixs[...,1], pixs[...,0]]
                overlap_maskid = np.unique(overlap_masks).tolist()
                overlap_maskid = [idi for idi in overlap_maskid if idi > -1]
                
                for k, idi in enumerate(overlap_maskid):
                    partid0 = part_dicts[(idn, idi)]
                        
                    ## corresponding parts in the current frame
                    ref_sam = pts_sam_[overlap_masks==idi]
                    ref_maskid = np.unique(ref_sam).tolist()
                    ref_maskid = [idf for idf in ref_maskid if idf > -1]
                        
                    for jk, idf in enumerate(ref_maskid):
                        cond1 = np.sum(ref_sam==idf)
                        cond2 = np.sum(pts_sam_==idf)
                        cond3 = np.sum(sam_==idi)
                        
                        if ol_rate * cond3 < cond1 and (cond1 > cond2 * ol_rate or cond1 < cond2 * 0.2):
                        
                            partid1 = part_dicts[(idx, idf)]
                            vertices[partid1].append(partid0)
                            vertices[partid0].append(partid1)
        
        for k,v in vertices.items():
            vertices[k] = np.unique(v).tolist()   # clear all edeges
        
        visited = {}
        for idx in range(len(vertices)):
            visited[idx] = False
        val_parts = {}
        largest = [0,0]
        find_connected_parts(vertices, visited, val_parts, mask_cents, largest)


        ### remove isolated parts with centers that are nearby to its isolated counterparts
        cents = []
        isolations, iso_cents = [], []
        if isoproc:
            for k,v in val_parts.items():
                if len(vertices[v]) > 0:
                    cents.append(mask_cents[v][0].reshape(1,-1))
                else:
                    isolations.append(v)
                    iso_cents.append(mask_cents[v][0].reshape(1,-1))
        else:
            for k,v in val_parts.items():
                cents.append(mask_cents[v][0].reshape(1,-1))
        
        # process isolated parts
        if len(iso_cents) > 0:

            val_parts_iso = {}
            count_k = 0
            for k in isolations:
                val_parts_iso[count_k] = k
                count_k += 1
            for k,v in val_parts_iso.items():
                cents.append(mask_cents[v][0].reshape(1,-1))
            # record the number of non-isolated parts
            solid_cents_num.append(np.max([len(cents) - len(val_parts_iso), 2]))

        else:
            solid_cents_num.append(len(cents))

        cents = np.concatenate(cents, axis=0)
        cents_lists.append(cents)
        
    
    cents_tosave = {}
    count = 0
    for ol_code in overlap_lists.keys():
        cent_ = cents_lists[count]
        if cent_.shape[0] < 2 or cent_.shape[0] > parts_upbd or cent_.shape[0] < parts_lobd:
            pass
        elif count > 0:
            cent_1 = cents_lists[count-1]
            if cent_.shape[0]==cent_1.shape[0] and np.sum(np.abs(cent_1-cent_)) < 1e-6:
                pass
            else:
                cents_tosave[ol_code] = cent_
                cents_tosave[ol_code + '_sol'] = np.array([solid_cents_num[count]])
        else:
            cents_tosave[ol_code] = cent_
            cents_tosave[ol_code + '_sol'] = np.array([solid_cents_num[count]])
        count += 1

    # in case no result meet the restrictions
    if len(cents_tosave) < 1:
        count = 0
        for ol_code in overlap_lists.keys():
            cent_ = cents_lists[count]
            if cent_.shape[0] < 2:
                pass
            elif count > 0:
                cent_1 = cents_lists[count-1]
                if cent_.shape[0]==cent_1.shape[0] and np.sum(np.abs(cent_1-cent_)) < 1e-6:
                    pass
                else:
                    cents_tosave[ol_code] = cent_
                    cents_tosave[ol_code + '_sol'] = np.array([solid_cents_num[count]])
            else:
                cents_tosave[ol_code] = cent_
                cents_tosave[ol_code + '_sol'] = np.array([solid_cents_num[count]])
            count += 1
    np.savez(os.path.join('output', renderdir, name, 'sam_cents.npz'), **cents_tosave)
