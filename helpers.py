import torch
import matplotlib
matplotlib.use('Agg')
from MLP import *


torch.autograd.set_detect_anomaly(True)


def update_lr(optimizer,lr_decay):
    for param_group in optimizer.param_groups:
        if param_group['lr'] > 0.0000001:
            param_group['lr'] = param_group['lr'] * lr_decay
            learning_rate = param_group['lr']
            print('learning rate is updated to ',learning_rate)
    return 0

def save_model(expID, model, i):
    # save model
    model_name = './experiments/{}/model/epoch.pt'.format(expID)
    torch.save(model, model_name)
    return 0


def get_arcs(H, W, phi_min, phi_max, r_min, r_max, c2w, n_selected_px, arc_n_samples, ray_n_samples, 
            hfov, px, r_increments, randomize_points, device, cube_center):
    
    i = px[:, 0]
    j = px[:, 1]

    # sample angle phi
    phi = torch.linspace(phi_min, phi_max, arc_n_samples).float().repeat(n_selected_px).reshape(n_selected_px, -1)

    dphi = (phi_max - phi_min) / arc_n_samples
    rnd = -dphi + torch.rand(n_selected_px, arc_n_samples)*2*dphi

    sonar_resolution = (r_max-r_min)/H
    if randomize_points:
        phi =  torch.clip(phi + rnd, min=phi_min, max=phi_max)

    # compute radius at each pixel
    r = i*sonar_resolution + r_min
    # compute bearing angle at each pixel
    theta = -hfov/2 + j*hfov/W
      

    # Need to calculate coords to figure out the ray direction 
    # the following operations mimick the cartesian product between the two lists [r, theta] and phi
    # coords is of size: n_selected_px x n_arc_n_samples x 3
    coords = torch.stack((r.repeat_interleave(arc_n_samples).reshape(n_selected_px, -1), 
                         theta.repeat_interleave(arc_n_samples).reshape(n_selected_px, -1), 
                          phi), dim = -1)
    coords = coords.reshape(-1, 3)

    holder = torch.empty(n_selected_px, arc_n_samples*ray_n_samples, dtype=torch.long).to(device)
    bitmask = torch.zeros(ray_n_samples, dtype=torch.bool)
    bitmask[ray_n_samples - 1] = True
    bitmask = bitmask.repeat(arc_n_samples)


    for n_px in range(n_selected_px):
        holder[n_px, :] = torch.randint(0, i[n_px]-1, (arc_n_samples*ray_n_samples,))
        holder[n_px, bitmask] = i[n_px] 
    
    holder = holder.reshape(n_selected_px, arc_n_samples, ray_n_samples)
    
    holder, _ = torch.sort(holder, dim=-1)

    holder = holder.reshape(-1)
        

    r_samples = torch.index_select(r_increments, 0, holder).reshape(n_selected_px, 
                                                                    arc_n_samples, 
                                                                    ray_n_samples)
    
    rnd = torch.rand((n_selected_px, arc_n_samples, ray_n_samples))*sonar_resolution
    
    if randomize_points:
        r_samples = r_samples + rnd

    rs = r_samples[:, :, -1]
    r_samples = r_samples.reshape(n_selected_px*arc_n_samples, ray_n_samples)

    theta_samples = coords[:, 1].repeat_interleave(ray_n_samples).reshape(-1, ray_n_samples)
    phi_samples = coords[:, 2].repeat_interleave(ray_n_samples).reshape(-1, ray_n_samples)

    # Note: r_samples is of size n_selected_px*arc_n_samples x ray_n_samples 
    # so each row of r_samples contain r values for points picked from the same ray (should have the same theta and phi values)
    # theta_samples is also of size  n_selected_px*arc_n_samples x ray_n_samples  
    # since all arc_n_samples x ray_n_samples  have the same value of theta, then the first n_selected_px rows have all the same value 
    # Finally phi_samples is  also of size  n_selected_px*arc_n_samples x ray_n_samples  
    # but not each ray has a different phi value
    
    # pts contain all points and is of size n_selected_px*arc_n_samples*ray_n_samples, 3 
    # the first ray_n_samples rows correspond to points along the same ray 
    # the first ray_n_samples*arc_n_samples row correspond to points along rays along the same arc 
    pts = torch.stack((r_samples, theta_samples, phi_samples), dim=-1).reshape(-1, 3)

    dists = torch.diff(r_samples, dim=1)
    dists = torch.cat([dists, torch.Tensor([sonar_resolution]).expand(dists[..., :1].shape)], -1)

    #r_samples_mid = r_samples + dists/2

    X_r_rand = pts[:,0]*torch.cos(pts[:,1])*torch.cos(pts[:,2])
    Y_r_rand = pts[:,0]*torch.sin(pts[:,1])*torch.cos(pts[:,2])
    Z_r_rand = pts[:,0]*torch.sin(pts[:,2])
    pts_r_rand = torch.stack((X_r_rand, Y_r_rand, Z_r_rand, torch.ones_like(X_r_rand)))


    pts_r_rand = torch.matmul(c2w, pts_r_rand)

    pts_r_rand = torch.stack((pts_r_rand[0,:], pts_r_rand[1,:], pts_r_rand[2,:]))

    # Centering step 
    pts_r_rand = pts_r_rand.T - cube_center

    # Transform to cartesian to apply pose transformation and get the direction
    # transformation as described in https://www.ri.cmu.edu/pub_files/2016/5/thuang_mastersthesis.pdf
    X = coords[:,0]*torch.cos(coords[:,1])*torch.cos(coords[:,2])
    Y = coords[:,0]*torch.sin(coords[:,1])*torch.cos(coords[:,2])
    Z = coords[:,0]*torch.sin(coords[:,2])

    dirs = torch.stack((X,Y,Z, torch.ones_like(X))).T
    dirs = dirs.repeat_interleave(ray_n_samples, 0)
    dirs = torch.matmul(c2w, dirs.T).T
    origin = torch.matmul(c2w, torch.tensor([0., 0., 0., 1.])).unsqueeze(dim=0)
    dirs = dirs - origin
    dirs = dirs[:, 0:3]
    dirs = torch.nn.functional.normalize(dirs, dim=1)

    return dirs, dphi, r, rs, pts_r_rand, dists


def select_coordinates(coords_all, target, N_rand, select_valid_px):
    if select_valid_px:
        coords = torch.nonzero(target)
    else: 
        select_inds = torch.randperm(coords_all.shape[0])[:N_rand]
        coords = coords_all[select_inds]
    return coords
