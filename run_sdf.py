import os, sys
import numpy as np
import json
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm, trange
import scipy.io
import matplotlib.pyplot as plt
from helpers import *
from MLP import *
#from PIL import Image
import cv2 as cv
import time
import random
import string 
from pyhocon import ConfigFactory
from models.fields import RenderingNetwork, SDFNetwork, SingleVarianceNetwork, NeRF
from models.renderer import NeuSRenderer
import trimesh
from itertools import groupby
from operator import itemgetter
from load_data import *
import logging
import argparse 

from math import ceil


def config_parser():
    import configargparse
    parser = configargparse.ArgumentParser()



class Runner:
    def __init__(self, conf, is_continue=False, write_config=True):
        conf_path = conf
        f = open(conf_path)
        conf_text = f.read()
        self.is_continue = is_continue
        self.conf = ConfigFactory.parse_string(conf_text)
        self.write_config = write_config

    def set_params(self):
        self.expID = self.conf.get_string('conf.expID') 

        dataset = self.conf.get_string('conf.dataset')
        self.image_setkeyname =  self.conf.get_string('conf.image_setkeyname') 

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dataset = dataset

        # Training parameters
        self.end_iter = self.conf.get_int('train.end_iter')
        self.N_rand = self.conf.get_int('train.num_select_pixels') #H*W 
        self.arc_n_samples = self.conf.get_int('train.arc_n_samples')
        self.save_freq = self.conf.get_int('train.save_freq')
        self.report_freq = self.conf.get_int('train.report_freq')
        self.val_mesh_freq = self.conf.get_int('train.val_mesh_freq')
        self.learning_rate = self.conf.get_float('train.learning_rate')
        self.learning_rate_alpha = self.conf.get_float('train.learning_rate_alpha')
        self.warm_up_end = self.conf.get_float('train.warm_up_end', default=0.0)
        self.anneal_end = self.conf.get_float('train.anneal_end', default=0.0)
        self.percent_select_true = self.conf.get_float('train.percent_select_true', default=0.5)
        self.r_div = self.conf.get_bool('train.r_div')
        # Weights
        self.igr_weight = self.conf.get_float('train.igr_weight')
        self.variation_reg_weight = self.conf.get_float('train.variation_reg_weight')
        self.px_sample_min_weight = self.conf.get_float('train.px_sample_min_weight')

        self.ray_n_samples = self.conf['model.neus_renderer']['n_samples']
        self.base_exp_dir = './experiments/{}'.format(self.expID)
        self.randomize_points = self.conf.get_float('train.randomize_points')
        self.select_px_method = self.conf.get_string('train.select_px_method')
        self.select_valid_px = self.conf.get_bool('train.select_valid_px')        
        self.x_max = self.conf.get_float('mesh.x_max')
        self.x_min = self.conf.get_float('mesh.x_min')
        self.y_max = self.conf.get_float('mesh.y_max')
        self.y_min = self.conf.get_float('mesh.y_min')
        self.z_max = self.conf.get_float('mesh.z_max')
        self.z_min = self.conf.get_float('mesh.z_min')
        self.level_set = self.conf.get_float('mesh.level_set')

        self.data = load_data(dataset)

        self.H, self.W = self.data[self.image_setkeyname][0].shape

        self.r_min = self.data["min_range"]
        self.r_max = self.data["max_range"]
        self.phi_min = -self.data["vfov"]/2
        self.phi_max = self.data["vfov"]/2
        self.vfov = self.data["vfov"]
        self.hfov = self.data["hfov"]


        self.cube_center = torch.Tensor([(self.x_max + self.x_min)/2, (self.y_max + self.y_min)/2, (self.z_max + self.z_min)/2])

        self.timef = self.conf.get_bool('conf.timef')
        self.end_iter = self.conf.get_int('train.end_iter')
        self.start_iter = self.conf.get_int('train.start_iter')
         
        self.object_bbox_min = self.conf.get_list('mesh.object_bbox_min')
        self.object_bbox_max = self.conf.get_list('mesh.object_bbox_max')

        r_increments = []
        self.sonar_resolution = (self.r_max-self.r_min)/self.H
        for i in range(self.H):
            r_increments.append(i*self.sonar_resolution + self.r_min)

        self.r_increments = torch.FloatTensor(r_increments).to(self.device)

        extrapath = './experiments/{}'.format(self.expID)
        if not os.path.exists(extrapath):
            os.makedirs(extrapath)

        extrapath = './experiments/{}/checkpoints'.format(self.expID)
        if not os.path.exists(extrapath):
            os.makedirs(extrapath)

        extrapath = './experiments/{}/model'.format(self.expID)
        if not os.path.exists(extrapath):
            os.makedirs(extrapath)

        if self.write_config:
            with open('./experiments/{}/config.json'.format(self.expID), 'w') as f:
                json.dump(self.conf.__dict__, f, indent = 2)

        # Create all image tensors beforehand to speed up process

        self.i_train = np.arange(len(self.data[self.image_setkeyname]))

        self.coords_all_ls = [(x, y) for x in np.arange(self.H) for y in np.arange(self.W)]
        self.coords_all_set = set(self.coords_all_ls)

        #self.coords_all = torch.from_numpy(np.array(self.coords_all_ls)).to(self.device)

        self.del_coords = []
        for y in np.arange(self.W):
            tmp = [(x, y) for x in np.arange(0, self.ray_n_samples)]
            self.del_coords.extend(tmp)

        self.coords_all = list(self.coords_all_set - set(self.del_coords))
        self.coords_all = torch.LongTensor(self.coords_all).to(self.device)

        self.criterion = torch.nn.L1Loss(reduction='sum')
        
        self.model_list = []
        self.writer = None

        # Networks
        params_to_train = []
        self.sdf_network = SDFNetwork(**self.conf['model.sdf_network']).to(self.device)

        self.deviation_network = SingleVarianceNetwork(**self.conf['model.variance_network']).to(self.device)
        self.color_network = RenderingNetwork(**self.conf['model.rendering_network']).to(self.device)
        params_to_train += list(self.sdf_network.parameters())
        params_to_train += list(self.deviation_network.parameters())
        params_to_train += list(self.color_network.parameters())

        self.optimizer = torch.optim.Adam(params_to_train, lr=self.learning_rate)


        self.iter_step = 0
        self.renderer = NeuSRenderer(self.sdf_network,
                                    self.deviation_network,
                                    self.color_network,
                                    self.base_exp_dir,
                                    self.expID,
                                    **self.conf['model.neus_renderer'])  

        latest_model_name = None
        if self.is_continue:
            model_list_raw = os.listdir(os.path.join(self.base_exp_dir, 'checkpoints'))
            model_list = []
            for model_name in model_list_raw:
                if model_name[-3:] == 'pth': #and int(model_name[5:-4]) <= self.end_iter:
                    model_list.append(model_name)
            model_list.sort()
            latest_model_name = model_list[-1]

        if latest_model_name is not None:
            logging.info('Find checkpoint: {}'.format(latest_model_name))
            self.load_checkpoint(latest_model_name)
    
    def getRandomImgCoordsByPercentage(self, target):
        true_coords = []
        for y in np.arange(self.W):
            col = target[:, y]
            gt0 = col > 0
            indTrue = np.where(gt0)[0]
            if len(indTrue) > 0:
                true_coords.extend([(x, y) for x in indTrue])

        sampling_perc = int(self.percent_select_true*len(true_coords))
        true_coords = random.sample(true_coords, sampling_perc)
        true_coords = list(set(true_coords) - set(self.del_coords))
        true_coords = torch.LongTensor(true_coords).to(self.device)
        target = torch.Tensor(target).to(self.device)
        if self.iter_step%len(self.data[self.image_setkeyname]) !=0:
            N_rand = 0
        else:
            N_rand = self.N_rand
        N_rand = self.N_rand
        coords = select_coordinates(self.coords_all, target, N_rand, self.select_valid_px)
        
        coords = torch.cat((coords, true_coords), dim=0)
            
        return coords, target

    def train(self):
        loss_arr = []

        for i in trange(self.start_iter, self.end_iter, len(self.data[self.image_setkeyname])):
            i_train = np.arange(len(self.data[self.image_setkeyname]))
            np.random.shuffle(i_train)
            loss_total = 0
            sum_intensity_loss = 0
            sum_eikonal_loss = 0
            sum_total_variational = 0
            
            for j in trange(0, len(i_train)):
                img_i = i_train[j]
                target = self.data[self.image_setkeyname][img_i]

                
                pose = self.data["sensor_poses"][img_i]  
                
                if self.select_px_method == "byprob":
                    coords, target = self.getRandomImgCoordsByProbability(target)
                else:
                    coords, target = self.getRandomImgCoordsByPercentage(target)

                n_pixels = len(coords)
                rays_d, dphi, r, rs, pts, dists = get_arcs(self.H, self.W, self.phi_min, self.phi_max, self.r_min, self.r_max,  torch.Tensor(pose), n_pixels,
                                                        self.arc_n_samples, self.ray_n_samples, self.hfov, coords, self.r_increments, 
                                                        self.randomize_points, self.device, self.cube_center)

                
                target_s = target[coords[:, 0], coords[:, 1]]

                render_out = self.renderer.render_sonar(rays_d, pts, dists, n_pixels, 
                                                        self.arc_n_samples, self.ray_n_samples,
                                                        cos_anneal_ratio=self.get_cos_anneal_ratio())
                

                intensityPointsOnArc = render_out["intensityPointsOnArc"]

                gradient_error = render_out['gradient_error'] #.reshape(n_pixels, self.arc_n_samples, -1)

                eikonal_loss = gradient_error.sum()*(1/(self.arc_n_samples*self.ray_n_samples*n_pixels))

                variation_regularization = render_out['variation_error']*(1/(self.arc_n_samples*self.ray_n_samples*n_pixels))

                if self.r_div:
                    intensity_fine = (torch.divide(intensityPointsOnArc, rs)*render_out["weights"]).sum(dim=1) 
                else:
                    intensity_fine = render_out['color_fine']

                intensity_error = self.criterion(intensity_fine, target_s)*(1/n_pixels)
                    
            
                loss = intensity_error + eikonal_loss * self.igr_weight  + variation_regularization*self.variation_reg_weight

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                with torch.no_grad():
                    lossNG = intensity_error + eikonal_loss * self.igr_weight 
                    loss_total += lossNG.cpu().numpy().item()
                    sum_intensity_loss += intensity_error.cpu().numpy().item()
                    sum_eikonal_loss += eikonal_loss.cpu().numpy().item()
                    sum_total_variational +=  variation_regularization.cpu().numpy().item()
                
                self.iter_step += 1
                self.update_learning_rate()

                del(target)
                del(target_s)
                del(rays_d)
                del(pts)
                del(dists)
                del(render_out)
                del(coords)
                
            with torch.no_grad():
                l = loss_total/len(i_train)
                iL =  sum_intensity_loss/len(i_train)
                eikL =  sum_eikonal_loss/len(i_train)
                varL =  sum_total_variational/len(i_train)
                loss_arr.append(l)

            if i ==0 or i % self.save_freq == 0:
                logging.info('iter:{} ********************* SAVING CHECKPOINT ****************'.format(self.optimizer.param_groups[0]['lr']))
                self.save_checkpoint()

            if i % self.report_freq == 0:
                print('iter:{:8>d} "Loss={} | intensity Loss={} " | eikonal loss={} | total variation loss = {} | lr={}'.format(self.iter_step, l, iL, eikL, varL, self.optimizer.param_groups[0]['lr']))

            if i == 0 or i % self.val_mesh_freq == 0:
                self.validate_mesh(threshold = self.level_set)
            


    def save_checkpoint(self):
        checkpoint = {
            'sdf_network_fine': self.sdf_network.state_dict(),
            'variance_network_fine': self.deviation_network.state_dict(),
            'color_network_fine': self.color_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'iter_step': self.iter_step,
        }

        os.makedirs(os.path.join(self.base_exp_dir, 'checkpoints'), exist_ok=True)
        torch.save(checkpoint, os.path.join(self.base_exp_dir, 'checkpoints', 'ckpt_{:0>6d}.pth'.format(self.iter_step)))

    def load_checkpoint(self, checkpoint_name):
        checkpoint = torch.load(os.path.join(self.base_exp_dir, 'checkpoints', checkpoint_name), map_location=self.device)
        self.sdf_network.load_state_dict(checkpoint['sdf_network_fine'])
        self.deviation_network.load_state_dict(checkpoint['variance_network_fine'])
        self.color_network.load_state_dict(checkpoint['color_network_fine'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.iter_step = checkpoint['iter_step']

    def update_learning_rate(self):
        if self.iter_step < self.warm_up_end:
            learning_factor = self.iter_step / self.warm_up_end
        else:
            alpha = self.learning_rate_alpha
            progress = (self.iter_step - self.warm_up_end) / (self.end_iter - self.warm_up_end)
            learning_factor = (np.cos(np.pi * progress) + 1.0) * 0.5 * (1 - alpha) + alpha

        for g in self.optimizer.param_groups:
            g['lr'] = self.learning_rate * learning_factor

    def get_cos_anneal_ratio(self):
        if self.anneal_end == 0.0:
            return 1.0
        else:
            return np.min([1.0, self.iter_step / self.anneal_end])

    def validate_mesh(self, world_space=False, resolution=64, threshold=0.0):
        bound_min = torch.tensor(self.object_bbox_min, dtype=torch.float32)
        bound_max = torch.tensor(self.object_bbox_max, dtype=torch.float32)

        vertices, triangles =\
            self.renderer.extract_geometry(bound_min, bound_max, resolution=resolution, threshold=threshold)

        os.makedirs(os.path.join(self.base_exp_dir, 'meshes'), exist_ok=True)

        if world_space:
            vertices = vertices * self.dataset.scale_mats_np[0][0, 0] + self.dataset.scale_mats_np[0][:3, 3][None]

        mesh = trimesh.Trimesh(vertices, triangles)
        mesh.export(os.path.join(self.base_exp_dir, 'meshes', '{:0>8d}.ply'.format(self.iter_step)))


if __name__=='__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
    logging.getLogger('matplotlib.font_manager').disabled = True
    logging.basicConfig(level=logging.DEBUG, format=FORMAT)
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', type=str, default="./confs/conf.conf")
    parser.add_argument('--is_continue', default=False, action="store_true")
    parser.add_argument('--gpu', type=int, default=0)

    args = parser.parse_args()

    torch.cuda.set_device(args.gpu)
    runner = Runner(args.conf, args.is_continue)
    runner.set_params()
    runner.train()
