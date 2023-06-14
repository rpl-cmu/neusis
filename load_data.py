import os
import cv2
import pickle 
import json 
import math
import scipy
from scipy.io import savemat
import numpy as np 
from scipy.spatial.transform import Rotation 


def load_data(target):
    dirpath = "./data/{}".format(target)
    pickle_loc = "{}/Data".format(dirpath)
    output_loc = "{}/UnzipData".format(dirpath)
    cfg_path = "{}/Config.json".format(dirpath)


    with open(cfg_path, 'r') as f:
        cfg = json.load(f)

    for agents in cfg["agents"][0]["sensors"]:
        if agents["sensor_type"] != "ImagingSonar": continue
        hfov = agents["configuration"]["Azimuth"]
        vfov = agents["configuration"]["Elevation"]
        min_range = agents["configuration"]["RangeMin"]
        max_range = agents["configuration"]["RangeMax"]
        hfov = math.radians(hfov)
        vfov = math.radians(vfov)

    if not os.path.exists(output_loc):
        os.makedirs(output_loc)
    images = []
    sensor_poses = []


    for pkls in os.listdir(pickle_loc):
        filename = "{}/{}".format(pickle_loc, pkls)
        with open(filename, 'rb') as f:
            state = pickle.load(f)
            image = state["ImagingSonar"]
            s = image.shape
            image[image < 0.2] = 0
            image[s[0]- 200:, :] = 0
            pose = state["PoseSensor"]
            images.append(image)
            sensor_poses.append(pose)

    data = {
        "images": images,
        "images_no_noise": [],
        "sensor_poses": sensor_poses,
        "min_range": min_range,
        "max_range": max_range,
        "hfov": hfov,
        "vfov": vfov
    }
    
    savemat('{}/{}.mat'.format(dirpath,target), data, oned_as='row')
    return data

def getPose(p, s):
    Rpm = Rotation.from_quat(np.array([p[3], p[1], p[2], p[0]]).squeeze())
    Rpm = np.array(Rpm.as_matrix())
    Rp = np.array([[Rpm[1, 1], Rpm[1, 2], Rpm[1, 0]],
                  [Rpm[2, 1], Rpm[2, 2], Rpm[2, 0]],
                  [Rpm[0, 1], Rpm[0, 2], Rpm[0, 0]]])    
    Tp = np.array([p[4], p[5], p[6]])


    Rsm = Rotation.from_quat(np.array([s[3], s[1], s[2], s[0]]).squeeze())
    Rsm = np.array(Rsm.as_matrix())
    Rs = np.array([[Rsm[1, 1], Rsm[1, 2], Rsm[1, 0]],
                  [Rsm[2, 1], Rsm[2, 2], Rsm[2, 0]],
                  [Rsm[0, 1], Rsm[0, 2], Rsm[0, 0]]])    
    Ts = np.array([s[4], s[5], s[6]])


    Pp = np.array([[Rp[0, 0], Rp[0, 1], Rp[0, 2], Tp[0][0]], 
                   [Rp[1, 0], Rp[1, 1], Rp[1, 2], Tp[1][0]],
                   [Rp[2, 0], Rp[2, 1], Rp[2, 2], Tp[2][0]],
                   [0, 0, 0, 1]])

    Ps = np.array([[Rs[0, 0], Rs[0, 1], Rs[0, 2], Ts[0][0]], 
                   [Rs[1, 0], Rs[1, 1], Rs[1, 2], Ts[1][0]],
                   [Rs[2, 0], Rs[2, 1], Rs[2, 2], Ts[2][0]],
                   [0, 0, 0, 1]])  

    return Pp@Ps

def load_data_real(target, conf):
    basedir = "./data/{}".format(target)
    bound_substract = 100
    max_row = 512
    min_int = conf.get_float("conf.filter_th")
    data = scipy.io.loadmat(basedir)
    
    images_raw = data["images"].squeeze()
    images = []
    skip_index = []
    
    skip_images = np.array(conf.get_list("conf.skip_range"))
    
    for l in skip_images:
        skip_index.extend(np.arange(l[0], l[1]+1))

    for i, image_raw in enumerate(images_raw): 
        if i in skip_index: continue
        im = image_raw
        im[0:bound_substract, :] = 0
        im[max_row-bound_substract:max_row, :] = 0
        im[im < min_int] = 0
        num_true = len(im[im > 0])
        if num_true < 20:
           skip_index.append(i)
           continue
        images.append(im)

    sensor_rels = data["sensor_rels"].squeeze()
    platform_poses = data["platform_poses"].squeeze()

    sensor_poses = []
    for i in range(len(sensor_rels)):
        if i in skip_index: continue
        P = getPose(platform_poses[i], sensor_rels[i])
        sensor_poses.append(P)

    min_range = data['min_range'][0][0]
    max_range = data['max_range'][0][0]
    hfov = data['hfov'][0][0]
    vfov = data['vfov'][0][0]

    data = {
        "images": images,
        "sensor_poses": sensor_poses,
        "min_range": min_range,
        "max_range": max_range,
        "hfov": hfov,
        "vfov": vfov
    }
    return data