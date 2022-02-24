import numpy as np
import random
import os
import json
import glob
import h5py
import PIL.Image as Image
from tqdm import tqdm
# import trimesh

mesh_path = 'lm_models/obj_000006.ply'
mm2m = True
output_path = 'output'
backgrounds_path = "background"


def hdf52png(hdf5):
    seg_save_path = os.path.join(output_path, "mask")
    rgb_save_path = os.path.join(output_path, "rgb")
    if not os.path.exists(seg_save_path):
        os.mkdir(seg_save_path)
    if not os.path.exists(rgb_save_path):
        os.mkdir(rgb_save_path)
    img_name = hdf5.split('.')[0].split('/')[-1] + ".png"
    # print(img_name)

    with h5py.File(hdf5) as f:
        colors = np.array(f["colors"])
        seg = np.array(f['instance_segmaps'])
    seg[seg != 0] = 255
    seg = np.expand_dims(seg, 2).repeat(3, axis=2)
    seg_img = Image.fromarray(seg)
    seg_img.save(os.path.join(seg_save_path, img_name))
    colors_img = Image.fromarray(colors)
    colors_img.save(os.path.join(rgb_save_path, img_name))


def paste_pic(png):
    img = Image.open(png)
    img_w, img_h = img.size
    save_path = ''.join(png.split('.')[:-1]) + ".jpg"

    background_img = random.choice([os.path.join(backgrounds_path, p) for p in os.listdir(backgrounds_path)])
    background = Image.open(background_img).resize([img_w, img_h])
    # a = np.asarray(background)
    # print(a.shape)
    background.paste(img, mask=img.convert('RGBA'))
    # background = background.convert('RGB')
    # background.save(rgb)
    background.save(save_path)
    os.remove(png)


def create_pose(gt_path, pose_count):
    save_pose_path = os.path.join(output_path, "pose")
    if not os.path.exists(save_pose_path):
        os.mkdir(save_pose_path)

    with open(gt_path) as f:
        gt = json.load(f)

    for img_id, poses in gt.items():
        RTs = []
        for pose in poses:
            R = np.asarray(pose['cam_R_m2c'])
            T = np.asarray(pose['cam_t_m2c'])
            R = R.reshape(3, 3)
            T = T.reshape(3, 1)
            RT = np.concatenate((R, T), axis=1)
            RTs.append(RT)
        RTs_name = os.path.join(save_pose_path, 'pose' + str(pose_count) + '.npy')
        pose_count += 1

        if len(poses) == 1:
            np.save(RTs_name, RT)

        else:
            np.save(RTs_name, RTs)
    return pose_count


def create_camera():
    camera_save_path = os.path.join(output_path, 'camera.txt')
    with open(os.path.join(output_path, 'camera.json')) as f:
        K = json.load(f)
    fx = K['fx']
    fy = K['fy']
    cx = K['cx']
    cy = K['cy']
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    np.savetxt(camera_save_path, K)


# def distance(point_one, point_two):
#     return ((point_one[0] - point_two[0])**2 + (point_one[1] - point_two[1])**2 + (point_one[2] - point_two[2])**2)**0.5

# def max_distance(points):
#     return max(distance(p1, p2) for p1, p2 in zip(points, points[1:]))

# def diameter():
#     mesh = trimesh.load(mesh_path)
#     vertices = mesh.vertices
#     print(vertices.shape)
#     maxD = max_distance(vertices.tolist())
#     print("Max vertice distance is: %f m." % maxD)

if __name__ == "__main__":

    hdf5s = glob.glob(output_path + '/hdf5' + '/*.hdf5')
    for hdf5 in tqdm(hdf5s):
        hdf52png(hdf5)
    print("hdf52png done!")

    pngs = glob.glob(os.path.join(output_path, "rgb", "*.png"))
    for png in tqdm(pngs):
        paste_pic(png)
    print("paste_pic done!")

    pose_count = 0
    train_pbr_path = os.path.join(output_path, 'train_pbr')
    train_pbr = os.listdir(train_pbr_path)
    train_pbr.sort()
    for gt_dir in tqdm(train_pbr):
        gt_path = os.path.join(train_pbr_path, gt_dir, 'scene_gt.json')
        print(gt_path)
        pose_count = create_pose(gt_path, pose_count)
    print("create_pose done!")

    create_camera()
    print("create_camera done!")

    #todo
    # diameter()
