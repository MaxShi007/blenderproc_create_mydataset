#obj_000006.ply
# mm2m = True
# obj.set_scale([0.001, 0.001, 0.001])
#light.set_type("POINT")
# light.set_location(bproc.sampler.shell(center=obj.get_location(), radius_min=2, radius_max=8, elevation_min=-30, elevation_max=89))
# light.set_energy(random.uniform(1000, 3000))
# location = bproc.sampler.shell(center=obj.get_location(), radius_min=0.3, radius_max=2.3, elevation_min=-30, elevation_max=89)

# mesh_path = './HTCxiaopin/lvluo_cut.obj'
# mm2m = False
# obj.set_scale([0.01, 0.01, 0.01])
# light.set_type("AREA")
# light.set_location(bproc.sampler.shell(center=obj.get_location(), radius_min=3, radius_max=3, elevation_min=10, elevation_max=89))
# light.set_energy(random.uniform(100, 500))
# location = bproc.sampler.shell(center=obj.get_location(), radius_min=0.7, radius_max=2, elevation_min=30, elevation_max=50)

import blenderproc as bproc
import numpy as np
import random
import os
import json
import glob
import h5py
import PIL.Image as Image

# import debugpy

# debugpy.listen(5678)
# debugpy.wait_for_client()
# mesh_path = 'lm_models/obj_000006.ply'
mesh_path = 'HTCxiaopin/xiaofangtuiche.obj'
intrinsics = 'intrinsics.json'
output_path = 'output'
pic_num = 100
mm2m = True
# mm2m = False


def generate_pic():

    bproc.init()

    # load the objects into the scene
    objs = bproc.loader.load_obj(mesh_path)
    obj_bvh_tree = bproc.object.create_bvh_tree_multi_objects(objs)

    # 加载相机内参
    with open(intrinsics) as f:
        dic_K = json.load(f)
    fx = dic_K['fx']
    fy = dic_K['fy']
    cx = dic_K['ppx']
    cy = dic_K['ppy']
    image_width = dic_K['width']
    image_height = dic_K['height']
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    bproc.camera.set_intrinsics_from_K_matrix(K, image_width, image_height)

    bproc.renderer.enable_depth_output(activate_antialiasing=False)

    for id, obj in enumerate(objs):
        obj.set_cp("category_id", id)

        obj_location = obj.get_location()
        obj_location = obj_location + np.array([0, 0, 0.4])  # 把视点中心从世界坐标系中心转移到物体中心（大概位置）
        print(obj_location)

        if mm2m:
            # Scale 3D model from mm to m
            obj.set_scale([0.01, 0.01, 0.01])

        # Create a new light
        light = bproc.types.Light()
        light.set_type("AREA")
        light.set_location(bproc.sampler.shell(center=obj_location, radius_min=3, radius_max=8, elevation_min=10, elevation_max=89))
        # Randomly set the color and energy
        light.set_color(np.random.uniform([0.5, 0.5, 0.5], [1, 1, 1]))
        light.set_energy(random.uniform(100, 500))

        poses = 0
        while poses < pic_num:
            # Sample random camera location around the object
            location = bproc.sampler.shell(center=obj_location, radius_min=2.4, radius_max=20, elevation_min=15, elevation_max=40)
            # Compute rotation based lookat point which is placed randomly around the object
            lookat_point = obj_location + np.random.uniform([-0.3, -0.3, -0.3], [0.3, 0.3, 0.3])
            rotation_matrix = bproc.camera.rotation_from_forward_vec(lookat_point - location, inplane_rot=np.random.uniform(-0.7854, 0.7854))
            # Add homog cam pose based on location an rotation
            cam2world_matrix = bproc.math.build_transformation_mat(location, rotation_matrix)
            # Only add camera pose if object is still visible
            if bproc.camera.perform_obstacle_in_view_check(cam2world_matrix, {"min": 0.2}, obj_bvh_tree, 100):  # (obj in bproc.camera.visible_objects(cam2world_matrix, sqrt_number_of_rays=100)) and
                bproc.camera.add_camera_pose(cam2world_matrix)
                poses += 1

    # Enable transparency so the background becomes transparent
    bproc.renderer.set_output_format(enable_transparency=True)

    data = bproc.renderer.render()
    data.update(bproc.renderer.render_segmap(map_by=["class", "instance", "name"]))
    hdf5_save_path = os.path.join(output_path, "hdf5")
    if not os.path.exists(hdf5_save_path):
        os.mkdir(hdf5_save_path)
    bproc.writer.write_hdf5(hdf5_save_path, output_data_dict=data, append_to_existing_output=True)
    bproc.writer.write_bop(output_path, depths=data["depth"], colors=data["colors"], color_file_format="JPEG", m2mm=False, append_to_existing_output=True)


if __name__ == '__main__':
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    generate_pic()
