import os
import logging
import numpy as np
from tqdm import tqdm
import math
import time
import json
import open3d as o3d
from skimage import measure
from scipy.spatial import KDTree
import torch_geometric.transforms as T

# torch imports
import torch
import torch.nn.functional as F

# lightconvpoint imports
from lightconvpoint.datasets.dataset import get_dataset
import lightconvpoint.utils.transforms as lcp_T
from lightconvpoint.utils.misc import dict_to_device

import networks
import datasets
import utils.argparseFromFile as argparse

def export_mesh_and_refine_vertices_region_growing_v2(
    network, latent,
    resolution,
    padding=0,
    mc_value=0,
    device=None,
    num_pts=50000, 
    refine_iter=10, 
    simplification_target=None,
    input_points=None,
    refine_threshold=None,
    out_value=np.nan,
    step=None,
    dilation_size=2,
    whole_negative_component=False,
    return_volume=False
):
    bmin = input_points.min()
    bmax = input_points.max()

    if step is None:
        step = (bmax - bmin) / (resolution - 1)
        resolutionX = resolution
        resolutionY = resolution
        resolutionZ = resolution
    else:
        bmin = input_points.min(axis=0)
        bmax = input_points.max(axis=0)
        resolutionX = math.ceil((bmax[0] - bmin[0]) / step)
        resolutionY = math.ceil((bmax[1] - bmin[1]) / step)
        resolutionZ = math.ceil((bmax[2] - bmin[2]) / step)

    bmin_pad = bmin - padding * step
    bmax_pad = bmax + padding * step

    pts_ids = (input_points - bmin) / step + padding
    pts_ids = pts_ids.astype(int)

    # create the volume
    volume = np.full((resolutionX + 2 * padding, resolutionY + 2 * padding, resolutionZ + 2 * padding), np.nan, dtype=np.float64)
    mask_to_see = np.full((resolutionX + 2 * padding, resolutionY + 2 * padding, resolutionZ + 2 * padding), True, dtype=bool)
    
    while pts_ids.shape[0] > 0:
        mask = np.full((resolutionX + 2 * padding, resolutionY + 2 * padding, resolutionZ + 2 * padding), False, dtype=bool)
        mask[pts_ids[:, 0], pts_ids[:, 1], pts_ids[:, 2]] = True

        # dilation
        for i in range(pts_ids.shape[0]):
            xc, yc, zc = int(pts_ids[i, 0]), int(pts_ids[i, 1]), int(pts_ids[i, 2])
            mask[max(0, xc - dilation_size):xc + dilation_size, 
                 max(0, yc - dilation_size):yc + dilation_size,
                 max(0, zc - dilation_size):zc + dilation_size] = True

        # get the valid points
        valid_points_coord = np.argwhere(mask).astype(np.float32)
        valid_points = valid_points_coord * step + bmin_pad

        # get the prediction for each valid point
        z = []
        near_surface_samples_torch = torch.tensor(valid_points, dtype=torch.float, device=device)
        for pnts in torch.split(near_surface_samples_torch, num_pts, dim=0):
            latent["pos_non_manifold"] = pnts.unsqueeze(0)
            occ_hat = network.from_latent(latent)

            # get class and max non class
            class_dim = 1
            occ_hat = torch.stack([occ_hat[:, class_dim], occ_hat[:, [i for i in range(occ_hat.shape[1]) if i != class_dim]].max(dim=1)[0]], dim=1)
            occ_hat = F.softmax(occ_hat, dim=1)
            occ_hat[:, 0] = occ_hat[:, 0] * (-1)
            if class_dim == 0:
                occ_hat = occ_hat * (-1)

            occ_hat = occ_hat.sum(dim=1)
            outputs = occ_hat.squeeze(0)
            z.append(outputs.detach().cpu().numpy())

        z = np.concatenate(z, axis=0)
        z = z.astype(np.float64)

        # update the volume
        volume[mask] = z

        # create the masks
        mask_pos = np.full((resolutionX + 2 * padding, resolutionY + 2 * padding, resolutionZ + 2 * padding), False, dtype=bool)
        mask_neg = np.full((resolutionX + 2 * padding, resolutionY + 2 * padding, resolutionZ + 2 * padding), False, dtype=bool)

        for i in range(pts_ids.shape[0]):
            xc, yc, zc = int(pts_ids[i, 0]), int(pts_ids[i, 1]), int(pts_ids[i, 2])
            mask_to_see[xc, yc, zc] = False
            if volume[xc, yc, zc] <= 0:
                mask_neg[max(0, xc - dilation_size):xc + dilation_size, 
                         max(0, yc - dilation_size):yc + dilation_size,
                         max(0, zc - dilation_size):zc + dilation_size] = True
            if volume[xc, yc, zc] >= 0:
                mask_pos[max(0, xc - dilation_size):xc + dilation_size, 
                         max(0, yc - dilation_size):yc + dilation_size,
                         max(0, zc - dilation_size):zc + dilation_size] = True

        new_mask = (mask_neg & (volume >= 0) & mask_to_see) | (mask_pos & (volume <= 0) & mask_to_see)
        pts_ids = np.argwhere(new_mask).astype(int)

    volume[0:padding, :, :] = out_value
    volume[-padding:, :, :] = out_value
    volume[:, 0:padding, :] = out_value
    volume[:, -padding:, :] = out_value
    volume[:, :, 0:padding] = out_value
    volume[:, :, -padding:] = out_value

    maxi = volume[~np.isnan(volume)].max()
    mini = volume[~np.isnan(volume)].min()

    if not (maxi > mc_value and mini < mc_value):
        return None

    if return_volume:
        return volume

    # compute the marching cubes
    verts, faces, _, _ = measure.marching_cubes(volume=volume.copy(), level=mc_value)

    # removing the nan values in the vertices
    values = verts.sum(axis=1)
    o3d_verts = o3d.utility.Vector3dVector(verts)
    o3d_faces = o3d.utility.Vector3iVector(faces)
    mesh = o3d.geometry.TriangleMesh(o3d_verts, o3d_faces)
    mesh.remove_vertices_by_mask(np.isnan(values))
    
    return mesh

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main(config):    
    config = eval(str(config))  
    logging.getLogger().setLevel(config["logging"])
    device = torch.device(config["device"])
    if config["device"] == "cuda":
        torch.backends.cudnn.benchmark = True

    savedir_root = config["save_dir"]

    # create the network
    N_LABELS = config["network_n_labels"]
    latent_size = config["network_latent_size"]
    backbone = config["network_backbone"]
    decoder = {'name': config["network_decoder"], 'k': config['network_decoder_k']}
    
    logging.info("Creating the network")
    def network_function():
        return networks.Network(3, latent_size, N_LABELS, backbone, decoder)
    net = network_function()
    checkpoint = torch.load(os.path.join(savedir_root, "checkpoint.pth"))
    net.load_state_dict(checkpoint["state_dict"])
    net.to(device)
    net.eval()
    logging.info(f"Network -- Number of parameters {count_parameters(net)}")
    
    logging.info("Getting the dataset")
    DatasetClass = get_dataset(eval("datasets." + config["dataset_name"]))
    test_transform = [
        lcp_T.FixedPoints(1, item_list=["pos_non_manifold", "occupancies", "y_v", "y_v_object"]),
        lcp_T.Permutation("pos", [1, 0]), lcp_T.Permutation("pos_non_manifold", [1, 0]),
        lcp_T.Permutation("normal", [1, 0]), lcp_T.Permutation("x", [1, 0]),
        lcp_T.ToDict()
    ]

    if config["manifold_points"] is not None and config["manifold_points"] > 0:
        test_transform.insert(0, lcp_T.FixedPoints(config["manifold_points"], item_list=["x", "pos", "normal", "y", "y_object"]))
    if config["random_noise"] is not None and config["random_noise"] > 0:
        test_transform.insert(0, lcp_T.RandomNoiseNormal(sigma=config["random_noise"]))
    if config["normals"]:
        test_transform.insert(0, lcp_T.FieldAsFeatures(["normal"]))

    test_transform = T.Compose(test_transform)
    gen_dataset = DatasetClass(config["dataset_root"], split=config["test_split"], transform=test_transform, network_function=network_function, filter_name=config["filter_name"], num_non_manifold_points=config["non_manifold_points"], dataset_size=config["num_mesh"])

    gen_loader = torch.utils.data.DataLoader(gen_dataset, batch_size=1, shuffle=False, num_workers=0)

    with torch.no_grad():
        gen_dir = f"gen_{config['dataset_name']}_{config['test_split']}_{config['manifold_points']}"
        savedir_mesh_root = os.path.join(savedir_root, gen_dir)

        timing_data = []
        sample_count = 0

        for data in tqdm(gen_loader, ncols=100):
            if sample_count >= 100:
                break
            sample_start_time = time.perf_counter()
            sample_count += 1

            shape_id = data["shape_id"].item()
            category_name = gen_dataset.get_category(shape_id)
            object_name = gen_dataset.get_object_name(shape_id)

            savedir_points = os.path.join(savedir_mesh_root, "input", category_name)
            os.makedirs(savedir_points, exist_ok=True)
            savedir_mesh = os.path.join(savedir_mesh_root, "meshes", category_name)
            os.makedirs(savedir_mesh, exist_ok=True)

            if config["resume"] and os.path.isfile(os.path.join(savedir_mesh, object_name + ".ply")):
                continue

            data = dict_to_device(data, device)
            if device.type == "cuda":
                torch.cuda.synchronize()
            inference_start_time = time.perf_counter()

            pts = data["pos"][0].transpose(1, 0).cpu().numpy()
            nls = data["x"][0].transpose(1, 0).cpu().numpy()
            np.savetxt(os.path.join(savedir_points, object_name + ".xyz"), np.concatenate([pts, nls], axis=1).astype(np.float16))

            latent = net.get_latent(data, with_correction=False)

            if device.type == "cuda":
                torch.cuda.synchronize()
            inference_end_time = time.perf_counter()
            inference_time = inference_end_time - inference_start_time

            if config["gen_resolution_global"]:
                resolution = config["gen_resolution_global"]
                step = None
            elif config["gen_resolution_metric"]:
                step = config['gen_resolution_metric']
                resolution = None
            else:
                raise ValueError("Specify either a global resolution or a metric resolution")

            if device.type == "cuda":
                torch.cuda.synchronize()
            mesh_generation_start_time = time.perf_counter()

            mesh = export_mesh_and_refine_vertices_region_growing_v2(net, latent, resolution=resolution, padding=1, mc_value=0, device=device, input_points=data["pos"][0].cpu().numpy().transpose(1, 0), refine_iter=config["gen_refine_iter"], out_value=1, step=step)

            if device.type == "cuda":
                torch.cuda.synchronize()
            mesh_generation_end_time = time.perf_counter()
            mesh_generation_time = mesh_generation_end_time - mesh_generation_start_time

            if mesh is not None:
                saved_mesh_path = os.path.join(savedir_mesh, object_name + ".ply")
                o3d.io.write_triangle_mesh(saved_mesh_path, mesh)
                
                timing_data.append({
                    'sample_index': sample_count,
                    'shape_id': shape_id,
                    'sample_time': time.perf_counter() - sample_start_time,
                    'inference_time': inference_time,
                    'mesh_generation_time': mesh_generation_time,
                    'saved_mesh_path': saved_mesh_path
                })
            else:
                logging.warning("Mesh is None")

            # Print timing information to the terminal
            print(f"Sample {sample_count}: Shape ID: {shape_id}, Sample Time: {time.perf_counter() - sample_start_time:.6f} seconds, Inference Time: {inference_time:.6f} seconds, Mesh Generation Time: {mesh_generation_time:.6f} seconds")

            # Save timing data to a JSON file after each sample is processed
            output_timing_file = os.path.join(savedir_root, 'inference_timing.json')
            with open(output_timing_file, 'w') as f:
                json.dump({'timing_data': timing_data}, f, indent=4)

        # After processing all samples, compute average times
        if sample_count > 0:
            avg_sample_time = sum(d['sample_time'] for d in timing_data) / sample_count
            avg_inference_time = sum(d['inference_time'] for d in timing_data) / sample_count
            avg_mesh_generation_time = sum(d['mesh_generation_time'] for d in timing_data) / sample_count

            # Update the JSON file with summary
            with open(output_timing_file, 'w') as f:
                json.dump({
                    'timing_data': timing_data,
                    'average_sample_time': avg_sample_time,
                    'average_inference_time': avg_inference_time,
                    'average_mesh_generation_time': avg_mesh_generation_time
                }, f, indent=4)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logging.getLogger("trimesh").setLevel(logging.CRITICAL)

    parser = argparse.ArgumentParserFromFile(description='Process some integers.')
    parser.add_argument('--config_default', type=str, default="configs/config_default.yaml")
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--num_mesh', type=int, default=None)
    parser.add_argument("--gen_refine_iter", type=int, default=10)

    parser.update_file_arg_names(["config_default", "config"])
    config = parser.parse(use_unknown=True)
    
    logging.getLogger().setLevel(config["logging"])
    if config["logging"] == "DEBUG":
        config["threads"] = 0
    
    config["save_dir"] = os.path.dirname(config["config"])

    main(config)
