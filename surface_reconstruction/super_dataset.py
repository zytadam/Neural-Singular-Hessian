import torch.utils.data as data
import numpy as np
import scipy.spatial as spatial
import open3d as o3d
import trimesh
import torch
import utils.utils as utils


class SuperDataset(data.Dataset):
    # A class to generate synthetic examples of basic shapes.
    # Generates clean and noisy point clouds sampled  + samples on a grid with their distance to the surface (not used in DiGS paper)
    def __init__(self, file_path, n_points, n_samples=128, res=128, sample_type='grid', sapmling_std=0.005,
                 requires_dist=False, requires_curvatures=False, grid_range=1.1):
        self.file_path = file_path
        self.n_points = n_points
        self.n_samples = n_samples
        # self.grid_res = res
        # self.sample_type = sample_type  # grid | gaussian | combined
        # self.sampling_std = sapmling_std
        # self.requires_dist = requires_dist
        # self.nonmnfld_dist, self.nonmnfld_n, self.mnfld_curvs = None, None, None
        # self.requires_curvatures = requires_curvatures  # assumes a subdirectory names "estimated props" in dataset path
        # load data
        self.o3d_point_cloud = o3d.io.read_point_cloud(self.file_path)
        self.grid_range = grid_range

        # extract center and scale points and normals
        self.points, self.mnfld_n = self.get_mnfld_points()
        self.bbox = np.array([np.min(self.points, axis=0), np.max(self.points, axis=0)]).transpose()
        self.bbox_trimesh = trimesh.PointCloud(self.points).bounding_box.copy()

        self.point_idxs = np.arange(self.points.shape[0], dtype=np.int32)
        # record sigma
        self.sample_gaussian_noise_around_shape()

    def get_mnfld_points(self):
        # Returns points on the manifold
        points = np.asarray(self.o3d_point_cloud.points, dtype=np.float32)
        normals = np.asarray(self.o3d_point_cloud.normals, dtype=np.float32)
        if normals.shape[0] == 0:
            normals = np.zeros_like(points)
        # center and scale data/point cloud
        self.cp = points.mean(axis=0)
        points = points - self.cp[None, :]
        self.scale = np.abs(points).max()
        points = points / self.scale

        return points, normals

    def sample_gaussian_noise_around_shape(self):
        kd_tree = spatial.KDTree(self.points)
        # query each point for sigma
        dist, _ = kd_tree.query(self.points, k=51, workers=-1)
        sigmas = dist[:, -1:]
        self.sigmas = sigmas
        return

    def __getitem__(self, index):
        manifold_points, manifold_values, manifold_normals, manifold_hessians = self.get_cylinder_points_normals(n=1, iso=False)

        nonmnfld_points = np.random.uniform(-self.grid_range, self.grid_range,
                                            size=(self.n_points, 3)).astype(np.float32)  # (n_points, 3)
        near_points = manifold_points

        # return {'points': manifold_points, 'mnfld_v': manifold_values, 'mnfld_n': manifold_normals, 'mnfld_h': manifold_hessians, 'nonmnfld_points': nonmnfld_points,
        #         'near_points': near_points}
        return {'points': manifold_points, 'mnfld_v': manifold_values, 'nonmnfld_points': nonmnfld_points,
                'near_points': near_points}

    def get_train_data(self, batch_size):
        manifold_idxes_permutation = np.random.permutation(self.points.shape[0])
        mnfld_idx = manifold_idxes_permutation[:batch_size]
        manifold_points = self.points[mnfld_idx]  # (n_points, 3)
        near_points = (manifold_points + self.sigmas[mnfld_idx] * np.random.randn(manifold_points.shape[0],
                                                                                  manifold_points.shape[1])).astype(
            np.float32)

        return manifold_points, near_points, self.points

    def gen_new_data(self, dense_pts):
        self.points = dense_pts
        kd_tree = spatial.KDTree(self.points)
        # query each point for sigma^2
        dist, _ = kd_tree.query(self.points, k=51, workers=-1)
        sigmas = dist[:, -1:]
        self.sigmas = sigmas

    def get_cylinder_points(self, r=0.4, h=0.8, n=2):

        def sdf_cylinder(points, radius, height):
            # Calculate the 2D distance to the z-axis (cylinder axis)
            d_xy = np.linalg.norm(points[:, :2], axis=1) - radius
            
            # Calculate the vertical distance to the top and bottom caps
            d_z = np.abs(points[:, 2]) - height / 2.0
            
            # Compute signed distances based on the different regions relative to the cylinder
            inside_cylinder = (d_xy < 0) & (d_z < 0)
            outside_side = (d_xy > 0) & (d_z < 0)
            outside_caps = (d_xy < 0) & (d_z > 0)
            outside_both = (d_xy > 0) & (d_z > 0)

            # Initialize the signed distance array
            distances = np.zeros(points.shape[0], dtype=np.float32)

            # Inside the cylinder: maximum of radial and vertical distances
            distances[inside_cylinder] = np.maximum(d_xy[inside_cylinder], d_z[inside_cylinder])

            # Outside side but within height bounds: radial distance
            distances[outside_side] = d_xy[outside_side]

            # Outside caps but within radius: vertical distance
            distances[outside_caps] = d_z[outside_caps]

            # Outside both the radius and height bounds: Euclidean distance to the corner
            distances[outside_both] = np.sqrt(d_xy[outside_both] ** 2 + d_z[outside_both] ** 2)
            
            return distances

        # Returns points on the manifold
        points = np.random.uniform(-self.grid_range, self.grid_range,
                                            size=(n*self.n_points, 3)).astype(np.float32)  # (n_points, 3)
        # center and scale data/point cloud
        self.scale = np.max([r, h/2])
        r /= self.scale
        h /= self.scale

        values = sdf_cylinder(points, r, h)

        return points, values
    
    def get_cylinder_points_normals(self, r=0.4, h=0.8, n=1, iso=False):

        def sdf_cylinder(points, radius, height):
            # Ensure that gradients can be computed for the input points
            points = torch.as_tensor(points, dtype=torch.float32)
            points.requires_grad_(True)
            
            # Calculate the 2D distance to the z-axis (cylinder axis)
            d_xy = torch.norm(points[:, :2], dim=1) - radius
            
            # Calculate the vertical distance to the top and bottom caps
            d_z = torch.abs(points[:, 2]) - height / 2.0
            
            # Compute signed distances based on the different regions relative to the cylinder
            inside_cylinder = (d_xy < 0) & (d_z < 0)
            outside_side = (d_xy > 0) & (d_z < 0)
            outside_caps = (d_xy < 0) & (d_z > 0)
            outside_both = (d_xy > 0) & (d_z > 0)

            # Initialize the signed distance tensor
            distances = torch.zeros(points.shape[0], device=points.device)

            # Inside the cylinder: maximum of radial and vertical distances
            distances[inside_cylinder] = torch.maximum(d_xy[inside_cylinder], d_z[inside_cylinder])

            # Outside side but within height bounds: radial distance
            distances[outside_side] = d_xy[outside_side]

            # Outside caps but within radius: vertical distance
            distances[outside_caps] = d_z[outside_caps]

            # Outside both the radius and height bounds: Euclidean distance to the corner
            distances[outside_both] = torch.sqrt(d_xy[outside_both] ** 2 + d_z[outside_both] ** 2)
            
            gradients = utils.gradient(points, distances)
            mnfld_dx = utils.gradient(points, gradients[:, 0])
            mnfld_dy = utils.gradient(points, gradients[:, 1])
            mnfld_dz = utils.gradient(points, gradients[:, 2])
            hessians = torch.stack((mnfld_dx, mnfld_dy, mnfld_dz), dim=-1)

            n_hat = torch.nn.functional.normalize(gradients, dim=-1)
            # grad_norm = torch.norm(gradients, dim=1, keepdim=True) + 1e-12
            P = torch.eye(3).unsqueeze(0) - n_hat.unsqueeze(-1) * n_hat.unsqueeze(-2)
            S = P @ hessians @ P

            S = S.reshape((-1,9))
            dS0 = utils.gradient(points, S[:, 0])
            dS1 = utils.gradient(points, S[:, 1])
            dS2 = utils.gradient(points, S[:, 2])
            dS3 = utils.gradient(points, S[:, 3])
            dS4 = utils.gradient(points, S[:, 4])
            dS5 = utils.gradient(points, S[:, 5])
            dS6 = utils.gradient(points, S[:, 6])
            dS7 = utils.gradient(points, S[:, 7])
            dS8 = utils.gradient(points, S[:, 8])
            dS = torch.stack((dS0, dS1, dS2, dS3, dS4, dS5, dS6, dS7, dS8), dim=-1)

            return distances.unsqueeze(-1).detach().numpy(), gradients.detach().numpy(), dS.detach().numpy()

        # Returns points on the manifold
        if iso:
            points = self.sample_isosurface(n*self.n_points, eps=0.02)
        else:
            points = np.random.uniform(-self.grid_range, self.grid_range,
                                            size=(n*self.n_points, 3)).astype(np.float32)  # (n_points, 3)

        # center and scale data/point cloud
        self.scale = np.max([r, h/2])
        r /= self.scale
        h /= self.scale

        values, normals, hessians = sdf_cylinder(points, r, h)

        return points, values, normals, hessians
    def __len__(self):
        return self.n_samples
    
    def set_model(self, model):
        self.model = model

    def sample_isosurface(self, num, eps):
        net = self.model
        device = torch.device("cuda:0")
        net.eval()
        with torch.no_grad():
            sample_list = []
            sample_number = 0
            
            while sample_number < num:
                points = 2*self.grid_range * torch.rand(num, 3, device=device) - self.grid_range
                preds = net(points)["nonmanifold_pnts_pred"]
                mask = torch.abs(preds).squeeze(-1) < eps
                samples = points[mask]
                sample_list.append(samples)
                sample_number += samples.shape[0]

            s_points = torch.cat(sample_list, dim=0)[:num]

        return s_points.detach().cpu().numpy()
