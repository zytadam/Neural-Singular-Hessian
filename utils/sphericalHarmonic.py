import torch
import numpy as np

class SphericalHarmonic:

    def __init__(self, device):

        self.f_0 = torch.tensor([0,0,0,0,np.sqrt(7/12),0,0,0,np.sqrt(5/12)], dtype=torch.float32, device=device)

        self.L_x = torch.tensor([[0, 0, 0, 0, 0, 0, 0,-np.sqrt(2), 0],
                        [0, 0, 0, 0, 0, 0,-np.sqrt(7/2), 0,-np.sqrt(2)],
                        [0, 0, 0, 0, 0,-3/np.sqrt(2), 0,-np.sqrt(7/2), 0],
                        [0, 0, 0, 0,-np.sqrt(10), 0, -3/np.sqrt(2), 0, 0],
                        [0, 0, 0, np.sqrt(10), 0, 0, 0, 0, 0],
                        [0, 0, 3/np.sqrt(2), 0, 0, 0, 0, 0, 0],
                        [0, np.sqrt(7/2), 0, 3/np.sqrt(2), 0, 0, 0, 0, 0],
                        [np.sqrt(2), 0, np.sqrt(7/2), 0, 0, 0, 0, 0, 0],
                        [0, np.sqrt(2), 0, 0, 0, 0, 0, 0, 0]], dtype=torch.float32, device=device)

        self.L_y = torch.tensor([[0, np.sqrt(2), 0, 0, 0, 0, 0, 0, 0],
                        [-np.sqrt(2), 0, np.sqrt(7/2), 0, 0, 0, 0, 0, 0],
                        [0,-np.sqrt(7/2), 0, 3/np.sqrt(2), 0, 0, 0, 0, 0],
                        [0, 0,-3/np.sqrt(2), 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0,-np.sqrt(10), 0, 0, 0],
                        [0, 0, 0, 0, np.sqrt(10), 0,-3/np.sqrt(2), 0, 0],
                        [0, 0, 0, 0, 0, 3/np.sqrt(2), 0,-np.sqrt(7/2), 0],
                        [0, 0, 0, 0, 0, 0, np.sqrt(7/2), 0,-np.sqrt(2)],
                        [0, 0, 0, 0, 0, 0, 0, np.sqrt(2), 0]], dtype=torch.float32, device=device)

        self.L_z = torch.tensor([[ 0, 0, 0, 0, 0, 0, 0, 0, 4],
                        [ 0, 0, 0, 0, 0, 0, 0, 3, 0],
                        [ 0, 0, 0, 0, 0, 0, 2, 0, 0],
                        [ 0, 0, 0, 0, 0, 1, 0, 0, 0],
                        [ 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [ 0, 0, 0,-1, 0, 0, 0, 0, 0],
                        [ 0, 0,-2, 0, 0, 0, 0, 0, 0],
                        [ 0,-3, 0, 0, 0, 0, 0, 0, 0],
                        [-4, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=torch.float32, device=device)
        
        
    def toDevice(self, device):
        self.L_x.to(device)
        self.L_y.to(device)
        self.L_z.to(device)
        self.f_0.to(device)
        
    def v2f(self, v):
        vL = v[0]*self.L_x + v[1]*self.L_y + v[2]*self.L_z
        exp_vL = torch.linalg.matrix_exp(vL)
        f = exp_vL @ self.f_0
        return f
    
    def v2f_batched(self, euler_angles_batched):
        """
        Convert batched Euler angles to a 9D SH vector using vectorized operations.
        
        Args:
            euler_angles_batched: A tensor of shape (b, n, 3) representing batched Euler angles.
        
        Returns:
            A tensor of shape (b, n, 9) representing the corresponding 9D SH vectors.
        """
        b, n, _ = euler_angles_batched.shape
        
        # Reshape euler angles into (b*n, 3) to process them in a vectorized manner
        euler_angles_flat = euler_angles_batched.reshape(-1, 3)

        # Create the rotation vectors 'v' from Euler angles
        # Assuming euler_angles correspond to (v_x, v_y, v_z) in each dimension
        v = euler_angles_flat

        # Compute vL for all (b*n) rotation vectors
        # Broadcast the matrix operations across all vectors
        vL = (v[:, 0].unsqueeze(1).unsqueeze(2) * self.L_x + 
            v[:, 1].unsqueeze(1).unsqueeze(2) * self.L_y + 
            v[:, 2].unsqueeze(1).unsqueeze(2) * self.L_z)

        # Compute the matrix exponential for all vL matrices
        exp_vL = torch.linalg.matrix_exp(vL)

        # Compute the 9D SH vector for all matrices
        # f_0 should be broadcasted across the batch
        f = torch.matmul(exp_vL, self.f_0)

        # Reshape the output back to (b, n, 9)
        output = f.reshape(b, n, 9)

        return output
    
    def as_euler_angles(self, rotation_matrix, convention='xyz'):
        """
        Convert a rotation matrix to Euler angles in the XYZ convention.
        The input rotation_matrix is a 3x3 tensor.
        """
        assert convention == 'xyz', "This implementation only supports XYZ convention."
        
        # Ensure the input is a valid rotation matrix
        assert rotation_matrix.shape == (3, 3), "Input rotation matrix must be 3x3."

        # Recover the angles in the XYZ convention
        # beta = arcsin(-R[2, 0])
        beta = torch.arcsin(-rotation_matrix[2, 0])
        
        # alpha = atan2(R[2, 1], R[2, 2])
        alpha = torch.atan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
        
        # gamma = atan2(R[1, 0], R[0, 0])
        gamma = torch.atan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
        
        # Stack the angles into a single tensor
        euler_angles = torch.stack([alpha, beta, gamma])
        
        return euler_angles
    
    def as_euler_angles_batched(self, rotation_matrices, convention='xyz'):
        """
        Convert batched rotation matrices to Euler angles in the XYZ convention.
        
        Args:
            rotation_matrices: A tensor of shape (b, n, 3, 3) where b is the batch size,
                            n is the number of rotation matrices per batch, and 3x3 are the rotation matrices.
            convention: Convention for Euler angles, defaults to 'xyz'.
            
        Returns:
            euler_angles: A tensor of shape (b, n, 3) where 3 corresponds to the Euler angles (alpha, beta, gamma).
        """
        assert convention == 'xyz', "This implementation only supports XYZ convention."
        assert rotation_matrices.shape[-2:] == (3, 3), "Input rotation matrices must be 3x3."
        device = rotation_matrices.device

        # Extract the elements needed for Euler angle computation
        r_20 = rotation_matrices[..., 2, 0]
        r_21 = rotation_matrices[..., 2, 1]
        r_22 = rotation_matrices[..., 2, 2]
        r_10 = rotation_matrices[..., 1, 0]
        r_00 = rotation_matrices[..., 0, 0]

        # Compute beta (Y rotation angle)
        beta = torch.arcsin(-r_20)
        
        # Compute alpha (X rotation angle)
        alpha = torch.atan2(r_21, r_22)
        
        # Compute gamma (Z rotation angle)
        gamma = torch.atan2(r_10, r_00)
        
        # Stack the Euler angles (alpha, beta, gamma) for each rotation matrix
        euler_angles = torch.stack([alpha, beta, gamma], dim=-1)
        
        return euler_angles


if __name__ == "__main__":
    SH = SphericalHarmonic()

    v = [0,0,torch.pi/6]
    f = SH.v2f(v)
    print(SH.f_0)
    print(f)