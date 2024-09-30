import torch
import torch.nn as nn
from torch.autograd import grad
import sphericalHarmonic as SH
from scipy.spatial.transform import Rotation as R

def gradient(inputs, outputs, create_graph=True, retain_graph=True):
    d_points = torch.ones_like(outputs, requires_grad=False, device=outputs.device)
    points_grad = grad(
        outputs=outputs,
        inputs=inputs,
        grad_outputs=d_points,
        create_graph=create_graph,
        retain_graph=retain_graph,
        only_inputs=True)[0]  # [:, -3:]
    return points_grad

# input (n,3)
def func(p):
    v = p[:,0]**2 + p[:,1]**2 + p[:,2]**2
    return v

if __name__ == "__main__":
    b = 1
    n = 1
    p = 2 * torch.rand(n, 3) - 1
    p.requires_grad = True
    v = func(p)

    g = gradient(inputs=p, outputs=v)
    g_dx = gradient(p, g[:, 0])
    g_dy = gradient(p, g[:, 1])
    g_dz = gradient(p, g[:, 2])
    h = torch.stack((g_dx, g_dy, g_dz), dim=-1)
    g_hat = torch.nn.functional.normalize(g, dim=-1)
    P = torch.eye(3).unsqueeze(0) - g_hat.unsqueeze(-1) * g_hat.unsqueeze(-2)
    S = P @ h @ P
    level_vals, level_vecs = torch.linalg.eigh(S)
    # similarity = torch.abs(torch.einsum('ni,nij->nj', g_hat, level_vecs))
    # similarity, sorted_indices = torch.sort(similarity, dim=1, descending=True)

    # level_vecs = torch.gather(level_vecs, 2, sorted_indices.unsqueeze(1).expand(-1, 3, -1))
    # level_vals = torch.gather(level_vals, 1, sorted_indices)
    # level_dk1dp = gradient(p, level_vals[:, 1])
    # level_dk2dp = gradient(p, level_vals[:, 2])
    # level_dk1de1 = torch.einsum("ni,ni->n", level_dk1dp, level_vecs[:, :, 1])
    # level_dk1de2 = torch.einsum("ni,ni->n", level_dk1dp, level_vecs[:, :, 2])
    # level_dk2de1 = torch.einsum("ni,ni->n", level_dk2dp, level_vecs[:, :, 1])
    # level_dk2de2 = torch.einsum("ni,ni->n", level_dk2dp, level_vecs[:, :, 2])
    # mvs_loss = (level_dk1de1**2 + level_dk1de2**2 + level_dk2de1**2 + level_dk2de2**2)

    # print(mvs_loss.min().item(), mvs_loss.max().item(), mvs_loss.mean().item())
    sh = SH.SphericalHarmonic()
    for i in range(n):
        eigvecs = level_vecs[0]
        print(eigvecs)
        v = sh.as_euler_angles(eigvecs)
        print(v)
        f = sh.v2f(v)
        print(f)
        df_0 = gradient(p, f[0]).squeeze(0)
        df_1 = gradient(p, f[1]).squeeze(0)
        df_2 = gradient(p, f[2]).squeeze(0)
        df_3 = gradient(p, f[3]).squeeze(0)
        df_4 = gradient(p, f[4]).squeeze(0)
        df_5 = gradient(p, f[5]).squeeze(0)
        df_6 = gradient(p, f[6]).squeeze(0)
        df_7 = gradient(p, f[7]).squeeze(0)
        df_8 = gradient(p, f[8]).squeeze(0)
        df = torch.stack((df_0, df_1, df_2, df_3, df_4, df_5, df_6, df_7, df_8), dim=0)
        print(df)

    p = p.unsqueeze(0)
    print(p.shape)
