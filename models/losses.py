import torch
import torch.nn as nn
import utils.utils as utils
import utils.sphericalHarmonic as SH


def eikonal_loss(nonmnfld_grad, mnfld_grad, eikonal_type='abs'):
    # Compute the eikonal loss that penalises when ||grad(f)|| != 1 for points on and off the manifold
    # shape is (bs, num_points, dim=3) for both grads
    # Eikonal
    if nonmnfld_grad is not None and mnfld_grad is not None:
        all_grads = torch.cat([nonmnfld_grad, mnfld_grad], dim=-2)
    elif nonmnfld_grad is not None:
        all_grads = nonmnfld_grad
    elif mnfld_grad is not None:
        all_grads = mnfld_grad

    if eikonal_type == 'abs':
        eikonal_term = ((all_grads.norm(2, dim=2) - 1).abs()).mean()
    else:
        eikonal_term = ((all_grads.norm(2, dim=2) - 1).square()).mean()

    return eikonal_term


def relax_eikonal_loss(nonmnfld_grad, mnfld_grad, min=.8, max=0.1, eikonal_type='abs', udf=False):
    # Compute the eikonal loss that penalises when ||grad(f)|| != 1 for points on and off the manifold
    # shape is (bs, num_points, dim=3) for both grads
    # Eikonal
    if nonmnfld_grad is not None and mnfld_grad is not None:
        all_grads = torch.cat([nonmnfld_grad, mnfld_grad], dim=-2)
    elif nonmnfld_grad is not None:
        all_grads = nonmnfld_grad
    elif mnfld_grad is not None:
        all_grads = mnfld_grad

    grad_norm = all_grads.norm(2, dim=-1) + 1e-12
    if udf:
        pass
    else:
        term = torch.relu(-(grad_norm - min))
    if eikonal_type == 'abs':
        eikonal_term = term.abs().mean()
    else:
        eikonal_term = term.square().mean()
    return eikonal_term


def latent_rg_loss(latent_reg, device):
    # compute the VAE latent representation regularization loss
    if latent_reg is not None:
        reg_loss = latent_reg.mean()
    else:
        reg_loss = torch.tensor([0.0], device=device)

    return reg_loss


def gaussian_curvature(nonmnfld_hessian_term, morse_nonmnfld_grad):
    device = morse_nonmnfld_grad.device
    nonmnfld_hessian_term = torch.cat((nonmnfld_hessian_term, morse_nonmnfld_grad[:, :, :, None]), dim=-1)
    zero_grad = torch.zeros(
        (morse_nonmnfld_grad.shape[0], morse_nonmnfld_grad.shape[1], 1, 1),
        device=device)
    zero_grad = torch.cat((morse_nonmnfld_grad[:, :, None, :], zero_grad), dim=-1)
    nonmnfld_hessian_term = torch.cat((nonmnfld_hessian_term, zero_grad), dim=-2)
    K_G = (-1. / (morse_nonmnfld_grad.norm(dim=-1) ** 2 + 1e-12)) * torch.det(
        nonmnfld_hessian_term)

    return K_G

def mean_curvature(hessian, grad):
    device = grad.device
    GHG_T = torch.einsum('bni,bnij,bnj->bn', grad, hessian, grad)
    Trace_H = torch.einsum('bnii->bn', hessian)
    grad_norm = grad.norm(dim=-1)
    K_M = (GHG_T - grad_norm ** 2 * Trace_H)/(2*grad_norm**2)
    return K_M


class MorseLoss(nn.Module):
    def __init__(self, weights=None, loss_type='siren_wo_n_w_morse', div_decay='none',
                 div_type='l1', bidirectional_morse=True, udf=False):
        super().__init__()
        if weights is None:
            weights = [3e3, 1e2, 1e2, 5e1, 1e2, 1e1]
        self.weights = weights  # sdf, intern, normal, eikonal, div
        self.loss_type = loss_type
        self.div_decay = div_decay
        self.div_type = div_type
        self.use_morse = True if 'morse' in self.loss_type else False
        self.bidirectional_morse = bidirectional_morse
        self.udf = udf

    def forward(self, output_pred, mnfld_points, nonmnfld_points, mnfld_n_gt=None, near_points=None):
        dims = mnfld_points.shape[-1]
        device = mnfld_points.device

        #########################################
        # Compute required terms
        #########################################

        non_manifold_pred = output_pred["nonmanifold_pnts_pred"]    # Now inter points in test
        manifold_pred = output_pred["manifold_pnts_pred"]
        near_pred = output_pred['near_points_pred'] # Now bounding points in test
        latent_reg = output_pred["latent_reg"]

        div_loss = torch.tensor([0.0], device=mnfld_points.device)
        morse_loss = torch.tensor([0.0], device=mnfld_points.device)
        curv_term = torch.tensor([0.0], device=mnfld_points.device)
        latent_reg_term = torch.tensor([0.0], device=mnfld_points.device)
        normal_term = torch.tensor([0.0], device=mnfld_points.device)

        # compute gradients for div (divergence), curl and curv (curvature)
        if manifold_pred is not None:
            mnfld_grad = utils.gradient(mnfld_points, manifold_pred)
        else:
            mnfld_grad = None

        nonmnfld_grad = utils.gradient(nonmnfld_points, non_manifold_pred)

        # morse_nonmnfld_points = None
        # morse_nonmnfld_grad = None
        # if self.use_morse and near_points is not None:
        #     morse_nonmnfld_points = near_points
        #     morse_nonmnfld_grad = utils.gradient(near_points, output_pred['near_points_pred'])
        # elif self.use_morse and near_points is None:
        #     morse_nonmnfld_points = nonmnfld_points
        #     morse_nonmnfld_grad = nonmnfld_grad
        morse_nonmnfld_points = nonmnfld_points
        morse_nonmnfld_grad = nonmnfld_grad

        if self.use_morse:
            nonmnfld_dx = utils.gradient(morse_nonmnfld_points, morse_nonmnfld_grad[:, :, 0])
            nonmnfld_dy = utils.gradient(morse_nonmnfld_points, morse_nonmnfld_grad[:, :, 1])

            mnfld_dx = utils.gradient(mnfld_points, mnfld_grad[:, :, 0])
            mnfld_dy = utils.gradient(mnfld_points, mnfld_grad[:, :, 1])
            if dims == 3:
                nonmnfld_dz = utils.gradient(morse_nonmnfld_points, morse_nonmnfld_grad[:, :, 2])
                nonmnfld_hessian_term = torch.stack((nonmnfld_dx, nonmnfld_dy, nonmnfld_dz), dim=-1)

                mnfld_dz = utils.gradient(mnfld_points, mnfld_grad[:, :, 2])
                mnfld_hessian_term = torch.stack((mnfld_dx, mnfld_dy, mnfld_dz), dim=-1)
            else:
                nonmnfld_hessian_term = torch.stack((nonmnfld_dx, nonmnfld_dy), dim=-1)
                mnfld_hessian_term = torch.stack((mnfld_dx, mnfld_dy), dim=-1)

            nonmnfld_det = torch.det(nonmnfld_hessian_term)
            mnfld_det = torch.det(mnfld_hessian_term)

            morse_mnfld = torch.tensor([0.0], device=mnfld_points.device)
            morse_nonmnfld = torch.tensor([0.0], device=mnfld_points.device)
            if self.div_type == 'l2':
                morse_nonmnfld = nonmnfld_det.square().mean()
                if self.bidirectional_morse:
                    morse_mnfld = mnfld_det.square().mean()
            elif self.div_type == 'l1':
                morse_nonmnfld = nonmnfld_det.abs().mean()
                # morse_nonmnfld = morse_nonmnfld_grad.norm(dim=-1).square().mean()
                # morse_nonmnfld = nonmnfld_hessian_term.norm(dim=[-1, -2]).square().mean()
                # nonmnfld_divergence = nonmnfld_dx[:, :, 0] + nonmnfld_dy[:, :, 1] + nonmnfld_dz[:, :, 2]
                # morse_nonmnfld = torch.clamp(torch.abs(nonmnfld_divergence), 0.1, 50).mean()
                if self.bidirectional_morse:
                    morse_mnfld = mnfld_det.abs().mean()

            morse_loss = 0.5 * (morse_nonmnfld + morse_mnfld)

        # latent regulariation for multiple shape learning
        latent_reg_term = latent_rg_loss(latent_reg, device)

        # normal term
        if mnfld_n_gt is not None:
            if 'igr' in self.loss_type:
                normal_term = ((mnfld_grad - mnfld_n_gt).abs()).norm(2, dim=1).mean()
            else:
                normal_term = (
                        1 - torch.abs(torch.nn.functional.cosine_similarity(mnfld_grad, mnfld_n_gt, dim=-1))).mean()

        # signed distance function term
        sdf_term = torch.abs(manifold_pred).mean()
        # sdf_term = (torch.abs(manifold_pred) * torch.exp(manifold_pred.abs())).mean()

        # eikonal term
        # eikonal_term = eikonal_loss(nonmnfld_grad, mnfld_grad=mnfld_grad, eikonal_type='abs')
        # Sometimes > relax may leading to bad results, use another type relax
        # eikonal_term = relax_eikonal_loss(None, mnfld_grad=mnfld_grad, udf=self.udf)
        eikonal_term = eikonal_loss(nonmnfld_grad=None, mnfld_grad=mnfld_grad, eikonal_type='abs')

        # inter term
        inter_term = torch.exp(-1e2 * torch.abs(near_pred)).mean()

        # smooth term
        # L2 Hessian
        nonmnfld_grad = utils.gradient(nonmnfld_points, non_manifold_pred)
        nonmnfld_dx = utils.gradient(nonmnfld_points, nonmnfld_grad[:, :, 0])
        nonmnfld_dy = utils.gradient(nonmnfld_points, nonmnfld_grad[:, :, 1])
        nonmnfld_dz = utils.gradient(nonmnfld_points, nonmnfld_grad[:, :, 2])
        nonmnfld_hessian_term = torch.stack((nonmnfld_dx, nonmnfld_dy, nonmnfld_dz), dim=-1)
        nonmnfld_hessian_norm = torch.linalg.matrix_norm(nonmnfld_hessian_term)

        # mnfld_grad = utils.gradient(mnfld_points, manifold_pred)
        # mnfld_dx = utils.gradient(mnfld_points, mnfld_grad[:, :, 0])
        # mnfld_dy = utils.gradient(mnfld_points, mnfld_grad[:, :, 1])
        # mnfld_dz = utils.gradient(mnfld_points, mnfld_grad[:, :, 2])
        # mnfld_hessian_term = torch.stack((mnfld_dx, mnfld_dy, mnfld_dz), dim=-1)
        # mnfld_hessian_norm = torch.linalg.matrix_norm(mnfld_hessian_term)

        # smooth_term = 0.5*(torch.abs(nonmnfld_hessian_norm).mean() + torch.abs(mnfld_hessian_norm).mean())

        # Bending energy
        # K_G = gaussian_curvature(nonmnfld_hessian_term, nonmnfld_grad)
        # K_M = mean_curvature(nonmnfld_hessian_term, nonmnfld_grad)

        eps = 0.02
        mask = (torch.abs(non_manifold_pred) < eps).detach().squeeze(-1)
        # bending_eng = 4 * K_M**2 - 2 * K_G
        # smooth_term = torch.mean(bending_eng[mask]) if mask.sum()>0 else torch.tensor(0.)

        # Shape operator
        n_hat = torch.nn.functional.normalize(nonmnfld_grad, dim=-1)
        grad_norm = torch.norm(nonmnfld_grad, dim=2, keepdim=True) + 1e-12
        P = torch.eye(3, device=nonmnfld_grad.device).unsqueeze(0).unsqueeze(0) - n_hat.unsqueeze(-1) * n_hat.unsqueeze(-2)
        level_hessian_term = P @ nonmnfld_hessian_term @ P
        level_vals, level_vecs = torch.linalg.eigh(level_hessian_term)

        # MVS loss
        similarity = torch.abs(torch.einsum('bni,bnij->bnj', n_hat, level_vecs))
        similarity, sorted_indices = torch.sort(similarity, dim=2, descending=True)
        level_vecs = torch.gather(level_vecs, 3, sorted_indices.unsqueeze(2).expand(-1, -1, 3, -1))
        level_vals = torch.gather(level_vals, 2, sorted_indices)
        # level_dk1dp = utils.gradient(nonmnfld_points, level_vals[:, :, 1]) * grad_norm
        # level_dk2dp = utils.gradient(nonmnfld_points, level_vals[:, :, 2]) * grad_norm
        # level_dk1de1 = torch.einsum("bni,bni->bn", level_dk1dp, level_vecs[:, :, :, 1])
        # level_dk1de2 = torch.einsum("bni,bni->bn", level_dk1dp, level_vecs[:, :, :, 2])
        # level_dk2de1 = torch.einsum("bni,bni->bn", level_dk2dp, level_vecs[:, :, :, 1])
        # level_dk2de2 = torch.einsum("bni,bni->bn", level_dk2dp, level_vecs[:, :, :, 2])
        # mvs_loss = (level_dk1de1**2 + level_dk1de2**2 + level_dk2de1**2 + level_dk2de2**2)
        # smooth_term = torch.mean(mvs_loss[mask]) if mask.sum()>0 else torch.tensor(0.)

        # SH loss
        # sh = SH.SphericalHarmonic(device=level_vecs.device)
        # v = sh.as_euler_angles_batched(rotation_matrices=level_vecs)
        # f = sh.v2f_batched(v)
        # df0 = utils.gradient(nonmnfld_points, f[:, :, 0])
        # df1 = utils.gradient(nonmnfld_points, f[:, :, 1])
        # df2 = utils.gradient(nonmnfld_points, f[:, :, 2])
        # df3 = utils.gradient(nonmnfld_points, f[:, :, 3])
        # df4 = utils.gradient(nonmnfld_points, f[:, :, 4])
        # df5 = utils.gradient(nonmnfld_points, f[:, :, 5])
        # df6 = utils.gradient(nonmnfld_points, f[:, :, 6])
        # df7 = utils.gradient(nonmnfld_points, f[:, :, 7])
        # df8 = utils.gradient(nonmnfld_points, f[:, :, 8])
        # df = torch.stack((df0, df1, df2, df3, df4, df5, df6, df7, df8), dim=-1) * grad_norm.unsqueeze(-1)
        
        # e1 = level_vecs[:, :, :, 1]
        # e2 = level_vecs[:, :, :, 2]
        # dfde1 = torch.einsum('bnik,bni->bnk', df, e1)
        # dfde2 = torch.einsum('bnik,bni->bnk', df, e2)
        
        # sh_loss = torch.linalg.matrix_norm(df)**2
        # sh_loss = torch.linalg.norm(dfde1, dim=-1) + torch.linalg.norm(dfde2, dim=-1)
        # smooth_term = torch.mean(sh_loss[mask]) if mask.sum()>0 else torch.tensor(0.)
        smooth_term = torch.tensor(0.0)
        #########################################
        # Losses
        #########################################

        # losses used in the paper
        if self.loss_type == 'siren':  # SIREN loss
            loss = self.weights[0] * sdf_term + self.weights[1] * inter_term + \
                   self.weights[2] * normal_term + self.weights[3] * eikonal_term
        elif self.loss_type == 'siren_w_morse':
            loss = self.weights[0] * sdf_term + self.weights[1] * inter_term + \
                   self.weights[2] * normal_term + self.weights[3] * eikonal_term + \
                   self.weights[4] * morse_loss
        elif self.loss_type == 'siren_wo_n':  # SIREN loss without normal constraint
            self.weights[2] = 0
            loss = self.weights[0] * sdf_term + self.weights[1] * inter_term + self.weights[3] * eikonal_term
        elif self.loss_type == 'siren_wo_n_w_morse':
            self.weights[2] = 0
            loss = self.weights[0] * sdf_term + self.weights[1] * inter_term + self.weights[3] * eikonal_term + \
                   self.weights[5] * morse_loss
        elif self.loss_type == 'siren_wo_n_wo_e_wo_morse':
            loss = self.weights[0] * sdf_term + self.weights[1] * inter_term
        elif self.loss_type == 'igr':  # IGR loss
            self.weights[1] = 0
            loss = self.weights[0] * sdf_term + self.weights[2] * normal_term + self.weights[3] * eikonal_term
        elif self.loss_type == 'igr_wo_n':  # IGR without normals loss
            self.weights[1] = 0
            self.weights[2] = 0
            loss = self.weights[0] * sdf_term + self.weights[3] * eikonal_term
        elif self.loss_type == 'igr_wo_n_w_morse':
            self.weights[1] = 0
            self.weights[2] = 0
            loss = self.weights[0] * sdf_term + self.weights[3] * eikonal_term + self.weights[5] * morse_loss
        elif self.loss_type == 'siren_w_div':  # SIREN loss with divergence term
            loss = self.weights[0] * sdf_term + self.weights[1] * inter_term + \
                   self.weights[2] * normal_term + self.weights[3] * eikonal_term + \
                   self.weights[4] * div_loss
        elif self.loss_type == 'siren_wo_e_w_morse':
            self.weights[3] = 0
            self.weights[4] = 0
            loss = self.weights[0] * sdf_term + self.weights[1] * inter_term + \
                   self.weights[2] * normal_term + self.weights[5] * morse_loss
        elif self.loss_type == 'siren_wo_e_wo_n_w_morse':
            self.weights[2] = 0
            self.weights[3] = 0
            self.weights[4] = 0
            loss = self.weights[0] * sdf_term + self.weights[1] * inter_term + self.weights[5] * morse_loss
        elif self.loss_type == 'siren_wo_n_w_div':  # SIREN loss without normals and with divergence constraint
            loss = self.weights[0] * sdf_term + self.weights[1] * inter_term + self.weights[3] * eikonal_term + \
                   self.weights[4] * div_loss
        elif self.loss_type == 'siren_test':
            loss = self.weights[0] * sdf_term + self.weights[1] * inter_term + self.weights[3] * eikonal_term + self.weights[5] * smooth_term
        elif self.loss_type == 'siren_test_morse':
            loss = self.weights[0] * sdf_term + self.weights[1] * inter_term + self.weights[3] * eikonal_term + self.weights[5] * morse_loss + self.weights[2] * smooth_term
        else:
            print(self.loss_type)
            raise Warning("unrecognized loss type")

        # If multiple surface reconstruction, then latent and latent_reg are defined so reg_term need to be used
        if latent_reg is not None:
            loss += self.weights[6] * latent_reg_term

        return {"loss": loss, 'sdf_term': sdf_term, 'inter_term': inter_term, 'latent_reg_term': latent_reg_term,
                'eikonal_term': eikonal_term, 'normals_loss': smooth_term, 'div_loss': div_loss,
                'curv_loss': curv_term.mean(), 'morse_term': morse_loss}, mnfld_grad

    def update_morse_weight(self, current_iteration, n_iterations, params=None):
        # `params`` should be (start_weight, *optional middle, end_weight) where optional middle is of the form [percent, value]*
        # Thus (1e2, 0.5, 1e2 0.7 0.0, 0.0) means that the weight at [0, 0.5, 0.75, 1] of the training process, the weight should
        #   be [1e2,1e2,0.0,0.0]. Between these points, the weights change as per the div_decay parameter, e.g. linearly, quintic, step etc.
        #   Thus the weight stays at 1e2 from 0-0.5, decay from 1e2 to 0.0 from 0.5-0.75, and then stays at 0.0 from 0.75-1.

        if not hasattr(self, 'decay_params_list'):
            assert len(params) >= 2, params
            assert len(params[1:-1]) % 2 == 0
            self.decay_params_list = list(zip([params[0], *params[1:-1][1::2], params[-1]], [0, *params[1:-1][::2], 1]))

        curr = current_iteration / n_iterations
        we, e = min([tup for tup in self.decay_params_list if tup[1] >= curr], key=lambda tup: tup[1])
        w0, s = max([tup for tup in self.decay_params_list if tup[1] <= curr], key=lambda tup: tup[1])

        # Divergence term anealing functions
        if self.div_decay == 'linear':  # linearly decrease weight from iter s to iter e
            if current_iteration < s * n_iterations:
                self.weights[5] = w0
            elif current_iteration >= s * n_iterations and current_iteration < e * n_iterations:
                self.weights[5] = w0 + (we - w0) * (current_iteration / n_iterations - s) / (e - s)
            else:
                self.weights[5] = we
        elif self.div_decay == 'quintic':  # linearly decrease weight from iter s to iter e
            if current_iteration < s * n_iterations:
                self.weights[5] = w0
            elif current_iteration >= s * n_iterations and current_iteration < e * n_iterations:
                self.weights[5] = w0 + (we - w0) * (1 - (1 - (current_iteration / n_iterations - s) / (e - s)) ** 5)
            else:
                self.weights[5] = we
        elif self.div_decay == 'step':  # change weight at s
            if current_iteration < s * n_iterations:
                self.weights[5] = w0
            else:
                self.weights[5] = we
        elif self.div_decay == 'none':
            pass
        else:
            raise Warning("unsupported div decay value")
