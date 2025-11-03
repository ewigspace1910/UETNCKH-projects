# from .gan_loss import gan_loss
# from .adversarial_loss import adversarial_loss
# from .perturbation_loss import perturbation_loss
import torch
import torch.nn.functional as F
import torch 
__all__ = [
    'gan_loss',
    'adversarial_loss',
    'perturbation_loss'
]

def l2norm(X, dim=-1, eps=1e-8):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X

def gan_loss(pred, target):
    pred = pred.view(-1, 1)
    target = target.view(-1, 1)
    loss = F.binary_cross_entropy_with_logits(pred, target)
    return loss


def adversarial_loss(vae, x_prime, watermark):
    latent_watermark = vae.encode(watermark)
    latent_adversarial = vae.encode(x_prime)
    return ((latent_watermark-latent_adversarial) ** 2).sum(dim=1).mean()
    # return torch.norm(latent_watermark-latent_adversarial, p=2, dim=1).mean()
    # return F.mse_loss(latent_adversarial, latent_watermark) 


def perturbation_loss(perturbation, watermark, c=0.1, watermark_region=4):
    # weighted_perturbation = torch.matmul(perturbation, (1 + watermark * watermark_region))-c
    weighted_perturbation = perturbation-c


    zeros = torch.zeros(perturbation.shape[0]).to(weighted_perturbation.device)
    weighted_perturbation = weighted_perturbation.view(perturbation.shape[0], -1)
    weighted_perturbation = torch.norm(weighted_perturbation, dim=1)
    return torch.mean(torch.maximum(zeros, weighted_perturbation))
    #||G(x) - x||_2 - c, where c is user-defined. Here it is set to 0.3

    # mask = ((weighted_perturbation - c)>0) + 0 
    # threshold_loss = (weighted_perturbation * mask).mean()
    # threshold_loss = torch.max(torch.zeros_like(l2_norm), l2_norm - c)
    # return threshold_loss # + small_perturbation_penalty

