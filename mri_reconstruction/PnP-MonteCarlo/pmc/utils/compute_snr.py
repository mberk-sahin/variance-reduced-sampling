import torch

def norm(x, dim):
    return (x ** 2).sum(dim=dim) ** 0.5

def compute_snr(xpred, xreal, is_batch=False):
    assert xpred.shape == xreal.shape, 'The shape of xpred and xreal should be the same.'
    if is_batch:
        dim = tuple(range(1, len(xpred.shape)))
    else:
        dim = tuple(range(0, len(xpred.shape)))
        
    # compte
    mse_ = ((xreal - xpred) ** 2).mean(dim=dim)
    diff_norm = norm(xreal - xpred, dim)
    psnr_ = 20 * torch.log10(1 / torch.sqrt(mse_))
    snr_ = 20 * torch.log10(norm(xreal, dim) / diff_norm)
    return torch.mean(psnr_), torch.mean(snr_), torch.mean(mse_)