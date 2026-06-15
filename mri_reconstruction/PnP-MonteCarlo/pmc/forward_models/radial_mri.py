import numpy as np
import math, decimal, torch

from typing import Optional
from torch.fft import fftshift, ifftshift, fftn, ifftn

from pmc.utils.add_awgn import addawgn
from pmc.forward_models.base import BaseForwardModel

import sigpy.mri as spmri
# for debugging
import inspect 
import os


class MRIRadial(BaseForwardModel):
    
    def __init__(self, input_snr, var, shape, num_lines, **kwargs) -> None:
        super().__init__(input_snr, var, **kwargs)
        self.shape = shape
        self.mask = self.generate_mask(shape, num_lines)
        print(f'lines: {num_lines}, acceleration: {1/self.mask.type(torch.float32).mean()}x')

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., H, W), real or complex
        y, _, noise_level = addawgn(self.forward(x), self.input_snr, mask=self.mask)
        print(f'Current noise level (beta=sqrt(variance)) is {noise_level}')
        return self.mask * y

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mask * self.fftn_(x)

    def adjoint(self, y: torch.Tensor) -> torch.Tensor:
        return self.ifftn_(self.mask * y)

    def grad(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # Base class uses forward/adjoint; we keep this behavior
        return super().grad(x, y).real
    
    @staticmethod
    def fftn_(image: torch.Tensor, dim=None) -> torch.Tensor:
        if dim is None:
            dim = tuple(range(1, len(image.shape)))
        return fftshift(fftn(ifftshift(image, dim=dim), dim=dim, norm='ortho'), dim=dim)
    
    @staticmethod
    def ifftn_(kspace: torch.Tensor, dim=None) -> torch.Tensor:
        if dim is None:
            dim = tuple(range(1, len(kspace.shape)))
        return fftshift(ifftn(ifftshift(kspace, dim=dim), dim=dim, norm='ortho'), dim=dim)
    
    @staticmethod
    def generate_mask(shape, num_lines):
        if shape[0] % 2 != 0 or shape[1] % 2 != 0:
            raise RuntimeError('image must be even sized! ')
        # convert to numpy
        shape = np.array(shape)
        center = np.array(shape / 2) + 1
        freqMax = np.ceil(
                    np.sum(
                        (shape/2) ** 2
                    ) ** 0.5
                ).astype(int)

        ang = np.linspace(0, math.pi, num=num_lines+1)
        mask = np.zeros(shape, dtype=bool)
        
        for indLine in range(0,num_lines):
            ix = np.zeros(2*freqMax + 1)
            iy = np.zeros(2*freqMax + 1)
            ind = np.zeros(2*freqMax + 1, dtype=bool)
            for i in range(2*freqMax + 1):
                ix[i] = decimal.Decimal(center[1] + (-freqMax+i)*math.cos(ang[indLine])).quantize(0,rounding=decimal.ROUND_HALF_UP)
            for i in range(2*freqMax + 1):
                iy[i] = decimal.Decimal(center[0] + (-freqMax+i)*math.sin(ang[indLine])).quantize(0,rounding=decimal.ROUND_HALF_UP)
                 
            for k in range(2*freqMax + 1):
                if (ix[k] >= 1) & (ix[k] <= shape[1]) & (iy[k] >= 1) & (iy[k] <= shape[0]):
                    ind[k] = True
                else:
                    ind[k] = False
                
            ix = ix[ind]
            iy = iy[ind]
            ix = ix.astype(np.int64)
            iy = iy.astype(np.int64)
            
            for i in range(len(ix)):
                mask[iy[i]-1][ix[i]-1] = True
        
        # shape -> (1, 1, H, W) for broadcasting over batch and coils
        return torch.Tensor(mask).unsqueeze(0).unsqueeze(0)


# -------------------------------------------------------------------------
# Multi-coil version with coil mini-batch support
# -------------------------------------------------------------------------

class MRIRadialParallel(BaseForwardModel):
    r"""
    Multi-coil radial MRI forward model:

        y_c = P F (s_c ⊙ x) + noise

    where:
        - x: single-coil "object" image, shape (B, H, W), real or complex
        - s_c: complex coil sensitivities, shape (C, H, W)
        - P: radial undersampling mask (same for all coils)
        - y_c: k-space data, shape (B, C, H, W)

    `forward`, `adjoint` and `grad` can operate on a *subset* of coils
    specified by `coil_idx`, which is ideal for coil mini-batching in
    posterior sampling.
    """

    def __init__(
        self,
        input_snr: float,
        var: float,
        shape,
        num_lines: int,
        num_coils: int,
        coil_radius: float = 1.5,
        coil_nzz: int = 1,
        std_coil_gain: Optional[float] = None,
        **kwargs
    ) -> None:
        super().__init__(input_snr, var, **kwargs)

        self.shape = tuple(shape)
        self.num_coils = num_coils

        # Same radial mask as before (broadcast over coils and batch)
        self.mask = MRIRadial.generate_mask(shape, num_lines)
        # Generate birdcage sensitivity maps with SigPy
        smaps_np = spmri.birdcage_maps(
            (num_coils, shape[0], shape[1]),
            r=coil_radius,
            nzz=coil_nzz,
            dtype=np.complex64,
        )  # (C, H, W), complex64
        self.smaps = torch.from_numpy(smaps_np)  # keep on CPU; move to device at call time
        print(f'lines: {num_lines}, acceleration: {1/self.mask.type(torch.float32).mean()}x')
        print(f'num_coils: {num_coils}, coil_radius: {coil_radius}, coil_nzz: {coil_nzz}')

        if std_coil_gain is not None:
            coil_gains = self.generate_coil_gains(sigma_gain=std_coil_gain).to(self.smaps.device)
            self.smaps *= coil_gains
            print(f'coil gain: {std_coil_gain}')

    def generate_coil_gains(self, sigma_gain: float=0.5):
        # coil gains: log-normal spread (tune sigma_gain)
        # sigma_gain: 0.2 mild, 0.5 strong, 1.0 very strong
        gains = torch.exp(sigma_gain * torch.randn(self.num_coils))  # positive
        phases = (2 * torch.pi) * torch.rand(self.num_coils)         # random phase
        complex_gain = gains * torch.exp(1j * phases)
        coil_gain = complex_gain.to(torch.complex64)  # (C,)
        return coil_gain[:, None, None]


    # ---------- helpers ----------
    @staticmethod
    def fftn_(x: torch.Tensor) -> torch.Tensor:
        # always FFT over last two dims (H, W), preserving batch and coil axes
        return fftshift(fftn(ifftshift(x, dim=(-2, -1)), dim=(-2, -1), norm='ortho'), dim=(-2, -1))

    @staticmethod
    def ifftn_(k: torch.Tensor) -> torch.Tensor:
        return fftshift(ifftn(ifftshift(k, dim=(-2, -1)), dim=(-2, -1), norm='ortho'), dim=(-2, -1))

    def _get_smaps(self, coil_idx, device, dtype) -> torch.Tensor:
        """Return sensitivity maps for selected coils on the correct device/dtype.

        coil_idx: 1D torch.Tensor/list/np array of indices or None (all coils).
        Output shape: (C_sel, H, W)
        """
        smaps = self.smaps.to(device=device, dtype=dtype).unsqueeze(dim=0)
        B, NC = coil_idx.shape
        _, C, H, W = smaps.shape
        smaps_b = smaps.expand(B, -1, -1, -1)

        idx = coil_idx[:,:,None,None].expand(B, NC, H, W)
        return torch.gather(smaps_b, dim=1, index=idx)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Simulate noisy multi-coil measurements for *all* coils.

        x: (B, H, W) or (H, W), real or complex.
        Returns: y of shape (B, C, H, W)
        """
        y_clean = self.forward(x)                    # all coils
        y_noisy, _, noise_level = addawgn(
            y_clean, self.input_snr, mask=self.mask
        )
        print(f'Current noise level (beta=sqrt(variance)) is {noise_level}')
        return self.mask * y_noisy

    def forward(self, x: torch.Tensor, coil_idx: torch.Tensor=None) -> torch.Tensor:
        """
        Forward operator restricted to selected coils.

        x: (B, 1, H, W) real or complex
        coil_idx: indices of coils, which should have (B, C_sel) shape
        Returns: k-space y of shape (B, C_sel, H, W).
        """
        assert x.ndim == 4, "x.shape must be (B, 1, H, W)"

        if coil_idx is None:
            coil_idx = torch.arange(self.num_coils, device=x.device)[None].expand(len(x), -1)
        smaps = self._get_smaps(coil_idx, x.device, dtype=torch.complex64)      # (B, C_sel, H, W)
        C_sel = smaps.shape[1]

        coil_imgs = x * smaps                         # (B, C_sel, H, W)

        # FFT and apply mask (broadcast over batch & coils)
        kspace = self.fftn_(coil_imgs)                        # (B, C_sel, H, W)
        return self.mask.to(device=x.device, dtype=kspace.dtype) * kspace

    def adjoint(self, y: torch.Tensor, coil_idx=None) -> torch.Tensor:
        """
        Adjoint of the coil-restricted forward operator.

        y: k-space data of shape (B, C_sel, H, W) or (C_sel, H, W).
        coil_idx: same coil subset used in `forward`.
        Returns: image-domain adjoint sum over selected coils, shape (B, H, W).
        """
        if coil_idx is None:
            coil_idx = torch.arange(self.num_coils, device=y.device)[None].expand(len(y), -1)

        smaps = self._get_smaps(coil_idx, y.device, dtype=torch.complex64)      # (C_sel, H, W)
        # Enforce mask consistency
        y_masked = self.mask.to(device=y.device, dtype=y.dtype) * y
        # Inverse FFT back to coil images
        coil_imgs = self.ifftn_(y_masked)                     # (B, C_sel, H, W)
        x_adj = (torch.conj(smaps) * coil_imgs).sum(dim=1, keepdim=True)           # (B, H, W)
        return x_adj

    def grad(self, x: torch.Tensor, y: torch.Tensor, coil_idx: torch.Tensor) -> torch.Tensor:
        """
        Stochastic data-fidelity gradient using a mini-batch of coils.

        Data term:  (1 / (2 * var)) * || A(x) - y ||_2^2
        Full gradient: (1 / var) * sum_c A_c^H (A_c x - y_c)

        We approximate this by using only coils in `coil_idx` and
        scaling by C / |B|, where B = coil_idx, C = total #coils:

            g_B(x) = (C / (|B| * var)) * sum_{c in B} A_c^H (A_c x - y_c)

        x: (B, H, W) or (H, W), real or complex.
        y: (B, C_sel, H, W) or (C_sel, H, W), k-space for the same subset `coil_idx`.
        coil_idx: 1D iterable of selected coil indices (B).
        Returns: gradient w.r.t x, same shape as x (batched), *real part*.
        """
        mini_batch = coil_idx.shape[1]
        y = torch.gather(
            y,
            dim=1,
            index=coil_idx[:, :, None, None].expand(-1, -1, y.shape[2], y.shape[3])
        )

        # Forward for subset, coyoksampute residual
        Ax_B = self.forward(x, coil_idx=coil_idx)       # (B, C_sel, H, W)
        residual = Ax_B - y                             # (B, C_sel, H, W)

        # Adjoint on subset residual
        adj_res = self.adjoint(residual, coil_idx=coil_idx)  # (B, 1, H, W)

        # Scale for unbiasedness and variance parameter
        g = (self.num_coils / (mini_batch * self.var)) * adj_res              # complex gradient
        return g.real
