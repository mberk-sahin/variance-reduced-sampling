import torch as th


def noisy_score(score_fn, x, noise=None):
    """
    Returns noisy estimate of ∇ log p(x).
    """
    score = score_fn(x)

    if noise is not None:
        score = score + noise

    return score


class SGD:
    def __init__(self, mini_batch_size=6):
        self.init_batch_size = 100
        self.mini_batch_size = mini_batch_size
        self.grads_used = []

        print("Estimator: SGD")
        print("-"*20)
        print("batch_size:", self.mini_batch_size)
        print("-"*20)

    def reset_grads(self):
        self.grads_used = []

    def score_update(self, x_k, m_k_1, score_fn, std_noise_grad, rng, **kwargs):

        if m_k_1 is None:
            ngrads_used = self.init_batch_size
        else:
            ngrads_used = self.mini_batch_size

        noise = th.randn(
            (ngrads_used,) + tuple(x_k.shape),
            device=x_k.device,
            generator=rng,
        ) * std_noise_grad

        m_k = noisy_score(score_fn, x_k, noise).mean(dim=0)
        self.grads_used.append(ngrads_used)
        return m_k


class SGDm:
    def __init__(self, beta=0.99, mini_batch_size=6):
        self.beta = beta
        self.init_batch_size = 100
        self.mini_batch_size = mini_batch_size
        self.grads_used = []

        print("Estimator: SGDm")
        print("-"*20)
        print("beta:", self.beta)
        print("batch_size:", self.mini_batch_size)
        print("-"*20)

    def reset_grads(self):
        self.grads_used = []

    def score_update(self, x_k, m_k_1, score_fn, std_noise_grad, rng, **kwargs):
        if m_k_1 is None:
            noise = th.randn(
                (self.init_batch_size,) + tuple(x_k.shape),
                device=x_k.device,
                generator=rng,
            ) * std_noise_grad
            m_k = noisy_score(score_fn, x_k, noise).mean(dim=0) # score_est
            ngrads_used = self.init_batch_size
        else:
            noise = th.randn(
                (self.mini_batch_size,) + tuple(x_k.shape),
                device=x_k.device,
                generator=rng,
            ) * std_noise_grad
            score_est = noisy_score(score_fn, x_k, noise).mean(dim=0)
            m_k = self.beta * m_k_1 + (1 - self.beta) * score_est
            ngrads_used = self.mini_batch_size

        self.grads_used.append(ngrads_used)
        return m_k


class STORM:
    def __init__(self, beta=0.99, mini_batch_size=3):
        self.beta = beta
        self.init_batch_size = 100
        self.mini_batch_size = mini_batch_size
        self.grads_used = []

        print("Estimator: STORM")
        print("-"*20)
        print("beta:", self.beta)
        print("batch_size:", self.mini_batch_size)
        print("-"*20)

    def reset_grads(self):
        self.grads_used = []

    def score_update(self, x_k, x_k_1, m_k_1, score_fn, std_noise_grad, rng, **kwargs):

        if m_k_1 is None:
            noise = th.randn(
                (self.init_batch_size,) + tuple(x_k.shape),
                device=x_k.device,
                generator=rng,
            ) * std_noise_grad

            m_k = noisy_score(score_fn, x_k, noise).mean(dim=0)
            ngrads_used = self.init_batch_size
        else:

            noise = th.randn(
                (self.mini_batch_size,) + tuple(x_k.shape),
                device=x_k.device,
                generator=rng,
            ) * std_noise_grad

            score_k = noisy_score(score_fn, x_k, noise).mean(dim=0)
            score_k_1 = noisy_score(score_fn, x_k_1, noise).mean(dim=0)
            m_k = score_k + self.beta * (m_k_1 - score_k_1)
            ngrads_used = self.mini_batch_size * 2

        self.grads_used.append(ngrads_used)
        return m_k


class SGDe:
    def __init__(self, beta=0.4, batch_size=10):
        self.beta = beta
        self.batch_size = batch_size
        self.init_batch_size = 100
        self.rng = th.Generator().manual_seed(42)
        self.grads_used = []

    def reset_grads(self):
        self.grads_used = []

    def score_update(self, x_k, m_k_1, score_fn, std_noise_grad, rng, **kwargs):

        if m_k_1 is None:
            noise = th.randn(
                (self.init_batch_size,) + tuple(x_k.shape),
                device=x_k.device,
                generator=rng,
            ) * std_noise_grad

            m_k = noisy_score(score_fn, x_k, noise).mean(dim=0) # score_est
            ngrads_used = self.init_batch_size
        else:
            if p < self.beta:
                m_k = m_k_1
                ngrads_used = 0 
            else:

                noise = th.randn(
                    (self.batch_size,) + tuple(x_k.shape),
                    device=x_k.device,
                    generator=rng,
                ) * std_noise_grad

                m_k = noisy_score(score_fn, x_k, noise).mean(dim=0)
                ngrads_used = self.batch_size
        
        self.grads_used.append(ngrads_used)
        return m_k


class PAGE:
    def __init__(self, beta=0.98, batch_size=100, mini_batch_size=2):
        self.beta = beta
        self.init_batch_size = 100
        self.batch_size = batch_size
        self.mini_batch_size = mini_batch_size
        self.rng = th.Generator().manual_seed(42)
        self.grads_used = []

    def reset_grads(self):
        self.grads_used = []

    def score_update(self, x_k, x_k_1, m_k_1, score_fn, std_noise_grad, rng, **kwargs):

        if m_k_1 is None:
            noise = th.randn(
                (self.init_batch_size,) + tuple(x_k.shape),
                device=x_k.device,
                generator=rng,
            ) * std_noise_grad

            m_k = noisy_score(score_fn, x_k, noise).mean(dim=0)
            ngrads_used = self.init_batch_size
        else:
            p = th.rand((), generator=self.rng).item()
            if p < self.beta:
                noise = th.randn(
                    (self.mini_batch_size,) + tuple(x_k.shape),
                    device=x_k.device,
                    generator=rng,
                ) * std_noise_grad

                score_k = noisy_score(score_fn, x_k, noise).mean(dim=0)
                score_k_1 = noisy_score(score_fn, x_k_1, noise).mean(dim=0)

                m_k = m_k_1 + score_k - score_k_1
                ngrads_used = self.mini_batch_size * 2

            else:
                noise = th.randn(
                    (self.batch_size,) + tuple(x_k.shape),
                    device=x_k.device,
                    generator=rng,
                ) * std_noise_grad

                m_k = noisy_score(score_fn, x_k, noise).mean(dim=0)
                ngrads_used = self.batch_size
                
        self.grads_used.append(ngrads_used)
        return m_k