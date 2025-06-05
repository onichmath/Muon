import torch
import torch.distributed as dist
import math


def _normalize_spectral(X):
    return X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)


def _newton_schulz(X, steps):
    a, b, c = (3.4445, -4.7750, 2.0315)
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X
    return X


def _maybe_transpose(X):
    return X.mT if X.size(-2) > X.size(-1) else X


def _power_iteration_svd(X, k, num_iter=3):
    """Fast approximate SVD using power iteration for top k components."""
    m, n = X.shape

    Q = torch.randn(n, k, device=X.device, dtype=X.dtype)
    Q, _ = torch.linalg.qr(Q)

    for _ in range(num_iter):
        Q = X @ Q
        Q, _ = torch.linalg.qr(Q)
        Q = X.T @ Q
        Q, _ = torch.linalg.qr(Q)

    Y = X @ Q
    Y, _ = torch.linalg.qr(Y)

    XTY = X.T @ Y
    S = torch.norm(XTY, dim=0)

    V = XTY / (S.unsqueeze(0) + 1e-10)

    U = Y
    Vh = V.T

    return U, S, Vh


def _spectral_norm_approximation(X, num_iter=1):
    """Fast spectral norm approximation using power iteration."""
    if X.dim() == 2:
        u = torch.randn(X.size(0), 1, device=X.device, dtype=X.dtype)
        for _ in range(num_iter):
            v = X.T @ u
            v = v / v.norm()
            u = X @ v
            u = u / u.norm()
        sigma = (u.T @ X @ v).item()
        return sigma
    else:
        return X.norm()


def _fast_spectral_filter(X, top_percent, method="power_iter"):
    """Fast spectral filtering using approximation methods."""
    m, n = X.shape

    if top_percent == 0.0:
        u = torch.randn(m, 1, device=X.device, dtype=X.dtype)
        u = u / u.norm()

        for _ in range(3):
            v = X.T @ u
            v = v / v.norm()
            u = X @ v
            u = u / u.norm()

        sigma = torch.norm(X.T @ u)
        return sigma * torch.outer(u.squeeze(), v.squeeze())

    k = max(1, int(top_percent * min(m, n)))

    if k <= 3 or k >= min(m, n) * 0.9:
        U, S, Vh = torch.linalg.svd(X, full_matrices=False)
    else:
        U, S, Vh = _power_iteration_svd(X, k)

    if k <= len(S):
        return U[:, :k] @ torch.diag(S[:k]) @ Vh[:k, :]
    else:
        return U @ torch.diag(S) @ Vh


def orthogonalize(G, method="ns", steps=10, top_percent=0.1):
    assert G.ndim >= 2
    orig_shape = G.shape
    dtype = G.dtype

    X = G.bfloat16() if method.startswith("ns") else G
    X = _maybe_transpose(X)
    if method != "svd":
        X = _normalize_spectral(X)

    if method.startswith("ns"):
        X = _newton_schulz(X, steps).float()

    if method in ["svd", "ns", "ns_unif", "topk", "spectralnorm"] and top_percent < 1.0:
        U, S, Vh = torch.linalg.svd(X, full_matrices=False)

        if top_percent == 0.0:
            k = 1
        else:
            total_singular = S.size(-1)
            k = max(1, math.ceil(top_percent * total_singular))
        S_trunc = torch.zeros_like(S)
        if method == "ns_unif":
            S_trunc[..., :k] = 1.0
        else:
            S_trunc[..., :k] = S[..., :k]
        X = U @ torch.diag_embed(S_trunc) @ Vh

    elif (
        method
        in ["svd_fast", "ns_fast", "ns_unif_fast", "topk_fast", "spectralnorm_fast"]
        and top_percent < 1.0
    ):
        if X.numel() > 10000:
            X = _fast_spectral_filter(X, top_percent, method="power_iter")
        else:
            U, S, Vh = torch.linalg.svd(X, full_matrices=False)
            if top_percent == 0.0:
                k = 1
            else:
                total_singular = S.size(-1)
                k = max(1, math.ceil(top_percent * total_singular))
            S_trunc = torch.zeros_like(S)
            if method == "ns_unif_fast":
                S_trunc[..., :k] = 1.0
            else:
                S_trunc[..., :k] = S[..., :k]
            X = U @ torch.diag_embed(S_trunc) @ Vh

    return _maybe_transpose(X).view(orig_shape).to(dtype)


def orthogonal_update(
    grad, momentum, beta, method, steps=10, top_percent=0.1, nesterov=True
):
    momentum.lerp_(grad, 1 - beta)
    update = grad.lerp(momentum, beta) if nesterov else momentum
    if update.ndim == 4:
        update = update.view(len(update), -1)
    update = orthogonalize(update, method=method, steps=steps, top_percent=top_percent)
    update *= max(1, grad.size(-2) / grad.size(-1)) ** 0.5
    return update


def adam_update(grad, buf1, buf2, step, betas, eps):
    buf1.lerp_(grad, 1 - betas[0])
    buf2.lerp_(grad.square(), 1 - betas[1])
    buf1c = buf1 / (1 - betas[0] ** step)
    buf2c = buf2 / (1 - betas[1] ** step)
    return buf1c / (buf2c.sqrt() + eps)


class UnifiedOptimizer(torch.optim.Optimizer):
    def __init__(self, param_groups, distributed=False):
        for group in param_groups:
            assert "use_muon" in group
            if group["use_muon"]:
                group.setdefault("lr", 0.02)
                group.setdefault("momentum", 0.95)
                group.setdefault("weight_decay", 0.0)
                group.setdefault("ns_steps", 10)
                group.setdefault("method", "ns")
                group.setdefault("top_percent", 1.0)
            else:
                group.setdefault("lr", 3e-4)
                group.setdefault("betas", (0.9, 0.95))
                group.setdefault("eps", 1e-10)
                group.setdefault("weight_decay", 0.0)
        super().__init__(param_groups, {})
        self.distributed = distributed

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            if group["use_muon"]:
                params = group["params"]
                if self.distributed:
                    pad = [torch.empty_like(params[-1])] * (
                        len(params) % dist.get_world_size()
                    )
                    params += pad
                for base_i in range(
                    0, len(params), dist.get_world_size() if self.distributed else 1
                ):
                    rank_i = base_i + (dist.get_rank() if self.distributed else 0)
                    if rank_i < len(params):
                        p = params[rank_i]
                        if p.grad is None:
                            continue
                        state = self.state.setdefault(
                            p, {"momentum_buffer": torch.zeros_like(p)}
                        )
                        update = orthogonal_update(
                            p.grad,
                            state["momentum_buffer"],
                            beta=group["momentum"],
                            method=group["method"],
                            steps=group["ns_steps"],
                            top_percent=group["top_percent"],
                        )
                        p.mul_(1 - group["lr"] * group["weight_decay"])
                        p.add_(update, alpha=-group["lr"])
                    if self.distributed:
                        dist.all_gather(
                            params[base_i : base_i + dist.get_world_size()],
                            params[rank_i],
                        )
            else:
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    state = self.state.setdefault(
                        p,
                        {
                            "exp_avg": torch.zeros_like(p),
                            "exp_avg_sq": torch.zeros_like(p),
                            "step": 0,
                        },
                    )
                    state["step"] += 1
                    update = adam_update(
                        p.grad,
                        state["exp_avg"],
                        state["exp_avg_sq"],
                        state["step"],
                        group["betas"],
                        group["eps"],
                    )
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update, alpha=-group["lr"])
