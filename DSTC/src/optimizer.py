from torch.optim import Adam, AdamW


def get_optimizer(
        params,
        lr=1e-4,
        wd=1e-2,
        betas=(0.9, 0.99),
        eps=1e-8,
):
    has_wd = wd > 0

    if not has_wd:
        return Adam(params, lr=lr, betas=betas, eps=eps)

    else:
        return AdamW(params, lr=lr, weight_decay=wd, betas=betas, eps=eps)

