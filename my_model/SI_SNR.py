import torch

EPS = 1e-4
def SISNR_loss(source, estimate_source):
    """Calcuate Scale-Invariant Source-to-Noise Ratio (SI-SNR)
    Args:
        source: torch tensor, [batch size, sequence length]
        estimate_source: torch tensor, [batch size, sequence length]
    Returns:
        SISNR, [batch size]
    """
    assert source.size() == estimate_source.size()
    # Step 1. Zero-mean norm
    source = source - torch.mean(source, axis = -1, keepdim=True)
    estimate_source = estimate_source - torch.mean(estimate_source, axis = -1, keepdim=True)
    # Step 2. SI-SNR
    # s_target = <s', s>s / ||s||^2
    ref_energy = torch.sum(source ** 2, axis = -1, keepdim=True) + EPS
    proj = torch.sum(source * estimate_source, axis = -1, keepdim=True) * source / ref_energy
    # e_noise = s' - s_target
    noise = estimate_source - proj
    # SI-SNR = 10 * log_10(||s_target||^2 / ||e_noise||^2)
    ratio = torch.sum(proj ** 2, axis = -1) / (torch.sum(noise ** 2, axis = -1) + EPS)

    # sisnr = 10 * torch.log10(ratio + EPS)
    sisnr = 10 * torch.log10(ratio + EPS) #no *10
    # if torch.isnan(sisnr).any():
    #     print("=== ratio 出现 NaN ===")
    #     print("source stats: min", source.min(), "max", source.max())
    #     print("estimate_source stats: min", estimate_source.min(), "max", estimate_source.max())
    #     print("ref_energy stats: min", ref_energy.min(), "max", ref_energy.max())
    #     print("proj stats: min", proj.min(), "max", proj.max())
    #     print("noise stats: min", noise.min(), "max", noise.max())
    #     print("ratio stats: ", ratio)
    #     raise ValueError(f"检测到 NaN 值！NaN 位置：{torch.isnan(sisnr).nonzero()}, ratio 值：{ratio[torch.isnan(ratio)]}")

    loss = torch.exp(- sisnr * 0.1)
    return torch.mean(loss)

def n_SISNR_loss(source, estimate_source):
    """Calcuate Scale-Invariant Source-to-Noise Ratio (SI-SNR)
    Args:
        source: torch tensor, [batch size, sequence length]
        estimate_source: torch tensor, [batch size, sequence length]
    Returns:
        SISNR, [batch size]
    """
    assert source.size() == estimate_source.size()
    # Step 1. Zero-mean norm
    source = source - torch.mean(source, axis = -1, keepdim=True)
    estimate_source = estimate_source - torch.mean(estimate_source, axis = -1, keepdim=True)
    # Step 2. SI-SNR
    # s_target = <s', s>s / ||s||^2
    ref_energy = torch.sum(source ** 2, axis = -1, keepdim=True) + EPS
    proj = torch.sum(source * estimate_source, axis = -1, keepdim=True) * source / ref_energy
    # e_noise = s' - s_target
    noise = estimate_source - proj
    # SI-SNR = 10 * log_10(||s_target||^2 / ||e_noise||^2)
    ratio = torch.sum(proj ** 2, axis = -1) / (torch.sum(noise ** 2, axis = -1) + EPS)

    # sisnr = 10 * torch.log10(ratio + EPS)
    sisnr = 10*torch.log10(ratio + EPS) #no *10

    return -torch.mean(sisnr)


def SISNR_score(source, estimate_source):
    assert source.size() == estimate_source.size()
    # Step 1. Zero-mean norm
    source = source - torch.mean(source, axis = -1, keepdim=True)
    estimate_source = estimate_source - torch.mean(estimate_source, axis = -1, keepdim=True)
    # Step 2. SI-SNR
    # s_target = <s', s>s / ||s||^2
    ref_energy = torch.sum(source ** 2, axis = -1, keepdim=True) + EPS
    proj = torch.sum(source * estimate_source, axis = -1, keepdim=True) * source / ref_energy
    # e_noise = s' - s_target
    noise = estimate_source - proj
    # SI-SNR = 10 * log_10(||s_target||^2 / ||e_noise||^2)
    ratio = torch.sum(proj ** 2, axis = -1) / (torch.sum(noise ** 2, axis = -1) + EPS)
    if (ratio < 0).any():
        raise ValueError(f"SI-SNR ratio has negative values: {ratio[ratio < 0]}")
    # sisnr = 10 * torch.log10(ratio + EPS)
    sisnr = 10 * torch.log10(ratio + EPS)

    return torch.mean(sisnr)

