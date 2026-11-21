"""跨粒度几何联合一致性损失 L_geo。"""
import torch
import torch.nn.functional as F


def _unit(x, dim = -1, eps = 1e-8):
    return x / x.norm(dim = dim, keepdim = True).clamp_min(eps)


def gram_volume(z, sigma = 1.0, eps = 1e-6):
    """RBF Gram 体积 V(Z) = sqrt(det(K + eps I))。z: (..., k, D)，返回 (...,)。"""
    zc = z.float()
    d2 = torch.cdist(zc, zc, p = 2.0).pow(2)                    # (...,k,k)
    K = torch.exp(-d2 / (2.0 * float(sigma) ** 2))
    k = K.shape[-1]
    I = torch.eye(k, device = K.device, dtype = K.dtype)
    sign, logabsdet = torch.linalg.slogdet(K + float(eps) * I)
    logabsdet = torch.where(sign > 0, logabsdet,
                            torch.full_like(logabsdet, -30.0))
    return torch.exp(0.5 * logabsdet.clamp(min = -30.0, max = 30.0))


def mask_to_patch_weight(mask, num_patch):
    """GT 掩码降采样到 patch 网格，返回软权重 (w_A, w_N)，形状 (N, num_patch)。"""
    if mask.dim() == 3:
        mask = mask.unsqueeze(1)
    g = int(round(num_patch ** 0.5))
    if g * g != num_patch:
        raise ValueError(f"patch 数 {num_patch} 不是完全平方数，无法还原成方形网格")
    w_A = F.adaptive_avg_pool2d(mask.float(), (g, g)).flatten(1)      # (N, g*g)
    return w_A, 1.0 - w_A


def masked_pool(patch, weight, eps = 1e-6):
    """按权重聚合 patch 特征。patch: (N,P,D)，weight: (N,P)，返回 (N,D)。"""
    w = weight.unsqueeze(-1)                                          # (N,P,1)
    return (patch * w).sum(1) / (w.sum(1) + eps)


def _drop_cls(patch):
    """patch 序列长度为平方数 + 1 时去掉 CLS token。"""
    P = patch.shape[1]
    if int(round(P ** 0.5)) ** 2 == P:
        return patch
    if int(round((P - 1) ** 0.5)) ** 2 == P - 1:
        return patch[:, 1:, :]
    raise ValueError(f"patch 序列长度 {P} 既不是完全平方数也不是平方数+1")


def geo_consistency_loss(streams, t_p, t_c, view_label, gt_mask,
                         sigma = 1.0, eps = 1e-6, delta = 0.02,
                         require_grad_check = True):
    """跨粒度几何联合一致性损失 L_geo。

    streams    : {模态名: (global_feat (N,D), patch_feat (N,P,D))}，N = b*nv
    t_p, t_c   : (N,2,D) 或 (2,D)，父/子文本 [正常, 异常]
    view_label : (N,) 或 (b,nv)，逐视角异常标签
    gt_mask    : (N,1,H,W) / (N,H,W) / (b,nv,1,H,W)，逐视角像素 GT
    delta      : hinge 间隔

    返回 (loss, stats)，stats 含 v_pos / v_neg / n_eff / n_total / hinge_active。
    """
    if not streams:
        raise ValueError("streams 不能为空")
    _ref = next(iter(streams.values()))[0]
    dev = _ref.device
    N = _ref.shape[0]

    y = view_label.reshape(-1)
    if y.shape[0] != N:
        raise ValueError(f"view_label 展平后 {y.shape[0]} 行，与特征 {N} 行不符")
    y = y.to(dev).long().clamp(0, 1)

    m = gt_mask
    if m.dim() == 5:                     # (b,nv,1,H,W)
        m = m.reshape(-1, *m.shape[2:])
    if m.dim() == 3:                     # (N,H,W)
        m = m.unsqueeze(1)
    if m.shape[0] != N:
        raise ValueError(f"gt_mask 展平后 {m.shape[0]} 行，与特征 {N} 行不符")
    m = m.to(dev)

    def _txt(t):
        t = t.to(dev)
        if t.dim() == 2:
            t = t.unsqueeze(0).expand(N, -1, -1)
        elif t.shape[0] != N:
            raise ValueError(f"文本特征 {tuple(t.shape)} 的第 0 维与 {N} 行不符")
        return _unit(t.float())

    tp, tc = _txt(t_p), _txt(t_c)
    idx = torch.arange(N, device = dev)
    tp_pos, tc_pos = tp[idx, y], tc[idx, y]                 # 正确配置的文本
    tp_neg, tc_neg = tp[idx, 1 - y], tc[idx, 1 - y]         # 错误配置的文本

    total, n_eff, v_pos_s, v_neg_s, n_act = 0.0, 0, 0.0, 0.0, 0
    for _name, (g_feat, p_feat) in streams.items():
        if g_feat.shape[0] != N or p_feat.shape[0] != N:
            raise ValueError(f"模态 {_name} 的行数与 {N} 不符")
        G = _unit(g_feat.float())                                     # (N,D)
        pf = _drop_cls(p_feat).float()                                # (N,P,D)
        w_A, w_N = mask_to_patch_weight(m, pf.shape[1])                # (N,P)
        # 异常视角取异常区域，正常视角取正常区域
        w = torch.where(y.bool().unsqueeze(-1), w_A, w_N)              # (N,P)
        L = _unit(masked_pool(pf, w))                                  # (N,D)
        valid = w.sum(1) > 1e-4

        Zp = torch.stack([G, L, tp_pos, tc_pos], dim = 1)              # (N,4,D)
        Zn = torch.stack([G, L, tp_neg, tc_neg], dim = 1)
        Vp = gram_volume(Zp, sigma = sigma, eps = eps)                 # (N,)
        Vn = gram_volume(Zn, sigma = sigma, eps = eps)
        hinge = F.relu(float(delta) + Vp - Vn)
        vm = valid.float()
        total = total + (hinge * vm).sum()
        n_eff += int(valid.sum().item())
        with torch.no_grad():
            v_pos_s += float((Vp * vm).sum().item())
            v_neg_s += float((Vn * vm).sum().item())
            n_act += int(((hinge > 0) & valid).sum().item())

    if n_eff == 0:
        loss = (tp.sum() + tc.sum()) * 0.0
    else:
        loss = total / float(n_eff)

    if require_grad_check:
        _any = any(t.requires_grad for t in
                   [g for g, _ in streams.values()] + [p for _, p in streams.values()]
                   + [t_p, t_c] if torch.is_tensor(t))
        if not _any and torch.is_grad_enabled():
            raise RuntimeError("L_geo 的所有输入都没有梯度")

    return loss, {
        'v_pos': v_pos_s / max(n_eff, 1),
        'v_neg': v_neg_s / max(n_eff, 1),
        'n_eff': float(n_eff),
        'n_total': float(N * len(streams)),
        'hinge_active': float(n_act) / max(n_eff, 1),
    }
