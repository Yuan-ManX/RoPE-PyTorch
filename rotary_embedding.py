from math import pi, log
from typing import Literal

import torch
from torch.amp import autocast
from torch.nn import Module, ModuleList
from torch import nn, einsum, broadcast_tensors, Tensor
from einops import rearrange, repeat


def exists(val):
    """
    检查变量是否存在（不为 None）。

    参数:
        val: 任意变量。

    返回:
        bool: 如果 val 不为 None，则返回 True，否则返回 False。
    """
    return val is not None


def default(val, d):
    """
    如果变量存在（不为 None），则返回变量本身；否则返回默认值。

    参数:
        val: 任意变量。
        d: 默认值。

    返回:
        任意类型: 如果 val 存在，则返回 val；否则返回 d。
    """
    return val if exists(val) else d


def broadcat(tensors, dim = -1):
    """
    在指定维度上连接多个张量，并进行广播。

    参数:
        tensors (Tuple[Tensor, ...]): 需要连接的多个张量。
        dim (int, 可选): 指定连接的维度，默认为最后一个维度（-1）。

    返回:
        Tensor: 连接后的张量。
    """
    # 使用 broadcast_tensors 对输入的张量进行广播，使其具有相同的形状
    broadcasted_tensors = broadcast_tensors(*tensors)
    # 在指定维度上连接广播后的张量
    return torch.cat(broadcasted_tensors, dim = dim)


def slice_at_dim(t, dim_slice: slice, *, dim):
    """
    在指定维度上对张量进行切片。

    参数:
        t (Tensor): 输入张量。
        dim_slice (slice): 切片对象，指定切片的范围。
        dim (int): 指定切片的维度。

    返回:
        Tensor: 切片后的张量。
    """
    # 调整维度索引，处理负数维度
    dim += (t.ndim if dim < 0 else 0)
    # 创建包含切片信息的列表
    colons = [slice(None)] * t.ndim
    colons[dim] = dim_slice
    # 应用切片并返回结果
    return t[tuple(colons)]


def rotate_half(x):
    """
    对输入张量的后半部分进行旋转。

    参数:
        x (Tensor): 输入张量，形状为 (..., d * 2)，其中 d 是特征维度。

    返回:
        Tensor: 旋转后的张量，形状与输入相同。
    """
    # 重塑张量形状为 (..., d, 2)
    x = rearrange(x, '... (d r) -> ... d r', r = 2)
    # 解绑最后一个维度，得到两个张量 x1 和 x2
    x1, x2 = x.unbind(dim = -1)
    # 构建旋转后的张量：(-x2, x1)
    x = torch.stack((-x2, x1), dim = -1)
    # 将张量重塑回原始形状 (..., d * 2)
    return rearrange(x, '... d r -> ... (d r)')


@autocast('cuda', enabled = False)
def apply_rotary_emb(
    freqs,
    t,
    start_index = 0,
    scale = 1.,
    seq_dim = -2,
    freqs_seq_dim = None
):
    """
    应用旋转位置嵌入。

    参数:
        freqs (Tensor): 旋转频率张量，形状为 (..., rot_dim)。
        t (Tensor): 输入张量，形状为 (..., seq_len, dim)。
        start_index (int, 可选): 开始索引，默认为0。
        scale (float, 可选): 缩放因子，默认为1。
        seq_dim (int, 可选): 序列维度，默认为倒数第二个维度。
        freqs_seq_dim (int, 可选): 频率序列维度，默认为 None。

    返回:
        Tensor: 应用旋转位置嵌入后的张量。
    """
    # 获取张量数据类型
    dtype = t.dtype
    
    # 确定频率序列维度
    if not exists(freqs_seq_dim):
        if freqs.ndim == 2 or t.ndim == 3:
            freqs_seq_dim = 0

    # 如果输入张量是三维的或指定了频率序列维度，则调整频率张量以匹配序列长度
    if t.ndim == 3 or exists(freqs_seq_dim):
        seq_len = t.shape[seq_dim]
        freqs = slice_at_dim(freqs, slice(-seq_len, None), dim = freqs_seq_dim)

    # 获取旋转维度
    rot_dim = freqs.shape[-1]
    # 计算结束索引
    end_index = start_index + rot_dim

    # 确保旋转维度小于等于输入张量的特征维度
    assert rot_dim <= t.shape[-1], f'feature dimension {t.shape[-1]} is not of sufficient size to rotate in all the positions {rot_dim}'

    # 将输入张量分割为三部分：左侧、中间（需要变换）和右侧
    t_left = t[..., :start_index]
    t_middle = t[..., start_index:end_index]
    t_right = t[..., end_index:]

    # 应用旋转位置嵌入，不修改 t 的原地数据
    t_transformed = (t_middle * freqs.cos() * scale) + (rotate_half(t_middle) * freqs.sin() * scale)
    
    # 将变换后的中间部分与左右两侧连接起来
    out = torch.cat((t_left, t_transformed, t_right), dim=-1)

    # 返回结果，并转换回原始数据类型
    return out.type(dtype)


def apply_learned_rotations(rotations, t, start_index = 0, freq_ranges = None):
    """
    应用学习到的旋转。

    参数:
        rotations (Tensor): 旋转张量。
        t (Tensor): 输入张量。
        start_index (int, 可选): 开始索引，默认为0。
        freq_ranges (Tensor, 可选): 频率范围张量。

    返回:
        Tensor: 应用旋转后的张量。
    """
    if exists(freq_ranges):
        # 使用 einsum 对旋转张量和频率范围张量进行批量操作
        rotations = einsum('..., f -> ... f', rotations, freq_ranges)
        # 重塑旋转张量形状为 (..., r * f)
        rotations = rearrange(rotations, '... r f -> ... (r f)')

    # 重复旋转张量，使其形状为 (..., n * 2)
    rotations = repeat(rotations, '... n -> ... (n r)', r = 2)
    # 应用旋转位置嵌入
    return apply_rotary_emb(rotations, t, start_index = start_index)


class RotaryEmbedding(Module):
    """
    旋转位置嵌入（Rotary Position Embedding, RoPE）模块。

    RoPE 是一种位置嵌入方法，通过对输入张量应用旋转矩阵，将位置信息编码到注意力机制中。
    该实现支持多种频率生成方式，并支持缓存和插值以提高效率。
    """
    def __init__(
        self,
        dim,
        custom_freqs: Tensor | None = None,
        freqs_for:  Literal['lang', 'pixel', 'constant'] = 'lang',
        theta = 10000,
        max_freq = 10,
        num_freqs = 1,
        learned_freq = False,
        use_xpos = False,
        xpos_scale_base = 512,
        interpolate_factor = 1.,
        theta_rescale_factor = 1.,
        seq_before_head_dim = False,
        cache_if_possible = True,
        cache_max_seq_len = 8192
    ):
        """
        初始化旋转位置嵌入模块。

        参数:
            dim (int): 嵌入维度，例如4096。
            custom_freqs (Tensor, 可选): 自定义频率张量，默认为 None。
            freqs_for (Literal['lang', 'pixel', 'constant'], 可选): 频率生成方式，默认为 'lang'。
                - 'lang': 语言模型使用的频率生成方式。
                - 'pixel': 像素模型使用的频率生成方式。
                - 'constant': 使用常数频率。
            theta (float, 可选): 旋转角度的基数，默认为10000。
            max_freq (float, 可选): 最大频率，默认为10。
            num_freqs (int, 可选): 频率的数量，默认为1。
            learned_freq (bool, 可选): 是否学习频率，默认为 False。
            use_xpos (bool, 可选): 是否使用 XPos 位置编码，默认为 False。
            xpos_scale_base (int, 可选): XPos 的缩放基数，默认为512。
            interpolate_factor (float, 可选): 插值因子，默认为1。
            theta_rescale_factor (float, 可选): θ 重缩放因子，默认为1。
            seq_before_head_dim (bool, 可选): 是否在多头维度之前应用序列维度，默认为 False。
            cache_if_possible (bool, 可选): 是否尽可能缓存频率，默认为 True。
            cache_max_seq_len (int, 可选): 缓存的最大序列长度，默认为8192。
        """
        super().__init__()
        # 对 θ 进行重缩放
        theta *= theta_rescale_factor ** (dim / (dim - 2))
        # 保存频率生成方式
        self.freqs_for = freqs_for

        if exists(custom_freqs):
            # 如果提供了自定义频率，则使用自定义频率
            freqs = custom_freqs
        elif freqs_for == 'lang':
            # 语言模型使用的频率生成方式
            freqs = 1. / (theta ** (torch.arange(0, dim, 2)[:(dim // 2)].float() / dim))
        elif freqs_for == 'pixel':
            # 像素模型使用的频率生成方式
            freqs = torch.linspace(1., max_freq / 2, dim // 2) * pi
        elif freqs_for == 'constant':
            # 使用常数频率
            freqs = torch.ones(num_freqs).float()

        # 是否缓存频率
        self.cache_if_possible = cache_if_possible
        # 缓存的最大序列长度
        self.cache_max_seq_len = cache_max_seq_len

        # 注册缓存频率张量，形状为 (cache_max_seq_len, dim)，并设置 persistent=False 以避免保存到模型状态中
        self.register_buffer('cached_freqs', torch.zeros(cache_max_seq_len, dim), persistent = False)
        # 初始化缓存频率的序列长度
        self.cached_freqs_seq_len = 0

        # 将频率参数注册为模型的可学习参数，如果 learned_freq 为 True，则需要梯度更新
        self.freqs = nn.Parameter(freqs, requires_grad = learned_freq)

        # 保存是否学习频率的标志
        self.learned_freq = learned_freq

        # 注册一个虚拟缓冲区，用于设备一致性（占位符）
        self.register_buffer('dummy', torch.tensor(0), persistent = False)

        # 默认的序列维度
        # 是否在多头维度之前应用序列维度
        self.seq_before_head_dim = seq_before_head_dim
        # 设置默认的序列维度
        self.default_seq_dim = -3 if seq_before_head_dim else -2

        # 插值因子
        assert interpolate_factor >= 1.
        # 保存插值因子
        self.interpolate_factor = interpolate_factor

        # XPos 位置编码
        self.use_xpos = use_xpos

        if not use_xpos:
            return

        # 计算 XPos 的缩放因子
        scale = (torch.arange(0, dim, 2) + 0.4 * dim) / (1.4 * dim)
        # 保存 XPos 的缩放基数
        self.scale_base = xpos_scale_base

        # 注册缩放因子和缓存缩放因子缓冲区
        self.register_buffer('scale', scale, persistent = False)
        self.register_buffer('cached_scales', torch.zeros(cache_max_seq_len, dim), persistent = False)
        # 初始化缓存缩放因子的序列长度
        self.cached_scales_seq_len = 0

        # 将 apply_rotary_emb 方法注册为静态方法
        self.apply_rotary_emb = staticmethod(apply_rotary_emb)

    @property
    def device(self):
        """
        获取当前设备。

        Returns:
            torch.device: 当前设备（CPU 或 GPU）。
        """
        return self.dummy.device

    def get_seq_pos(self, seq_len, device, dtype, offset = 0):
        """
        生成序列位置的嵌入。

        参数:
            seq_len (int): 序列长度。
            device (torch.device): 设备（CPU 或 GPU）。
            dtype (torch.dtype): 数据类型。
            offset (int, 可选): 位置偏移量，默认为0。

        Returns:
            torch.Tensor: 生成的序列位置嵌入，形状为 (seq_len,)。
        """
        return (torch.arange(seq_len, device = device, dtype = dtype) + offset) / self.interpolate_factor

    def rotate_queries_or_keys(self, t, seq_dim = None, offset = 0, scale = None):
        """
        对查询或键张量应用旋转位置嵌入。

        参数:
            t (torch.Tensor): 输入张量，形状为 (..., seq_len, dim)。
            seq_dim (int, 可选): 序列维度，默认为 None。如果为 None，则使用默认的序列维度。
            offset (int, 可选): 位置偏移量，默认为0。
            scale (float, 可选): 缩放因子，默认为 None。

        Returns:
            torch.Tensor: 应用旋转位置嵌入后的张量。
        """
        # 设置序列维度，如果未指定，则使用默认的序列维度
        seq_dim = default(seq_dim, self.default_seq_dim)

        assert not self.use_xpos or exists(scale), 'you must use `.rotate_queries_and_keys` method instead and pass in both queries and keys, for length extrapolatable rotary embeddings'

        # 获取设备、数据类型和序列长度
        device, dtype, seq_len = t.device, t.dtype, t.shape[seq_dim]

        # 生成序列位置嵌入
        seq = self.get_seq_pos(seq_len, device = device, dtype = dtype, offset = offset)

        # 生成频率张量
        freqs = self.forward(seq, seq_len = seq_len, offset = offset)

        # 如果序列维度是 -3（多头维度），则调整频率张量形状
        if seq_dim == -3:
            freqs = rearrange(freqs, 'n d -> n 1 d')

        # 应用旋转位置嵌入
        return apply_rotary_emb(freqs, t, scale = default(scale, 1.), seq_dim = seq_dim)

    def rotate_queries_with_cached_keys(self, q, k, seq_dim = None, offset = 0):
        """
        对查询和键张量应用旋转位置嵌入，使用缓存的键进行优化。

        参数:
            q (torch.Tensor): 查询张量，形状为 (..., q_seq_len, dim)。
            k (torch.Tensor): 键张量，形状为 (..., k_seq_len, dim)。
            seq_dim (int, 可选): 序列维度，默认为 None。如果为 None，则使用默认的序列维度。
            offset (int, 可选): 位置偏移量，默认为0。

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: 应用旋转位置嵌入后的查询和键张量。
        """
        # 获取数据类型、设备和序列维度
        dtype, device, seq_dim = q.dtype, q.device, default(seq_dim, self.default_seq_dim)

        # 获取查询和键的序列长度
        q_len, k_len = q.shape[seq_dim], k.shape[seq_dim]
        assert q_len <= k_len

        # 初始化查询和键的缩放因子
        q_scale = k_scale = 1.

        if self.use_xpos:
            # 如果使用 XPos，则生成序列位置嵌入，并计算查询和键的缩放因子
            seq = self.get_seq_pos(k_len, dtype = dtype, device = device)

            q_scale = self.get_scale(seq[-q_len:]).type(dtype)
            k_scale = self.get_scale(seq).type(dtype)

        # 对查询和键应用旋转位置嵌入
        rotated_q = self.rotate_queries_or_keys(q, seq_dim = seq_dim, scale = q_scale, offset = k_len - q_len + offset)
        rotated_k = self.rotate_queries_or_keys(k, seq_dim = seq_dim, scale = k_scale ** -1)

        # 转换查询张量类型
        rotated_q = rotated_q.type(q.dtype)
        # 转换键张量类型
        rotated_k = rotated_k.type(k.dtype)

        # 返回旋转后的查询和键张量
        return rotated_q, rotated_k

    def rotate_queries_and_keys(self, q, k, seq_dim = None):
        """
        对查询和键张量应用旋转位置嵌入。

        参数:
            q (torch.Tensor): 查询张量，形状为 (..., q_seq_len, dim)。
            k (torch.Tensor): 键张量，形状为 (..., k_seq_len, dim)。
            seq_dim (int, 可选): 序列维度，默认为 None。如果为 None，则使用默认的序列维度。

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: 应用旋转位置嵌入后的查询和键张量。
        """
        # 设置序列维度，如果未指定，则使用默认的序列维度
        seq_dim = default(seq_dim, self.default_seq_dim)

        assert self.use_xpos
        # 获取设备、数据类型和序列长度
        device, dtype, seq_len = q.device, q.dtype, q.shape[seq_dim]

        # 生成序列位置嵌入
        seq = self.get_seq_pos(seq_len, dtype = dtype, device = device)

        # 生成频率张量
        freqs = self.forward(seq, seq_len = seq_len)
        # 计算缩放因子
        scale = self.get_scale(seq, seq_len = seq_len).to(dtype)

        # 如果序列维度是 -3（多头维度），则调整频率和缩放因子张量形状
        if seq_dim == -3:
            freqs = rearrange(freqs, 'n d -> n 1 d')
            scale = rearrange(scale, 'n d -> n 1 d')

        # 对查询和键应用旋转位置嵌入
        rotated_q = apply_rotary_emb(freqs, q, scale = scale, seq_dim = seq_dim)
        rotated_k = apply_rotary_emb(freqs, k, scale = scale ** -1, seq_dim = seq_dim)

        # 转换查询张量类型
        rotated_q = rotated_q.type(q.dtype)
        # 转换键张量类型
        rotated_k = rotated_k.type(k.dtype)

        # 返回旋转后的查询和键张量
        return rotated_q, rotated_k

    def get_scale(
        self,
        t: Tensor,
        seq_len: int | None = None,
        offset = 0
    ):
        """
        计算缩放因子，用于 XPos 位置编码。

        参数:
            t (Tensor): 输入张量，形状为 (seq_len,)。
            seq_len (int, 可选): 序列长度，默认为 None。
            offset (int, 可选): 位置偏移量，默认为0。

        返回:
            Tensor: 计算得到的缩放因子，形状为 (seq_len, 1)。
        """
        assert self.use_xpos

        # 判断是否需要缓存缩放因子
        should_cache = (
            self.cache_if_possible and
            exists(seq_len) and
            (offset + seq_len) <= self.cache_max_seq_len
        )

        # 如果需要缓存，并且缓存的缩放因子已经存在且满足条件，则返回缓存的缩放因子
        if (
            should_cache and \
            exists(self.cached_scales) and \
            (seq_len + offset) <= self.cached_scales_seq_len
        ):
            return self.cached_scales[offset:(offset + seq_len)]

        # 初始化缩放因子为1
        scale = 1.
        if self.use_xpos:
            # 计算幂指数： (t - len(t) // 2) / scale_base
            power = (t - len(t) // 2) / self.scale_base
            # 计算缩放因子： scale ** power，并重塑形状为 (n, 1)
            scale = self.scale ** rearrange(power, 'n -> n 1')
            # 重复缩放因子，使其形状为 (n, d * 2)
            scale = repeat(scale, 'n d -> n (d r)', r = 2)

        # 如果需要缓存，并且偏移量为0，则将缩放因子缓存到 cached_scales 中
        if should_cache and offset == 0:
            # 缓存缩放因子，并分离梯度
            self.cached_scales[:seq_len] = scale.detach()
            # 更新缓存的序列长度
            self.cached_scales_seq_len = seq_len
        # 返回缩放因子
        return scale

    def get_axial_freqs(self, *dims):
        """
        生成轴向频率，用于多维数据的旋转位置嵌入。

        参数:
            *dims (int): 每个维度的长度。

        返回:
            Tensor: 生成的轴向频率，形状为 (..., dim1, dim2, ..., dimN, freq_dim)。
        """
        # 定义切片对象，用于选择所有元素
        Colon = slice(None)
        # 初始化频率列表
        all_freqs = []

        for ind, dim in enumerate(dims):
            if self.freqs_for == 'pixel':
                # 如果频率生成方式为 'pixel'，则生成线性空间频率
                pos = torch.linspace(-1, 1, steps = dim, device = self.device)
            else:
                # 否则，生成整数序列作为位置
                pos = torch.arange(dim, device = self.device)

            # 生成频率张量
            freqs = self.forward(pos, seq_len = dim)

            # 创建所有轴的切片列表，除了当前维度
            all_axis = [None] * len(dims)
            all_axis[ind] = Colon

            # 创建新的轴切片，用于选择当前维度的频率
            new_axis_slice = (Ellipsis, *all_axis, Colon)
            all_freqs.append(freqs[new_axis_slice])

        # 对所有频率张量进行广播，使其具有相同的形状
        all_freqs = broadcast_tensors(*all_freqs)
        # 将所有频率张量在最后一个维度上连接起来
        return torch.cat(all_freqs, dim = -1)

    @autocast('cuda', enabled = False)
    def forward(
        self,
        t: Tensor,
        seq_len: int | None = None,
        offset = 0
    ):
        """
        前向传播方法，生成旋转位置嵌入频率。

        参数:
            t (Tensor): 输入张量，形状为 (seq_len,) 或 (..., seq_len)。
            seq_len (int, 可选): 序列长度，默认为 None。
            offset (int, 可选): 位置偏移量，默认为0。

        返回:
            Tensor: 生成的旋转位置嵌入频率，形状与输入张量相同。
        """
        # 判断是否需要缓存频率
        should_cache = (
            self.cache_if_possible and
            not self.learned_freq and
            exists(seq_len) and
            self.freqs_for != 'pixel' and
            (offset + seq_len) <= self.cache_max_seq_len
        )

        # 如果需要缓存，并且缓存的频率已经存在且满足条件，则返回缓存的频率
        if (
            should_cache and \
            exists(self.cached_freqs) and \
            (offset + seq_len) <= self.cached_freqs_seq_len
        ):
            return self.cached_freqs[offset:(offset + seq_len)].detach()

        # 获取频率参数
        freqs = self.freqs

        # 将输入张量与频率参数进行批量操作，生成频率张量
        freqs = einsum('..., f -> ... f', t.type(freqs.dtype), freqs)
        # 重复频率张量，使其形状为 (..., n * 2)
        freqs = repeat(freqs, '... n -> ... (n r)', r = 2)

        # 如果需要缓存，并且偏移量为0，则将频率缓存到 cached_freqs 中
        if should_cache and offset == 0:
            # 缓存频率，并分离梯度
            self.cached_freqs[:seq_len] = freqs.detach()
            # 更新缓存的序列长度
            self.cached_freqs_seq_len = seq_len

        # 返回频率张量
        return freqs
