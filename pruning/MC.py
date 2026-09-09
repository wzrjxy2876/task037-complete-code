import csv
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import numpy as np
from timm.models.layers import DropPath, trunc_normal_
from functools import reduce, lru_cache
from operator import mul
from einops import rearrange
import math
import tqdm
from collections import Counter, defaultdict

from contribution_field_loader import load_contribution_fields
from cost_decoupled_selection import (
    mark_selected,
    rank_candidates,
    validate_selection_cost_mode,
)
from coverage_selector import functional_coverage, greedy_coverage_repair
from functional_competition_pruning import (
    EXPECTED_DESCRIPTOR_UNITS,
    EXPECTED_NPZ_ARRAYS,
    EXPECTED_PRUNING_LAYERS,
    EXPECTED_VIDEO_FIELDS,
    FunctionalCompetitionPruner,
    LayerWiseContributionFieldArchive,
    run_one_domain_numerical_preflight,
    validate_functional_score_mode,
    write_functional_selection_artifacts,
)
from temporal_dynamicity import (
    assemble_descriptor_variant,
    compute_temporal_dynamicity,
    descriptor_variant_label,
)
from task012_diagnostics import minimum_keep_units




# =============================================================================
# 基于三维描述符与力场Mean-Shift的结构化剪枝器
# =============================================================================

class InteractionPruner:
    """
    三维描述符体系 + 力场Mean-Shift分组 + 流形一致性评分 + 迭代剪枝微调
    
    核心创新：
    1. 绝对重要性：时序平均激活幅度 + 时序激活频率（加权融合）
    2. 相对重要性：绝对重要性 × (1 - 可替代性分数)（舒尔补计算）
    3. 时空信息：响应轨迹各向异性 × 时间覆盖度
    4. 力场Mean-Shift：高斯吸引力场驱动，自然形成功能簇
    5. 流形一致性：流线平行度 × 功能强度
    """
    
    def __init__(self, model, target_sparsity=0.5, iter_prune_steps=5, 
                 finetune_epochs=5, finetune_lr=1e-5, gamma_decay=0.5,
                 min_keep_ratio=0.1, sigma=0.09, importance_alpha=0.5,
                 activation_threshold_mode='mean', importance_norm='robust_minmax',
                 norm_q_low=0.01, norm_q_high=0.99, selection_mode="bms",
                 contribution_npz=None, descriptor_variant="dynamic3d",
                 descriptor_statistics_path=None,
                 diagnostic_min_attention_heads=0,
                 selection_cost_mode="coupled",
                 selection_trace_enabled=False,
                 functional_cache_dir=None,
                 functional_audit_dir=None,
                 functional_score="domain_average",
                 functional_descriptor_cache=None):
        """
        Args:
            model: 待剪枝模型
            target_sparsity: 目标稀疏度（0-1）
            gamma_decay: 旧版跨层描述符参数，仅为调用接口兼容而保留
            min_keep_ratio: 每层最小保留比例
            sigma: Mean-Shift高斯核带宽
            importance_alpha: 激活幅值在绝对重要性中的融合权重，频率权重为1-alpha
            activation_threshold_mode: 激活频率阈值模式，可选'mean'、'median'、'fixed'
            importance_norm: 幅值与频率独立归一化方式，可选'robust_minmax'、'minmax'
            norm_q_low: robust_minmax的下分位点
            norm_q_high: robust_minmax的上分位点
            selection_mode: 剪枝选择后端，"bms"、"coverage"或"functional"
            contribution_npz: Coverage后端使用的Contribution Field文件
            descriptor_variant: "abs_rel"、"old3d"或"dynamic3d"
            descriptor_statistics_path: 每单元描述符统计CSV输出路径
            diagnostic_min_attention_heads: Task012诊断专用；0保持原规则，
                2/3仅提高Attention最小保留头数，不改变MLP或参数预算
            selection_cost_mode: "coupled"保持Task012选择；"decoupled"仅让
                参数代价参与全局预算累计与停止，不参与候选准入
            selection_trace_enabled: Task013审计专用，记录少量选择标量
            functional_cache_dir: Task014对齐Field的共享mmap目录
            functional_audit_dir: Task014严格映射与选择证据输出目录
            functional_score: Task016跨域功能损失，domain_average精确保留
                Task014，domain_total移除固定有效需求数的平均因子
            functional_descriptor_cache: Task016可验证复用的Dynamic3D逐单元CSV
        """
        if selection_mode not in ("bms", "coverage", "functional"):
            raise ValueError(
                "selection_mode must be 'bms', 'coverage', or 'functional', "
                f"got {selection_mode!r}"
            )
        if descriptor_variant not in ("abs_rel", "old3d", "dynamic3d"):
            raise ValueError(
                "descriptor_variant must be 'abs_rel', 'old3d', or "
                f"'dynamic3d', got {descriptor_variant!r}"
            )
        self.model = model
        self.target_sparsity = target_sparsity
        self.iter_prune_steps = iter_prune_steps
        self.finetune_epochs = finetune_epochs
        self.finetune_lr = finetune_lr
        self.gamma_decay = gamma_decay
        self.min_keep_ratio = min_keep_ratio
        self.sigma = sigma

        if not 0.0 <= importance_alpha <= 1.0:
            raise ValueError(f"importance_alpha必须位于[0, 1]，当前为{importance_alpha}")
        if activation_threshold_mode not in {'mean', 'median', 'fixed'}:
            raise ValueError(
                "activation_threshold_mode仅支持'mean'、'median'或'fixed'，"
                f"当前为{activation_threshold_mode}"
            )
        if importance_norm not in {'robust_minmax', 'minmax'}:
            raise ValueError(
                "importance_norm仅支持'robust_minmax'或'minmax'，"
                f"当前为{importance_norm}"
            )
        if not 0.0 <= norm_q_low < norm_q_high <= 1.0:
            raise ValueError(
                f"归一化分位点需满足0<=q_low<q_high<=1，当前为"
                f"({norm_q_low}, {norm_q_high})"
            )

        self.importance_alpha = float(importance_alpha)
        self.activation_threshold_mode = activation_threshold_mode
        self.importance_norm = importance_norm
        self.norm_q_low = float(norm_q_low)
        self.norm_q_high = float(norm_q_high)
        self.importance_eps = 1e-8
        self.selection_mode = selection_mode
        self.contribution_npz = contribution_npz
        self.descriptor_variant = descriptor_variant
        self.descriptor_statistics_path = (
            Path(descriptor_statistics_path)
            if descriptor_statistics_path is not None else None
        )
        if int(diagnostic_min_attention_heads) not in (0, 2, 3):
            raise ValueError(
                "diagnostic_min_attention_heads must be one of (0, 2, 3)"
            )
        self.diagnostic_min_attention_heads = int(
            diagnostic_min_attention_heads
        )
        self.selection_cost_mode = validate_selection_cost_mode(
            selection_cost_mode
        )
        self.selection_trace_enabled = bool(selection_trace_enabled)
        self.selection_candidates = []
        self.functional_cache_dir = Path(
            functional_cache_dir or "task014_functional_pruning/field_cache"
        )
        self.functional_audit_dir = Path(
            functional_audit_dir or "task014_functional_pruning/functional_domains"
        )
        self.functional_mapping_audit = None
        self.functional_zero_field_audit = None
        self.functional_preflight_result = None
        self.functional_selection_result = None
        self.functional_score = validate_functional_score_mode(functional_score)
        self.functional_descriptor_cache = (
            Path(functional_descriptor_cache)
            if functional_descriptor_cache is not None else None
        )
        self.max_activation_samples = 2048
        self.collect_activation_samples = True
        
        self.activations = {}
        self.spatiotemporal_scores = {}
        self.temporal_dynamicity_scores = {}
        self.hooks = []
        self.device = next(model.parameters()).device
        self.ordered_layer_names = []
        self._tdd_logged_layers = set()
        
        # 统计信息
        self.total_original_params = sum(p.numel() for p in model.parameters())
        self.current_sparsity = 0.0
        self.pruning_history = []

    def _minimum_keep_units(self, module, total_units):
        """Return Task011's rule plus the explicit Task012 Attention probe."""
        unit_type = (
            'head'
            if "WindowAttention3D" in module.__class__.__name__
            else 'neuron'
        )
        return minimum_keep_units(
            total_units,
            self.min_keep_ratio,
            unit_type,
            self.diagnostic_min_attention_heads,
        )
        
    # =========================================================================
    # Hook 注册与激活捕获
    # =========================================================================
    
    def register_hooks(self):
        """
        注册前向钩子，捕获注意力头和FFN神经元的激活特征
        """
        self.hooks = []
        self.activations = {}
        self.spatiotemporal_scores = {}
        self.temporal_dynamicity_scores = {}
        self.ordered_layer_names = []
        self._tdd_logged_layers = set()
        
        for name, m in self.model.named_modules():
            classname = m.__class__.__name__
            
            # WindowAttention3D: 捕获每个注意力头的激活
            if "WindowAttention3D" in classname:
                num_heads = m.num_heads if hasattr(m, 'num_heads') else \
                    m.qkv.out_features // (3 * m.head_dim)
                h = m.proj.register_forward_pre_hook(
                    self._get_activation_hook(
                        name, 'attention', num_heads, m.head_dim,
                        attention_module=m
                    )
                )
                self.hooks.append(h)
                self.ordered_layer_names.append(name)
                
            # Mlp: 捕获每个神经元的激活
            elif "Mlp" in classname:
                h = m.fc2.register_forward_pre_hook(
                    self._get_activation_hook(name, 'mlp')
                )
                self.hooks.append(h)
                self.ordered_layer_names.append(name)

    def remove_hooks(self):
        """移除所有钩子"""
        for h in self.hooks:
            h.remove()
        self.hooks = []
        print(">>> 所有 Hooks 已摘除")

    def _attention_response_volume(
            self, x, attention_module, num_heads, head_dim):
        """将窗口注意力输出恢复为每个注意力头的全局时空响应体。

        Args:
            x: [B*nW, Wd*Wh*Ww, U*E]，投影前窗口注意力输出；
               B为视频批量，nW为每个视频的窗口数，U为注意力头数，
               E为每个头的嵌入维度。
            attention_module: 当前WindowAttention3D模块，提供本次前向的几何信息。
            num_heads: 注意力头数U。
            head_dim: 每个注意力头的嵌入维度E。

        Returns:
            response: [B, U, T, H, W]，轴依次为视频、剪枝单元、时间、
                      空间高度和空间宽度。
        """
        geometry = getattr(attention_module, '_pruning_geometry', None)
        if geometry is None:
            raise RuntimeError(
                "注意力模块缺少_pruning_geometry，无法恢复真实时间与空间轴"
            )
        if x.ndim != 3:
            raise ValueError(
                f"注意力投影输入应为[B*nW, N, C]，当前形状为{tuple(x.shape)}"
            )

        batch_size = int(geometry['batch_size'])
        depth = int(geometry['depth'])
        height = int(geometry['height'])
        width = int(geometry['width'])
        padded_depth = int(geometry['padded_depth'])
        padded_height = int(geometry['padded_height'])
        padded_width = int(geometry['padded_width'])
        window_size = tuple(int(v) for v in geometry['window_size'])
        shift_size = tuple(int(v) for v in geometry['shift_size'])

        expected_tokens = reduce(mul, window_size)
        expected_channels = int(num_heads) * int(head_dim)
        expected_windows = (
            batch_size
            * (padded_depth // window_size[0])
            * (padded_height // window_size[1])
            * (padded_width // window_size[2])
        )
        if x.shape != (expected_windows, expected_tokens, expected_channels):
            raise ValueError(
                "注意力窗口形状与前向几何信息不一致: "
                f"got={tuple(x.shape)}, expected="
                f"({expected_windows}, {expected_tokens}, {expected_channels})"
            )

        # [B*nW,Wd,Wh,Ww,U,E] -> [B*nW,Wd,Wh,Ww,U]
        local_response = x.reshape(
            expected_windows, *window_size, num_heads, head_dim
        ).abs().mean(dim=-1)
        # [B*nW,Wd,Wh,Ww,U] -> [B,Tp,Hp,Wp,U]
        response = window_reverse(
            local_response, window_size, batch_size,
            padded_depth, padded_height, padded_width
        )
        if any(v > 0 for v in shift_size):
            response = torch.roll(response, shifts=shift_size, dims=(1, 2, 3))
        response = response[:, :depth, :height, :width, :]
        return response.permute(0, 4, 1, 2, 3).contiguous()

    def _attention_signed_response_volume(
            self, x, attention_module, num_heads, head_dim):
        """恢复投影前Attention Head的有符号时空特征响应。

        Args:
            x: ``[B*nW,N,U*E]``，其中N为窗口token数，U为头数，
               E为head_dim；该张量是输出投影混合各头之前的响应。

        Returns:
            ``A [B,U,T,H,W,E]``；轴依次为视频、Attention Head、时间、
            空间高度、空间宽度和头特征。TDD在E轴仍保留时先构造
            ``A-mean_T(A)``。
        """
        geometry = getattr(attention_module, '_pruning_geometry', None)
        if geometry is None:
            raise RuntimeError(
                "注意力模块缺少_pruning_geometry，无法恢复真实时间与空间轴"
            )
        if x.ndim != 3:
            raise ValueError(
                f"注意力投影输入应为[B*nW,N,C]，当前形状为{tuple(x.shape)}"
            )

        batch_size = int(geometry['batch_size'])
        depth = int(geometry['depth'])
        height = int(geometry['height'])
        width = int(geometry['width'])
        padded_depth = int(geometry['padded_depth'])
        padded_height = int(geometry['padded_height'])
        padded_width = int(geometry['padded_width'])
        window_size = tuple(int(v) for v in geometry['window_size'])
        shift_size = tuple(int(v) for v in geometry['shift_size'])

        expected_tokens = reduce(mul, window_size)
        expected_channels = int(num_heads) * int(head_dim)
        expected_windows = (
            batch_size
            * (padded_depth // window_size[0])
            * (padded_height // window_size[1])
            * (padded_width // window_size[2])
        )
        if x.shape != (expected_windows, expected_tokens, expected_channels):
            raise ValueError(
                "注意力窗口形状与前向几何信息不一致: "
                f"got={tuple(x.shape)}, expected="
                f"({expected_windows}, {expected_tokens}, {expected_channels})"
            )

        # [B*nW,N,U*E] -> [B,Tp,Hp,Wp,U*E]，不提前取绝对值或压缩E。
        restored = window_reverse(
            x, window_size, batch_size,
            padded_depth, padded_height, padded_width
        )
        if any(v > 0 for v in shift_size):
            restored = torch.roll(
                restored, shifts=shift_size, dims=(1, 2, 3)
            )
        restored = restored[:, :depth, :height, :width, :]
        restored = restored.reshape(
            batch_size, depth, height, width, num_heads, head_dim
        )
        return restored.permute(0, 4, 1, 2, 3, 5).contiguous()

    @staticmethod
    def _mlp_response_volume(x):
        """将FFN激活转换为[B,U,T,H,W]时空响应体。

        输入x为[B,T,H,W,U]，U是FFN神经元轴；输出轴依次为视频、
        剪枝单元、时间、空间高度和空间宽度。
        """
        if x.ndim != 5:
            raise ValueError(
                f"FFN激活应为[B,T,H,W,U]，当前形状为{tuple(x.shape)}"
            )
        return x.abs().permute(0, 4, 1, 2, 3).contiguous()

    @staticmethod
    def _mlp_signed_response_volume(x):
        """将GELU后、fc2前FFN响应恢复为``[B,U,T,H,W,1]``。"""
        if x.ndim != 5:
            raise ValueError(
                f"FFN激活应为[B,T,H,W,U]，当前形状为{tuple(x.shape)}"
            )
        return x.permute(0, 4, 1, 2, 3).unsqueeze(-1).contiguous()

    def _append_activation_samples(self, name, samples):
        """在校准CUDA设备为每层维护至多2048个``[M,U]``样本。"""
        if samples.ndim != 2:
            raise ValueError(
                f"激活样本应为[M,U]，当前形状为{tuple(samples.shape)}"
            )
        if samples.shape[0] > self.max_activation_samples:
            indices = torch.randperm(samples.shape[0], device=samples.device)[
                :self.max_activation_samples
            ]
            samples = samples.index_select(0, indices)
        samples = samples.detach().to(device=self.device, dtype=torch.float32)

        if name in self.activations and self.activations[name]:
            samples = torch.cat([self.activations[name][0], samples], dim=0)
            if samples.shape[0] > self.max_activation_samples:
                # Preserve Task009's CPU RNG sequence for reservoir membership;
                # only the small index vector crosses to CUDA, not activations.
                indices = torch.randperm(samples.shape[0])[
                    :self.max_activation_samples
                ].to(samples.device)
                samples = samples.index_select(0, indices)
        self.activations[name] = [samples]

    @staticmethod
    def compute_spatiotemporal_information(response_volume, eps=1e-8):
        """计算每个视频、每个剪枝单元的时空响应轨迹一致性。

        对非负响应体A建立t-x-y质量分布，利用全部时间与空间位置的一、
        二阶矩构造3x3加权协方差。该计算同时覆盖相邻与非相邻帧，且不将
        head/channel/embedding轴误当作时间轴。

        Args:
            response_volume: [B,U,T,H,W]，B为视频数，U为剪枝单元数，
                             T/H/W分别为时间、高度和宽度。
            eps: 数值稳定常数，不属于方法超参数。

        Returns:
            score: [B,U]，sqrt(anisotropy * temporal_extent)，范围[0,1]。
            anisotropy: [B,U]，时空轨迹各向异性，范围[0,1]。
            temporal_extent: [B,U]，归一化时间覆盖度，范围[0,1]。
        """
        if response_volume.ndim != 5:
            raise ValueError(
                "response_volume应为[B,U,T,H,W]，当前形状为"
                f"{tuple(response_volume.shape)}"
            )
        if any(size <= 0 for size in response_volume.shape):
            raise ValueError(
                f"response_volume各轴不能为空，当前形状为{tuple(response_volume.shape)}"
            )

        response = torch.nan_to_num(
            response_volume.detach().float().abs(),
            nan=0.0, posinf=0.0, neginf=0.0
        )
        _, _, time_size, height, width = response.shape

        def coordinate(size):
            if size <= 1:
                return torch.zeros(1, device=response.device, dtype=response.dtype)
            return torch.linspace(
                0.0, 1.0, size, device=response.device, dtype=response.dtype
            )

        time_coord = coordinate(time_size)
        height_coord = coordinate(height)
        width_coord = coordinate(width)

        mass = response.sum(dim=(2, 3, 4))  # [B,U]
        valid = mass > eps
        denominator = mass.clamp_min(eps)

        # 直接累计矩，避免显式构造[B,U,T*H*W,3]中心化张量。
        mean_t = torch.einsum('buthw,t->bu', response, time_coord) / denominator
        mean_h = torch.einsum('buthw,h->bu', response, height_coord) / denominator
        mean_w = torch.einsum('buthw,w->bu', response, width_coord) / denominator

        second_t = torch.einsum(
            'buthw,t->bu', response, time_coord.square()
        ) / denominator
        second_h = torch.einsum(
            'buthw,h->bu', response, height_coord.square()
        ) / denominator
        second_w = torch.einsum(
            'buthw,w->bu', response, width_coord.square()
        ) / denominator
        second_th = torch.einsum(
            'buthw,t,h->bu', response, time_coord, height_coord
        ) / denominator
        second_tw = torch.einsum(
            'buthw,t,w->bu', response, time_coord, width_coord
        ) / denominator
        second_hw = torch.einsum(
            'buthw,h,w->bu', response, height_coord, width_coord
        ) / denominator

        covariance = response.new_zeros((*mass.shape, 3, 3))  # [B,U,3,3]
        covariance[..., 0, 0] = (second_t - mean_t.square()).clamp_min(0.0)
        covariance[..., 1, 1] = (second_h - mean_h.square()).clamp_min(0.0)
        covariance[..., 2, 2] = (second_w - mean_w.square()).clamp_min(0.0)
        covariance[..., 0, 1] = covariance[..., 1, 0] = second_th - mean_t * mean_h
        covariance[..., 0, 2] = covariance[..., 2, 0] = second_tw - mean_t * mean_w
        covariance[..., 1, 2] = covariance[..., 2, 1] = second_hw - mean_h * mean_w

        eigenvalues = torch.linalg.eigvalsh(covariance).clamp_min(0.0)
        trace = eigenvalues.sum(dim=-1)
        principal_ratio = eigenvalues[..., -1] / trace.clamp_min(eps)
        anisotropy = ((3.0 * principal_ratio - 1.0) / 2.0).clamp(0.0, 1.0)

        # 归一化坐标位于[0,1]，其理论最大方差为1/4。
        temporal_extent = (4.0 * covariance[..., 0, 0]).clamp(0.0, 1.0)
        score = torch.sqrt((anisotropy * temporal_extent).clamp_min(0.0))

        score = torch.where(valid, score, torch.zeros_like(score))
        anisotropy = torch.where(valid, anisotropy, torch.zeros_like(anisotropy))
        temporal_extent = torch.where(
            valid, temporal_extent, torch.zeros_like(temporal_extent)
        )
        return (
            torch.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0),
            torch.nan_to_num(anisotropy, nan=0.0, posinf=0.0, neginf=0.0),
            torch.nan_to_num(temporal_extent, nan=0.0, posinf=0.0, neginf=0.0),
        )

    def _get_activation_hook(
            self, name, module_type, num_heads=None, head_dim=None,
            attention_module=None):
        """
        生成激活捕获钩子
        
        绝对/相对维继续使用``[B,U,T,H,W]``非负幅值响应。Dynamic3D
        额外保留``A [B,U,T,H,W,D]``的符号与特征轴，统一计算``[B,U]``
        TDD；D是Attention head_dim或FFN的1。
        """
        def hook(module, input):
            x = input[0].detach()
            if module_type == 'attention':
                if self.descriptor_variant == "dynamic3d":
                    signed_response = self._attention_signed_response_volume(
                        x, attention_module, num_heads, head_dim
                    )
                    # Reuse the exact Task009 path for D_abs/D_rel isolation.
                    response = self._attention_response_volume(
                        x, attention_module, num_heads, head_dim
                    )
                else:
                    signed_response = None
                    response = self._attention_response_volume(
                        x, attention_module, num_heads, head_dim
                    )
            else:
                if self.descriptor_variant == "dynamic3d":
                    signed_response = self._mlp_signed_response_volume(x)
                    # Reuse the exact Task009 path for D_abs/D_rel isolation.
                    response = self._mlp_response_volume(x)
                else:
                    signed_response = None
                    response = self._mlp_response_volume(x)

            # [B,U,T,H,W] -> [B*T*H*W,U]，仅剪枝单元轴U保持不变。
            if self.collect_activation_samples:
                samples = response.permute(0, 2, 3, 4, 1).reshape(
                    -1, response.shape[1]
                )
                self._append_activation_samples(name, samples)

            if self.descriptor_variant == "old3d":
                score, _, _ = self.compute_spatiotemporal_information(response)
                self.spatiotemporal_scores.setdefault(name, []).append(
                    score.detach()
                )
            elif self.descriptor_variant == "dynamic3d":
                score = compute_temporal_dynamicity(signed_response)
                self.temporal_dynamicity_scores.setdefault(name, []).append(
                    score.detach()
                )
                if name not in self._tdd_logged_layers:
                    _, units, time_size, height, width, feature_dim = (
                        signed_response.shape
                    )
                    peak_mib = (
                        torch.cuda.max_memory_allocated(signed_response.device)
                        / (1024 ** 2)
                        if signed_response.is_cuda else 0.0
                    )
                    print(
                        ">>> TDD extraction memory: "
                        f"layer={name}, units={units}, temporal={time_size}, "
                        f"spatial={height}x{width}, feature_dim={feature_dim}, "
                        f"peak_cuda={peak_mib:.1f} MiB"
                    )
                    self._tdd_logged_layers.add(name)
            
        return hook

    def run_calibration(
            self, loader, device, num_batches=10, video_transform=None):
        """
        运行校准，收集激活统计
        
        Args:
            loader: 数据加载器
            device: 计算设备
            num_batches: 校准批次数量
            video_transform: 可选的CUDA端视频变换，仅供独立TDD sanity探针使用
        """
        self.register_hooks()
        self.model.eval()
        
        print(f">>> 开始校准 (收集 {num_batches} 个批次)...")
        try:
            with torch.no_grad():
                for i, batch in enumerate(
                        tqdm.tqdm(loader, desc="Calibration", total=num_batches)):
                    if i >= num_batches:
                        break
                    
                    if isinstance(batch, (list, tuple)):
                        vids = batch[0]
                    else:
                        vids = batch

                    vids = vids.float().to(device, non_blocking=True)
                    if video_transform is not None:
                        vids = video_transform(vids)
                    self.model(vids)
                    del vids
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
        finally:
            self.remove_hooks()
        print(f">>> 校准完成，共收集 {len(self.activations)} 层的激活数据")

    # =========================================================================
    # 三维描述符计算
    # =========================================================================
    
    def _compute_activation_statistics(self, Z):
        """计算尚未融合的激活幅值与激活频率统计量。

        Args:
            Z: [M, C] 非负激活幅度矩阵。

        Returns:
            mean_amp: [C] 平均激活幅值，取值范围为[0, +inf)。
            freq: [C] 激活频率，取值范围为[0, 1]。
            threshold: [C] 或标量，频率统计使用的阈值。
        """
        if Z.ndim != 2:
            raise ValueError(f"Z应为二维矩阵[M, C]，当前形状为{tuple(Z.shape)}")
        if Z.shape[0] == 0 or Z.shape[1] == 0:
            raise ValueError(f"Z不能为空，当前形状为{tuple(Z.shape)}")

        # Hook阶段已经取绝对值，此处再次clamp用于防止外部调用传入负值。
        Z = Z.float().clamp_min(0.0)
        mean_amp = Z.mean(dim=0)

        # 采用每个剪枝单元自身的统计量作为相对阈值，避免固定1e-4使频率大面积饱和为1。
        if self.activation_threshold_mode == 'mean':
            threshold = mean_amp
        elif self.activation_threshold_mode == 'median':
            threshold = Z.median(dim=0).values
        else:
            threshold = torch.full_like(mean_amp, 1e-4)

        freq = (Z > threshold.unsqueeze(0)).float().mean(dim=0)
        return mean_amp, freq, threshold

    def _normalize_importance_statistic(self, x, name='statistic'):
        """将一维统计量独立映射至[0, 1]，避免不同统计量直接相加产生尺度偏置。"""
        if x.ndim != 1:
            raise ValueError(f"{name}应为一维向量，当前形状为{tuple(x.shape)}")
        x = torch.nan_to_num(x.float(), nan=0.0, posinf=0.0, neginf=0.0)

        if self.importance_norm == 'robust_minmax' and x.numel() > 1:
            lower = torch.quantile(x, self.norm_q_low)
            upper = torch.quantile(x, self.norm_q_high)
        else:
            lower = x.min()
            upper = x.max()

        scale = upper - lower
        if (not torch.isfinite(scale)) or scale.abs() <= self.importance_eps:
            # 所有单元统计值相同时，该统计量不提供区分信息；置零而非置常数，
            # 防止其对融合结果引入无意义的统一偏移。
            return torch.zeros_like(x)

        return ((x - lower) / (scale + self.importance_eps)).clamp_(0.0, 1.0)

    def compute_absolute_importance(self, Z, global_amp_bounds=None, global_freq_bounds=None):
        """计算尺度对齐后的绝对重要性。

        D_abs = alpha * normalized(mean_amp)
                + (1-alpha) * normalized(freq)

        两个组成统计量在融合前分别归一化至[0, 1]。默认使用相对均值阈值计算
        激活频率，从而避免固定阈值导致频率饱和。

        Args:
            Z: [M, C] 激活幅度矩阵。
            global_amp_bounds: 保留接口兼容，当前由调用方统一拼接后归一化。
            global_freq_bounds: 保留接口兼容，当前由调用方统一拼接后归一化。

        Returns:
            D_abs: [C] 绝对重要性分数，范围[0, 1]。
            mean_amp: [C] 原始平均激活幅值。
            freq: [C] 原始激活频率。
        """
        del global_amp_bounds, global_freq_bounds
        mean_amp, freq, _ = self._compute_activation_statistics(Z)
        mean_amp_norm = self._normalize_importance_statistic(mean_amp, 'mean_amp')
        freq_norm = self._normalize_importance_statistic(freq, 'freq')
        alpha = self.importance_alpha
        D_abs = alpha * mean_amp_norm + (1.0 - alpha) * freq_norm
        return D_abs, mean_amp, freq

    def compute_substitutability(self, Z):
        """
        计算可替代性分数（基于舒尔补思想）
        
        若某单元可被其他单元的线性组合有效重构，则可替代性高
        
        Args:
            Z: [M, C] 激活矩阵
            
        Returns:
            substitutability: [C] 可替代性分数 (0-1)
        """
        device = Z.device
        Z_centered = Z - Z.mean(dim=0, keepdim=True)
        
        # 计算协方差矩阵（带正则化）
        eps = 1e-5
        cov = torch.matmul(Z_centered.T, Z_centered) / Z.shape[0] + \
              eps * torch.eye(Z.shape[1], device=device)
        
        try:
            # 求逆矩阵
            inv_cov = torch.linalg.inv(cov)
            
            # 舒尔补：误差方差 = 1 / 逆矩阵对角线元素
            err_var = 1.0 / torch.diag(inv_cov)
            orig_var = torch.diag(cov)
            
            # 可替代分数 = 1 - (误差方差 / 原始方差)
            # 若可替代性高（可被其他单元重构），err_var很小，分数接近1
            substitutability = torch.clamp(
                1.0 - err_var / (orig_var + eps), min=0.0, max=1.0
            )
        except:
            # 矩阵求逆失败时使用简化估计
            substitutability = torch.zeros(Z.shape[1], device=device)
            
        return substitutability

    def compute_relative_importance(self, D_abs, Z):
        """
        计算相对重要性
        
        D_rel = D_abs * (1 - substitutability)
        
        高度可替代的单元，折减系数趋近于零，重要性显著降低
        
        Args:
            D_abs: [C] 绝对重要性
            Z: [M, C] 激活矩阵
            
        Returns:
            D_rel: [C] 相对重要性
            substitutability: [C] 可替代性分数
        """
        substitutability = self.compute_substitutability(Z)
        reduction_factor = 1.0 - substitutability
        D_rel = D_abs * reduction_factor
        
        return D_rel, substitutability

    # =========================================================================
    # 力场Mean-Shift聚类
    # =========================================================================
    
    def mean_shift_clustering(self, V_norm):
        """
        力场Mean-Shift聚类
        
        将每个描述符视为多维空间中的质点，描述符间相互作用形成力场。
        通过高斯核计算吸引力，沿合力方向迭代移动，追踪运动轨迹至收敛。
        
        Args:
            V_norm: [N, 3] 归一化后的三维描述符（绝对、相对、时空信息）
            
        Returns:
            groups: 分组列表，每个元素是组成员索引列表
            Trajectories: [N, 3] 运动轨迹（终点 - 起点）
            P_final: [N, 3] 运动终点位置
        """
        P = V_norm.clone()
        N = P.shape[0]
        sigma = self.sigma
        max_iters = 60
        tol = 1e-4
        
        print(f">>> Mean-Shift迭代 (N={N}个剪枝单元, sigma={sigma})...")
        
        for step in range(max_iters):
            # 逐行分块计算与完整[N,N]实现完全相同的高斯核更新，避免同时
            # 常驻dist和K两个全矩阵。1024仅为内存实现常数，不进入方法定义。
            chunk_size = min(1024, N)
            P_new = torch.empty_like(P)
            for start in range(0, N, chunk_size):
                end = min(start + chunk_size, N)
                # [Q,N]，Q<=1024
                dist = torch.cdist(P[start:end], P)
                kernel = torch.exp(- (dist ** 2) / (2 * sigma ** 2))
                P_new[start:end] = (
                    torch.matmul(kernel, P)
                    / (kernel.sum(dim=1, keepdim=True) + 1e-8)
                )
                del dist, kernel
            
            # 检查收敛
            max_movement = torch.max(torch.norm(P_new - P, dim=1))
            P = P_new
            
            if max_movement < tol:
                print(f">>> Mean-Shift收敛于第 {step+1} 轮 (最大位移: {max_movement:.6f})")
                break
        
        # 根据运动终点分组（汇点聚类）
        # 对终点进行量化，合并接近的点
        P_rounded = torch.round(P * 100) / 100
        unique_sinks, inverse_indices = torch.unique(
            P_rounded, dim=0, return_inverse=True
        )
        
        groups = []
        for i in range(len(unique_sinks)):
            members = (inverse_indices == i).nonzero(as_tuple=True)[0].tolist()
            if members:
                groups.append(members)
        
        print(f">>> 聚类完成，共生成 {len(groups)} 个功能簇")
        
        # 计算运动轨迹
        Trajectories = P - V_norm
        
        return groups, Trajectories, P

    # =========================================================================
    # 流形一致性评分
    # =========================================================================
    
    def compute_group_manifold_score(self, group_members, V, Trajectories):
        """
        计算组的流形一致性得分
        
        score = 动态一致性(流线平行度) × 静态功能强度
        
        Args:
            group_members: 组成员索引列表
            V: [N, 3] 原始三维描述符
            Trajectories: [N, 3] 运动轨迹
            
        Returns:
            score: 最终得分
            dyn_consist: 流线平行度（动态一致性）
            static_strength: 功能强度（静态功能价值）
        """
        if len(group_members) == 0:
            return 0.0, 0.0, 0.0
        
        # (1) 动态一致性：流线平行度
        if len(group_members) > 1:
            # 提取组内运动轨迹
            T_G = Trajectories[group_members]  # [G, 3]
            T_G_n = F.normalize(T_G, p=2, dim=1)  # 归一化方向向量
            
            # 计算所有向量对的余弦相似度
            sim_matrix = torch.matmul(T_G_n, T_G_n.T)  # [G, G]
            n_m = len(group_members)
            
            # 平均余弦相似度（排除对角线）
            dyn_consist = (sim_matrix.sum() - n_m) / (n_m * (n_m - 1) + 1e-8)
            dyn_consist = torch.clamp(dyn_consist, min=0.0, max=1.0).item()
        else:
            # 单点一致性为1
            dyn_consist = 1.0
        
        # (2) 静态功能强度：极值平均
        V_G = V[group_members]  # [G, 3]
        max_vals = V_G.max(dim=0)[0]  # [3]
        min_vals = V_G.min(dim=0)[0]  # [3]
        
        # 各维度极值平均后加权融合
        dim_scores = (max_vals + min_vals) / 2.0
        static_strength = dim_scores.mean().item()
        
        # (3) 融合得分
        score = dyn_consist * static_strength
        
        return score, dyn_consist, static_strength

    # =========================================================================
    # 剪枝决策与执行
    # =========================================================================
    
    def estimate_unit_cost(self, module, unit_type):
        """
        估计剪枝单元的参数代价
        """
        try:
            if unit_type == 'head':
                # 确保模块确实有这些属性
                if not hasattr(module, 'qkv') or not hasattr(module, 'head_dim'):
                    return 0
                d = module.qkv.in_features
                h_dim = module.head_dim
                # QKV权重 + QKV偏置 + Proj权重
                qkv_bias_params = 3 * h_dim if module.qkv.bias is not None else 0
                return 3 * d * h_dim + qkv_bias_params + d * h_dim
            else:  # neuron
                # 确保模块确实有 fc1
                if not hasattr(module, 'fc1'):
                    return 0
                in_f = module.fc1.in_features
                out_f = module.fc2.out_features
                # fc1权重 + fc1偏置 + fc2权重
                bias_params = 1 if module.fc1.bias is not None else 0
                return in_f + bias_params + out_f
        except Exception as e:
            print(f"Warning: Error estimating cost for {module}: {e}")
            return 0


    def _write_descriptor_statistics(
            self, valid_layer_names, layer_D_abs, layer_D_rel,
            layer_D_third):
        """原子写出一行一个剪枝单元的描述符调试表。"""
        if self.descriptor_statistics_path is None:
            return

        output_path = self.descriptor_statistics_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = output_path.with_suffix(output_path.suffix + ".tmp")
        modules = dict(self.model.named_modules())
        third_name = {
            "abs_rel": "none",
            "old3d": "D_st_old",
            "dynamic3d": "D_dyn",
        }[self.descriptor_variant]
        fieldnames = [
            "global_index", "layer", "unit_type", "unit_index",
            "D_abs", "D_rel", "D_third", "third_descriptor_name", "D_dyn",
        ]

        rows = []
        global_index = 0
        for name in valid_layer_names:
            d_abs = layer_D_abs[name].detach().double().cpu().tolist()
            d_rel = layer_D_rel[name].detach().double().cpu().tolist()
            third_tensor = layer_D_third.get(name)
            d_third = (
                third_tensor.detach().double().cpu().tolist()
                if third_tensor is not None else [None] * len(d_abs)
            )
            module = modules[name]
            unit_type = (
                "attention_head"
                if "WindowAttention3D" in module.__class__.__name__
                else "ffn_neuron"
            )
            for unit_index, (d_a, d_r, d_3) in enumerate(
                    zip(d_abs, d_rel, d_third)):
                rows.append({
                    "global_index": global_index,
                    "layer": name,
                    "unit_type": unit_type,
                    "unit_index": unit_index,
                    "D_abs": f"{d_a:.17g}",
                    "D_rel": f"{d_r:.17g}",
                    "D_third": "" if d_3 is None else f"{d_3:.17g}",
                    "third_descriptor_name": third_name,
                    "D_dyn": (
                        f"{d_3:.17g}"
                        if self.descriptor_variant == "dynamic3d" else ""
                    ),
                })
                global_index += 1

        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(output_path)
        print(f">>> Descriptor statistics: {output_path.resolve()}")

    def _load_functional_descriptor_cache(self, valid_layer_names):
        """Load exact ``V[N,3]`` and row mapping from a verified Task014 CSV."""
        path = self.functional_descriptor_cache
        if path is None:
            raise RuntimeError("functional_descriptor_cache is not configured")
        if not path.is_file():
            raise FileNotFoundError(path)
        modules = dict(self.model.named_modules())
        expected_layers = list(valid_layer_names)
        rows = []
        with path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        if len(rows) != EXPECTED_DESCRIPTOR_UNITS:
            raise ValueError(
                f"Descriptor cache has {len(rows)} rows, expected "
                f"{EXPECTED_DESCRIPTOR_UNITS}"
            )
        values = []
        unit_info = []
        observed_layers = []
        previous_layer = None
        expected_unit_index = 0
        for expected_global, row in enumerate(rows):
            if int(row["global_index"]) != expected_global:
                raise ValueError("Descriptor cache global indices are not contiguous")
            layer = row["layer"]
            if layer != previous_layer:
                observed_layers.append(layer)
                previous_layer = layer
                expected_unit_index = 0
            if int(row["unit_index"]) != expected_unit_index:
                raise ValueError(f"Descriptor cache unit mapping is invalid for {layer}")
            expected_unit_index += 1
            module = modules.get(layer)
            if module is None:
                raise ValueError(f"Descriptor cache layer is absent from model: {layer}")
            expected_type = (
                "attention_head"
                if "WindowAttention3D" in module.__class__.__name__
                else "ffn_neuron"
            )
            if row["unit_type"] != expected_type:
                raise ValueError(f"Descriptor cache type mismatch for {layer}")
            if row["third_descriptor_name"] != "D_dyn":
                raise ValueError("Task016 descriptor cache must contain Dynamic3D")
            triple = (float(row["D_abs"]), float(row["D_rel"]), float(row["D_dyn"]))
            if not all(math.isfinite(value) for value in triple):
                raise ValueError("Descriptor cache contains NaN or infinity")
            values.append(triple)
            unit_info.append({"layer": layer, "idx": int(row["unit_index"])})
        if observed_layers != expected_layers:
            raise ValueError("Descriptor cache layer order differs from calibrated model")
        V = torch.as_tensor(values, dtype=torch.float32, device=self.device)  # [N,3]
        print(f">>> Reused verified Dynamic3D descriptor cache: {path.resolve()}")
        return V, unit_info, {}, {}, {}

    def build_3d_descriptors(self, valid_layer_names):
        """构建受控描述符变体``V in R^[N,d]``，其中``d in {2,3}``。

        ``abs_rel``返回``[D_abs,D_rel]``；``old3d``返回旧时空第三维；
        ``dynamic3d``返回``[D_abs,D_rel,D_dyn]``。前两维、逐维标准化及
        后续BMS数学实现保持不变。
        """
        print(f"\n>>> Descriptor variant: {descriptor_variant_label(self.descriptor_variant)}")
        dimension_names = {
            "abs_rel": ("D_abs", "D_rel"),
            "old3d": ("D_abs", "D_rel", "D_st_old"),
            "dynamic3d": ("D_abs", "D_rel", "D_dyn"),
        }[self.descriptor_variant]
        print(">>> Descriptor dimensions:")
        for dimension_name in dimension_names:
            print(f"    {dimension_name}")

        layer_D_abs = {}
        layer_D_rel = {}
        layer_D_third = {}
        layer_Z = {}
        layer_mean_amp = {}
        layer_freq = {}

        # 统计保留在校准CUDA设备；仅最终CSV写盘时一次性转CPU。
        for name in valid_layer_names:
            Z = torch.cat(self.activations[name], dim=0).to(
                self.device, dtype=torch.float32
            )
            mean_amp, freq, _ = self._compute_activation_statistics(Z)
            layer_Z[name] = Z
            layer_mean_amp[name] = mean_amp
            layer_freq[name] = freq

        all_mean_amp = torch.cat(
            [layer_mean_amp[name] for name in valid_layer_names], dim=0
        )
        all_freq = torch.cat(
            [layer_freq[name] for name in valid_layer_names], dim=0
        )
        all_mean_amp_norm = self._normalize_importance_statistic(
            all_mean_amp, 'global_mean_amp'
        )
        all_freq_norm = self._normalize_importance_statistic(
            all_freq, 'global_freq'
        )

        print(
            f">>> 绝对重要性尺度对齐: alpha={self.importance_alpha:.3f}, "
            f"threshold={self.activation_threshold_mode}, norm={self.importance_norm}"
        )
        print(
            f"    原始幅值范围=[{all_mean_amp.min().item():.6g}, "
            f"{all_mean_amp.max().item():.6g}], "
            f"原始频率范围=[{all_freq.min().item():.6g}, "
            f"{all_freq.max().item():.6g}]"
        )

        offset = 0
        for name in valid_layer_names:
            count = layer_mean_amp[name].numel()
            amp_norm = all_mean_amp_norm[offset:offset + count]
            freq_norm = all_freq_norm[offset:offset + count]
            offset += count

            D_abs = (
                self.importance_alpha * amp_norm
                + (1.0 - self.importance_alpha) * freq_norm
            )
            layer_D_abs[name] = D_abs
            D_rel, _ = self.compute_relative_importance(D_abs, layer_Z[name])
            layer_D_rel[name] = D_rel

            if self.descriptor_variant == "old3d":
                if name not in self.spatiotemporal_scores:
                    raise RuntimeError(f"{name}缺少旧时空信息统计，请重新运行校准")
                per_video = torch.cat(
                    self.spatiotemporal_scores[name], dim=0
                ).to(self.device, dtype=torch.float32)
                layer_D_third[name] = per_video.mean(dim=0)
            elif self.descriptor_variant == "dynamic3d":
                if name not in self.temporal_dynamicity_scores:
                    raise RuntimeError(f"{name}缺少TDD统计，请重新运行校准")
                per_video = torch.cat(
                    self.temporal_dynamicity_scores[name], dim=0
                ).to(self.device, dtype=torch.float32)
                layer_D_third[name] = per_video.mean(dim=0)

            if self.descriptor_variant != "abs_rel":
                D_third = layer_D_third[name]
                if D_third.numel() != D_abs.numel():
                    raise ValueError(
                        f"{name}第三维单元数{D_third.numel()}与重要性单元数"
                        f"{D_abs.numel()}不一致"
                    )
                if not torch.isfinite(D_third).all():
                    raise ValueError(f"{name}第三维包含NaN或Inf")
                if self.descriptor_variant == "dynamic3d" and (
                        torch.any(D_third < -1e-6)
                        or torch.any(D_third > 1.0 + 1e-6)):
                    raise ValueError(f"{name} D_dyn超出[0,1]")

        descriptor_blocks = []
        unit_info = []
        for name in valid_layer_names:
            D_a = layer_D_abs[name]  # [U]
            D_r = layer_D_rel[name]  # [U]
            D_third = layer_D_third.get(name)
            descriptor_blocks.append(
                assemble_descriptor_variant(
                    D_a,
                    D_r,
                    D_third if self.descriptor_variant == "old3d" else None,
                    D_third if self.descriptor_variant == "dynamic3d" else None,
                    self.descriptor_variant,
                )
            )
            unit_info.extend(
                {'layer': name, 'idx': unit_index}
                for unit_index in range(D_a.numel())
            )

        V = torch.cat(descriptor_blocks, dim=0).to(self.device)
        expected_dimensions = 2 if self.descriptor_variant == "abs_rel" else 3
        if (
                V.ndim != 2
                or V.shape[1] != expected_dimensions
                or len(unit_info) != V.shape[0]):
            raise RuntimeError(
                f"描述符与单元元数据不一致: V={tuple(V.shape)}, "
                f"unit_info={len(unit_info)}"
            )
        if not torch.isfinite(V).all():
            raise ValueError("描述符包含NaN或Inf")

        print(f">>> 共构建 {len(V)} 个剪枝单元的{expected_dimensions}维描述符")
        if self.descriptor_variant == "dynamic3d":
            print(
                f">>> D_dyn range: [{V[:, 2].min().item():.6g}, "
                f"{V[:, 2].max().item():.6g}]"
            )
            print(
                f">>> D_dyn mean/std: {V[:, 2].mean().item():.6g} / "
                f"{V[:, 2].std(unbiased=False).item():.6g}"
            )

        self._write_descriptor_statistics(
            valid_layer_names, layer_D_abs, layer_D_rel, layer_D_third
        )
        return V, unit_info, layer_D_abs, layer_D_rel, layer_D_third

    def _check_layer_constraints(self, grp, unit_info, layer_info):
        """
        检查如果移除当前组(grp)，是否会违反各层的最小保留比例限制
        """
        # 统计这一组如果在各层实施剪枝，会分别剪掉多少个单元
        grp_layer_counts = Counter()
        for idx in grp['members']:
            layer_name = unit_info[idx]['layer']
            grp_layer_counts[layer_name] += 1
        
        # 逐层校验
        for layer_name, count in grp_layer_counts.items():
            current_pruned = layer_info[layer_name]['pruned']
            max_prunable = layer_info[layer_name]['max_prunable']
            
            # 如果 已经剪掉的 + 这一组贡献的 > 该层允许剪掉的最大数量
            if current_pruned + count > max_prunable:
                return False, grp_layer_counts
                
        return True, grp_layer_counts

    def _build_selection_candidates(self, group_scores, unit_info, V, registry):
        """Build Task013 unit evidence without changing BMS group decisions.

        ``V`` has shape ``[N,3]`` (unit, descriptor dimension).  Every member
        inherits its group's raw manifold score; the within-group evidence
        order is the existing ascending ``mean(V[i,:])`` critical-group order.
        """
        if not self.selection_trace_enabled:
            self.selection_candidates = []
            return
        modules = dict(self.model.named_modules())
        candidates = []
        for group in group_scores:
            members = sorted(
                group['members'],
                key=lambda index: (float(V[index].mean().item()), int(index)),
            )
            for index in members:
                info = unit_info[index]
                module = modules[info['layer']]
                is_attention = (
                    "WindowAttention3D" in module.__class__.__name__
                )
                cost_type = 'head' if is_attention else 'neuron'
                candidates.append({
                    'layer': info['layer'],
                    'unit_type': (
                        'attention_head' if is_attention else 'ffn_neuron'
                    ),
                    'unit_index': int(info['idx']),
                    'group_id': int(group['group_id']),
                    'raw_pruning_score': float(group['score']),
                    'parameter_cost': int(
                        self.estimate_unit_cost(module, cost_type)
                    ),
                })
        ranked = rank_candidates(candidates, self.selection_cost_mode)
        self.selection_candidates = mark_selected(ranked, registry)

    def group_pruning(self, V, unit_info, valid_layer_names, step_target_sparsity):
        """
        组级结构剪枝
        
        Args:
            V: [N, 3] 三维描述符
            unit_info: 剪枝单元信息
            valid_layer_names: 有效层名称
            step_target_sparsity: 本轮目标稀疏度
            
        Returns:
            registry: 剪枝注册表 {layer_name: set(pruned_indices)}
        """
        print("\n>>> 步骤2: 力场Mean-Shift聚类...")
        

        # 1. 初始化各层计数器，用于硬性约束
        layer_info = {}
        for name in valid_layer_names:
            # 获取该层总单元数（如特征图通道数或神经元数）
            # 这里假设 unit_info 已经包含了全量信息
            total_units = sum(1 for info in unit_info if info['layer'] == name)
            layer_info[name] = {
                'total': total_units,
                'pruned': 0,
                'max_prunable': int(total_units * (1 - self.min_keep_ratio)) # 至少保留 min_keep_ratio
            }

        # 归一化描述符
        V_norm = (V - V.mean(dim=0, keepdim=True)) / (V.std(dim=0, keepdim=True) + 1e-8)
        
        # Mean-Shift聚类
        groups, Trajectories, P_final = self.mean_shift_clustering(V_norm)
        
        print("\n>>> 步骤3: 计算流形一致性并生成剪枝决策...")
        
        # 3. 流形一致性评分
        group_scores = []
        
        for g_idx, g_members in enumerate(groups):
            score, dyn_consist, static_strength = self.compute_group_manifold_score(
                g_members, V, Trajectories
            )
            
            # 计算组参数代价
            cost = 0
            for idx in g_members:
                m_info = unit_info[idx]
                m = dict(self.model.named_modules())[m_info['layer']]
                
                if "WindowAttention3D" in m.__class__.__name__:
                    cost += self.estimate_unit_cost(m, 'head')
                else:
                    cost += self.estimate_unit_cost(m, 'neuron')
            
            group_scores.append({
                'group_id': g_idx,
                'members': g_members,
                'score': score,
                'dyn_consist': dyn_consist,
                'static_strength': static_strength,
                'cost': cost
            })
        
        # 4. 构建层信息（限制每层最大剪枝比例）
        layer_info = {}
        for name in valid_layer_names:
            m = dict(self.model.named_modules())[name]
            
            if "WindowAttention3D" in m.__class__.__name__:
                if not hasattr(m, 'num_heads'):
                    m.num_heads = m.qkv.out_features // (3 * m.head_dim)
                total_u = m.num_heads
            else:
                if not hasattr(m, 'original_hidden_features'):
                    m.original_hidden_features = m.fc1.out_features
                total_u = m.original_hidden_features
            
            max_prune = total_u - self._minimum_keep_units(m, total_u)
            layer_info[name] = {'total': total_u, 'pruned': 0, 'max_prunable': max_prune}
        
        # 5. 全局组级剪枝
        total_p = self.total_original_params
        target_red = total_p * step_target_sparsity
        current_red = 0
        registry = {name: set() for name in valid_layer_names}
        
        # 按流形一致性得分排序（低到高）
        group_scores.sort(key=lambda x: x['score'])

        
        if self.selection_cost_mode == 'decoupled':
            # Task013: raw score decides which feasible group is next.  Cost
            # never gates admission and is read only after selection to update
            # the shared parameter budget and stopping condition.
            for grp in group_scores:
                valid_prune, layer_counts = self._check_layer_constraints(
                    grp, unit_info, layer_info
                )
                if not valid_prune:
                    continue
                for idx in grp['members']:
                    info = unit_info[idx]
                    registry[info['layer']].add(info['idx'])
                    layer_info[info['layer']]['pruned'] += 1
                current_red += grp['cost']
                if current_red >= target_red:
                    break
        else:
            # Coupled is the exact Task012 admission/critical-group path.
            for grp in group_scores:
                # 检查如果加入这一整组，是否会显著超过目标
                if current_red + grp['cost'] <= target_red:
                    # 检查每层最小保留比例约束 (保持你原有的逻辑)
                    valid_prune, layer_counts = self._check_layer_constraints(grp, unit_info, layer_info)

                    if valid_prune:
                        for idx in grp['members']:
                            info = unit_info[idx]
                            registry[info['layer']].add(info['idx'])
                            layer_info[info['layer']]['pruned'] += 1
                        current_red += grp['cost']
                else:
                    # --- 关键改进：处理“临界组” ---
                    # 计算还需要减少多少参数量
                    remaining_gap = target_red - current_red
                    if remaining_gap <= 0: break

                    # 对该组内的成员按“个体静态强度”进行排序
                    members_with_val = []
                    for idx in grp['members']:
                        # 使用 V[idx] 的范数或均值作为个体重要性参考
                        individual_val = V[idx].mean().item()
                        members_with_val.append((idx, individual_val))

                    # 从最不重要的个体开始剪
                    members_with_val.sort(key=lambda x: x[1])

                    for idx, val in members_with_val:
                        info = unit_info[idx]
                        cost = self.estimate_unit_cost(dict(self.model.named_modules())[info['layer']],
                                                    'head' if 'Attention' in info['layer'] else 'neuron')

                        # 再次检查层约束
                        if layer_info[info['layer']]['pruned'] + 1 <= layer_info[info['layer']]['max_prunable']:
                            registry[info['layer']].add(info['idx'])
                            layer_info[info['layer']]['pruned'] += 1
                            current_red += cost

                        if current_red >= target_red: break
                    if current_red >= target_red:
                        break
                    if self.diagnostic_min_attention_heads:
                        # Task012诊断保护可能使临界组中的Attention成员不可剪。
                        # 继续按既有全局组排序把剩余预算分配给后续可行单元。
                        continue
                    break # 原始路径保持Task011行为：处理一个临界组后退出

        self._build_selection_candidates(
            group_scores, unit_info, V, registry
        )

        return registry, current_red / total_p


    def _functional_mapping_context(self, unit_info, valid_layer_names):
        """Build strict Task014 layer/type/cost metadata without changing BMS.

        ``unit_info`` has one row per descriptor unit and remains in the exact
        layer-major order produced by :meth:`build_3d_descriptors`.
        """
        if self.contribution_npz is None:
            raise ValueError(
                "selection_mode='functional' requires --contribution_npz"
            )
        modules = dict(self.model.named_modules())
        model_unit_types = {}
        layer_capacities = {}
        layer_unit_counts = Counter(
            str(info['layer']) for info in unit_info
        )
        for layer_name in valid_layer_names:
            if layer_name not in modules:
                raise KeyError(f"Model module not found for layer {layer_name!r}")
            module = modules[layer_name]
            class_name = module.__class__.__name__
            total_units = int(layer_unit_counts[layer_name])
            if "WindowAttention3D" in class_name:
                model_unit_types[layer_name] = "attention_head"
            elif "Mlp" in class_name:
                model_unit_types[layer_name] = "ffn_neuron"
            else:
                raise TypeError(
                    f"Unsupported Task014 pruning module {class_name} for "
                    f"layer {layer_name!r}"
                )
            layer_capacities[layer_name] = (
                total_units - self._minimum_keep_units(module, total_units)
            )

        archive = LayerWiseContributionFieldArchive(self.contribution_npz)
        audit = archive.audit_descriptor_mapping(
            unit_info,
            valid_layer_names,
            model_unit_types,
            expected_total_units=EXPECTED_DESCRIPTOR_UNITS,
            expected_layer_count=EXPECTED_PRUNING_LAYERS,
            expected_array_count=EXPECTED_NPZ_ARRAYS,
            expected_video_fields=EXPECTED_VIDEO_FIELDS,
        )
        audit.write(self.functional_audit_dir)
        self.functional_mapping_audit = audit
        zero_field_audit = archive.audit_zero_function_fields(
            audit,
            self.device,
            self.functional_audit_dir,
        )
        zero_field_audit.require_valid_null_semantics()
        self.functional_zero_field_audit = zero_field_audit

        unit_costs = []
        for global_index, info in enumerate(unit_info):
            layer_name = str(info['layer'])
            module = modules[layer_name]
            cost_type = (
                'head'
                if model_unit_types[layer_name] == 'attention_head'
                else 'neuron'
            )
            cost = int(self.estimate_unit_cost(module, cost_type))
            if cost <= 0:
                raise ValueError(
                    f"Non-positive unit cost for descriptor row {global_index}: "
                    f"{layer_name}[{int(info['idx'])}]"
                )
            unit_costs.append(cost)
        return archive, audit, unit_costs, layer_capacities

    def _functional_bms_domains(self, V):
        """Run the unchanged Dynamic3D standardization and BMS grouping only."""
        if V.ndim != 2 or tuple(V.shape[1:]) != (3,):
            raise ValueError(f"Task014 requires Dynamic3D V[N,3], got {tuple(V.shape)}")
        V_norm = (
            (V - V.mean(dim=0, keepdim=True))
            / (V.std(dim=0, keepdim=True) + 1e-8)
        )
        groups, _, _ = self.mean_shift_clustering(V_norm)
        return groups

    def run_functional_preflight(self):
        """Run mapping audit and one real BMS-domain numerical gate only.

        This method does not create a registry, apply a pruning mask, validate
        the model, or accumulate parameter cost.  It is the mandatory first
        server step before the six Task014 prune-only experiments.
        """
        valid_layer_names = [
            name for name in self.ordered_layer_names if name in self.activations
        ]
        if not valid_layer_names:
            raise RuntimeError("Task014 preflight found no calibrated pruning layers")
        if self.selection_mode == "functional" and self.functional_descriptor_cache:
            V, unit_info, _, _, _ = self._load_functional_descriptor_cache(
                valid_layer_names
            )
        else:
            V, unit_info, _, _, _ = self.build_3d_descriptors(valid_layer_names)
        if self.descriptor_variant != "dynamic3d":
            raise ValueError("Task014 preflight requires descriptor_variant=dynamic3d")
        groups = self._functional_bms_domains(V)
        archive, audit, _, _ = self._functional_mapping_context(
            unit_info, valid_layer_names
        )
        result = run_one_domain_numerical_preflight(
            archive,
            audit,
            self.functional_zero_field_audit,
            groups,
            self.device,
            self.functional_audit_dir,
        )
        self.functional_preflight_result = result
        print(">>> Task014 mapping audit: PASSED")
        print(
            f">>> Task014 one-domain numerical preflight: PASSED "
            f"(domain={result['domain_id']}, size={result['domain_size']})"
        )
        return result

    def functional_pruning(self, V, unit_info, valid_layer_names,
                           step_target_sparsity=None):
        """Select units by dynamic marginal functional loss inside BMS domains.

        ``V in R^[N,3]`` is used only to construct BMS domains.  Cached fields
        have source shape ``[9,U_l,16,H_l,W_l]``.  The nine per-video fields of
        every unit are independently pooled to ``[16,7,7]``, concatenated to a
        signed vector in ``R^[9*16*7*7]``, and L2-normalized only when its norm
        exceeds the existing normalization epsilon.
        Signed cosine is converted to same-direction similarity as
        ``clamp(cosine,0,1)``.  The immutable mask
        ``valid_function_mask in {0,1}^[|G_k|]`` marks fields whose complete
        pooled signed vector norm exceeds the existing normalization epsilon.
        Exact-zero and raw-near-zero fields are numerical Null Functional
        Units under the current functional calibration set.  Within each
        domain ``G_k``, matrix
        ``A^(k) in R^[|G_k|,|G_k|]`` defines active-demand coverage and the
        set-dependent loss ``Delta_i(S_k)``.  Cost is used only after the next
        feasible minimum-loss unit is chosen.
        """
        print("\n>>> Selection mode: FUNCTIONAL COVERAGE")
        print(">>> Descriptor: Dynamic3D")
        print(f">>> Functional score: {self.functional_score}")
        if self.descriptor_variant != "dynamic3d":
            raise ValueError("Functional selection requires Dynamic3D")
        if step_target_sparsity is None:
            step_target_sparsity = self.target_sparsity
        if not 0.0 <= float(step_target_sparsity) <= 1.0:
            raise ValueError("step_target_sparsity must be in [0,1]")
        if len(unit_info) != V.shape[0]:
            raise ValueError("Descriptor rows and unit_info rows differ")

        # BMS is called directly for membership only.  The old group manifold
        # score and group_pruning() priority are intentionally not invoked.
        groups = self._functional_bms_domains(V)
        print(f">>> BMS domains: {len(groups)}")
        archive, audit, unit_costs, layer_capacities = (
            self._functional_mapping_context(unit_info, valid_layer_names)
        )
        print(f">>> Contribution Fields loaded: {audit.mapped_units}")

        # The real-domain numerical gate runs before mmap construction and
        # before a single production unit can be selected.
        self.functional_preflight_result = run_one_domain_numerical_preflight(
            archive,
            audit,
            self.functional_zero_field_audit,
            groups,
            self.device,
            self.functional_audit_dir,
        )
        print(">>> One-domain numerical gate: PASSED")

        vector_memmap, vector_mask_memmap = archive.prepare_vector_memmap(
            audit, self.functional_cache_dir, self.device
        )
        target_budget = float(self.total_original_params) * float(
            step_target_sparsity
        )
        print(f">>> Target estimated parameter budget: {target_budget:.0f}")
        selector = FunctionalCompetitionPruner(
            groups=groups,
            audit=audit,
            vector_memmap=vector_memmap,
            vector_mask_memmap=vector_mask_memmap,
            unit_costs=unit_costs,
            max_prunable_by_layer=layer_capacities,
            target_parameter_budget=target_budget,
            preferred_device=self.device,
            functional_score=self.functional_score,
            total_original_parameters=self.total_original_params,
        )
        result = selector.select()
        write_functional_selection_artifacts(self.functional_audit_dir, result)
        self.functional_selection_result = result
        return (
            result.registry,
            result.removed_parameter_cost / float(self.total_original_params),
        )


    def coverage_pruning(self, V, unit_info, valid_layer_names,
                         step_target_sparsity=None):
        """用D_rel候选集合与Contribution Field覆盖修正生成registry。

        Args:
            V: ``[N,3]``三维描述符，列依次为D_abs、D_rel和D_st
            unit_info: 长度为N的单元信息，与V逐行对应
            valid_layer_names: 当前有效剪枝层名称
            step_target_sparsity: 本轮沿用的参数剪枝目标

        Returns:
            registry: ``{layer_name: set(pruned_indices)}``
            actual_sparsity: 按现有参数代价定义计算的实际剪枝率

        Contribution Field ``F``不进入V，也不生成新的importance score；
        它只在每层内把``initial_remove``修正为等规模的``final_remove``。
        """
        print("\n>>> 步骤2: D_rel标量候选剪枝与Contribution Field覆盖修正...")

        if self.contribution_npz is None:
            raise ValueError(
                "selection_mode='coverage' requires --contribution_npz"
            )
        if step_target_sparsity is None:
            step_target_sparsity = self.target_sparsity
        if not 0.0 <= step_target_sparsity <= 1.0:
            raise ValueError(
                "step_target_sparsity must be between 0 and 1, "
                f"got {step_target_sparsity}"
            )
        if V.ndim != 2 or V.shape[1] != 3:
            raise ValueError(
                f"V must have shape [N,3], got {tuple(V.shape)}"
            )
        if len(unit_info) != V.shape[0]:
            raise ValueError(
                f"unit_info has {len(unit_info)} rows but V has {V.shape[0]}"
            )

        # scalar_scores: [N].  Only V[:,1] (D_rel) creates initial_remove.
        scalar_scores = V[:, 1].detach().to(dtype=torch.float32)
        if not torch.isfinite(scalar_scores).all():
            raise ValueError("D_rel scalar scores contain NaN or infinity")

        modules = dict(self.model.named_modules())
        registry = {name: set() for name in valid_layer_names}
        layer_info = {}
        layer_global_indices = defaultdict(list)
        unit_costs = []

        for global_index, info in enumerate(unit_info):
            layer_name = info['layer']
            unit_index = int(info['idx'])
            if layer_name not in registry:
                raise KeyError(f"unit_info references invalid layer {layer_name!r}")
            if layer_name not in modules:
                raise KeyError(f"Model module not found for layer {layer_name!r}")

            module = modules[layer_name]
            if "WindowAttention3D" in module.__class__.__name__:
                unit_type = 'head'
            elif "Mlp" in module.__class__.__name__:
                unit_type = 'neuron'
            else:
                raise TypeError(
                    f"Unsupported pruning module {module.__class__.__name__} "
                    f"for layer {layer_name!r}"
                )
            unit_cost = self.estimate_unit_cost(module, unit_type)
            if unit_cost <= 0:
                raise ValueError(
                    f"Non-positive pruning cost for {layer_name}[{unit_index}]"
                )

            unit_costs.append(unit_cost)
            layer_global_indices[layer_name].append((unit_index, global_index))

        for layer_name in valid_layer_names:
            pairs = sorted(layer_global_indices[layer_name])
            local_indices = [unit_index for unit_index, _ in pairs]
            if local_indices != list(range(len(pairs))):
                raise ValueError(
                    f"Unit indices for {layer_name!r} are not contiguous from zero"
                )
            total_units = len(pairs)
            min_keep = max(1, int(total_units * self.min_keep_ratio))
            layer_info[layer_name] = {
                'total': total_units,
                'pruned': 0,
                'max_prunable': total_units - min_keep,
            }

        total_p = self.total_original_params
        if total_p <= 0:
            raise ValueError("The model has no parameters for sparsity accounting")
        target_red = total_p * step_target_sparsity
        current_red = 0

        # Scalar candidate pruning: global D_rel order, existing parameter
        # costs, and the existing per-layer minimum-retention constraint.
        score_values = scalar_scores.cpu().tolist()
        scalar_order = sorted(
            range(len(unit_info)),
            key=lambda index: (score_values[index], index),
        )
        for global_index in scalar_order:
            if current_red >= target_red:
                break
            info = unit_info[global_index]
            layer_name = info['layer']
            if layer_info[layer_name]['pruned'] >= layer_info[layer_name]['max_prunable']:
                continue
            registry[layer_name].add(int(info['idx']))
            layer_info[layer_name]['pruned'] += 1
            current_red += unit_costs[global_index]

        initial_remove_count = sum(len(indices) for indices in registry.values())
        if current_red < target_red:
            print(
                "  警告: 受每层最小保留约束限制，D_rel候选集合未达到目标参数量"
            )

        # Snapshot the D_rel-only set.  It supplies selected_before in the CSV
        # and is never overwritten by a coverage-derived scalar score.
        initial_registry = {
            layer_name: set(indices) for layer_name, indices in registry.items()
        }
        contribution_fields = load_contribution_fields(self.contribution_npz)
        missing_field_layers = [
            layer_name for layer_name in valid_layer_names
            if layer_name not in contribution_fields
        ]
        if missing_field_layers:
            print(
                f"  警告: {len(missing_field_layers)}个有效层没有Contribution "
                "Field，将保留其D_rel初始删除集合"
            )
        matched_layers = 0
        repaired_layers = 0
        weighted_initial_coverage = 0.0
        weighted_final_coverage = 0.0
        coverage_unit_count = 0

        for layer_name in valid_layer_names:
            if layer_name not in contribution_fields:
                continue

            matched_layers += 1
            pairs = sorted(layer_global_indices[layer_name])
            global_indices = [global_index for _, global_index in pairs]
            layer_scores = scalar_scores[global_indices]  # [U]
            fields = contribution_fields[layer_name]  # [U,T,H,W]
            if fields.shape[0] != layer_scores.numel():
                raise ValueError(
                    f"Contribution Field for {layer_name!r} has {fields.shape[0]} "
                    f"units but unit_info has {layer_scores.numel()}"
                )

            remove_num = len(initial_registry[layer_name])
            local_score_values = layer_scores.detach().cpu().tolist()
            expected_initial = set(sorted(
                range(layer_scores.numel()),
                key=lambda index: (local_score_values[index], index),
            )[:remove_num])
            if expected_initial != initial_registry[layer_name]:
                raise RuntimeError(
                    f"Global D_rel selection is inconsistent with the per-layer "
                    f"initial_remove set for {layer_name!r}"
                )

            # Similarity work runs on the model device (CPU or CUDA) and is
            # chunked inside coverage_selector.py.  F remains [U,T,H,W].
            fields_device = fields.to(device=self.device, non_blocking=True)
            layer_scores_device = layer_scores.to(
                device=self.device, dtype=torch.float32
            )
            repaired_remove = greedy_coverage_repair(
                layer_scores_device, fields_device, remove_num
            )  # [R]
            repaired_set = set(repaired_remove.tolist())
            if len(repaired_set) != remove_num:
                raise RuntimeError(
                    f"Coverage repair changed the removal count for {layer_name!r}"
                )

            num_units = layer_scores.numel()
            initial_keep = [
                index for index in range(num_units)
                if index not in initial_registry[layer_name]
            ]  # [K]
            final_keep = [
                index for index in range(num_units)
                if index not in repaired_set
            ]  # [K]
            initial_coverage = float(functional_coverage(
                fields_device, initial_keep
            ).item())
            final_coverage = float(functional_coverage(
                fields_device, final_keep
            ).item())
            if final_coverage + 1e-7 < initial_coverage:
                raise RuntimeError(
                    f"Coverage decreased for {layer_name!r}: "
                    f"{initial_coverage:.8f} -> {final_coverage:.8f}"
                )

            if repaired_set != initial_registry[layer_name]:
                repaired_layers += 1
            registry[layer_name] = repaired_set
            weighted_initial_coverage += initial_coverage * num_units
            weighted_final_coverage += final_coverage * num_units
            coverage_unit_count += num_units

            print(
                f"  {layer_name}: {initial_coverage:.6f} -> "
                f"{final_coverage:.6f} "
                f"({final_coverage - initial_coverage:+.6f})"
            )

        if matched_layers == 0:
            available = ", ".join(sorted(contribution_fields))
            raise ValueError(
                "No Contribution Field layer matches the current pruning layers. "
                f"Available layers: {available}"
            )

        final_remove_count = sum(len(indices) for indices in registry.values())
        if final_remove_count != initial_remove_count:
            raise RuntimeError(
                "Coverage repair changed the global number of removed units"
            )

        initial_coverage = weighted_initial_coverage / coverage_unit_count
        final_coverage = weighted_final_coverage / coverage_unit_count
        print(f"Initial functional coverage: {initial_coverage:.6f}")
        print(f"Final functional coverage: {final_coverage:.6f}")
        print(f"Coverage improvement: {final_coverage - initial_coverage:.6f}")

        csv_rows = []
        for layer_name in valid_layer_names:
            pairs = sorted(layer_global_indices[layer_name])
            for unit_index, global_index in pairs:
                csv_rows.append({
                    'layer': layer_name,
                    'unit_index': unit_index,
                    'D_rel': float(scalar_scores[global_index].item()),
                    'selected_before': int(
                        unit_index in initial_registry[layer_name]
                    ),
                    'selected_after': int(unit_index in registry[layer_name]),
                })

        result_path = Path("coverage_selection_result.csv")
        with result_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=(
                    'layer', 'unit_index', 'D_rel',
                    'selected_before', 'selected_after',
                ),
            )
            writer.writeheader()
            writer.writerows(csv_rows)

        print(
            f">>> Coverage修正完成: 匹配{matched_layers}层，"
            f"发生交换{repaired_layers}层，删除单元数保持{final_remove_count}"
        )
        print(f">>> Coverage选择记录: {result_path.resolve()}")
        return registry, current_red / total_p


    def apply_pruning_masks(self, registry, valid_layer_names):
        """
        将剪枝掩码应用到模型
        
        Args:
            registry: {layer_name: set(pruned_indices)}
            valid_layer_names: 有效层名称
        """
        print("\n>>> 步骤4: 应用剪枝掩码...")
        
        for name in valid_layer_names:
            m = dict(self.model.named_modules())[name]
            pruned = registry[name]
            
            if "WindowAttention3D" in m.__class__.__name__:
                # 确保至少保留1个头
                all_heads = set(range(m.num_heads))
                remaining = all_heads - pruned
                if len(remaining) == 0:
                    # 如果全部要被剪枝，至少保留1个（保留索引0）
                    remaining = {0}
                    print(f"  警告: {name} 尝试剪枝所有头，强制保留1个")
                m.keep_heads = sorted(list(remaining))
                if len(pruned) > 0:
                    actual_pruned = len(all_heads) - len(remaining)
                    print(f"  {name}: Heads {m.num_heads} -> {len(m.keep_heads)} "
                          f"(-{actual_pruned/m.num_heads:.1%})")
                          
            elif "Mlp" in m.__class__.__name__:
                # 确保至少保留1个神经元
                all_neurons = set(range(m.original_hidden_features))
                remaining = all_neurons - pruned
                if len(remaining) == 0:
                    # 如果全部要被剪枝，至少保留1个
                    remaining = {0}
                    print(f"  警告: {name} 尝试剪枝所有神经元，强制保留1个")
                m.keep_neurons = sorted(list(remaining))
                if len(pruned) > 0:
                    actual_pruned = len(all_neurons) - len(remaining)
                    print(f"  {name}: Neurons {m.original_hidden_features} -> {len(m.keep_neurons)} "
                          f"(-{actual_pruned/m.original_hidden_features:.1%})")

    def prune(self, step_target_sparsity=None):
        """
        执行单次剪枝流程（供外部直接调用）
        
        流程：构建三维描述符 -> Mean-Shift聚类 -> 流形一致性评分 -> 组级剪枝 -> 应用掩码
        
        Args:
            step_target_sparsity: 本轮目标稀疏度，默认为self.target_sparsity
            
        Returns:
            actual_sparsity: 实际达到的稀疏度
        """
        if step_target_sparsity is None:
            step_target_sparsity = self.target_sparsity
            
        print("\n" + "="*60)
        print(">>> 开始执行剪枝算法...")
        print("="*60)
        
        # 获取有效层
        valid_layer_names = [n for n in self.ordered_layer_names 
                             if n in self.activations]
        
        if len(valid_layer_names) == 0:
            print("警告: 没有有效的层可以剪枝")
            return 0.0
        
        # 1. 构建三维描述符
        V, unit_info, _, _, _ = self.build_3d_descriptors(valid_layer_names)
        
        if len(V) == 0:
            print("警告: 没有有效的剪枝单元")
            return 0.0
        
        # 2. 选择剪枝后端；默认BMS路径保持原有行为。
        if self.selection_mode == "bms":
            registry, actual_sparsity = self.group_pruning(
                V, unit_info, valid_layer_names, step_target_sparsity
            )
        elif self.selection_mode == "coverage":
            registry, actual_sparsity = self.coverage_pruning(
                V, unit_info, valid_layer_names, step_target_sparsity
            )
        elif self.selection_mode == "functional":
            registry, actual_sparsity = self.functional_pruning(
                V, unit_info, valid_layer_names, step_target_sparsity
            )
        else:
            raise RuntimeError(
                f"Unsupported selection_mode at prune time: {self.selection_mode!r}"
            )
        
        # 3. 应用掩码
        self.apply_pruning_masks(registry, valid_layer_names)
        
        # 4. 更新当前稀疏度
        self.current_sparsity += actual_sparsity
        self.pruning_history.append({
            'sparsity': self.current_sparsity,
            'step_sparsity': actual_sparsity
        })
        
        print(f">>> 剪枝完成！当前累计稀疏度: {self.current_sparsity:.2%}")
        
        return actual_sparsity


    def get_pruning_report(self):
        """
        生成剪枝报告
        
        Returns:
            report: 包含剪枝统计信息的字典
        """
        total_original_params = self.total_original_params
        
        current_heads_total = 0
        original_heads_total = 0
        current_neurons_total = 0
        original_neurons_total = 0
        total_reduction = 0
        
        for name, module in self.model.named_modules():
            if "WindowAttention3D" in str(type(module)):
                n_heads = len(module.keep_heads)
                orig_heads = module.num_heads
                head_dim = module.head_dim
                dim = module.qkv.in_features
                
                current_heads_total += n_heads
                original_heads_total += orig_heads
                
                if n_heads < orig_heads:
                    pruned_count = orig_heads - n_heads
                    qkv_weight_red = (3 * pruned_count * head_dim) * dim
                    qkv_bias_red = (3 * pruned_count * head_dim) if module.qkv.bias is not None else 0
                    proj_weight_red = module.proj.out_features * (pruned_count * head_dim)
                    total_reduction += (qkv_weight_red + qkv_bias_red + proj_weight_red)
                    
            elif "Mlp" in str(type(module)):
                if hasattr(module, 'keep_neurons'):
                    n_neurons = len(module.keep_neurons)
                    orig_neurons = module.original_hidden_features
                    in_features = module.fc1.in_features
                    out_features = module.fc2.out_features
                    
                    current_neurons_total += n_neurons
                    original_neurons_total += orig_neurons
                    
                    if n_neurons < orig_neurons:
                        pruned_count = orig_neurons - n_neurons
                        fc1_red = pruned_count * in_features
                        if module.fc1.bias is not None:
                            fc1_red += pruned_count
                        fc2_red = out_features * pruned_count
                        total_reduction += (fc1_red + fc2_red)
        
        current_params = total_original_params - total_reduction
        head_ratio = 1 - (current_heads_total / original_heads_total) if original_heads_total > 0 else 0
        neuron_ratio = 1 - (current_neurons_total / original_neurons_total) if original_neurons_total > 0 else 0
        actual_sparsity = total_reduction / total_original_params
        
        print(f"\n" + "="*50)
        print(f"{'结构剪枝报告':^50}")
        print(f"-"*50)
        print(f"总参数量 (原始):     {total_original_params/1e6:>10.2f} M")
        print(f"总参数量 (剪枝后):   {current_params/1e6:>10.2f} M")
        print(f"参数减少量:          {total_reduction/1e6:>10.2f} M")
        print(f"实际压缩率:          {actual_sparsity:>10.2%}")
        print(f"-"*50)
        print(f"注意力头:            {original_heads_total} -> {current_heads_total} (-{head_ratio:.1%})")
        print(f"FFN神经元:           {original_neurons_total} -> {current_neurons_total} (-{neuron_ratio:.1%})")
        print(f"="*50)
        
        return {
            'total_params_m': current_params / 1e6,
            'original_params_m': total_original_params / 1e6,
            'sparsity': actual_sparsity,
            'remaining_heads': current_heads_total,
            'original_heads': original_heads_total,
            'remaining_neurons': current_neurons_total,
            'original_neurons': original_neurons_total,
            'pruning_history': self.pruning_history
        }


# =============================================================================
# Mlp 模块（支持神经元剪枝）
# =============================================================================

class Mlp(nn.Module):
    """支持结构化剪枝的MLP模块"""
    
    def __init__(self, in_features, hidden_features=None, out_features=None, 
                 act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        
        # 剪枝相关属性
        self.original_hidden_features = hidden_features
        self.keep_neurons = list(range(hidden_features))

    def forward(self, x):
        if len(self.keep_neurons) < self.original_hidden_features:
            # 动态索引切片（剪枝模式）
            idx = torch.tensor(self.keep_neurons, device=x.device, dtype=torch.long)
            
            # fc1: 输出维度被剪枝
            w1 = self.fc1.weight.index_select(0, idx)
            b1 = self.fc1.bias.index_select(0, idx) if self.fc1.bias is not None else None
            x = F.linear(x, w1, b1)
            x = self.act(x)
            x = self.drop(x)
            
            # fc2: 输入维度被剪枝
            w2 = self.fc2.weight.index_select(1, idx)
            x = F.linear(x, w2, self.fc2.bias)
        else:
            # 正常前向
            x = self.fc1(x)
            x = self.act(x)
            x = self.drop(x)
            x = self.fc2(x)
            
        x = self.drop(x)
        return x


# =============================================================================
# WindowAttention3D 模块（支持注意力头剪枝）
# =============================================================================

class WindowAttention3D(nn.Module):
    """支持注意力头剪枝的3D窗口注意力模块"""
    
    def __init__(self, dim, window_size, num_heads, qkv_bias=False, 
                 qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.head_dim = head_dim
        self.scale = qk_scale or head_dim ** -0.5

        # 相对位置偏置表
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1) * 
                       (2 * window_size[2] - 1), num_heads))
        
        self.register_buffer("relative_position_index", self._make_relative_position_index())

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

        # 剪枝相关属性
        self.keep_heads = list(range(num_heads))

    def _make_relative_position_index(self):
        """生成相对位置索引"""
        coords_d = torch.arange(self.window_size[0])
        coords_h = torch.arange(self.window_size[1])
        coords_w = torch.arange(self.window_size[2])
        coords = torch.stack(torch.meshgrid(coords_d, coords_h, coords_w, indexing='ij'))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 2] += self.window_size[2] - 1
        relative_coords[:, :, 0] *= (2 * self.window_size[1] - 1) * (2 * self.window_size[2] - 1)
        relative_coords[:, :, 1] *= (2 * self.window_size[2] - 1)
        return relative_coords.sum(-1)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        
        # QKV投影并reshape为多头形式
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # 应用剪枝：只保留选中的头
        if len(self.keep_heads) < self.num_heads:
            head_idx = torch.tensor(self.keep_heads, device=x.device, dtype=torch.long)
            q = q.index_select(1, head_idx)
            k = k.index_select(1, head_idx)
            v = v.index_select(1, head_idx)
            curr_heads = len(self.keep_heads)
        else:
            curr_heads = self.num_heads

        # 缩放点积注意力
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        # 相对位置偏置
        rel_pos_bias = self.relative_position_bias_table[
            self.relative_position_index[:N, :N].reshape(-1)].reshape(N, N, -1)
        rel_pos_bias = rel_pos_bias.permute(2, 0, 1).contiguous()
        
        if len(self.keep_heads) < self.num_heads:
            rel_pos_bias = rel_pos_bias.index_select(0, head_idx)
            
        attn = attn + rel_pos_bias.unsqueeze(0)

        # 应用mask
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, curr_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, curr_heads, N, N)
            
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        # 注意力输出
        x = (attn @ v).transpose(1, 2).reshape(B_, N, -1)
        
        # 输出投影（处理剪枝后的维度）
        if len(self.keep_heads) < self.num_heads:
            # 构造权重索引
            keep_channels = []
            for h in self.keep_heads:
                keep_channels.extend(range(h * self.head_dim, (h + 1) * self.head_dim))
            c_idx = torch.tensor(keep_channels, device=x.device, dtype=torch.long)
            
            # proj权重: [out_dim, in_dim]，输入维度被剪枝
            w_proj = self.proj.weight.index_select(1, c_idx)
            x = F.linear(x, w_proj, self.proj.bias)
        else:
            x = self.proj(x)
            
        x = self.proj_drop(x)
        return x


# =============================================================================
# 辅助函数
# =============================================================================

def window_partition(x, window_size):
    """
    Args:
        x: (B, D, H, W, C)
        window_size: (Wd, Wh, Ww)
    Returns:
        windows: (B*num_windows, Wd*Wh*Ww, C)
    """
    B, D, H, W, C = x.shape
    x = x.view(B, D // window_size[0], window_size[0], 
               H // window_size[1], window_size[1], 
               W // window_size[2], window_size[2], C)
    windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(
        -1, reduce(mul, window_size), C)
    return windows


def window_reverse(windows, window_size, B, D, H, W):
    """
    Args:
        windows: (B*num_windows, Wd*Wh*Ww, C)
        window_size: (Wd, Wh, Ww)
    Returns:
        x: (B, D, H, W, C)
    """
    x = windows.view(B, D // window_size[0], H // window_size[1], W // window_size[2],
                     window_size[0], window_size[1], window_size[2], -1)
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(B, D, H, W, -1)
    return x


def get_window_size(x_size, window_size, shift_size=None):
    """获取实际使用的窗口大小"""
    use_window_size = list(window_size)
    if shift_size is not None:
        use_shift_size = list(shift_size)
    for i in range(len(x_size)):
        if x_size[i] <= window_size[i]:
            use_window_size[i] = x_size[i]
            if shift_size is not None:
                use_shift_size[i] = 0
    if shift_size is None:
        return tuple(use_window_size)
    else:
        return tuple(use_window_size), tuple(use_shift_size)


# =============================================================================
# Swin Transformer 3D 模块
# =============================================================================

class SwinTransformerBlock3D(nn.Module):
    """支持剪枝的3D Swin Transformer块"""
    
    def __init__(self, dim, num_heads, window_size=(2, 7, 7), shift_size=(0, 0, 0),
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., 
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, 
                 use_checkpoint=False):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.use_checkpoint = use_checkpoint

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention3D(
            dim, window_size=self.window_size, num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, 
                      act_layer=act_layer, drop=drop)

    def forward_part1(self, x, mask_matrix):
        """注意力部分"""
        B, D, H, W, C = x.shape
        window_size, shift_size = get_window_size((D, H, W), self.window_size, self.shift_size)

        x = self.norm1(x)
        
        # Padding
        pad_l = pad_t = pad_d0 = 0
        pad_d1 = (window_size[0] - D % window_size[0]) % window_size[0]
        pad_b = (window_size[1] - H % window_size[1]) % window_size[1]
        pad_r = (window_size[2] - W % window_size[2]) % window_size[2]
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b, pad_d0, pad_d1))
        _, Dp, Hp, Wp, _ = x.shape

        # Cyclic shift
        if any(i > 0 for i in shift_size):
            shifted_x = torch.roll(x, shifts=(-shift_size[0], -shift_size[1], -shift_size[2]), 
                                  dims=(1, 2, 3))
            attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = None

        # Window partition
        x_windows = window_partition(shifted_x, window_size)
        
        # 仅记录本次前向的整数几何信息，供校准Hook将窗口响应恢复为
        # [B,U,T,H,W]；不参与模型计算，也不改变注意力输出。
        self.attn._pruning_geometry = {
            'batch_size': B,
            'depth': D,
            'height': H,
            'width': W,
            'padded_depth': Dp,
            'padded_height': Hp,
            'padded_width': Wp,
            'window_size': tuple(window_size),
            'shift_size': tuple(shift_size),
        }

        # Window Attention
        attn_windows = self.attn(x_windows, mask=attn_mask)
        
        # Window reverse
        attn_windows = attn_windows.view(-1, *(window_size + (C,)))
        shifted_x = window_reverse(attn_windows, window_size, B, Dp, Hp, Wp)
        
        # Reverse cyclic shift
        if any(i > 0 for i in shift_size):
            x = torch.roll(shifted_x, shifts=(shift_size[0], shift_size[1], shift_size[2]), 
                          dims=(1, 2, 3))
        else:
            x = shifted_x

        if pad_d1 > 0 or pad_r > 0 or pad_b > 0:
            x = x[:, :D, :H, :W, :].contiguous()
        return x

    def forward_part2(self, x):
        """MLP部分"""
        return self.drop_path(self.mlp(self.norm2(x)))

    def forward(self, x, mask_matrix):
        # 残差连接 + Attention
        use_checkpoint = (
            self.use_checkpoint and self.training and torch.is_grad_enabled()
        )
        if use_checkpoint:
            x = x + self.drop_path(checkpoint.checkpoint(self.forward_part1, x, mask_matrix))
        else:
            x = x + self.drop_path(self.forward_part1(x, mask_matrix))

        # 残差连接 + MLP
        if use_checkpoint:
            x = x + checkpoint.checkpoint(self.forward_part2, x)
        else:
            x = x + self.forward_part2(x)

        return x


class PatchMerging(nn.Module):
    """Patch Merging层"""
    
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        B, D, H, W, C = x.shape

        # Padding
        pad_input = (H % 2 == 1) or (W % 2 == 1)
        if pad_input:
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))

        x0 = x[:, :, 0::2, 0::2, :]
        x1 = x[:, :, 1::2, 0::2, :]
        x2 = x[:, :, 0::2, 1::2, :]
        x3 = x[:, :, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], -1)

        x = self.norm(x)
        x = self.reduction(x)

        return x


@lru_cache()
def compute_mask(D, H, W, window_size, shift_size, device):
    """计算注意力mask"""
    img_mask = torch.zeros((1, D, H, W, 1), device=device)
    cnt = 0
    for d in slice(-window_size[0]), slice(-window_size[0], -shift_size[0]), slice(-shift_size[0], None):
        for h in slice(-window_size[1]), slice(-window_size[1], -shift_size[1]), slice(-shift_size[1], None):
            for w in slice(-window_size[2]), slice(-window_size[2], -shift_size[2]), slice(-shift_size[2], None):
                img_mask[:, d, h, w, :] = cnt
                cnt += 1
    mask_windows = window_partition(img_mask, window_size)
    mask_windows = mask_windows.squeeze(-1)
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
    return attn_mask


class BasicLayer(nn.Module):
    """Swin Transformer的一个阶段"""
    
    def __init__(self, dim, depth, num_heads, window_size=(1, 7, 7), mlp_ratio=4.,
                 qkv_bias=False, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False):
        super().__init__()
        self.window_size = window_size
        self.shift_size = tuple(i // 2 for i in window_size)
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # 构建blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock3D(
                dim=dim, num_heads=num_heads, window_size=window_size,
                shift_size=(0, 0, 0) if (i % 2 == 0) else self.shift_size,
                mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop, attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer, use_checkpoint=use_checkpoint)
            for i in range(depth)])
        
        self.downsample = downsample
        if self.downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)

    def forward(self, x):
        B, C, D, H, W = x.shape
        window_size, shift_size = get_window_size((D, H, W), self.window_size, self.shift_size)
        x = rearrange(x, 'b c d h w -> b d h w c')
        Dp = int(np.ceil(D / window_size[0])) * window_size[0]
        Hp = int(np.ceil(H / window_size[1])) * window_size[1]
        Wp = int(np.ceil(W / window_size[2])) * window_size[2]
        attn_mask = compute_mask(Dp, Hp, Wp, window_size, shift_size, x.device)
        
        for blk in self.blocks:
            x = blk(x, attn_mask)
        x = x.view(B, D, H, W, -1)

        if self.downsample is not None:
            x = self.downsample(x)
        x = rearrange(x, 'b d h w c -> b c d h w')
        return x


class PatchEmbed3D(nn.Module):
    """3D Patch Embedding"""
    
    def __init__(self, patch_size=(2, 4, 4), in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else None

    def forward(self, x):
        _, _, D, H, W = x.size()
        if W % self.patch_size[2] != 0:
            x = F.pad(x, (0, self.patch_size[2] - W % self.patch_size[2]))
        if H % self.patch_size[1] != 0:
            x = F.pad(x, (0, 0, 0, self.patch_size[1] - H % self.patch_size[1]))
        if D % self.patch_size[0] != 0:
            x = F.pad(x, (0, 0, 0, 0, 0, self.patch_size[0] - D % self.patch_size[0]))

        x = self.proj(x)
        if self.norm is not None:
            D, Wh, Ww = x.size(2), x.size(3), x.size(4)
            x = x.flatten(2).transpose(1, 2)
            x = self.norm(x)
            x = x.transpose(1, 2).view(-1, self.embed_dim, D, Wh, Ww)

        return x


class I3DHead(nn.Module):
    """I3D分类头"""
    
    def __init__(self, num_classes, in_channels, spatial_type='avg', 
                 dropout_ratio=0.5, init_std=0.01, **kwargs):
        super().__init__()
        self.spatial_type = spatial_type
        self.dropout_ratio = dropout_ratio
        self.init_std = init_std
        
        if self.dropout_ratio != 0:
            self.dropout = nn.Dropout(p=self.dropout_ratio)
        else:
            self.dropout = None
        self.fc_cls = nn.Linear(in_channels, num_classes)

        if self.spatial_type == 'avg':
            self.avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        else:
            self.avg_pool = None

    def forward(self, x):
        if self.avg_pool is not None:
            x = self.avg_pool(x)
        if self.dropout is not None:
            x = self.dropout(x)
        x = x.view(x.shape[0], -1)
        cls_score = self.fc_cls(x)
        return cls_score


class SwinTransformer3D(nn.Module):
    """支持结构化剪枝的Swin Transformer 3D"""
    
    def __init__(self, pretrained=None, pretrained2d=True, patch_size=(2, 4, 4),
                 in_chans=3, embed_dim=96, depths=[2, 2, 6, 2], 
                 num_heads=[3, 6, 12, 24], window_size=(8, 7, 7), mlp_ratio=4.,
                 qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0.2, norm_layer=nn.LayerNorm, patch_norm=False,
                 frozen_stages=-1, use_checkpoint=False, num_classes=400):
        super().__init__()

        self.pretrained = pretrained
        self.pretrained2d = pretrained2d
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.frozen_stages = frozen_stages
        self.window_size = window_size
        self.patch_size = patch_size
        
        # 分类头
        self.num_features = int(embed_dim * 2**(self.num_layers - 1))
        self.cls_head = I3DHead(num_classes, self.num_features).cuda()

        # Patch embedding
        self.patch_embed = PatchEmbed3D(
            patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # Stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        # 构建各层
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=int(embed_dim * 2**i_layer),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging if i_layer < self.num_layers - 1 else None,
                use_checkpoint=use_checkpoint)
            self.layers.append(layer)

        self.norm = norm_layer(self.num_features)

    def forward(self, x):
        features = []
        x = self.patch_embed(x)
        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x.contiguous())
            features.append(x)
            
        x = rearrange(x, 'n c d h w -> n d h w c')
        x = self.norm(x)
        x = rearrange(x, 'n d h w c -> n c d h w')
        x = self.cls_head(x)
        return x, features

    def get_detailed_pruning_report(self):
        """
        生成详细的剪枝报告
        
        Returns:
            report: 包含剪枝统计信息的字典
        """
        total_original_params = sum(p.numel() for p in self.parameters())
        
        current_heads_total = 0
        original_heads_total = 0
        current_neurons_total = 0
        original_neurons_total = 0
        total_reduction = 0
        
        layer_details = []
        
        for name, module in self.named_modules():
            if "WindowAttention3D" in str(type(module)):
                n_heads = len(module.keep_heads) if hasattr(module, 'keep_heads') else module.num_heads
                orig_heads = module.num_heads
                head_dim = module.head_dim
                dim = module.qkv.in_features
                
                current_heads_total += n_heads
                original_heads_total += orig_heads
                
                if n_heads < orig_heads:
                    pruned_count = orig_heads - n_heads
                    qkv_weight_red = (3 * pruned_count * head_dim) * dim
                    qkv_bias_red = (3 * pruned_count * head_dim) if module.qkv.bias is not None else 0
                    proj_weight_red = module.proj.out_features * (pruned_count * head_dim)
                    total_reduction += (qkv_weight_red + qkv_bias_red + proj_weight_red)
                    
                layer_details.append({
                    'name': name,
                    'type': 'attention',
                    'original': orig_heads,
                    'remaining': n_heads,
                    'pruned_ratio': 1 - n_heads/orig_heads if orig_heads > 0 else 0
                })
                    
            elif "Mlp" in str(type(module)):
                if hasattr(module, 'keep_neurons'):
                    n_neurons = len(module.keep_neurons)
                    orig_neurons = module.original_hidden_features
                    in_features = module.fc1.in_features
                    out_features = module.fc2.out_features
                    
                    current_neurons_total += n_neurons
                    original_neurons_total += orig_neurons
                    
                    if n_neurons < orig_neurons:
                        pruned_count = orig_neurons - n_neurons
                        fc1_red = pruned_count * in_features
                        if module.fc1.bias is not None:
                            fc1_red += pruned_count
                        fc2_red = out_features * pruned_count
                        total_reduction += (fc1_red + fc2_red)
                        
                    layer_details.append({
                        'name': name,
                        'type': 'mlp',
                        'original': orig_neurons,
                        'remaining': n_neurons,
                        'pruned_ratio': 1 - n_neurons/orig_neurons if orig_neurons > 0 else 0
                    })
        
        current_params = total_original_params - total_reduction
        head_ratio = 1 - (current_heads_total / original_heads_total) if original_heads_total > 0 else 0
        neuron_ratio = 1 - (current_neurons_total / original_neurons_total) if original_neurons_total > 0 else 0
        actual_sparsity = total_reduction / total_original_params if total_original_params > 0 else 0
        
        print(f"\n" + "="*50)
        print(f"{'结构剪枝报告':^50}")
        print(f"-"*50)
        print(f"总参数量 (原始):     {total_original_params/1e6:>10.2f} M")
        print(f"总参数量 (剪枝后):   {current_params/1e6:>10.2f} M")
        print(f"参数减少量:          {total_reduction/1e6:>10.2f} M")
        print(f"实际压缩率:          {actual_sparsity:>10.2%}")
        print(f"-"*50)
        print(f"注意力头:            {original_heads_total} -> {current_heads_total} (-{head_ratio:.1%})")
        print(f"FFN神经元:           {original_neurons_total} -> {current_neurons_total} (-{neuron_ratio:.1%})")
        print(f"="*50)
        
        return {
            'total_params_m': current_params / 1e6,
            'original_params_m': total_original_params / 1e6,
            'sparsity': actual_sparsity,
            'remaining_heads': current_heads_total,
            'original_heads': original_heads_total,
            'remaining_neurons': current_neurons_total,
            'original_neurons': original_neurons_total,
            'layer_details': layer_details
        }


# =============================================================================
# 使用示例
# =============================================================================

if __name__ == "__main__":
    # 示例：创建模型并进行剪枝
    print("=" * 60)
    print("Swin Transformer 3D 结构化剪枝示例")
    print("=" * 60)
    
    # 创建模型
    model = SwinTransformer3D(
        patch_size=(2, 4, 4),
        in_chans=3,
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=(8, 7, 7),
        mlp_ratio=4.,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.2,
        patch_norm=True,
        num_classes=400
    )
    
    # 统计原始参数
    total_params = sum(p.numel() for p in model.parameters())
    print(f"原始模型参数量: {total_params/1e6:.2f} M")
    
    # 创建剪枝器
    pruner = InteractionPruner(
        model=model,
        target_sparsity=0.3,  # 目标稀疏度30%
        min_keep_ratio=0.1,   # 每层最少保留10%
        sigma=0.5             # Mean-Shift带宽
    )
    
    print("\n剪枝器配置:")
    print(f"  目标稀疏度: {pruner.target_sparsity:.2%}")
    print(f"  迭代剪枝步数: {pruner.iter_prune_steps}")
    print("  第三维: 时空响应轨迹一致性")
    
    print("\n>>> 使用说明:")
    print("1. 准备校准数据加载器 calib_loader (用于计算重要性)")
    print("2. 准备训练数据加载器 train_loader ")
    print("3. 调用: pruner.iterative_prune(calib_loader, train_loader, device)")
    print("4. 查看剪枝报告: pruner.get_pruning_report()")
