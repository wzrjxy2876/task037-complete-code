from __future__ import annotations

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
import csv
import json
import os
import re
import time

from function_bms_utils import (
    MAX_ACTIVATION_SAMPLES,
    NUMERICAL_EPS,
    REFERENCE_TIME_BINS,
    assert_finite,
    build_local_global_masks,
    compute_frame_overlap,
    compute_function_uniqueness,
    compute_function_similarity,
    compute_intra_cluster_functional_redundancy,
    compute_motion_retention,
    groups_to_labels,
    mean_shift_bms,
    normalize_frame_probabilities,
    resize_relation_matrix,
    robust_scale_01,
    safe_standardize_descriptors,
    select_descriptor_cluster_representatives,
    summarize_functional_redundancy,
    summarize_matrix_distribution,
    tensor_summary,
)




# =============================================================================
# 基于三维描述符与力场Mean-Shift的结构化剪枝器
# =============================================================================

class InteractionPruner:
    """
    三维描述符体系 + 力场Mean-Shift分组 + 流形一致性评分 + 迭代剪枝微调
    
    核心创新：
    1. 绝对重要性：时序平均激活幅度 + 时序激活频率（加权融合）
    2. 相对重要性：绝对重要性 × (1 - 可替代性分数)（舒尔补计算）
    3. 跨层影响：反向传播累加与指数衰减
    4. 力场Mean-Shift：高斯吸引力场驱动，自然形成功能簇
    5. 流形一致性：流线平行度 × 功能强度
    """
    
    def __init__(self, model, target_sparsity=0.5, iter_prune_steps=5,
                 finetune_epochs=5, finetune_lr=1e-5, gamma_decay=0.5,
                 min_keep_ratio=0.1, sigma=0.1):
        """
        Args:
            model: 待剪枝模型
            target_sparsity: 目标稀疏度（0-1）
            gamma_decay: 跨层影响衰减因子
            min_keep_ratio: 每层最小保留比例
            sigma: Mean-Shift高斯核带宽
        """
        self.model = model
        self.target_sparsity = target_sparsity
        self.iter_prune_steps = iter_prune_steps
        self.finetune_epochs = finetune_epochs
        self.finetune_lr = finetune_lr
        self.gamma_decay = gamma_decay
        self.min_keep_ratio = min_keep_ratio
        self.sigma = sigma
        
        self.activations = {}
        self.hooks = []
        self.device = next(model.parameters()).device
        self.ordered_layer_names = []
        self.unit_metadata = {}
        self.frame_relation_accumulators = {}
        self.current_input_shape = None
        self.latest_analysis = None
        
        # 统计信息
        self.total_original_params = sum(p.numel() for p in model.parameters())
        self.current_sparsity = 0.0
        self.pruning_history = []
        
    # =========================================================================
    # Hook 注册与激活捕获
    # =========================================================================
    
    def register_hooks(self):
        """Register hooks for static samples and online frame-relation fields."""
        self.hooks = []
        self.activations = {}
        self.ordered_layer_names = []
        self.unit_metadata = {}
        self.frame_relation_accumulators = {}

        for name, module in self.model.named_modules():
            classname = module.__class__.__name__
            if "WindowAttention3D" in classname:
                stage, block = self._parse_stage_block(name)
                self.unit_metadata[name] = {
                    "unit_type": "head",
                    "stage": stage,
                    "block": block,
                    "num_units": module.num_heads,
                }
                handle = module.proj.register_forward_pre_hook(
                    self._get_activation_hook(name, "head", module)
                )
            elif "Mlp" in classname:
                stage, block = self._parse_stage_block(name)
                self.unit_metadata[name] = {
                    "unit_type": "neuron",
                    "stage": stage,
                    "block": block,
                    "num_units": module.original_hidden_features,
                }
                handle = module.fc2.register_forward_pre_hook(
                    self._get_activation_hook(name, "neuron", module)
                )
            else:
                continue
            self.hooks.append(handle)
            self.ordered_layer_names.append(name)

    def remove_hooks(self):
        """Remove all calibration hooks."""
        for handle in self.hooks:
            handle.remove()
        self.hooks = []
        print(">>> All calibration hooks removed")

    @staticmethod
    def _parse_stage_block(layer_name):
        match = re.search(r"layers\.(\d+)\.blocks\.(\d+)", layer_name)
        if match is None:
            return -1, -1
        return int(match.group(1)), int(match.group(2))

    def _store_static_samples(self, name, response):
        """Maintain at most 2048 flattened samples [M, U] per layer."""
        samples = response.permute(0, 2, 3, 4, 1).reshape(-1, response.shape[1])
        if samples.shape[0] > MAX_ACTIVATION_SAMPLES:
            indices = torch.randperm(samples.shape[0], device=samples.device)
            samples = samples[indices[:MAX_ACTIVATION_SAMPLES]]
        samples = samples.detach().to(device="cpu", dtype=torch.float32)
        if name in self.activations:
            samples = torch.cat((self.activations[name], samples), dim=0)
            if samples.shape[0] > MAX_ACTIVATION_SAMPLES:
                indices = torch.randperm(samples.shape[0])[:MAX_ACTIVATION_SAMPLES]
                samples = samples[indices]
        self.activations[name] = samples

    def _window_reverse_3d(self, windows, geometry):
        """Reverse window partition and cyclic shift into [B,U,T,H,W]."""
        response = window_reverse(
            windows,
            geometry["window_size"],
            geometry["batch_size"],
            geometry["padded_depth"],
            geometry["padded_height"],
            geometry["padded_width"],
        )
        if any(value > 0 for value in geometry["shift_size"]):
            response = torch.roll(
                response, shifts=geometry["shift_size"], dims=(1, 2, 3)
            )
        response = response[
            :,
            :geometry["depth"],
            :geometry["height"],
            :geometry["width"],
            :,
        ]
        return response.permute(0, 4, 1, 2, 3).contiguous()

    def _extract_attention_spatiotemporal_response(self, x, source_module):
        """Recover head responses from pre-projection [B*nW,N,H*Dh]."""
        geometry = getattr(source_module, "_pruning_geometry", None)
        if geometry is None:
            raise RuntimeError("attention geometry was not recorded by its parent block")
        if x.ndim != 3:
            raise ValueError(
                f"attention pre-projection input must be 3D, got {tuple(x.shape)}"
            )
        expected_channels = source_module.num_heads * source_module.head_dim
        if x.shape[-1] != expected_channels:
            raise ValueError(
                f"attention channel mismatch: {x.shape[-1]} != {expected_channels}"
            )
        head_response = x.reshape(
            x.shape[0], x.shape[1], source_module.num_heads, source_module.head_dim
        ).abs().mean(dim=-1)
        response = self._window_reverse_3d(head_response, geometry)
        expected_shape = (
            geometry["batch_size"],
            source_module.num_heads,
            geometry["depth"],
            geometry["height"],
            geometry["width"],
        )
        if tuple(response.shape) != expected_shape:
            raise AssertionError(
                f"attention response shape {tuple(response.shape)} != {expected_shape}"
            )
        return response

    @staticmethod
    def _extract_mlp_spatiotemporal_response(x, source_module):
        """Recover neuron responses from the actual global MLP input layout."""
        if x.ndim != 5:
            raise ValueError(
                f"MLP fc2 pre-hook expects [B,T,H,W,U], got {tuple(x.shape)}"
            )
        if x.shape[-1] != source_module.original_hidden_features:
            raise ValueError(
                "MLP hidden dimension does not match original_hidden_features"
            )
        return x.abs().permute(0, 4, 1, 2, 3).contiguous()

    def _get_activation_hook(self, name, unit_type, source_module):
        """Consume a high-dimensional response within its current batch."""
        def hook(module, inputs):
            x = inputs[0].detach()
            if unit_type == "head":
                response = self._extract_attention_spatiotemporal_response(
                    x, source_module
                )
            else:
                response = self._extract_mlp_spatiotemporal_response(
                    x, source_module
                )
            assert_finite(f"{name}.spatiotemporal_response", response)
            self._store_static_samples(name, response)
            self._accumulate_frame_relation_fields(name, response)
            del response

        return hook

    def _accumulate_frame_relation_fields(self, name, response):
        """Accumulate O/R [U,8,8] on CPU and release [B,U,T,H,W]."""
        probabilities = normalize_frame_probabilities(response)
        overlap = resize_relation_matrix(
            compute_frame_overlap(probabilities), REFERENCE_TIME_BINS
        )
        motion = resize_relation_matrix(
            compute_motion_retention(probabilities), REFERENCE_TIME_BINS
        )
        overlap_sum = overlap.sum(dim=0).to(device="cpu", dtype=torch.float64)
        motion_sum = motion.sum(dim=0).to(device="cpu", dtype=torch.float64)
        batch_size = response.shape[0]
        if name not in self.frame_relation_accumulators:
            self.frame_relation_accumulators[name] = {
                "overlap_sum": overlap_sum,
                "motion_sum": motion_sum,
                "count": batch_size,
                "source_time": response.shape[2],
                "response_geometry": tuple(int(value) for value in response.shape[2:]),
            }
        else:
            accumulator = self.frame_relation_accumulators[name]
            if accumulator["overlap_sum"].shape != overlap_sum.shape:
                raise ValueError(f"relation shape changed during calibration for {name}")
            accumulator["overlap_sum"].add_(overlap_sum)
            accumulator["motion_sum"].add_(motion_sum)
            accumulator["count"] += batch_size
        del probabilities, overlap, motion, overlap_sum, motion_sum

    def _log_memory(self, label):
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(self.device) / (1024 ** 2)
            reserved = torch.cuda.memory_reserved(self.device) / (1024 ** 2)
            print(
                f">>> Memory {label}: CUDA allocated={allocated:.1f} MiB, "
                f"reserved={reserved:.1f} MiB"
            )
        else:
            print(f">>> Memory {label}: CPU execution")

    def run_calibration(self, loader, device, num_batches=10):
        """Collect static samples and online LG-FRF statistics."""
        self.register_hooks()
        self.model.eval()
        print(f">>> Calibration: collecting {num_batches} batches")
        try:
            with torch.no_grad():
                for index, batch in enumerate(
                    tqdm.tqdm(loader, desc="Calibration", total=num_batches)
                ):
                    if index >= num_batches:
                        break
                    vids = batch[0] if isinstance(batch, (list, tuple)) else batch
                    vids = vids.float().to(device, non_blocking=True)
                    if vids.ndim != 5:
                        raise ValueError(
                            "calibration videos must be [B,C,T,H,W], "
                            f"got {tuple(vids.shape)}"
                        )
                    self.current_input_shape = tuple(vids.shape)
                    self.model(vids)
                    del vids
                    if torch.cuda.is_available() and (index + 1) % 4 == 0:
                        torch.cuda.empty_cache()
        finally:
            self.remove_hooks()
        self._log_memory("after calibration")
        print(
            f">>> Calibration complete: {len(self.activations)} layers, "
            f"relation fields use T_ref={REFERENCE_TIME_BINS}"
        )

    def _robust_scale_01(self, x):
        return robust_scale_01(x)

    def compute_absolute_importance(self, Z):
        """
        计算绝对重要性
        
        D_abs = 0.5 * 时序平均激活幅度 + 0.5 * 时序激活频率
        
        Args:
            Z: [M, C] 激活矩阵，M是样本数，C是剪枝单元数
            
        Returns:
            D_abs: [C] 绝对重要性
            mean_amp: [C] 时序平均激活幅度
            freq: [C] 时序激活频率
        """
        # 时序平均激活幅度
        assert_finite("activation_samples", Z)
        mean_amp = Z.abs().mean(dim=0)
        
        # 时序激活频率（阈值判断显著激活）
        threshold = 1e-4
        freq = (Z.abs() > threshold).float().mean(dim=0)
        
        # 加权融合（权重可调整）
        mean_amp_scaled = self._robust_scale_01(mean_amp)
        freq_scaled = self._robust_scale_01(freq)
        D_abs = 0.5 * mean_amp_scaled + 0.5 * freq_scaled
        assert_finite("mean_amp_scaled", mean_amp_scaled, (0.0, 1.0))
        assert_finite("freq_scaled", freq_scaled, (0.0, 1.0))
        assert_finite("D_abs", D_abs, (0.0, 1.0))
        print(tensor_summary("mean_amp_scaled", mean_amp_scaled))
        print(tensor_summary("freq_scaled", freq_scaled))
        print(tensor_summary("D_abs", D_abs))
        return D_abs, mean_amp_scaled, freq_scaled

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
        centered = Z - Z.mean(dim=0, keepdim=True)
        scale = centered.std(dim=0, unbiased=False, keepdim=True)
        standardized = centered / scale.clamp_min(1e-5)
        standardized[:, scale.squeeze(0) <= 1e-5] = 0.0
        eps = 1e-5
        cov = (standardized.T @ standardized) / max(Z.shape[0], 1)
        cov = cov + eps * torch.eye(Z.shape[1], device=device)
        assert_finite("substitutability_covariance", cov)
        factor, info = torch.linalg.cholesky_ex(cov)
        if int(info.max()) == 0:
            inv_cov = torch.cholesky_inverse(factor)
        else:
            inv_cov = torch.linalg.pinv(cov)
        err_var = 1.0 / torch.diag(inv_cov).clamp_min(eps)
        orig_var = torch.diag(cov)
        substitutability = torch.clamp(
            1.0 - err_var / (orig_var + eps), min=0.0, max=1.0
        )
        assert_finite("substitutability", substitutability, (0.0, 1.0))
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
        assert_finite("D_rel", D_rel, (0.0, 1.0))
        return D_rel, substitutability

    def compute_cross_layer_impact(self, valid_layer_names, layer_D_abs, layer_D_rel):
        """
        计算跨层影响（反向传播累加与指数衰减）
        
        D_cross_l = D_abs_l + D_rel_l + gamma * mean(D_cross_{l+1})
        
        从输出层开始，逐层向前传播
        
        Args:
            valid_layer_names: 有效层名称列表（从输入到输出的顺序）
            layer_D_abs: 各层绝对重要性字典
            layer_D_rel: 各层相对重要性字典
            
        Returns:
            layer_D_cross: 各层跨层影响字典
        """
        layer_D_cross = {}
        gamma = self.gamma_decay
        
        # 从最后一层（输出层）开始向前传播
        prev_cross_mean = 0.0
        
        for name in reversed(valid_layer_names):
            D_abs = layer_D_abs[name]
            D_rel = layer_D_rel[name]
            
            # 当前层跨层影响 = 自身重要性 + 衰减后的下层影响均值
            D_cross = D_abs + D_rel + gamma * prev_cross_mean
            assert_finite(f"{name}.D_cross", D_cross)
            layer_D_cross[name] = D_cross
            
            # 更新用于下一层的下层影响均值
            prev_cross_mean = D_cross.mean()
            print(tensor_summary(f"{name}.D_cross", D_cross))
            
        return layer_D_cross

    # =========================================================================
    # 力场Mean-Shift聚类
    # =========================================================================
    
    def _mean_shift_clustering_impl(
        self, V_norm, function_similarity=None, return_stats=False
    ):
        """Run original or LG-FRF-guided BMS on fixed 3D coordinates."""
        groups, trajectories, sinks, stats = mean_shift_bms(
            V_norm,
            sigma=self.sigma,
            function_similarity=function_similarity,
            chunk_size=256,
        )
        print(
            ">>> descriptor kernel: "
            f"min={stats['descriptor_kernel_min']:.6f}, "
            f"max={stats['descriptor_kernel_max']:.6f}, "
            f"mean={stats['descriptor_kernel_mean']:.6f}"
        )
        print(
            ">>> joint kernel: "
            f"min={stats['joint_kernel_min']:.6f}, "
            f"max={stats['joint_kernel_max']:.6f}, "
            f"mean={stats['joint_kernel_mean']:.6f}"
        )
        print(
            ">>> kernel difference: "
            f"mean_abs={stats['mean_abs_kernel_difference']:.6g}, "
            f"max_abs={stats['max_abs_kernel_difference']:.6g}, "
            f"relative={stats['relative_kernel_difference']:.6g}, "
            f"joint<desc={stats['ratio_joint_lower_than_descriptor']:.6f}, "
            f"joint<0.5*desc="
            f"{stats['ratio_joint_less_than_half_descriptor']:.6f}"
        )
        print(
            f">>> BMS clusters={len(groups)}, iterations={stats['iterations']}, "
            f"max_movement={stats['max_movement']:.6g}"
        )
        result = (groups, trajectories, sinks)
        return result + (stats,) if return_stats else result

    def mean_shift_clustering(self, V_norm):
        """Run descriptor BMS without any functional clustering input."""
        return self._mean_shift_clustering_impl(V_norm, None)

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


    def _build_unit_metadata(self, layer_name, num_units):
        """Build stable metadata shared by descriptors and relation fields."""
        if layer_name not in self.unit_metadata:
            raise KeyError(f"missing unit metadata for {layer_name}")
        metadata = self.unit_metadata[layer_name]
        if metadata["num_units"] != num_units:
            raise ValueError(
                f"unit count mismatch for {layer_name}: "
                f"{metadata['num_units']} != {num_units}"
            )
        response_geometry = None
        if layer_name in self.frame_relation_accumulators:
            response_geometry = self.frame_relation_accumulators[layer_name].get(
                "response_geometry"
            )
        return [
            {
                "unit_id": (
                    f"{layer_name}::{metadata['unit_type']}::{local_index}"
                ),
                "layer": layer_name,
                "idx": local_index,
                "unit_type": metadata["unit_type"],
                "stage": metadata["stage"],
                "block": metadata["block"],
                "response_geometry": response_geometry,
            }
            for local_index in range(num_units)
        ]

    def build_3d_descriptors(self, valid_layer_names):
        """
        构建所有剪枝单元的三维描述符
        
        Returns:
            V: [N, 3] 三维描述符矩阵（绝对、相对、跨层影响）
            unit_info: 剪枝单元信息列表
        """
        print("\n>>> 步骤1: 构建三维描述符 (绝对重要性 / 相对重要性 / 跨层影响)...")
        
        layer_D_abs = {}
        layer_D_rel = {}
        
        # 1. 计算绝对重要性和相对重要性
        for name in valid_layer_names:
            # 合并所有批次的激活 [M, C]
            Z = self.activations[name].to(self.device).float()
            
            # 绝对重要性
            D_abs, mean_amp, freq = self.compute_absolute_importance(Z)
            layer_D_abs[name] = D_abs
            
            # 相对重要性（含可替代性计算）
            D_rel, subst = self.compute_relative_importance(D_abs, Z)
            layer_D_rel[name] = D_rel
            
        # 2. 计算跨层影响（反向传播）
        layer_D_cross = self.compute_cross_layer_impact(
            valid_layer_names, layer_D_abs, layer_D_rel
        )
        
        # 3. 聚合所有剪枝单元的三维描述符
        all_descriptors = []
        unit_info = []
        
        for name in valid_layer_names:
            D_a = layer_D_abs[name]  # [C]
            D_r = layer_D_rel[name]  # [C]
            D_c = layer_D_cross[name]  # [C]
            
            layer_units = self._build_unit_metadata(name, len(D_a))
            for i, metadata in enumerate(layer_units):
                all_descriptors.append([D_a[i].item(), D_r[i].item(), D_c[i].item()])
                unit_info.append(metadata)
        
        V = torch.tensor(all_descriptors, device=self.device)
        
        if V.ndim != 2 or V.shape[1] != 3 or len(unit_info) != V.shape[0]:
            raise AssertionError("descriptor/unit metadata alignment failed")
        assert_finite("formal_descriptors", V)
        print(tensor_summary("formal_descriptors", V))
        
        return V, unit_info, layer_D_abs, layer_D_rel, layer_D_cross

    def _safe_standardize_descriptors(self, descriptors):
        return safe_standardize_descriptors(descriptors, logger=print)

    @staticmethod
    def _print_distribution(name, summary):
        ordered = (
            "min", "q01", "q05", "q25", "median", "q75", "q95", "q99",
            "max", "mean", "std", "diagonal_mean", "off_diagonal_mean",
            "off_diagonal_std",
        )
        values = ", ".join(
            f"{key}={summary[key]:.6g}" for key in ordered if key in summary
        )
        print(f">>> {name}: shape={tuple(summary['shape'])}, {values}")

    @staticmethod
    def _print_centered_feature_summary(name, summary):
        print(
            f">>> {name}: shape={tuple(summary['shape'])}, "
            f"mean_abs={summary['mean_absolute_value']:.6g}, "
            f"std={summary['std']:.6g}, "
            f"zero_norm_units={summary['zero_norm_unit_count']}, "
            f"finite_ratio={summary['finite_ratio']:.6f}"
        )

    def _finalize_frame_relation_fields(self, unit_info):
        """Finalize online sums as O/R [N,8,8] in descriptor unit order."""
        overlap_fields = []
        motion_fields = []
        seen_layers = []
        for info in unit_info:
            if info["layer"] not in seen_layers:
                seen_layers.append(info["layer"])
        for layer_name in seen_layers:
            if layer_name not in self.frame_relation_accumulators:
                raise RuntimeError(
                    f"functional relation field is unavailable for {layer_name}"
                )
            accumulator = self.frame_relation_accumulators[layer_name]
            count = accumulator["count"]
            if count <= 0:
                raise RuntimeError(f"relation count is zero for {layer_name}")
            overlap = (accumulator["overlap_sum"] / count).float()
            motion = (accumulator["motion_sum"] / count).float()
            expected_units = self.unit_metadata[layer_name]["num_units"]
            expected_shape = (
                expected_units,
                REFERENCE_TIME_BINS,
                REFERENCE_TIME_BINS,
            )
            if tuple(overlap.shape) != expected_shape or tuple(motion.shape) != expected_shape:
                raise AssertionError(
                    f"{layer_name} relation shape mismatch: "
                    f"O={tuple(overlap.shape)}, R={tuple(motion.shape)}, "
                    f"expected={expected_shape}"
                )
            assert_finite(f"{layer_name}.overlap_relation", overlap, (0.0, 1.0))
            assert_finite(f"{layer_name}.motion_retention", motion, (0.0, 1.0))
            print(
                f">>> {layer_name}: relation source T={accumulator['source_time']} "
                f"-> T_ref={REFERENCE_TIME_BINS}, shape={expected_shape}"
            )
            overlap_fields.append(overlap)
            motion_fields.append(motion)
        overlap_relation = torch.cat(overlap_fields, dim=0)
        motion_relation = torch.cat(motion_fields, dim=0)
        if overlap_relation.shape[0] != len(unit_info):
            raise AssertionError("relation fields and unit_info are not aligned")
        self._print_distribution(
            "overlap_relation",
            summarize_matrix_distribution(overlap_relation),
        )
        self._print_distribution(
            "motion_retention_relation",
            summarize_matrix_distribution(motion_relation),
        )
        return overlap_relation, motion_relation

    @staticmethod
    def _build_local_global_masks(time_bins=REFERENCE_TIME_BINS):
        return build_local_global_masks(time_bins)

    def _compute_function_similarity(self, overlap_relation, motion_relation):
        local_mask, global_mask = self._build_local_global_masks(
            overlap_relation.shape[-1]
        )
        storage_dtype = (
            torch.float16 if overlap_relation.shape[0] > 4096 else torch.float32
        )
        s_local, s_global, s_function, details = compute_function_similarity(
            overlap_relation,
            motion_relation,
            local_mask,
            global_mask,
            chunk_size=256,
            storage_dtype=storage_dtype,
            return_details=True,
        )
        self._print_distribution(
            "displacement_relation",
            summarize_matrix_distribution(details["displacement_relation"]),
        )
        self._print_distribution(
            "displacement_rate_relation",
            summarize_matrix_distribution(details["displacement_rate_relation"]),
        )
        for name, summary in details["feature_summaries"].items():
            self._print_centered_feature_summary(name, summary)
        for name, summary in details["similarity_summaries"].items():
            self._print_distribution(name, summary)
        return (
            local_mask,
            global_mask,
            s_local,
            s_global,
            s_function,
            details,
        )

    @staticmethod
    def _metadata_redundancy_summary(functional_redundancy, unit_info):
        values = functional_redundancy.detach().float().cpu()
        by_stage = defaultdict(list)
        by_type = defaultdict(list)
        for index, info in enumerate(unit_info):
            value = float(values[index])
            by_stage[str(info.get("stage", -1))].append(value)
            by_type[str(info.get("unit_type", "unknown"))].append(value)

        def summarize(groups):
            return {
                key: {
                    "count": len(group_values),
                    "mean": float(np.mean(group_values)),
                    "min": float(np.min(group_values)),
                    "max": float(np.max(group_values)),
                }
                for key, group_values in sorted(groups.items())
            }

        return {
            "stage_wise_redundancy": summarize(by_stage),
            "unit_type_redundancy": summarize(by_type),
        }

    def analyze_functional_redundancy(self, output_dir=None):
        """Find and protect one functional representative per descriptor cluster."""
        valid_layer_names = [
            name for name in self.ordered_layer_names if name in self.activations
        ]
        if not valid_layer_names:
            raise RuntimeError("calibration produced no prunable layer statistics")
        descriptors, unit_info, d_abs, d_rel, d_cross = self.build_3d_descriptors(
            valid_layer_names
        )
        descriptors_normalized = self._safe_standardize_descriptors(descriptors)
        relation_available = all(
            name in self.frame_relation_accumulators for name in valid_layer_names
        )
        if not relation_available:
            raise RuntimeError(
                "LG-FRF relation fields are required for functional redundancy"
            )

        analysis_start = time.perf_counter()
        overlap, motion = self._finalize_frame_relation_fields(unit_info)
        local_mask, global_mask, s_local, s_global, s_function, details = (
            self._compute_function_similarity(overlap, motion)
        )
        print("\n>>> Running descriptor BMS (the only clustering path)")
        descriptor = self._mean_shift_clustering_impl(
            descriptors_normalized,
            function_similarity=None,
            return_stats=True,
        )
        descriptor_groups = descriptor[0]
        descriptor_importance = descriptors.float().mean(dim=1)
        functional_redundancy = compute_intra_cluster_functional_redundancy(
            descriptor_groups,
            s_function,
        )
        functional_uniqueness = compute_function_uniqueness(
            functional_redundancy
        )
        representatives = select_descriptor_cluster_representatives(
            descriptor_groups,
            descriptor_importance,
            functional_uniqueness,
        )
        protected_units = set(representatives)
        for unit_index, info in enumerate(unit_info):
            info["protected"] = unit_index in protected_units
        redundancy_summary = summarize_functional_redundancy(
            descriptor_groups,
            functional_redundancy,
            representatives,
        )
        redundancy_summary.update(
            self._metadata_redundancy_summary(
                functional_redundancy,
                unit_info,
            )
        )
        redundancy_summary["protected_unit_ids"] = [
            unit_info[index]["unit_id"] for index in representatives
        ]
        representative_index = torch.as_tensor(
            representatives,
            dtype=torch.long,
        )
        redundancy_summary.update(
            {
                "protected_heads": sum(
                    unit_info[index]["unit_type"] == "head"
                    for index in representatives
                ),
                "protected_neurons": sum(
                    unit_info[index]["unit_type"] == "neuron"
                    for index in representatives
                ),
                "protected_ratio": len(representatives) / len(unit_info),
                "average_representative_importance": float(
                    descriptor_importance.detach().cpu()[
                        representative_index
                    ].mean()
                ),
                "average_representative_uniqueness": float(
                    functional_uniqueness.detach().cpu()[
                        representative_index
                    ].mean()
                ),
            }
        )

        print(f">>> Cluster Count: {redundancy_summary['cluster_count']}")
        print(
            ">>> Average Cluster Size: "
            f"{redundancy_summary['average_cluster_size']:.6f}"
        )
        print(
            ">>> Average Functional Redundancy: "
            f"{redundancy_summary['average_functional_redundancy']:.6f}"
        )
        print(
            f">>> Representative Count: "
            f"{redundancy_summary['representative_count']}"
        )
        print(
            f">>> Protected Units: {len(representatives)}"
        )
        print(
            f">>> Protected Heads: {redundancy_summary['protected_heads']}"
        )
        print(
            f">>> Protected Neurons: {redundancy_summary['protected_neurons']}"
        )
        print(
            f">>> Protected Ratio: {redundancy_summary['protected_ratio']:.6f}"
        )
        print(
            ">>> Avg Representative Importance: "
            f"{redundancy_summary['average_representative_importance']:.6f}"
        )
        print(
            ">>> Avg Representative Uniqueness: "
            f"{redundancy_summary['average_representative_uniqueness']:.6f}"
        )
        print(
            ">>> Avg Cluster Size: "
            f"{redundancy_summary['average_cluster_size']:.6f}"
        )
        print(
            ">>> Average Redundancy: "
            f"{redundancy_summary['average_functional_redundancy']:.6f}"
        )
        print(
            ">>> Stage-wise Redundancy: "
            + json.dumps(
                redundancy_summary["stage_wise_redundancy"],
                ensure_ascii=False,
            )
        )
        type_summary = redundancy_summary["unit_type_redundancy"]
        print(
            ">>> Head Redundancy: "
            f"{type_summary.get('head', {}).get('mean', 0.0):.6f}"
        )
        print(
            ">>> Neuron Redundancy: "
            f"{type_summary.get('neuron', {}).get('mean', 0.0):.6f}"
        )

        analysis = {
            "valid_layer_names": valid_layer_names,
            "descriptors_raw": descriptors,
            "descriptors_normalized": descriptors_normalized,
            "descriptor_importance": descriptor_importance,
            "unit_info": unit_info,
            "D_abs": d_abs,
            "D_rel": d_rel,
            "D_cross": d_cross,
            "overlap_relation": overlap,
            "motion_retention_relation": motion,
            "displacement_relation": details["displacement_relation"],
            "displacement_rate_relation": details["displacement_rate_relation"],
            "local_mask": local_mask,
            "global_mask": global_mask,
            "S_loc": s_local,
            "S_glo": s_global,
            "S_func": s_function,
            "function_diagnostics": details,
            "descriptor": descriptor,
            "active_clustering": descriptor,
            "functional_redundancy": functional_redundancy,
            "functional_uniqueness": functional_uniqueness,
            "cluster_representatives": representatives,
            "protected_unit_indices": protected_units,
            "redundancy_summary": redundancy_summary,
            "original": descriptor,
        }
        if output_dir is not None:
            analysis["analysis_output_dir"] = os.fspath(output_dir)
        self.latest_analysis = analysis
        if output_dir is not None:
            self.export_functional_redundancy_analysis(output_dir, analysis)
        self._log_memory("after functional-redundancy analysis")
        print(
            ">>> Descriptor clustering + LG-FRF redundancy elapsed="
            f"{time.perf_counter() - analysis_start:.3f}s"
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return analysis

    def export_functional_redundancy_analysis(self, output_dir, analysis=None):
        """Export descriptor clusters and intra-cluster redundancy artifacts."""
        if analysis is None:
            analysis = self.latest_analysis
        required = {
            "descriptor",
            "descriptor_importance",
            "functional_redundancy",
            "functional_uniqueness",
            "cluster_representatives",
        }
        if analysis is None or not required.issubset(analysis):
            raise RuntimeError("functional redundancy analysis is unavailable")
        os.makedirs(output_dir, exist_ok=True)
        unit_info = analysis["unit_info"]
        unit_ids = np.asarray([info["unit_id"] for info in unit_info])
        serialized_info = np.asarray(
            [json.dumps(info, ensure_ascii=False) for info in unit_info]
        )

        np.savez_compressed(
            os.path.join(output_dir, "formal_descriptors.npz"),
            unit_ids=unit_ids,
            descriptors_raw=analysis["descriptors_raw"].detach().cpu().numpy(),
            descriptors_normalized=(
                analysis["descriptors_normalized"].detach().cpu().numpy()
            ),
            unit_info=serialized_info,
        )
        np.savez_compressed(
            os.path.join(output_dir, "frame_relation_fields.npz"),
            unit_ids=unit_ids,
            overlap_relation=analysis["overlap_relation"].cpu().numpy(),
            motion_retention_relation=(
                analysis["motion_retention_relation"].cpu().numpy()
            ),
            displacement_relation=(
                analysis["displacement_relation"].cpu().numpy()
            ),
            displacement_rate_relation=(
                analysis["displacement_rate_relation"].cpu().numpy()
            ),
            local_mask=analysis["local_mask"].cpu().numpy(),
            global_mask=analysis["global_mask"].cpu().numpy(),
        )
        np.savez_compressed(
            os.path.join(output_dir, "function_similarity.npz"),
            unit_ids=unit_ids,
            S_loc=analysis["S_loc"].cpu().numpy(),
            S_glo=analysis["S_glo"].cpu().numpy(),
            S_func=analysis["S_func"].cpu().numpy(),
        )

        descriptor_groups, descriptor_trajectories, descriptor_sinks = (
            analysis["descriptor"][:3]
        )
        descriptor_labels = groups_to_labels(descriptor_groups, len(unit_info))
        representatives = [int(index) for index in analysis["cluster_representatives"]]
        protected_mask = np.zeros(len(unit_info), dtype=np.bool_)
        protected_mask[representatives] = True
        np.savez_compressed(
            os.path.join(output_dir, "descriptor_clusters.npz"),
            unit_ids=unit_ids,
            descriptor_bms_labels=descriptor_labels.numpy(),
            descriptor_trajectories=(
                descriptor_trajectories.detach().cpu().numpy()
            ),
            descriptor_sinks=descriptor_sinks.detach().cpu().numpy(),
            representative_indices=np.asarray(representatives, dtype=np.int64),
            protected_mask=protected_mask,
        )
        np.savez_compressed(
            os.path.join(output_dir, "functional_redundancy.npz"),
            unit_ids=unit_ids,
            descriptor_bms_labels=descriptor_labels.numpy(),
            descriptor_importance=(
                analysis["descriptor_importance"].detach().cpu().numpy()
            ),
            functional_redundancy=(
                analysis["functional_redundancy"].detach().cpu().numpy()
            ),
            functional_uniqueness=(
                analysis["functional_uniqueness"].detach().cpu().numpy()
            ),
            representative_indices=np.asarray(representatives, dtype=np.int64),
            protected_mask=protected_mask,
        )
        np.savez_compressed(
            os.path.join(output_dir, "function_uniqueness.npz"),
            unit_ids=unit_ids,
            descriptor_bms_labels=descriptor_labels.numpy(),
            functional_uniqueness=(
                analysis["functional_uniqueness"].detach().cpu().numpy()
            ),
            representative_indices=np.asarray(representatives, dtype=np.int64),
            protected_mask=protected_mask,
        )

        importance = analysis["descriptor_importance"].detach().float().cpu()
        redundancy = analysis["functional_redundancy"].detach().float().cpu()
        uniqueness = analysis["functional_uniqueness"].detach().float().cpu()
        cluster_by_unit = descriptor_labels.tolist()
        pruning_registry = analysis.get("pruning_registry")
        model_modules = dict(self.model.named_modules())

        def unit_kept(info, protected):
            if pruning_registry is None:
                return True if protected else ""
            return info["idx"] not in pruning_registry.get(info["layer"], set())

        def unit_parameter_count(info):
            module = model_modules.get(info["layer"])
            if module is None:
                return 0
            return int(self.estimate_unit_cost(module, info["unit_type"]))

        unit_rows = []
        for unit_index, info in enumerate(unit_info):
            protected = bool(protected_mask[unit_index])
            unit_rows.append(
                {
                    "unit_index": unit_index,
                    "unit_id": info["unit_id"],
                    "descriptor_importance": float(importance[unit_index]),
                    "function_redundancy": float(redundancy[unit_index]),
                    "function_uniqueness": float(uniqueness[unit_index]),
                    "protected": protected,
                    "cluster_id": int(cluster_by_unit[unit_index]),
                    "stage": info.get("stage", -1),
                    "block": info.get("block", -1),
                    "unit_type": info.get("unit_type", ""),
                    "layer": info.get("layer", ""),
                    "parameter_count": unit_parameter_count(info),
                    "kept": unit_kept(info, protected),
                }
            )
        with open(
            os.path.join(output_dir, "unit_redundancy.csv"),
            "w",
            newline="",
            encoding="utf-8",
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(unit_rows[0]))
            writer.writeheader()
            writer.writerows(unit_rows)

        representative_rows = []
        protected_rows = []
        for cluster_id, representative in enumerate(representatives):
            info = unit_info[representative]
            parameter_count = unit_parameter_count(info)
            representative_rows.append(
                {
                    "cluster_id": cluster_id,
                    "representative_id": info["unit_id"],
                    "importance": float(importance[representative]),
                    "redundancy": float(redundancy[representative]),
                    "uniqueness": float(uniqueness[representative]),
                    "stage": info.get("stage", -1),
                    "block": info.get("block", -1),
                    "unit_type": info.get("unit_type", ""),
                    "parameter_count": parameter_count,
                }
            )
            protected_rows.append(
                {
                    "unit_id": info["unit_id"],
                    "cluster_id": cluster_id,
                    "importance": float(importance[representative]),
                    "uniqueness": float(uniqueness[representative]),
                    "stage": info.get("stage", -1),
                    "block": info.get("block", -1),
                    "type": info.get("unit_type", ""),
                    "kept": unit_kept(info, True),
                }
            )
        with open(
            os.path.join(output_dir, "representative.csv"),
            "w",
            newline="",
            encoding="utf-8",
        ) as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=list(representative_rows[0]),
            )
            writer.writeheader()
            writer.writerows(representative_rows)
        with open(
            os.path.join(output_dir, "protected_units.csv"),
            "w",
            newline="",
            encoding="utf-8",
        ) as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=list(protected_rows[0]),
            )
            writer.writeheader()
            writer.writerows(protected_rows)

        cluster_rows = []
        for cluster_id, members in enumerate(descriptor_groups):
            member_values = redundancy[
                torch.as_tensor(members, dtype=torch.long)
            ]
            representative = representatives[cluster_id]
            infos = [unit_info[index] for index in members]
            cluster_rows.append(
                {
                    "cluster_id": cluster_id,
                    "size": len(members),
                    "mean_redundancy": float(member_values.mean()),
                    "max_redundancy": float(member_values.max()),
                    "min_redundancy": float(member_values.min()),
                    "mean_uniqueness": float(
                        uniqueness[torch.as_tensor(members, dtype=torch.long)].mean()
                    ),
                    "representative_id": unit_info[representative]["unit_id"],
                    "protected_unit": unit_info[representative]["unit_id"],
                    "stage": ";".join(
                        sorted({str(info.get("stage", -1)) for info in infos})
                    ),
                    "block": ";".join(
                        sorted({str(info.get("block", -1)) for info in infos})
                    ),
                    "unit_type": ";".join(
                        sorted({str(info.get("unit_type", "")) for info in infos})
                    ),
                    "head_count": sum(
                        info.get("unit_type") == "head" for info in infos
                    ),
                    "neuron_count": sum(
                        info.get("unit_type") == "neuron" for info in infos
                    ),
                    "layers": ";".join(
                        sorted({str(info.get("layer", "")) for info in infos})
                    ),
                }
            )
        with open(
            os.path.join(output_dir, "cluster_redundancy.csv"),
            "w",
            newline="",
            encoding="utf-8",
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(cluster_rows[0]))
            writer.writeheader()
            writer.writerows(cluster_rows)

        with open(
            os.path.join(output_dir, "functional_redundancy_summary.json"),
            "w",
            encoding="utf-8",
        ) as handle:
            json.dump(
                analysis["redundancy_summary"],
                handle,
                ensure_ascii=False,
                indent=2,
            )
        print(f">>> Functional redundancy analysis exported to {output_dir}")

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

    def group_pruning(
        self,
        V,
        unit_info,
        valid_layer_names,
        step_target_sparsity,
        clustering_result=None,
        descriptor_importance=None,
        protected_units=None,
    ):
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

        if clustering_result is None:
            raise RuntimeError(
                "group_pruning requires prepared active_clustering; "
                "BMS must not be rerun during pruning"
            )
        if (
            descriptor_importance is None
            or protected_units is None
        ):
            raise RuntimeError(
                "group_pruning requires prepared descriptor importance and "
                "protected representatives"
            )
        groups, Trajectories, P_final = clustering_result[:3]
        protected_units = {int(index) for index in protected_units}
        descriptor_importance = torch.as_tensor(
            descriptor_importance,
            dtype=torch.float64,
        ).detach().cpu()
        if descriptor_importance.shape != (len(unit_info),):
            raise ValueError("descriptor_importance must have shape [N]")
        for unit_index, info in enumerate(unit_info):
            info["protected"] = unit_index in protected_units
        
        print("\n>>> 步骤3: 计算流形一致性并生成剪枝决策...")
        
        # 3. 流形一致性评分
        group_scores = []
        model_modules = dict(self.model.named_modules())

        for g_idx, g_members in enumerate(groups):
            protected_in_group = protected_units.intersection(g_members)
            if len(protected_in_group) != 1:
                raise AssertionError(
                    "every descriptor cluster must have exactly one "
                    "protected representative"
                )
            candidate_members = []
            for index in g_members:
                protected = bool(unit_info[index].get("protected", False))
                if protected:
                    continue
                candidate_members.append(index)
            ordered_candidates = sorted(
                candidate_members,
                key=lambda index: (float(descriptor_importance[index]), index),
            )
            score, dyn_consist, static_strength = self.compute_group_manifold_score(
                g_members, V, Trajectories
            )
            
            # 计算组参数代价
            cost = 0
            for idx in ordered_candidates:
                m_info = unit_info[idx]
                m = model_modules[m_info['layer']]
                
                cost += self.estimate_unit_cost(m, m_info["unit_type"])
            
            group_scores.append({
                'group_id': g_idx,
                'members': ordered_candidates,
                'all_members': g_members,
                'representative': next(iter(protected_in_group)),
                'score': score,
                'dyn_consist': dyn_consist,
                'static_strength': static_strength,
                'cost': cost
            })
        print(
            f">>> Protected Units: {len(protected_units)} | "
            f"Global Candidate Count: "
            f"{sum(len(group['members']) for group in group_scores)}"
        )
        
        # 4. 构建层信息（限制每层最大剪枝比例）
        layer_info = {}
        for name in valid_layer_names:
            m = model_modules[name]
            
            if self.unit_metadata[name]["unit_type"] == "head":
                if not hasattr(m, 'num_heads'):
                    m.num_heads = m.qkv.out_features // (3 * m.head_dim)
                total_u = m.num_heads
            else:
                if not hasattr(m, 'original_hidden_features'):
                    m.original_hidden_features = m.fc1.out_features
                total_u = m.original_hidden_features
            
            max_prune = total_u - max(1, int(total_u * self.min_keep_ratio))
            layer_info[name] = {'total': total_u, 'pruned': 0, 'max_prunable': max_prune}
        
        # 5. 全局组级剪枝
        total_p = self.total_original_params
        target_red = total_p * step_target_sparsity
        current_red = 0
        registry = {name: set() for name in valid_layer_names}
        
        # 按流形一致性得分排序（低到高）
        group_scores.sort(key=lambda x: x['score'])

        for grp in group_scores:
            if not grp["members"]:
                continue
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
                
                # Preserve the original descriptor-only boundary-group order.
                for idx in grp['members']:
                    info = unit_info[idx]
                    cost = self.estimate_unit_cost(
                        model_modules[info["layer"]],
                        info["unit_type"],
                    )
                    
                    # 再次检查层约束
                    if layer_info[info['layer']]['pruned'] + 1 <= layer_info[info['layer']]['max_prunable']:
                        registry[info['layer']].add(info['idx'])
                        layer_info[info['layer']]['pruned'] += 1
                        current_red += cost
                        
                    if current_red >= target_red: break
                break # 达到目标，退出组循环

        for unit_index in protected_units:
            info = unit_info[unit_index]
            if info["idx"] in registry[info["layer"]]:
                raise AssertionError("a protected representative was pruned")
        actually_pruned = sum(len(indices) for indices in registry.values())
        print(f">>> Protected Units: {len(protected_units)}")
        print(f">>> Actually Pruned: {actually_pruned}")
        print(">>> Representative Survival: 100%")
        self.latest_pruning_registry = registry
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
            
            if self.unit_metadata[name]["unit_type"] == "head":
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
                          
            elif self.unit_metadata[name]["unit_type"] == "neuron":
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

    def prune(self, step_target_sparsity=None, analysis_output_dir=None,
              prepared_analysis=None):
        """Prune by the original budget while masking LGFR representatives."""
        if step_target_sparsity is None:
            step_target_sparsity = self.target_sparsity
        analysis = prepared_analysis
        if analysis is None:
            raise RuntimeError(
                "prune requires prepared_analysis; descriptor clustering and "
                "redundancy analysis must finish before pruning"
            )
        clustering_result = analysis.get("active_clustering")
        if clustering_result is None:
            raise RuntimeError("prepared active_clustering is unavailable")
        if clustering_result is not analysis.get("descriptor"):
            raise RuntimeError(
                "active_clustering must be the prepared descriptor BMS result"
            )
        mode = "descriptor BMS + LGFR representative protection"
        print(f"\n>>> Applying pruning groups from {mode}")
        registry, actual_sparsity = self.group_pruning(
            analysis["descriptors_raw"],
            analysis["unit_info"],
            analysis["valid_layer_names"],
            step_target_sparsity,
            clustering_result=clustering_result,
            descriptor_importance=analysis["descriptor_importance"],
            protected_units=analysis["protected_unit_indices"],
        )
        analysis["pruning_registry"] = registry
        self.apply_pruning_masks(registry, analysis["valid_layer_names"])
        self.current_sparsity += actual_sparsity
        self.pruning_history.append(
            {
                "sparsity": self.current_sparsity,
                "step_sparsity": actual_sparsity,
                "mode": mode,
            }
        )
        self._log_memory("after pruning masks")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        export_dir = analysis_output_dir or analysis.get("analysis_output_dir")
        if export_dir is not None:
            self.export_functional_redundancy_analysis(export_dir, analysis)
        print(f">>> Pruning complete: cumulative sparsity={self.current_sparsity:.2%}")
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
        
        # Window Attention
        self.attn._pruning_geometry = {
            "batch_size": B,
            "depth": D,
            "height": H,
            "width": W,
            "padded_depth": Dp,
            "padded_height": Hp,
            "padded_width": Wp,
            "window_size": tuple(window_size),
            "shift_size": tuple(shift_size),
        }
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
        if self.use_checkpoint:
            x = x + self.drop_path(checkpoint.checkpoint(self.forward_part1, x, mask_matrix))
        else:
            x = x + self.drop_path(self.forward_part1(x, mask_matrix))

        # 残差连接 + MLP
        if self.use_checkpoint:
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
        self.cls_head = I3DHead(num_classes, self.num_features)

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
        gamma_decay=0.85,     # 跨层衰减因子
        min_keep_ratio=0.1,   # 每层最少保留10%
        sigma=0.5             # Mean-Shift带宽
    )
    
    print("\n剪枝器配置:")
    print(f"  目标稀疏度: {pruner.target_sparsity:.2%}")
    print(f"  迭代剪枝步数: {pruner.iter_prune_steps}")
    print(f"  跨层衰减因子: {pruner.gamma_decay}")
    
    print("\n>>> 使用说明:")
    print("1. 准备校准数据加载器 calib_loader (用于计算重要性)")
    print("2. 准备训练数据加载器 train_loader ")
    print("3. 调用: pruner.iterative_prune(calib_loader, train_loader, device)")
    print("4. 查看剪枝报告: pruner.get_pruning_report()")
