# 剪枝技巧:各层conv3部分狠狠的剪枝 点数就上去了
"""
SlowFast Networks for Video Recognition
ICCV 2019, https://arxiv.org/abs/1812.03982
Code adapted from https://github.com/r1ch88/SlowFastNetworks

Modified: 使用3D描述符 + Mean-Shift聚类的剪枝方法 (MC_v1.0.2)
          + BN层同步 + 残差连接保护 + 分路径校准
"""
import torch
import torch.nn as nn
from torch.nn import BatchNorm3d
import math
import tqdm
import torch.nn.functional as F


class InteractionPruner:
    """
    使用3D描述符 + Mean-Shift聚类 + 流形一致性的结构化剪枝器
    适配SlowFast网络的Conv3d层
    
    核心改进:
    1. BN层同步: 被剪通道对应的BN weight/bias置零，梯度冻结
    2. 残差连接保护: conv3与downsample的mask严格一致
    3. 分路径校准: Fast/Slow路径的D_cross独立计算
    """
    def __init__(self, model, target_sparsity=0.5, gamma_decay=0.5, 
                 min_keep_ratio=0.1, sigma=0.1):
            # 在 __init__ 中添加
            self.trajectories = {} 
            self.model = model
            self.target_sparsity = target_sparsity
            self.gamma_decay = gamma_decay
            self.min_keep_ratio = min_keep_ratio
            self.sigma = sigma
            self.activations = {}
            self.hooks = []
            self.bn_hooks = []      # BN梯度hook handles
            self.device = next(model.parameters()).device
            self.ordered_layer_names = []
            self.fast_layer_names = []   # Fast Path 层名
            self.slow_layer_names = []   # Slow Path 层名
            self.lateral_names = []      # Lateral Connection 层名
            self.next_layer_map = {}     # 同路径下一层映射
            self.fast_to_lateral_map = {} # Fast输出 -> Lateral输入映射
            self.lateral_to_slow_map = {} # Lateral输出 -> Slow输入映射
            self.layer_avg_scores = {}
            self.total_original_params = sum(p.numel() for p in model.parameters())

    # ==================== Hook & Calibration ====================

    def register_hooks(self):
        """注册钩子以收集Conv3d层的激活值"""
        self.hooks = []
        self.activations = {}
        self.ordered_layer_names = []
        self.fast_layer_names = []
        self.slow_layer_names = []
        self.lateral_names = []
        self.next_layer_map = {}
        self.fast_to_lateral_map = {}
        self.lateral_to_slow_map = {}
        
        module_dict = dict(self.model.named_modules())
        
        # 手动定义 Fast Path 层级顺序 (确保与实际数据流一致)
        for stage_idx in range(2, 6):
            stage_name = f'fast_res{stage_idx}'
            stage = module_dict.get(stage_name)
            if isinstance(stage, nn.Sequential):
                for block_idx in range(len(stage)):
                    for conv_idx in [1, 2, 3]:
                        conv_name = f'{stage_name}.{block_idx}.conv{conv_idx}'
                        if conv_name in module_dict and isinstance(module_dict[conv_name], nn.Conv3d):
                            m = module_dict[conv_name]
                            h = m.register_forward_hook(self._get_activation_hook(conv_name))
                            self.hooks.append(h)
                            self.ordered_layer_names.append(conv_name)
                            self.fast_layer_names.append(conv_name)
        
        # 手动定义 Lateral Connection 层级顺序
        for lat_name in ['lateral_p1.0', 'lateral_res2.0', 'lateral_res3.0', 'lateral_res4.0']:
            if lat_name in module_dict and isinstance(module_dict[lat_name], nn.Conv3d):
                m = module_dict[lat_name]
                h = m.register_forward_hook(self._get_activation_hook(lat_name))
                self.hooks.append(h)
                self.ordered_layer_names.append(lat_name)
                self.lateral_names.append(lat_name)
        
        # 手动定义 Slow Path 层级顺序
        for stage_idx in range(2, 6):
            stage_name = f'slow_res{stage_idx}'
            stage = module_dict.get(stage_name)
            if isinstance(stage, nn.Sequential):
                for block_idx in range(len(stage)):
                    for conv_idx in [1, 2, 3]:
                        conv_name = f'{stage_name}.{block_idx}.conv{conv_idx}'
                        if conv_name in module_dict and isinstance(module_dict[conv_name], nn.Conv3d):
                            m = module_dict[conv_name]
                            h = m.register_forward_hook(self._get_activation_hook(conv_name))
                            self.hooks.append(h)
                            self.ordered_layer_names.append(conv_name)
                            self.slow_layer_names.append(conv_name)
        
        # 构建同路径下一层映射 (分路径校准)
        # 注意: Lateral layers之间没有直接数据流，不加入next_layer_map
        for path_layers in [self.fast_layer_names, self.slow_layer_names]:
            for i in range(len(path_layers) - 1):
                self.next_layer_map[path_layers[i]] = path_layers[i + 1]
        
        # 构建侧向连接依赖映射
        self._build_lateral_maps(module_dict)

    def _build_lateral_maps(self, module_dict):
        """建立Fast路径输出 -> Lateral输入 -> Slow路径输入的依赖映射"""
        # lateral_p1: fast_conv1(pool) -> slow_res2
        if 'lateral_p1.0' in module_dict:
            if 'fast_conv1' in module_dict:
                self.fast_to_lateral_map['fast_conv1'] = 'lateral_p1.0'
            if 'slow_res2.0.conv1' in module_dict:
                self.lateral_to_slow_map['lateral_p1.0'] = 'slow_res2.0.conv1'
        
        # lateral_res2 -> slow_res3, lateral_res3 -> slow_res4, lateral_res4 -> slow_res5
        for stage_idx in range(2, 5):
            lateral_name = f'lateral_res{stage_idx}.0'
            fast_stage_name = f'fast_res{stage_idx}'
            slow_stage_name = f'slow_res{stage_idx + 1}'
            
            fast_stage = module_dict.get(fast_stage_name)
            if isinstance(fast_stage, nn.Sequential) and len(fast_stage) > 0:
                last_block_idx = len(fast_stage) - 1
                fast_conv3_name = f'{fast_stage_name}.{last_block_idx}.conv3'
                if fast_conv3_name in module_dict:
                    self.fast_to_lateral_map[fast_conv3_name] = lateral_name
            
            slow_conv1_name = f'{slow_stage_name}.0.conv1'
            if slow_conv1_name in module_dict:
                self.lateral_to_slow_map[lateral_name] = slow_conv1_name

    def _get_activation_hook(self, name):
        """收集激活值 Z [B, T, C] 和 轨迹 Trajectories [B, T-1, C]"""
        def hook(module, input, output):
            z = output.detach()  # [B, C, T, H, W]
            
            # 1. 空间平均池化 [B, C, T]
            z_pool = torch.mean(z, dim=(3, 4))
            
            # 2. 时间轴对齐
            if z_pool.shape[2] > 8:
                z_pool = F.adaptive_avg_pool1d(z_pool, 8)
            
            # 3. 转换为 [B, T, C]
            z_final = z_pool.permute(0, 2, 1) 
            
            # 4. 限制 Batch 大小
            if z_final.shape[0] > 64:
                z_final = z_final[:64]
                
            # 5. 计算轨迹 (Temporal Difference)
            # Z_diff [B, T-1, C]
            if z_final.shape[1] > 1:
                z_diff = z_final[:, 1:, :] - z_final[:, :-1, :]
            else:
                z_diff = torch.zeros_like(z_final)
                
            # 6. 更新激活值和轨迹 (EMA)
            self._update_activations(name, z_final.to("cpu", dtype=torch.float32))
            self._update_trajectories(name, z_diff.to("cpu", dtype=torch.float32))
            
        return hook

    def _update_trajectories(self, name, t_new):
        """EMA 累积轨迹"""
        if name not in self.trajectories:
            self.trajectories[name] = t_new
        else:
            # 确保形状一致
            if self.trajectories[name].shape == t_new.shape:
                self.trajectories[name] = 0.9 * self.trajectories[name] + 0.1 * t_new
            else:
                # 如果形状不一致（例如最后一个 batch 较小），只更新重叠部分
                min_b = min(self.trajectories[name].shape[0], t_new.shape[0])
                self.trajectories[name][:min_b] = 0.9 * self.trajectories[name][:min_b] + 0.1 * t_new[:min_b]

    # def _get_activation_hook(self, name):
    #     """收集激活值用于3D描述符计算 [B, T, C]"""
    #     def hook(module, input, output):
    #         z = output.detach()  # [B, C, T, H, W]
    #         z_pool = torch.mean(z, dim=(3, 4))  # [B, C, T] - 空间平均池化
            
    #         if z_pool.shape[2] > 8:
    #             z_pool = F.adaptive_avg_pool1d(z_pool, 8)
            
    #         z_final = z_pool.permute(0, 2, 1)  # [B, T, C]
            
    #         if z_final.shape[0] > 64:
    #             z_final = z_final[:64]
            
    #         self._update_activations(name, z_final.to("cpu", dtype=torch.float32))
    #     return hook

    def _update_activations(self, name, z_new):
        """EMA累积激活值"""
        if name not in self.activations:
            self.activations[name] = z_new
        else:
            if self.activations[name].shape[0] == z_new.shape[0]:
                self.activations[name] = 0.9 * self.activations[name] + 0.1 * z_new
            else:
                min_b = min(self.activations[name].shape[0], z_new.shape[0])
                self.activations[name][:min_b] = 0.9 * self.activations[name][:min_b] + 0.1 * z_new[:min_b]

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []

    def run_calibration(self, loader, device):
        """运行校准以收集激活统计信息"""
        self.register_hooks()
        self.model.eval()
        print(">>> 正在采集层间交互特征(3D描述符)...")
        with torch.no_grad():
            for i, batch in enumerate(tqdm.tqdm(loader, desc="Calibration")):
                if i >= 10:
                    break
                vids = batch[0]
                self.model(vids.float().to(device, non_blocking=True))
        self.remove_hooks()

    def estimate_unit_cost(self, module):
        """
        计算剪掉一个输出通道节省的参数数量
        包括: Conv权重 + BN参数 (gamma, beta, running_mean, running_var)
        """
        in_c = module.in_channels
        k = module.kernel_size
        # Conv 权重节省: in_c * kT * kH * kW
        conv_params = in_c * k[0] * k[1] * k[2]
        
        # BN 参数节省: 每个通道对应 4 个浮点数
        bn_params = 4 
        
        return conv_params + bn_params

    # ==================== BN同步与残差保护 ====================

    def _get_bn_name(self, conv_name):
        """根据conv名称推导对应的BN名称"""
        if '.conv1' in conv_name:
            return conv_name.replace('.conv1', '.bn1')
        elif '.conv2' in conv_name:
            return conv_name.replace('.conv2', '.bn2')
        elif '.conv3' in conv_name:
            return conv_name.replace('.conv3', '.bn3')
        elif '.downsample.0' in conv_name:
            return conv_name.replace('.downsample.0', '.downsample.1')
        return None

    def sync_bn_with_masks(self):
        """
        BN层同步：将被掩码通道对应的BN weight和bias置零。
        这是物理剪枝模拟，确保微调时BN统计量正确。
        """
        module_dict = dict(self.model.named_modules())
        sync_count = 0
        
        for name, conv in module_dict.items():
            if not (isinstance(conv, nn.Conv3d) and hasattr(conv, 'interaction_mask')):
                continue
            
            bn_name = self._get_bn_name(name)
            if bn_name is None:
                continue
            
            bn = module_dict.get(bn_name)
            if bn is None or not isinstance(bn, (nn.BatchNorm3d, nn.BatchNorm2d, nn.BatchNorm1d)):
                continue
            
            mask = conv.interaction_mask.to(bn.weight.device)  # [C]
            
            with torch.no_grad():
                # 置零BN weight (gamma) 和 bias (beta)
                if bn.weight is not None:
                    bn.weight.data *= mask
                if bn.bias is not None:
                    bn.bias.data *= mask
                # 同步running_mean和running_var (可选但推荐)
                if hasattr(bn, 'running_mean') and bn.running_mean is not None:
                    bn.running_mean.data *= mask
                if hasattr(bn, 'running_var') and bn.running_var is not None:
                    # running_var置为1避免除零，running_mean已为0
                    bn.running_var.data = bn.running_var.data * mask + (1 - mask)
            
            sync_count += 1
        
        print(f">>> BN同步完成: {sync_count} 个BN层已同步")

    def freeze_pruned_bn_params(self):
        """
        冻结被剪通道的BN参数梯度：注册hook屏蔽被剪通道的梯度更新。
        调用后进入微调阶段，被剪通道的BN参数不会再更新。
        """
        # 先清理旧的hook
        for h in self.bn_hooks:
            h.remove()
        self.bn_hooks = []
        
        module_dict = dict(self.model.named_modules())
        frozen_count = 0
        
        for name, conv in module_dict.items():
            if not (isinstance(conv, nn.Conv3d) and hasattr(conv, 'interaction_mask')):
                continue
            
            bn_name = self._get_bn_name(name)
            if bn_name is None:
                continue
            
            bn = module_dict.get(bn_name)
            if bn is None or not isinstance(bn, (nn.BatchNorm3d, nn.BatchNorm2d, nn.BatchNorm1d)):
                continue
            
            mask = conv.interaction_mask  # [C]
            
            # 为weight注册梯度hook
            if bn.weight is not None and bn.weight.requires_grad:
                def make_hook(m):
                    return lambda grad: grad * m.to(grad.device)
                h = bn.weight.register_hook(make_hook(mask))
                self.bn_hooks.append(h)
                frozen_count += 1
            
            # 为bias注册梯度hook
            if bn.bias is not None and bn.bias.requires_grad:
                def make_hook(m):
                    return lambda grad: grad * m.to(grad.device)
                h = bn.bias.register_hook(make_hook(mask))
                self.bn_hooks.append(h)
        
        print(f">>> BN梯度冻结完成: {frozen_count} 个BN参数已保护")

    def sync_downsample_masks(self):
        """
        残差连接保护：将每个Bottleneck的conv3掩码同步到其downsample conv。
        确保 identity 和 out 在相加前具有完全一致的有效通道。
        """
        module_dict = dict(self.model.named_modules())
        sync_count = 0
        
        for name, conv3 in module_dict.items():
            # 只处理有mask的conv3
            if '.conv3' not in name or not hasattr(conv3, 'interaction_mask'):
                continue
            
            # 找到该Bottleneck的downsample
            block_name = name.rsplit('.', 1)[0]  # e.g., fast_res2.0
            block = module_dict.get(block_name)
            if block is None or not hasattr(block, 'downsample') or block.downsample is None:
                continue
            
            down_conv = block.downsample[0]  # nn.Conv3d
            if not isinstance(down_conv, nn.Conv3d):
                continue
            
            mask = conv3.interaction_mask  # [C]
            # downsample输出通道数 = conv3输出通道数 = planes*expansion
            # 直接复用mask
            down_conv.register_buffer('interaction_mask', mask.clone())
            sync_count += 1
        
        print(f">>> 残差连接保护完成: {sync_count} 个downsample已同步")

    def sync_lateral_connections(self):
        """
        侧向连接同步:
        1. Fast路径Stage输出mask -> Lateral Conv输入权重mask
        2. Lateral Conv输出mask -> Slow路径对应Stage输入权重mask
        防止Fast剪枝后Lateral接收'死通道'输入，或Lateral输出污染Slow路径。
        """
        module_dict = dict(self.model.named_modules())
        sync_count = 0
        
        # 1. Fast输出 -> Lateral输入同步
        for fast_conv3_name, lateral_name in self.fast_to_lateral_map.items():
            fast_conv3 = module_dict.get(fast_conv3_name)
            lateral_conv = module_dict.get(lateral_name)
            if fast_conv3 is None or lateral_conv is None:
                continue
            if not hasattr(fast_conv3, 'interaction_mask'):
                continue
            
            mask = fast_conv3.interaction_mask.to(lateral_conv.weight.device)
            if lateral_conv.in_channels != mask.shape[0]:
                print(f"警告: {lateral_name} 输入通道数({lateral_conv.in_channels})与 {fast_conv3_name} 输出通道数({mask.shape[0]})不匹配，跳过侧向同步")
                continue
            
            with torch.no_grad():
                lateral_conv.weight.data *= mask.view(1, -1, 1, 1, 1)
            
            if lateral_conv.weight.requires_grad:
                def make_input_hook(m):
                    return lambda grad: grad * m.to(grad.device).view(1, -1, 1, 1, 1)
                h = lateral_conv.weight.register_hook(make_input_hook(mask))
                self.bn_hooks.append(h)
            sync_count += 1
        
        # 2. Lateral输出 -> Slow输入同步
        for lateral_name, slow_conv1_name in self.lateral_to_slow_map.items():
            lateral_conv = module_dict.get(lateral_name)
            slow_conv1 = module_dict.get(slow_conv1_name)
            if lateral_conv is None or slow_conv1 is None:
                continue
            if not hasattr(lateral_conv, 'interaction_mask'):
                continue
            
            lateral_mask = lateral_conv.interaction_mask.to(slow_conv1.weight.device)
            slow_in_c = slow_conv1.in_channels
            lateral_out_c = lateral_conv.out_channels
            
            if slow_in_c < lateral_out_c:
                print(f"警告: {slow_conv1_name} 输入通道数({slow_in_c})小于 {lateral_name} 输出通道数({lateral_out_c})，跳过侧向同步")
                continue
            
            slow_prev_out_c = slow_in_c - lateral_out_c
            with torch.no_grad():
                full_mask = torch.ones(slow_in_c, device=slow_conv1.weight.device)
                full_mask[slow_prev_out_c:] = lateral_mask
                slow_conv1.weight.data *= full_mask.view(1, -1, 1, 1, 1)
            
            if slow_conv1.weight.requires_grad:
                def make_input_hook(m):
                    return lambda grad: grad * m.to(grad.device).view(1, -1, 1, 1, 1)
                h = slow_conv1.weight.register_hook(make_input_hook(full_mask))
                self.bn_hooks.append(h)
            sync_count += 1
        
        print(f">>> 侧向连接同步完成: {sync_count} 个连接已同步")

    def apply_all_post_prune_sync(self):
        """剪枝后一次性调用所有同步操作"""
        self.sync_downsample_masks()
        self.sync_lateral_connections()
        self.sync_bn_with_masks()
        self.freeze_pruned_bn_params()

    # ==================== 3D描述符计算核心方法 ====================
    
    def compute_absolute_importance(self, Z):
        """计算绝对重要性: 时序平均激活幅度 + 激活频率 + 时序方差"""
        mean_amp = Z.mean(dim=(0, 1))
        freq = (Z.abs() > 1e-4).float().mean(dim=(0, 1))
        
        # 引入时序方差: 对时间轴计算方差后取平均
        # 高方差表示通道在捕捉动态特征，重要性应提高
        temporal_var = Z.var(dim=1).mean(dim=0)
        if temporal_var.max() > 0:
            temporal_var = temporal_var / temporal_var.max()
        
        return 0.35 * mean_amp + 0.25 * freq + 0.4 * temporal_var

    def compute_substitutability(self, Z):
        """
        计算通道可替代性 (基于Schur complement)
        修复版：处理维度问题，避免 .T 警告，增加鲁棒性
        """
        # 确保 Z 是 3D: [B, T, C]
        if Z.ndim != 3:
            raise ValueError(f"Expected 3D input [B, T, C], got shape {Z.shape}")
        
        B, T, C = Z.shape
        
        # 如果通道数太少，无法计算协方差，直接返回低可替代性（即重要）
        if C < 2:
            return torch.zeros(C, device=Z.device)
            
        # Reshape to [N, C] where N = B * T
        Z_reshaped = Z.reshape(-1, C)
        
        # Centering
        mean = Z_reshaped.mean(dim=0, keepdim=True)
        Z_centered = Z_reshaped - mean
        
        # 计算协方差矩阵 [C, C]
        # 使用 permute 代替 .T 以避免警告并明确维度
        # Z_centered: [N, C]
        # Z_centered.permute(1, 0): [C, N]
        cov = torch.matmul(Z_centered.permute(1, 0), Z_centered) / Z_reshaped.shape[0]
        
        # 添加正则化项
        eps = 1e-5
        cov += eps * torch.eye(C, device=Z.device)
        
        try:
            # 求逆
            inv_cov = torch.linalg.inv(cov)
            
            # 舒尔补余项
            diag_inv = torch.diag(inv_cov)
            diag_cov = torch.diag(cov)
            
            # 避免除以零
            err_var = 1.0 / (diag_inv + 1e-8)
            
            # 可替代性分数
            substitutability = 1.0 - err_var / (diag_cov + 1e-8)
            
            # Clamp to [0, 1]
            substitutability = torch.clamp(substitutability, min=0.0, max=1.0)
            
        except Exception as e:
            # 如果求逆失败，返回中等可替代性
            print(f"Warning: Covariance inversion failed for layer with C={C}. Error: {e}")
            substitutability = torch.ones(C, device=Z.device) * 0.5
            
        return substitutability

    def compute_relative_importance(self, D_abs, Z):
        """计算相对重要性: 绝对重要性 x (1 - 可替代性)"""
        substitutability = self.compute_substitutability(Z)
        return D_abs * (1 - substitutability)

    def compute_cross_layer_impact(self, Z_current, Z_next):
        
        """计算跨层影响分数"""
        if Z_next is None:
            return torch.ones(Z_current.shape[-1], device=Z_current.device)
        
        B, T, C_curr = Z_current.shape
        B2, T2, C_next = Z_next.shape
        
        # 对齐时间维度
        if T != T2:
            target_T = min(T, T2)
            # Permute to [B, C, T] for pooling
            Z_current = Z_current.permute(0, 2, 1)
            Z_next = Z_next.permute(0, 2, 1)
            
            Z_current = F.adaptive_avg_pool1d(Z_current, target_T)
            Z_next = F.adaptive_avg_pool1d(Z_next, target_T)
            
            # Permute back to [B, T, C]
            Z_current = Z_current.permute(0, 2, 1)
            Z_next = Z_next.permute(0, 2, 1)
            
            # Update shapes
            T = target_T
            T2 = target_T

        # 对齐 Batch 维度
        min_B = min(B, B2)
        Z_current = Z_current[:min_B]
        Z_next = Z_next[:min_B]
        
        # Flatten to [N, C]
        Z_curr_flat = Z_current.reshape(-1, C_curr)
        Z_next_flat = Z_next.reshape(-1, C_next)
        
        # Centering
        Z_curr_centered = Z_curr_flat - Z_curr_flat.mean(dim=0, keepdim=True)
        Z_next_centered = Z_next_flat - Z_next_flat.mean(dim=0, keepdim=True)
        
        # Cross-correlation [C_curr, C_next]
        # Use permute instead of .T
        cross_corr = torch.matmul(Z_curr_centered.permute(1, 0), Z_next_centered) / Z_curr_flat.shape[0]
        
        # Impact on current channels: mean absolute correlation with next layer
        impact = cross_corr.abs().mean(dim=1)
        
        # Normalize to [0, 1]
        if impact.max() > impact.min():
            impact = (impact - impact.min()) / (impact.max() - impact.min() + 1e-8)
        else:
            impact = torch.zeros_like(impact)
            
        return impact

    def mean_shift_clustering(self, descriptors):
        """
        Mean-Shift聚类: 将通道分组为功能聚类
        聚类前进行StandardScaler归一化 (零均值, 单位方差)
        descriptors: [N_units, 3] - 每个通道的3D描述符 (abs, rel, cross)
        Returns: 聚类标签列表
        """
        N, D = descriptors.shape
        if N <= 5:
            return [[i] for i in range(N)]
        
        # StandardScaler: 对每维特征独立进行零均值单位方差归一化
        descriptors_norm = (descriptors - descriptors.mean(dim=0, keepdim=True)) / (descriptors.std(dim=0, keepdim=True) + 1e-8)
        
        labels = torch.zeros(N, dtype=torch.long, device=descriptors.device) - 1
        cluster_id = 0
        visited = torch.zeros(N, dtype=torch.bool, device=descriptors.device)
        
        for i in range(N):
            if visited[i]:
                continue
            
            center = descriptors_norm[i].clone()
            
            for _ in range(10):
                distances = torch.norm(descriptors_norm - center.unsqueeze(0), dim=1)
                weights = torch.exp(-distances**2 / (2 * self.sigma**2))
                new_center = (weights.unsqueeze(1) * descriptors_norm).sum(dim=0) / (weights.sum() + 1e-8)
                
                if torch.norm(new_center - center) < 1e-3:
                    break
                center = new_center
            
            within_cluster = torch.norm(descriptors_norm - center.unsqueeze(0), dim=1) < self.sigma * 2.5
            members = []
            for j in range(N):
                if within_cluster[j] and not visited[j]:
                    labels[j] = cluster_id
                    visited[j] = True
                    members.append(j)
            
            if len(members) > 0:
                cluster_id += 1
            else:
                labels[i] = cluster_id
                visited[i] = True
                cluster_id += 1
        
        groups = {}
        for i in range(N):
            cid = labels[i].item()
            if cid not in groups:
                groups[cid] = []
            groups[cid].append(i)
        
        # 防御性检查: 如果某个聚类过大(超过70%)，说明可能过度聚合，进行拆分
        final_groups = []
        for group in list(groups.values()):
            if len(group) > int(N * 0.7) and len(group) > 5:
                group_desc = descriptors_norm[group]
                norms = torch.norm(group_desc, dim=1)
                median_norm = norms.median()
                sub_g1 = [group[i] for i in range(len(group)) if norms[i] <= median_norm]
                sub_g2 = [group[i] for i in range(len(group)) if norms[i] > median_norm]
                if len(sub_g1) > 0:
                    final_groups.append(sub_g1)
                if len(sub_g2) > 0:
                    final_groups.append(sub_g2)
            else:
                final_groups.append(group)
        
        return final_groups

    def compute_group_manifold_score(self, group_descriptors):
        """
        计算组的流形一致性分数
        group_descriptors: [G, 3] 组内通道的3D描述符
        Returns: (dynamic_score, static_score)
        """
        G = len(group_descriptors)
        if G <= 1:
            return 1.0, 1.0
        
        # 确保输入是 Tensor
        if not isinstance(group_descriptors[0], torch.Tensor):
            desc = torch.stack([torch.tensor(d) for d in group_descriptors])
        else:
            desc = torch.stack(group_descriptors)
            
        if G >= 2:
            desc_norm = desc / (torch.norm(desc, dim=1, keepdim=True) + 1e-8)
            cos_sim = torch.matmul(desc_norm, desc_norm.T)
            mask = 1 - torch.eye(G, device=desc.device)
            dynamic_score = (cos_sim * mask).sum() / (mask.sum() + 1e-8)
        else:
            dynamic_score = 1.0
        
        static_score = desc.abs().mean().item()
        
        return dynamic_score.item(), static_score

    def compute_layer_descriptors(self, Z, Z_next=None, Z_prev=None):
        """
        计算层的3D描述符
        Returns: [N, 3] 描述符矩阵 (D_abs, D_rel, D_cross)
        """
        D_abs = self.compute_absolute_importance(Z)
        D_rel = self.compute_relative_importance(D_abs, Z)
        D_cross = self.compute_cross_layer_impact(Z, Z_next)
        
        descriptors = torch.stack([D_abs, D_rel, D_cross], dim=1)
        return descriptors

    # ==================== 剪枝核心方法 ====================


    def prune_exact(self, target_sparsity, tolerance=0.001):
        """
        精确控制稀疏度的剪枝 (二分搜索 + 后处理微调)
        """
        valid_layer_names = [n for n in self.ordered_layer_names if n in self.activations]
        device = next(self.model.parameters()).device
        
        if not valid_layer_names:
            return 0.0
        
        print(f">>> 精确剪枝 (目标: {target_sparsity:.4%}, 容差: {tolerance:.4%})...")
        
        import numpy as np
        
        layer_norm_scores = {}
        layer_costs = {}
        layer_max_prunable = {}
        layer_masks = {} # 存储初始掩码状态
        
        # 1. 预计算每层的分数和成本
        for l_idx, name in enumerate(valid_layer_names):
            Z = self.activations[name].to(device)
            
            Z_next = None
            next_name = self.next_layer_map.get(name)
            if next_name is not None and next_name in self.activations:
                Z_next = self.activations[next_name].to(device)
            
            descriptors = self.compute_layer_descriptors(Z, Z_next)
            groups = self.mean_shift_clustering(descriptors)
            
            N = descriptors.shape[0]
            unit_scores = torch.zeros(N, device=device)
            
            for group in groups:
                group_desc = [descriptors[i] for i in group]
                # 只传入一个参数
                dynamic_score, static_score = self.compute_group_manifold_score(group_desc)
                
                for idx in group:
                    unit_scores[idx] = dynamic_score * static_score
            
            self.layer_avg_scores[name] = unit_scores.mean().item()
            
            s_min, s_max = unit_scores.min(), unit_scores.max()
            if s_max > s_min:
                norm_scores = ((unit_scores - s_min) / (s_max - s_min)).cpu().numpy()
            else:
                norm_scores = unit_scores.cpu().numpy()
            
            module = dict(self.model.named_modules())[name]
            cost = self.estimate_unit_cost(module)
            min_keep = max(1, int(N * self.min_keep_ratio))
            
            layer_norm_scores[name] = norm_scores
            layer_costs[name] = cost
            layer_max_prunable[name] = N - min_keep
            layer_masks[name] = torch.ones(N, dtype=torch.bool) # True = Keep

        # 2. 二分搜索寻找最佳阈值
        low, high = 0.0, 1.0
        best_threshold = 0.5
        best_registry = None
        best_sparsity_diff = float('inf')
        
        # 计算总参数量用于二分搜索的目标判断
        total_original_params = self.total_original_params
        
        for iteration in range(30): # 增加迭代次数以提高精度
            mid = (low + high) / 2.0
            current_pruned_params = 0
            temp_registry = {name: set() for name in valid_layer_names}
            
            for name in valid_layer_names:
                scores = layer_norm_scores[name]
                cost = layer_costs[name]
                max_p = layer_max_prunable[name]
                
                # 找出低于阈值的索引
                pruned_indices = np.where(scores < mid)[0]
                
                # 限制最大可剪枝数量
                if len(pruned_indices) > max_p:
                    # 按分数从小到大排序，只剪最弱的 max_p 个
                    sorted_indices = pruned_indices[np.argsort(scores[pruned_indices])]
                    pruned_indices = sorted_indices[:max_p]
                
                for idx in pruned_indices:
                    temp_registry[name].add(int(idx))
                    current_pruned_params += cost
            
            actual_sparsity = current_pruned_params / total_original_params
            diff = abs(actual_sparsity - target_sparsity)
            
            if diff < best_sparsity_diff:
                best_sparsity_diff = diff
                best_threshold = mid
                best_registry = temp_registry
                best_actual_sparsity = actual_sparsity
                
            if diff <= tolerance:
                break
                
            if actual_sparsity < target_sparsity:
                low = mid
            else:
                high = mid

        # 3. 后处理微调 (Post-processing Fine-tuning)
        # 如果二分搜索的结果还不够精确，通过逐个调整通道来逼近目标
        if best_registry is not None:
            current_pruned_params = sum([len(best_registry[n]) * layer_costs[n] for n in valid_layer_names])
            current_sparsity = current_pruned_params / total_original_params
            target_pruned_params = int(target_sparsity * total_original_params)
            
            # 构建一个全局的“可操作通道列表”，按重要性排序
            # 格式: (score, layer_name, channel_idx, cost)
            candidate_list = []
            for name in valid_layer_names:
                scores = layer_norm_scores[name]
                cost = layer_costs[name]
                for idx in range(len(scores)):
                    # 如果当前在 best_registry 中，说明它是“被剪枝”的候选者（弱）
                    # 如果不在，说明它是“保留”的候选者（强）
                    # 我们的目标是：如果剪多了，从 registry 里捞出最重要的恢复；如果剪少了，把 registry 外最弱的加入
                    is_pruned = idx in best_registry[name]
                    candidate_list.append((scores[idx], name, idx, cost, is_pruned))
            
            # 按分数升序排列 (分数越低越应该被剪)
            candidate_list.sort(key=lambda x: x[0])
            
            # 重新构建 Registry 以精确匹配 target_pruned_params
            final_registry = {name: set() for name in valid_layer_names}
            accumulated_params = 0
            
            # 先加入所有肯定要被剪的（分数极低的），直到接近目标
            # 这里采用贪心策略：从分数最低的開始添加，直到参数量达到 target
            for score, name, idx, cost, _ in candidate_list:
                if accumulated_params + cost <= target_pruned_params:
                    # 检查是否超过该层最大可剪枝数
                    if len(final_registry[name]) < layer_max_prunable[name]:
                        final_registry[name].add(idx)
                        accumulated_params += cost
                else:
                    break
                    
            # 应用最终 Registry
            for name in valid_layer_names:
                module = dict(self.model.named_modules())[name]
                mask = torch.ones(module.out_channels, device=device)
                for idx in final_registry[name]:
                    mask[idx] = 0.0
                module.register_buffer('interaction_mask', mask)
                
            # 计算最终实际稀疏度用于日志
            final_pruned_params = sum([len(final_registry[n]) * layer_costs[n] for n in valid_layer_names])
            final_sparsity = final_pruned_params / total_original_params
            print(f">>> 后处理微调完成。目标参数剪枝量: {target_pruned_params}, 实际: {final_pruned_params}")
            print(f">>> 最终稀疏度: {final_sparsity:.4%}")

        # 剪枝后同步: BN + 残差连接 + 梯度冻结
        self.apply_all_post_prune_sync()
        
        # 返回最终稀疏度
        if best_registry is not None:
            final_pruned_params = sum([len(final_registry[n]) * layer_costs[n] for n in valid_layer_names])
            return final_pruned_params / total_original_params
        return 0.0

    def prune(self, target_sparsity=None, tolerance=0.01):
        """
        剪枝入口方法
        自动完成: 掩码生成 -> downsample同步 -> BN同步 -> 梯度冻结
        """
        if target_sparsity is None:
            target_sparsity = self.target_sparsity
        return self.prune_exact(target_sparsity, tolerance)

    def get_stats(self):
        """打印每层剪枝统计"""
        print("\n" + "=" * 85)
        print(f"{'Layer Name':<40} | {'Score':<10} | {'Kept/Total':<15} | {'Ratio'}")
        print("-" * 85)
        
        all_scores = list(self.layer_avg_scores.values())
        max_score = max(all_scores) if all_scores else 1.0
        
        for name, m in self.model.named_modules():
            if hasattr(m, 'interaction_mask'):
                kept = int(m.interaction_mask.sum())
                total = m.out_channels
                avg_score = self.layer_avg_scores.get(name, 0.0)
                bar_len = int((avg_score / max_score) * 10)
                score_bar = "█" * bar_len
                print(f"{name:40} | {avg_score:10.4f} | {kept:>4}/{total:<4} | {kept/total:>6.1%} {score_bar}")
        print("=" * 85 + "\n")


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, strides=1, downsample=None, head_conv=1,
                 norm_layer=BatchNorm3d, norm_kwargs=None, layer_name=''):
        super(Bottleneck, self).__init__()

        if head_conv == 1:
            self.conv1 = nn.Conv3d(in_channels=inplanes, out_channels=planes, kernel_size=1, bias=False)
        elif head_conv == 3:
            self.conv1 = nn.Conv3d(in_channels=inplanes, out_channels=planes, kernel_size=(3, 1, 1), 
                                   padding=(1, 0, 0), bias=False)
        else:
            raise ValueError("Unsupported head_conv!")
        
        self.bn1 = norm_layer(num_features=planes, **({} if norm_kwargs is None else norm_kwargs))
        
        self.conv2 = nn.Conv3d(in_channels=planes, out_channels=planes, kernel_size=(1, 3, 3),
                               stride=(1, strides, strides), padding=(0, 1, 1), bias=False)
        self.bn2 = norm_layer(num_features=planes, **({} if norm_kwargs is None else norm_kwargs))
        
        self.conv3 = nn.Conv3d(in_channels=planes, out_channels=planes * self.expansion, 
                               kernel_size=1, stride=1, bias=False)
        self.bn3 = norm_layer(num_features=planes * self.expansion, **({} if norm_kwargs is None else norm_kwargs))
        
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def _apply_mask(self, x, layer):
        """应用来自 InteractionPruner 的剪枝掩码"""
        if hasattr(layer, 'interaction_mask'):
            mask = layer.interaction_mask.view(1, -1, 1, 1, 1)
            return x * mask
        return x

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self._apply_mask(out, self.conv1)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self._apply_mask(out, self.conv2)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self._apply_mask(out, self.conv3) 
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)
            # 残差连接保护: 对downsample的输出也应用掩码
            identity = self._apply_mask(identity, self.downsample[0])

        out = self.relu(out + identity)
        return out


class SlowFast(nn.Module):
    def __init__(self,
                 num_classes,
                 block=Bottleneck,
                 layers=[3, 4, 6, 3],
                 num_block_temp_kernel_fast=None,
                 num_block_temp_kernel_slow=None,
                 dropout_ratio=0.5,
                 alpha=8,
                 beta_inv=8,
                 fusion_conv_channel_ratio=2,
                 fusion_kernel_size=5,
                 width_per_group=64,
                 slow_temporal_stride=16,
                 norm_layer=BatchNorm3d,
                 norm_kwargs=None,
                 **kwargs):
        super(SlowFast, self).__init__()
        
        self.alpha = alpha
        self.beta_inv = beta_inv
        self.fusion_conv_channel_ratio = fusion_conv_channel_ratio
        self.fusion_kernel_size = fusion_kernel_size
        self.width_per_group = width_per_group
        self.dim_inner = width_per_group
        self.out_dim_ratio = beta_inv // fusion_conv_channel_ratio
        self.slow_temporal_stride = slow_temporal_stride
        self.dropout_ratio = dropout_ratio

        # Fast Pathway
        fast_in_c = width_per_group // beta_inv
        self.fast_conv1 = nn.Conv3d(3, fast_in_c, kernel_size=(5, 7, 7), stride=(1, 2, 2), padding=(2, 3, 3), bias=False)
        self.fast_bn1 = norm_layer(num_features=fast_in_c, **({} if norm_kwargs is None else norm_kwargs))
        self.fast_relu = nn.ReLU(inplace=True)
        self.fast_maxpool = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        
        self.fast_res2 = self._make_layer_fast(fast_in_c, self.dim_inner // beta_inv, layers[0], head_conv=3, norm_layer=norm_layer)
        self.fast_res3 = self._make_layer_fast(self.dim_inner * 4 // beta_inv, self.dim_inner * 2 // beta_inv, layers[1], strides=2, head_conv=3, norm_layer=norm_layer)
        self.fast_res4 = self._make_layer_fast(self.dim_inner * 8 // beta_inv, self.dim_inner * 4 // beta_inv, layers[2], strides=2, head_conv=3, norm_layer=norm_layer)
        self.fast_res5 = self._make_layer_fast(self.dim_inner * 16 // beta_inv, self.dim_inner * 8 // beta_inv, layers[3], strides=2, head_conv=3, norm_layer=norm_layer)

        # Lateral Connections
        self.lateral_p1 = self._make_lateral_conv(fast_in_c, norm_layer)
        self.lateral_res2 = self._make_lateral_conv(self.dim_inner * 4 // beta_inv, norm_layer)
        self.lateral_res3 = self._make_lateral_conv(self.dim_inner * 8 // beta_inv, norm_layer)
        self.lateral_res4 = self._make_lateral_conv(self.dim_inner * 16 // beta_inv, norm_layer)

        # Slow Pathway
        self.slow_conv1 = nn.Conv3d(3, width_per_group, kernel_size=(1, 7, 7), stride=(1, 2, 2), padding=(0, 3, 3), bias=False)
        self.slow_bn1 = norm_layer(num_features=width_per_group, **({} if norm_kwargs is None else norm_kwargs))
        self.slow_relu = nn.ReLU(inplace=True)
        self.slow_maxpool = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))

        self.slow_res2 = self._make_layer_slow(width_per_group + (fast_in_c * fusion_conv_channel_ratio), self.dim_inner, layers[0], head_conv=1, norm_layer=norm_layer)
        self.slow_res3 = self._make_layer_slow(self.dim_inner * 4 + (self.dim_inner * 4 // beta_inv * fusion_conv_channel_ratio), self.dim_inner * 2, layers[1], strides=2, head_conv=1, norm_layer=norm_layer)
        self.slow_res4 = self._make_layer_slow(self.dim_inner * 8 + (self.dim_inner * 8 // beta_inv * fusion_conv_channel_ratio), self.dim_inner * 4, layers[2], strides=2, head_conv=3, norm_layer=norm_layer)
        self.slow_res5 = self._make_layer_slow(self.dim_inner * 16 + (self.dim_inner * 16 // beta_inv * fusion_conv_channel_ratio), self.dim_inner * 8, layers[3], strides=2, head_conv=3, norm_layer=norm_layer)

        # Classifier
        self.avg = nn.AdaptiveAvgPool3d(1)
        self.dp = nn.Dropout(p=self.dropout_ratio)
        self.feat_dim = (self.dim_inner * 32 // beta_inv) + (self.dim_inner * 32)
        self.fc = nn.Linear(self.feat_dim, num_classes)

    def _make_lateral_conv(self, in_c, norm_layer):
        return nn.Sequential(
            nn.Conv3d(in_c, in_c * self.fusion_conv_channel_ratio, kernel_size=(self.fusion_kernel_size, 1, 1), 
                      stride=(self.alpha, 1, 1), padding=(self.fusion_kernel_size // 2, 0, 0), bias=False),
            norm_layer(num_features=in_c * self.fusion_conv_channel_ratio),
            nn.ReLU(inplace=True)
        )

    def _make_layer_fast(self, inplanes, planes, num_blocks, strides=1, head_conv=1, norm_layer=BatchNorm3d):
        downsample = None
        if strides != 1 or inplanes != planes * Bottleneck.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(inplanes, planes * Bottleneck.expansion, kernel_size=1, stride=(1, strides, strides), bias=False),
                norm_layer(num_features=planes * Bottleneck.expansion)
            )
        layers = [Bottleneck(inplanes, planes, strides, downsample, head_conv, norm_layer)]
        inplanes = planes * Bottleneck.expansion
        for _ in range(1, num_blocks):
            layers.append(Bottleneck(inplanes, planes, 1, None, head_conv, norm_layer))
        return nn.Sequential(*layers)

    def _make_layer_slow(self, inplanes, planes, num_blocks, strides=1, head_conv=1, norm_layer=BatchNorm3d):
        downsample = None
        if strides != 1 or inplanes != planes * Bottleneck.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(inplanes, planes * Bottleneck.expansion, kernel_size=1, stride=(1, strides, strides), bias=False),
                norm_layer(num_features=planes * Bottleneck.expansion)
            )
        layers = [Bottleneck(inplanes, planes, strides, downsample, head_conv, norm_layer)]
        inplanes = planes * Bottleneck.expansion
        for _ in range(1, num_blocks):
            layers.append(Bottleneck(inplanes, planes, 1, None, head_conv, norm_layer))
        return nn.Sequential(*layers)

    def forward(self, x):
        fast_input = x
        slow_input = x[:, :, ::self.slow_temporal_stride // 2, :, :]

        # Fast Path
        f = self.fast_conv1(fast_input)
        f = self.fast_bn1(f)
        f = self.fast_relu(f)
        f_pool = self.fast_maxpool(f)
        
        l1 = self.lateral_p1(f_pool)
        f_res2 = self.fast_res2(f_pool)
        l2 = self.lateral_res2(f_res2)
        f_res3 = self.fast_res3(f_res2)
        l3 = self.lateral_res3(f_res3)
        f_res4 = self.fast_res4(f_res3)
        l4 = self.lateral_res4(f_res4)
        f_res5 = self.fast_res5(f_res4)
        f_out = self.avg(f_res5).view(f_res5.size(0), -1)

        # Slow Path
        s = self.slow_conv1(slow_input)
        s = self.slow_bn1(s)
        s = self.slow_relu(s)
        s_pool = self.slow_maxpool(s)
        
        s_res2 = self.slow_res2(torch.cat([s_pool, l1], dim=1))
        s_res3 = self.slow_res3(torch.cat([s_res2, l2], dim=1))
        s_res4 = self.slow_res4(torch.cat([s_res3, l3], dim=1))
        s_res5 = self.slow_res5(torch.cat([s_res4, l4], dim=1))
        s_out = self.avg(s_res5).view(s_res5.size(0), -1)

        # Fusion
        out = torch.cat([s_out, f_out], dim=1)
        out = self.dp(out)
        out = self.fc(out)
        return out

    def get_detailed_pruning_report(self):
        """获取详细的剪枝报告"""
        total_model_params = sum(p.numel() for p in self.parameters())
        total_pruned_params = 0
        layer_count = 0

        for name, module in self.named_modules():
            if hasattr(module, 'interaction_mask'):
                mask = module.interaction_mask
                pruned_channels = mask.numel() - int(mask.sum().item())
                
                weight_shape = module.weight.shape
                params_per_channel = torch.prod(torch.tensor(weight_shape[1:])).item()
                
                total_pruned_params += pruned_channels * params_per_channel
                layer_count += 1

        if total_model_params == 0:
            return {'sparsity': 0.0, 'pruned_layers': 0}

        actual_sparsity = total_pruned_params / total_model_params

        report = {
            'sparsity': actual_sparsity,
            'total_model_params': total_model_params,
            'total_pruned_params': total_pruned_params,
            'pruned_layers': layer_count
        }

        print(f"\n" + "=" * 40)
        print(f">>> 全模型参数量: {total_model_params:,}")
        print(f">>> 已剪掉参数量: {total_pruned_params:,}")
        print(f">>> 实际总剪枝率: {actual_sparsity:.2%}")
        print(f">>> 涉及剪枝层数: {layer_count}")
        print("=" * 40 + "\n")
        
        return report


def slowfast_16x8_resnet101_kinetics400(num_classes):
    model = SlowFast(num_classes=num_classes,
                     layers=[3, 4, 23, 3],
                     pretrained=False,
                     alpha=4,
                     beta_inv=8,
                     fusion_conv_channel_ratio=2,
                     fusion_kernel_size=5,
                     width_per_group=64,
                     num_groups=1,
                     slow_temporal_stride=8,
                     fast_temporal_stride=2,
                     slow_frames=16,
                     fast_frames=64,
                     bn_eval=False,
                     partial_bn=False,
                     bn_frozen=False)
    return model
