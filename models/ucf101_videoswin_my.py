import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import torch
import numpy as np
import math
import torch.nn as nn
import sys
from types import ModuleType

# --- 1. 修复 PIL.Image 常量缺失问题 ---
try:
    import PIL.Image as PIL_Image # 使用别名避开命名空间混淆
    if not hasattr(PIL_Image, 'LINEAR'):
        # 将旧版常量映射到新版的 Resampling 枚举上
        resampling = getattr(PIL_Image, 'Resampling', None)
        PIL_Image.LINEAR = resampling.BILINEAR if resampling else 2
        PIL_Image.BILINEAR = resampling.BILINEAR if resampling else 2
        PIL_Image.BICUBIC = resampling.BICUBIC if resampling else 3
        PIL_Image.NEAREST = resampling.NEAREST if resampling else 0
        
        # 关键一步：确保全局 PIL.Image 也被注入
        import PIL
        PIL.Image = PIL_Image
        sys.modules["PIL.Image"] = PIL_Image
        print(">>> 已手动修复 PIL.Image 兼容性补丁")
except ImportError:
    print(">>> 未检测到 Pillow 库")

# --- 2. 修复 torch._six 缺失问题 ---
try:
    import torch._six
except ImportError:
    _six_mod = ModuleType("torch._six")
    sys.modules["torch._six"] = _six_mod
    
if not hasattr(torch._six, 'int_classes'):
    torch._six.int_classes = (int,)
if not hasattr(torch._six, 'string_classes'):
    torch._six.string_classes = (str,)
    print(">>> 已手动修复 torch._six 兼容性补丁")

from dataset.ucf101 import get_dataset
from gluoncv.torch.model_zoo import get_model
from utils import CONFIG_PATHS, OPT_PATH, get_cfg_custom, MODEL_TO_CKPTS
import tqdm
import torch.nn.functional as F
from thop import profile
import random
import time
from MC import SwinTransformer3D,I3DHead,InteractionPruner

checkpoint_path = os.environ.get('TRAIN_OUTPUT_DIR', 'outlog/swin_transformer')
train_state_path = checkpoint_path
def resume_training(resume, model, optimizer):
    start_epoch = 1
    if resume > 0:
        start_epoch += resume
        model_path = os.path.join(
            checkpoint_path, 'checkpoint-{}.ckpt'.format(resume))
        model.module.load_state_dict(torch.load(model_path))
        train_path = os.path.join(
            train_state_path, 'checkpoint-{}_optimizer.ckpt'.format(resume))
        state_dict = torch.load(train_path)
        optimizer.load_state_dict(state_dict['optimizer'])
    return start_epoch

def run_one_epoch(epoch, net, optimizer, data_loader, epoch_step_num):
    net.train()
    optimizer.zero_grad()
    total_loss=0.0
    total_correct=0
    with tqdm.tqdm(data_loader, total=math.floor(epoch_step_num), ncols=0) as pbar:
        for n_iter, (input, target,index) in enumerate(pbar):
            input = input.float().cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            optimizer.zero_grad()
            #print(target)
            logits_student,_= net(input)
            loss_ce =  F.cross_entropy(logits_student, target)
            total_loss+=loss_ce
            loss_ce.backward()
            predictions = torch.argmax(logits_student, dim=1)
            correct = (predictions == target).sum().item()
            total_correct += correct
            pbar.set_description(f"Epoch {epoch},  CEloss:{loss_ce.item():.4f},Accuracy: {correct / cfg.CONFIG.TRAIN.BATCH_SIZE:.4f}")
    avg_loss = total_loss / len(data_loader)
    avg_accuracy = total_correct / len(data_loader.dataset)
    print(f"Epoch: {epoch}, Loss: {avg_loss}, Accuracy: {avg_accuracy}")
    return avg_loss, avg_accuracy

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].contiguous().view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res
def validate_rgb(val_loader, net, top1, top5):
    with torch.no_grad():
        with tqdm.tqdm(val_loader, total=len(val_loader), ncols=0) as pbar:
            for n_iter, (input, target,index) in enumerate(pbar):
                input = input.float().cuda(non_blocking=True)
                #print(input.size())
                target = target.cuda(non_blocking=True)

                output,_ = net(input)
                fusion = F.softmax(output, dim=1)

                prec1, prec5 = accuracy(fusion, target, topk=(1, 5))
                top1.update(prec1.item())
                top5.update(prec5.item())

                del input, target, output
                torch.cuda.empty_cache()
def get_rng_states():
    states = []
    states.append(random.getstate())
    states.append(np.random.get_state())
    states.append(torch.get_rng_state())
    if torch.cuda.is_available():
        states.append(torch.cuda.get_rng_state())
    return states
def save_model(epoch, model, optimizer):#改改best
    torch.save(model.module.state_dict(),
               os.path.join(args.adv_path, 'checkpoint-{}.ckpt'.format(epoch)))
    torch.save({'optimizer': optimizer.state_dict(),
                'state': get_rng_states()},
               os.path.join(args.adv_path, 'checkpoint-{}_optimizer.ckpt'.format(epoch)))
def arg_parse():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument(
        '--gpu', type=str, default=None,
        help='物理GPU编号，例如1,2；未提供时沿用CUDA_VISIBLE_DEVICES或默认0'
    )
    parser.add_argument('--batch_size', type=int, default=4, metavar='N')
    parser.add_argument(
        '--calib_batch_size', type=int, default=1,
        help='校准阶段批量；12GB显卡建议固定为1'
    )
    parser.add_argument(
        '--calib_batches', type=int, default=40,
        help='校准批次数；默认40x1与旧设置10x4保持40个视频'
    )
    parser.add_argument(
        '--accum_steps', type=int, default=1,
        help='梯度累积步数；两卡总batch=2时设为2可保持有效batch=4'
    )
    parser.add_argument(
        '--disable_amp', action='store_true',
        help='关闭默认启用的混合精度训练'
    )
    parser.add_argument('--model', type=str)
    parser.add_argument('--file_prefix', type=str, default='')
    parser.add_argument('--sparsity', type=float, default=0.5, help='剪枝比例')
    parser.add_argument(
        '--selection_mode', choices=('bms', 'coverage', 'functional'), default='bms',
        help='剪枝选择后端，默认保持原BMS流程'
    )
    parser.add_argument(
        '--selection_cost_mode', choices=('coupled', 'decoupled'),
        default='coupled',
        help='Task013参数代价交互；默认coupled完全保持Task012选择逻辑'
    )
    parser.add_argument(
        '--contribution_npz',
        default=None,
        help='Coverage/Functional后端读取的Contribution Field NPZ路径'
    )
    parser.add_argument(
        '--functional-score', '--functional_score',
        dest='functional_score',
        choices=('domain_average', 'domain_total'),
        default='domain_average',
        help='Task016跨BMS域功能损失尺度；默认精确保留Task014'
    )
    parser.add_argument(
        '--descriptor_variant',
        choices=('abs_rel', 'old3d', 'dynamic3d'),
        default='dynamic3d',
        help='受控描述符变体；Task010默认dynamic3d'
    )
    parser.add_argument('--seed', type=int, default=3407)
    parser.add_argument(
        '--output_root', default='tdd_pruning_validation',
        help='Task010各变体隔离输出的根目录'
    )
    parser.add_argument('--sigma', type=float, default=0.09)
    parser.add_argument('--gamma_decay', type=float, default=0.5)
    parser.add_argument('--min_keep_ratio', type=float, default=0.1)
    parser.add_argument('--importance_alpha', type=float, default=0.5)
    parser.add_argument(
        '--prune_only', action='store_true',
        help='完成校准、剪枝和微调前评估后停止，便于先做离线比较'
    )
    parser.add_argument(
        '--checkpoint_path',
        default=os.environ.get('MODEL_CHECKPOINT', 'pretrained/checkpoint-68.ckpt'),
        help='模型检查点；可通过MODEL_CHECKPOINT覆盖默认相对路径'
    )
    parser.add_argument(
        '--task011_diagnosis_dir', default=None,
        help='可选Task011只读审计输出目录；不改变剪枝数学或默认行为'
    )
    parser.add_argument(
        '--task012_run_dir', default=None,
        help='可选Task012单次prune-only诊断目录；每个配置必须独立'
    )
    parser.add_argument(
        '--task012_baseline_cache', default=None,
        help='Task012同模型/验证/GPU配置共享的未剪枝基线缓存'
    )
    parser.add_argument(
        '--task013_run_dir', default=None,
        help='可选Task013单次cost-mode prune-only诊断目录'
    )
    parser.add_argument(
        '--task013_baseline_cache', default=None,
        help='Task013六个受控运行共享的精确匹配未剪枝基线缓存'
    )
    parser.add_argument(
        '--task014_run_dir', default=None,
        help='可选Task014单次prune-only运行目录'
    )
    parser.add_argument(
        '--task014_baseline_cache', default=None,
        help='Task014六个受控运行共享的精确匹配未剪枝基线缓存'
    )
    parser.add_argument(
        '--task016_run_dir', default=None,
        help='可选Task016单次prune-only运行目录'
    )
    parser.add_argument(
        '--task016_baseline_cache', default=None,
        help='Task016运行复用的精确匹配未剪枝基线缓存'
    )
    parser.add_argument(
        '--functional_cache_dir', default='task014_functional_pruning/field_cache',
        help='Task014由完整NPZ逐层构建的共享对齐向量mmap目录'
    )
    parser.add_argument(
        '--functional_descriptor_cache', default=None,
        help='Task016复用的已验证Task014 Dynamic3D descriptor_statistics.csv'
    )
    parser.add_argument(
        '--task014_preflight_only', action='store_true',
        help='仅运行36,378单元映射审计和一个真实BMS域数值门禁'
    )
    parser.add_argument(
        '--diagnostic_min_attention_heads', type=int, choices=(0, 2, 3),
        default=0,
        help='仅Task012诊断：0保持原规则，2/3临时保护Attention；MLP不变'
    )
    parser.add_argument(
        '--evaluate_baseline_before_pruning', action='store_true',
        help='Task011/Task012受控运行：剪枝前用同一验证管线测量或复用基线'
    )
    parser.add_argument('--visualize', action='store_true', help='是否生成可视化图表')
    args = parser.parse_args()
    if args.batch_size <= 0 or args.calib_batch_size <= 0:
        parser.error('batch_size和calib_batch_size必须大于0')
    if args.calib_batches <= 0 or args.accum_steps <= 0:
        parser.error('calib_batches和accum_steps必须大于0')
    if args.selection_mode in ('coverage', 'functional') and \
            args.contribution_npz is None:
        parser.error(
            'selection_mode=coverage/functional时必须显式传入--contribution_npz'
        )
    if args.sigma <= 0.0:
        parser.error('sigma必须大于0')
    if not 0.0 < args.min_keep_ratio <= 1.0:
        parser.error('min_keep_ratio必须位于(0,1]')
    if not 0.0 <= args.importance_alpha <= 1.0:
        parser.error('importance_alpha必须位于[0,1]')
    audit_modes = sum(bool(value) for value in (
        args.task011_diagnosis_dir, args.task012_run_dir, args.task013_run_dir,
        args.task014_run_dir, args.task016_run_dir,
    ))
    if audit_modes > 1:
        parser.error('Task011、Task012、Task013、Task014与Task016审计输出模式不能同时启用')
    if bool(args.task011_diagnosis_dir) != bool(
            args.evaluate_baseline_before_pruning) and not (
                args.task012_run_dir or args.task013_run_dir or
                args.task014_run_dir or args.task016_run_dir):
        parser.error(
            'Task011诊断必须同时提供--task011_diagnosis_dir和'
            '--evaluate_baseline_before_pruning'
        )
    if args.task012_run_dir:
        if not args.evaluate_baseline_before_pruning:
            parser.error('Task012必须启用--evaluate_baseline_before_pruning')
        if not args.task012_baseline_cache:
            parser.error('Task012必须提供--task012_baseline_cache')
        if not args.prune_only:
            parser.error('Task012只允许--prune_only，禁止完整微调')
        if args.selection_mode != 'bms':
            parser.error('Task012主诊断固定使用现有BMS后端')
        if args.descriptor_variant not in ('old3d', 'dynamic3d'):
            parser.error('Task012主诊断只允许old3d或dynamic3d')
        if not math.isclose(args.sigma, 0.1, rel_tol=0.0, abs_tol=1e-12):
            parser.error('Task012固定sigma=0.1')
        if not math.isclose(
                args.min_keep_ratio, 0.1, rel_tol=0.0, abs_tol=1e-12):
            parser.error('Task012固定min_keep_ratio=0.1')
        if args.seed != 3407:
            parser.error('Task012固定seed=3407')
        if args.selection_cost_mode != 'coupled':
            parser.error('Task012必须保持selection_cost_mode=coupled')
    if args.task013_run_dir:
        if not args.evaluate_baseline_before_pruning:
            parser.error('Task013必须启用--evaluate_baseline_before_pruning')
        if not args.task013_baseline_cache:
            parser.error('Task013必须提供--task013_baseline_cache')
        if not args.prune_only:
            parser.error('Task013只允许--prune_only，禁止完整微调')
        if args.selection_mode != 'bms':
            parser.error('Task013固定使用现有BMS后端')
        if args.descriptor_variant != 'dynamic3d':
            parser.error('Task013主实验固定使用dynamic3d')
        if not any(math.isclose(
                args.sparsity, value, rel_tol=0.0, abs_tol=1e-12
                ) for value in (0.10, 0.20, 0.30)):
            parser.error('Task013主实验sparsity只允许0.10、0.20或0.30')
        if not math.isclose(args.sigma, 0.1, rel_tol=0.0, abs_tol=1e-12):
            parser.error('Task013固定sigma=0.1')
        if not math.isclose(
                args.min_keep_ratio, 0.1, rel_tol=0.0, abs_tol=1e-12):
            parser.error('Task013固定min_keep_ratio=0.1')
        if args.seed != 3407:
            parser.error('Task013固定seed=3407')
        if args.diagnostic_min_attention_heads != 0:
            parser.error('Task013必须使用原始最小保留规则，禁止min2/min3')
    if args.task014_run_dir:
        if not args.evaluate_baseline_before_pruning:
            parser.error('Task014必须启用--evaluate_baseline_before_pruning')
        if not args.task014_baseline_cache:
            parser.error('Task014必须提供--task014_baseline_cache')
        if not args.prune_only:
            parser.error('Task014只允许--prune_only，禁止完整微调')
        if args.selection_mode not in ('bms', 'functional'):
            parser.error('Task014主实验只允许bms或functional选择后端')
        if args.selection_cost_mode != 'decoupled':
            parser.error('Task014固定沿用Task013 cost-decoupled原则')
        if args.descriptor_variant != 'dynamic3d':
            parser.error('Task014主实验固定使用dynamic3d')
        if not any(math.isclose(
                args.sparsity, value, rel_tol=0.0, abs_tol=1e-12
                ) for value in (0.10, 0.20, 0.30)):
            parser.error('Task014主实验sparsity只允许0.10、0.20或0.30')
        if not math.isclose(args.sigma, 0.1, rel_tol=0.0, abs_tol=1e-12):
            parser.error('Task014固定sigma=0.1')
        if not math.isclose(
                args.min_keep_ratio, 0.1, rel_tol=0.0, abs_tol=1e-12):
            parser.error('Task014固定min_keep_ratio=0.1')
        if args.seed != 3407:
            parser.error('Task014固定seed=3407')
        if args.diagnostic_min_attention_heads != 0:
            parser.error('Task014必须使用原始最小保留规则')
        if args.functional_score != 'domain_average':
            parser.error('Task014必须保持functional_score=domain_average')
    if args.task016_run_dir:
        if not args.evaluate_baseline_before_pruning:
            parser.error('Task016必须启用--evaluate_baseline_before_pruning')
        if not args.task016_baseline_cache:
            parser.error('Task016必须提供--task016_baseline_cache')
        if not args.prune_only:
            parser.error('Task016只允许--prune_only，禁止完整微调')
        if args.selection_mode != 'functional':
            parser.error('Task016固定使用functional选择后端')
        if args.selection_cost_mode != 'decoupled':
            parser.error('Task016固定沿用Task013 cost-decoupled预算原则')
        if args.descriptor_variant != 'dynamic3d':
            parser.error('Task016固定使用dynamic3d')
        if not any(math.isclose(
                args.sparsity, value, rel_tol=0.0, abs_tol=1e-12
                ) for value in (0.10, 0.20, 0.30)):
            parser.error('Task016 sparsity只允许0.10、0.20或0.30')
        if not math.isclose(args.sigma, 0.1, rel_tol=0.0, abs_tol=1e-12):
            parser.error('Task016固定sigma=0.1')
        if not math.isclose(
                args.min_keep_ratio, 0.1, rel_tol=0.0, abs_tol=1e-12):
            parser.error('Task016固定min_keep_ratio=0.1')
        if args.seed != 3407:
            parser.error('Task016固定seed=3407')
        if args.diagnostic_min_attention_heads != 0:
            parser.error('Task016必须使用原始最小保留规则')
    if args.task014_preflight_only:
        if args.selection_mode != 'functional':
            parser.error('Task014 preflight固定使用selection_mode=functional')
        if args.descriptor_variant != 'dynamic3d':
            parser.error('Task014 preflight固定使用dynamic3d')
        if args.task014_run_dir:
            parser.error('Task014 preflight不得与完整run_dir同时启用')
        if not args.prune_only:
            parser.error('Task014 preflight必须启用--prune_only')
    if not (args.task012_run_dir or args.task013_run_dir or
            args.task014_run_dir or args.task016_run_dir) and \
            args.diagnostic_min_attention_heads != 0:
        parser.error('--diagnostic_min_attention_heads仅允许用于Task012诊断')
    args.adv_path = os.path.join(
        args.output_root, args.descriptor_variant, f'seed{args.seed}'
    )
    if not os.path.exists(args.adv_path):
        os.makedirs(args.adv_path)
    return args

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


# --- 路径配置 ---
checkpoint_path = os.environ.get('TRAIN_OUTPUT_DIR', 'outlog/swin_transformer')
train_state_path = checkpoint_path

def resume_training(resume, model, optimizer):
    start_epoch = 1
    if resume > 0:
        start_epoch += resume
        model_path = os.path.join(checkpoint_path, f'checkpoint-{resume}.ckpt')
        # 注意：DataParallel 包装后的模型加载需要 .module
        state_dict = torch.load(model_path)
        model.module.load_state_dict(state_dict)
        
        train_path = os.path.join(train_state_path, f'checkpoint-{resume}_optimizer.ckpt')
        opt_state = torch.load(train_path)
        optimizer.load_state_dict(opt_state['optimizer'])
    return start_epoch


def run_one_epoch(
        epoch, net, optimizer, data_loader, scaler, accum_steps=1,
        use_amp=True):
    net.train()
    total_loss = 0.0
    total_correct = 0
    optimizer.zero_grad(set_to_none=True)
    total_steps = len(data_loader)
    final_group_size = total_steps % accum_steps

    with tqdm.tqdm(data_loader, ncols=0) as pbar:
        for n_iter, (input, target, index) in enumerate(pbar):
            input = input.float().cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            with torch.cuda.amp.autocast(enabled=use_amp):
                logits_student, _ = net(input)
                loss_ce = F.cross_entropy(logits_student, target)
                divisor = accum_steps
                if final_group_size and n_iter >= total_steps - final_group_size:
                    divisor = final_group_size
                loss_for_backward = loss_ce / divisor

            scaler.scale(loss_for_backward).backward()
            should_step = (
                (n_iter + 1) % accum_steps == 0
                or (n_iter + 1) == len(data_loader)
            )
            if should_step:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            total_loss += loss_ce.item()
            predictions = torch.argmax(logits_student, dim=1)
            correct = (predictions == target).sum().item()
            total_correct += correct
            
            pbar.set_description(f"Epoch {epoch} | Loss: {loss_ce.item():.4f} | Acc: {correct/input.size(0):.4f}")
            
    avg_loss = total_loss / len(data_loader)
    avg_accuracy = total_correct / len(data_loader.dataset)
    return avg_loss, avg_accuracy

class AverageMeter(object):
    def __init__(self): self.reset()
    def reset(self): self.val = self.avg = self.sum = self.count = 0
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def validate_rgb(val_loader, net, top1, top5, use_amp=True):
    net.eval()
    with torch.no_grad():
        for input, target, _ in tqdm.tqdm(val_loader, desc="Validating", ncols=0):
            input = input.float().cuda()
            target = target.cuda()
            with torch.cuda.amp.autocast(enabled=use_amp):
                output, _ = net(input)
            prec1, prec5 = accuracy(output, target, topk=(1, 5))
            top1.update(prec1.item(), input.size(0))
            top5.update(prec5.item(), input.size(0))


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
def count_sparsity(model):
    """统计模型各层及总体的稀疏度"""
    total_params = 0
    zero_params = 0
    print("\n--- Layer-wise Sparsity Statistics ---")
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear):
            w = m.weight.data
            layer_params = w.numel()
            layer_zero = (w == 0).sum().item()
            total_params += layer_params
            zero_params += layer_zero
            print(f"Layer: {name:30} | Size: {list(w.shape)} | Sparsity: {100 * layer_zero/layer_params:.2f}%")
    
    total_sparsity = 100 * zero_params / total_params if total_params > 0 else 0
    return total_params, total_sparsity

def log_to_file(file_path, message):
    """简单的日志写入工具"""
    with open(file_path, 'a') as f:
        f.write(message + '\n')
def extract_keep_indices(model):
    """提取模型中所有被剪枝层的索引状态，用于后续恢复"""
    indices = {}
    target_model = model.module if isinstance(model, nn.DataParallel) else model
    for name, m in target_model.named_modules():
        if hasattr(m, 'keep_heads') and m.keep_heads is not None:
            indices[name] = {'type': 'head', 'data': m.keep_heads}
        elif hasattr(m, 'keep_neurons') and m.keep_neurons is not None:
            indices[name] = {'type': 'neuron', 'data': m.keep_neurons}
    return indices

def reset_cuda_peak_memory():
    """重置所有逻辑GPU的峰值分配显存统计。"""
    for gpu_id in range(torch.cuda.device_count()):
        torch.cuda.reset_peak_memory_stats(gpu_id)

def format_cuda_peak_memory():
    """返回每张逻辑GPU的峰值分配/保留显存（MiB）。"""
    values = []
    for gpu_id in range(torch.cuda.device_count()):
        allocated_mib = torch.cuda.max_memory_allocated(gpu_id) / (1024 ** 2)
        reserved_mib = torch.cuda.max_memory_reserved(gpu_id) / (1024 ** 2)
        values.append(
            f"cuda:{gpu_id}=allocated {allocated_mib:.1f} MiB/"
            f"reserved {reserved_mib:.1f} MiB"
        )
    return ", ".join(values)


TDD_SUMMARY_FIELDS = [
    'variant', 'seed', 'descriptor_dim', 'target_sparsity',
    'estimated_actual_sparsity', 'pre_ft_top1', 'pre_ft_top5',
    'best_top1', 'best_top5', 'final_top1', 'final_top5', 'best_epoch',
    'num_groups', 'singleton_ratio', 'removed_heads', 'removed_neurons',
]


def _git_commit():
    result = subprocess.run(
        ['git', 'rev-parse', 'HEAD'], check=False, capture_output=True, text=True
    )
    return result.stdout.strip() if result.returncode == 0 else 'unavailable'


def _atomic_write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + '\n',
        encoding='utf-8'
    )
    temporary.replace(path)


def _update_tdd_summary(output_root, metrics):
    """Update one ``variant x seed`` row without disturbing other runs."""
    output_path = Path(output_root) / 'summary.csv'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    if output_path.is_file():
        with output_path.open('r', encoding='utf-8', newline='') as handle:
            rows = list(csv.DictReader(handle))
    key = (str(metrics['variant']), str(metrics['seed']))
    rows = [
        row for row in rows
        if (str(row.get('variant')), str(row.get('seed'))) != key
    ]
    rows.append({name: metrics.get(name, '') for name in TDD_SUMMARY_FIELDS})
    rows.sort(key=lambda row: (str(row['variant']), int(row['seed'])))

    temporary = output_path.with_suffix('.csv.tmp')
    with temporary.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=TDD_SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(output_path)


def _write_final_metrics(args, metrics):
    output = dict(metrics)
    output['variant'] = args.descriptor_variant
    output['seed'] = args.seed
    output['descriptor_dim'] = 2 if args.descriptor_variant == 'abs_rel' else 3
    output['target_sparsity'] = args.sparsity
    _atomic_write_json(Path(args.adv_path) / 'final_metrics.json', output)
    _update_tdd_summary(args.output_root, output)


def _write_run_metadata(args, gpu_ids, status):
    metadata = {
        'status': status,
        'timestamp_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'git_commit': _git_commit(),
        'command_line': sys.argv,
        'descriptor_variant': args.descriptor_variant,
        'descriptor_dim': 2 if args.descriptor_variant == 'abs_rel' else 3,
        'selection_mode': args.selection_mode,
        'selection_cost_mode': args.selection_cost_mode,
        'functional_score': args.functional_score,
        'functional_descriptor_cache': args.functional_descriptor_cache,
        'seed': args.seed,
        'target_sparsity': args.sparsity,
        'checkpoint_path': str(Path(args.checkpoint_path).expanduser()),
        'train_split': os.environ.get(
            'UCF101_TRAIN_SPLIT', 'dataset/UCF101_Frame/train_rgb_split1.txt'
        ),
        'validation_split': os.environ.get(
            'UCF101_VAL_SPLIT', 'dataset/UCF101_Frame/val_rgb_split1.txt'
        ),
        'sigma': args.sigma,
        'gamma_decay': args.gamma_decay,
        'min_keep_ratio': args.min_keep_ratio,
        'importance_alpha': args.importance_alpha,
        'torch_version': torch.__version__,
        'cuda_version': torch.version.cuda,
        'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
        'gpu_names': [torch.cuda.get_device_name(index) for index in gpu_ids],
        'calibration_device': 'cuda:0',
        'training_devices': [f'cuda:{index}' for index in gpu_ids],
        'calibration_batch_size': args.calib_batch_size,
        'calibration_batches': args.calib_batches,
        'training_batch_size': args.batch_size,
        'amp_enabled': not args.disable_amp,
        'determinism_note': (
            'Seeds and deterministic cuDNN are configured; some CUDA kernels '
            'may still be nondeterministic on the server runtime.'
        ),
    }
    _atomic_write_json(Path(args.adv_path) / 'run_metadata.json', metadata)

if __name__ == '__main__':
    # 1. 环境初始化
    args = arg_parse()

    # 显式--gpu优先；未提供时沿用启动命令环境，二者都没有则使用物理卡0。
    visible_gpus = args.gpu or os.environ.get("CUDA_VISIBLE_DEVICES") or "0"
    os.environ["CUDA_VISIBLE_DEVICES"] = visible_gpus
    set_seed(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("当前环境未检测到CUDA，Video Swin剪枝无法运行")
    gpu_ids = list(range(torch.cuda.device_count()))
    if not gpu_ids:
        raise RuntimeError("CUDA_VISIBLE_DEVICES未暴露任何可用GPU")
    if (args.task012_run_dir or args.task013_run_dir or
            args.task014_run_dir) and len(gpu_ids) < 2:
        task_name = (
            'Task014' if args.task014_run_dir
            else ('Task013' if args.task013_run_dir else 'Task012')
        )
        raise RuntimeError(
            f"{task_name}要求同时暴露两张GPU，例如CUDA_VISIBLE_DEVICES=0,1"
        )
    device = torch.device("cuda:0")
    task011_runtime = None
    task012_runtime = None
    task013_runtime = None
    task014_runtime = None
    task016_runtime = None
    if (args.task011_diagnosis_dir or args.task012_run_dir or
            args.task013_run_dir or args.task014_run_dir or
            args.task016_run_dir):
        import task011_runtime_audit as task011_runtime
    if args.task012_run_dir:
        import task012_runtime_audit as task012_runtime
    if args.task013_run_dir:
        import task013_runtime_audit as task013_runtime
    if args.task014_run_dir:
        import task014_runtime_audit as task014_runtime
    if args.task016_run_dir:
        import task016_runtime_audit as task016_runtime

    print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
    print("Visible GPU count:", torch.cuda.device_count())
    print("DataParallel device_ids:", gpu_ids)
    for gpu_id in gpu_ids:
        print(f"cuda:{gpu_id}: {torch.cuda.get_device_name(gpu_id)}")
    _write_run_metadata(args, gpu_ids, status='in_progress')

    # 2. 日志工具
    if not os.path.exists(args.adv_path): os.makedirs(args.adv_path)
    log_path = os.path.join(args.adv_path, 'pruning_report.txt')
    training_log_path = os.path.join(args.adv_path, 'training_log.txt')
    Path(log_path).write_text('', encoding='utf-8')
    Path(training_log_path).write_text('', encoding='utf-8')

    def logger(msg, training=False):
        print(msg)
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(msg + '\n')
        if training:
            with open(training_log_path, 'a', encoding='utf-8') as f:
                f.write(msg + '\n')

    logger(
        f"==== Task010启动: {args.model} | "
        f"descriptor={args.descriptor_variant} | seed={args.seed} ===="
    )

    # 3. 构建模型 (SwinTransformer3D)
    # 请确保你在 IPvit.py 263 行已按上述方法修改了 meshgrid
    student = SwinTransformer3D(
        patch_size=(2, 4, 4), embed_dim=96, depths=[2, 2, 18, 2],
        num_heads=[3, 6, 12, 24], window_size=(8, 7, 7),
        mlp_ratio=4., qkv_bias=True, patch_norm=True,
        drop_path_rate=0.2, use_checkpoint=True
    ).to(device)

    # 4. 加载权重
    pretrained_path = str(Path(args.checkpoint_path).expanduser())
    checkpoint_metadata = None

    if os.path.exists(pretrained_path):
        checkpoint = torch.load(pretrained_path, map_location=device)
        state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
        new_state_dict = {k.replace('module.', '').replace('backbone.', ''): v for k, v in state_dict.items()}
        msg = student.load_state_dict(new_state_dict, strict=False)
        logger(f"成功加载预训练权重。")
        if args.task011_diagnosis_dir:
            checkpoint_metadata = task011_runtime.checkpoint_audit(
                Path(pretrained_path), checkpoint, new_state_dict, msg, student
            )
        elif (task012_runtime is not None or task013_runtime is not None or
              task014_runtime is not None or task016_runtime is not None):
            runtime = (
                task016_runtime or task014_runtime or task013_runtime
                or task012_runtime
            )
            baseline_cache_path = Path(
                args.task016_baseline_cache
                if task016_runtime is not None else (
                args.task014_baseline_cache
                if task014_runtime is not None else (
                    args.task013_baseline_cache
                    if task013_runtime is not None else args.task012_baseline_cache
                ))
            )
            checkpoint_metadata = runtime.checkpoint_audit_cached(
                Path(pretrained_path),
                new_state_dict,
                msg,
                baseline_cache_path.parent / 'checkpoint_sha256_cache.json',
            )
    else:
        if task011_runtime is not None or args.task014_preflight_only:
            raise FileNotFoundError(pretrained_path)
        logger("警告: 未找到预训练权重！")

    parameter_count_before = sum(parameter.numel() for parameter in student.parameters())

    # 5. 准备数据集
    cfg_path = CONFIG_PATHS[args.model]
    cfg = get_cfg_custom(cfg_path, args.batch_size)

    print(cfg)

    train_split = os.environ.get(
        "UCF101_TRAIN_SPLIT", "dataset/UCF101_Frame/train_rgb_split1.txt"
    )
    validation_split = os.environ.get(
        "UCF101_VAL_SPLIT", "dataset/UCF101_Frame/val_rgb_split1.txt"
    )
    train_loader = get_dataset(train_split, args.batch_size)
    calib_loader = get_dataset(
        train_split,
        args.calib_batch_size
    )
    val_loader = get_dataset(validation_split, args.batch_size)

    # Task011 only: evaluate the unpruned checkpoint with the exact current
    # validation path.  Restore every RNG stream afterwards so this mandatory
    # diagnostic cannot alter the shuffled calibration sample sequence.
    baseline_top1_value = None
    baseline_top5_value = None
    use_amp = not args.disable_amp
    validation_metadata = None
    if task011_runtime is not None:
        validation_metadata = task011_runtime.dataset_audit(val_loader)
        cached_baseline = None
        baseline_cache_identity = None
        if (task012_runtime is not None or task013_runtime is not None or
                task014_runtime is not None or task016_runtime is not None):
            runtime = (
                task016_runtime or task014_runtime or task013_runtime
                or task012_runtime
            )
            if checkpoint_metadata is None:
                raise RuntimeError('受控检查点审计未正确初始化')
            baseline_cache_identity = runtime.baseline_identity(
                args=args,
                checkpoint_metadata=checkpoint_metadata,
                validation_metadata=validation_metadata,
                gpu_ids=gpu_ids,
            )
            baseline_cache_path = Path(
                args.task016_baseline_cache
                if task016_runtime is not None else (
                args.task014_baseline_cache
                if task014_runtime is not None else (
                    args.task013_baseline_cache
                    if task013_runtime is not None else args.task012_baseline_cache
                ))
            )
            cached_baseline = runtime.load_baseline_cache(
                baseline_cache_path, baseline_cache_identity
            )
        if cached_baseline is None:
            rng_state = task011_runtime.capture_rng_state()
            baseline_model = (
                nn.DataParallel(student, device_ids=gpu_ids, output_device=0)
                if len(gpu_ids) > 1 else student
            )
            baseline_top1 = AverageMeter()
            baseline_top5 = AverageMeter()
            validate_rgb(
                val_loader, baseline_model, baseline_top1, baseline_top5,
                use_amp=use_amp
            )
            baseline_top1_value = float(baseline_top1.avg)
            baseline_top5_value = float(baseline_top5.avg)
            if isinstance(baseline_model, nn.DataParallel):
                student = baseline_model.module
            del baseline_model
            task011_runtime.restore_rng_state(rng_state)
            torch.cuda.empty_cache()
            if (task012_runtime is not None or task013_runtime is not None or
                    task014_runtime is not None or task016_runtime is not None):
                runtime = (
                    task016_runtime or task014_runtime or task013_runtime
                    or task012_runtime
                )
                runtime.write_baseline_cache(
                    baseline_cache_path,
                    baseline_cache_identity,
                    baseline_top1_value,
                    baseline_top5_value,
                )
            baseline_source = 'evaluated on all visible GPUs'
        else:
            baseline_top1_value = cached_baseline['baseline_top1']
            baseline_top5_value = cached_baseline['baseline_top5']
            baseline_source = 'reused from exact-compatible cache'
        task_name = (
            'Task016' if task016_runtime is not None
            else ('Task014' if task014_runtime is not None
            else ('Task013' if task013_runtime is not None
            else ('Task012' if task012_runtime is not None else 'Task011')
            ))
        )
        logger(
            f">>> {task_name} unpruned baseline ({baseline_source}) | Top1: "
            f"{baseline_top1_value:.3f}% | Top5: {baseline_top5_value:.3f}%"
        )

    # -----------------------------------------------------
    # 6. InteractionPruner 剪枝阶段 (核心步骤)
    # -----------------------------------------------------
    # Hook几何恢复与BMS在cuda:0完成；微调/验证随后使用所有可见GPU。
    pruner = InteractionPruner(
        student,
        target_sparsity=args.sparsity,
        gamma_decay=args.gamma_decay,
        min_keep_ratio=args.min_keep_ratio,
        sigma=args.sigma,
        importance_alpha=args.importance_alpha,
        selection_mode=args.selection_mode,
        contribution_npz=args.contribution_npz,
        descriptor_variant=args.descriptor_variant,
        descriptor_statistics_path=(
            Path(args.adv_path) / 'descriptor_statistics.csv'
        ),
        diagnostic_min_attention_heads=args.diagnostic_min_attention_heads,
        selection_cost_mode=args.selection_cost_mode,
        selection_trace_enabled=(
            task013_runtime is not None
            or (task014_runtime is not None and args.selection_mode == 'bms')
        ),
        functional_cache_dir=args.functional_cache_dir,
        functional_audit_dir=(
            Path(args.task016_run_dir or args.task014_run_dir)
            if (args.task016_run_dir or args.task014_run_dir)
            else Path(args.output_root) / 'preflight'
        ),
        functional_score=args.functional_score,
        functional_descriptor_cache=args.functional_descriptor_cache,
    )
    captured_groups = {'groups': None}
    calibration_input = calib_loader
    calibration_audit = None
    if task011_runtime is not None:
        captured_groups = task011_runtime.install_bms_group_capture(pruner)
        calibration_audit = task011_runtime.CalibrationAuditLoader(
            calib_loader, args.calib_batches
        )
        calibration_input = calibration_audit
    #pruner.analyze_parameter_distribution(student) 
    #pruner.check_model_structure(student)
    student.eval()
    logger("\n>>> 开始校准与特征收集 (Calibration)...")
    reset_cuda_peak_memory()
    
    # 直接调用这个完整方法，它内部会自动处理 hook 的生命周期
    pruner.run_calibration(
        calibration_input, device=device, num_batches=args.calib_batches
    )

    if args.task014_preflight_only:
        pruner.run_functional_preflight()
        _write_run_metadata(args, gpu_ids, status='task014_preflight_complete')
        logger(">>> Task014 preflight完成；未生成registry、未应用剪枝、未运行验证。")
        raise SystemExit(0)

    logger(">>> 步骤 B: 执行剪枝算法并更新模型索引...")
    # 注意：此时不需要再手动 remove_hooks，因为 run_calibration 应该已经做过了
    pruner.prune()
    
    # 打印剪枝后的报告 (调用你模型里的统计函数)
    actual_report = student.get_detailed_pruning_report()
    logger(f"剪枝完成！理论目标: {args.sparsity}, 实际参数量减少: {actual_report['sparsity']:.2%}")
    logger(f"校准与剪枝峰值显存: {format_cuda_peak_memory()}")

    # -----------------------------------------------------
    # 7. 微调阶段 (Recovery Fine-tuning)
    # -----------------------------------------------------
    # 剪枝后包装 DataParallel
    if len(gpu_ids) > 1:
        student = nn.DataParallel(
            student, device_ids=gpu_ids, output_device=0
        )

    # 同一参数预算下的微调前精度直接衡量剪枝决策质量。
    pre_ft_top1 = AverageMeter()
    pre_ft_top5 = AverageMeter()
    validate_rgb(
        val_loader, student, pre_ft_top1, pre_ft_top5, use_amp=use_amp
    )
    removed_heads = (
        actual_report['original_heads'] - actual_report['remaining_heads']
    )
    removed_neurons = (
        actual_report['original_neurons'] - actual_report['remaining_neurons']
    )
    logger(
        f">>> Pre-finetune | Top1: {pre_ft_top1.avg:.3f}% | "
        f"Top5: {pre_ft_top5.avg:.3f}% | removed_heads={removed_heads} | "
        f"removed_neurons={removed_neurons}"
    )

    task011_audit = None
    task012_audit = None
    task013_audit = None
    task014_audit = None
    task016_audit = None
    if args.task011_diagnosis_dir:
        if checkpoint_metadata is None or calibration_audit is None:
            raise RuntimeError('Task011运行审计未正确初始化')
        task011_audit = task011_runtime.write_controlled_run_artifacts(
            diagnosis_dir=Path(args.task011_diagnosis_dir),
            args=args,
            model=student,
            pruner=pruner,
            actual_report=actual_report,
            baseline_top1=baseline_top1_value,
            baseline_top5=baseline_top5_value,
            pre_ft_top1=float(pre_ft_top1.avg),
            pre_ft_top5=float(pre_ft_top5.avg),
            parameter_count_before=parameter_count_before,
            checkpoint_metadata=checkpoint_metadata,
            calibration_loader=calibration_audit,
            validation_metadata=validation_metadata,
            captured_groups=captured_groups,
        )
    elif task012_runtime is not None:
        if checkpoint_metadata is None or calibration_audit is None:
            raise RuntimeError('Task012运行审计未正确初始化')
        task012_audit = task012_runtime.write_task012_run_artifacts(
            run_dir=Path(args.task012_run_dir),
            args=args,
            model=student,
            pruner=pruner,
            actual_report=actual_report,
            baseline_top1=baseline_top1_value,
            baseline_top5=baseline_top5_value,
            pre_ft_top1=float(pre_ft_top1.avg),
            pre_ft_top5=float(pre_ft_top5.avg),
            parameter_count_before=parameter_count_before,
            checkpoint_metadata=checkpoint_metadata,
            calibration_loader=calibration_audit,
            validation_metadata=validation_metadata,
            captured_groups=captured_groups,
        )
    elif task013_runtime is not None:
        if checkpoint_metadata is None or calibration_audit is None:
            raise RuntimeError('Task013运行审计未正确初始化')
        task013_audit = task013_runtime.write_task013_run_artifacts(
            run_dir=Path(args.task013_run_dir),
            args=args,
            model=student,
            pruner=pruner,
            actual_report=actual_report,
            baseline_top1=baseline_top1_value,
            baseline_top5=baseline_top5_value,
            pre_ft_top1=float(pre_ft_top1.avg),
            pre_ft_top5=float(pre_ft_top5.avg),
            parameter_count_before=parameter_count_before,
            checkpoint_metadata=checkpoint_metadata,
            calibration_loader=calibration_audit,
            validation_metadata=validation_metadata,
            captured_groups=captured_groups,
        )
    elif task014_runtime is not None:
        if checkpoint_metadata is None or calibration_audit is None:
            raise RuntimeError('Task014运行审计未正确初始化')
        task014_audit = task014_runtime.write_task014_run_artifacts(
            run_dir=Path(args.task014_run_dir),
            args=args,
            model=student,
            pruner=pruner,
            actual_report=actual_report,
            baseline_top1=baseline_top1_value,
            baseline_top5=baseline_top5_value,
            pre_ft_top1=float(pre_ft_top1.avg),
            pre_ft_top5=float(pre_ft_top5.avg),
            parameter_count_before=parameter_count_before,
            checkpoint_metadata=checkpoint_metadata,
            calibration_loader=calibration_audit,
            validation_metadata=validation_metadata,
            captured_groups=captured_groups,
        )
    elif task016_runtime is not None:
        if checkpoint_metadata is None or calibration_audit is None:
            raise RuntimeError('Task016运行审计未正确初始化')
        task016_audit = task016_runtime.write_task016_run_artifacts(
            run_dir=Path(args.task016_run_dir),
            args=args,
            model=student,
            pruner=pruner,
            actual_report=actual_report,
            baseline_top1=baseline_top1_value,
            baseline_top5=baseline_top5_value,
            pre_ft_top1=float(pre_ft_top1.avg),
            pre_ft_top5=float(pre_ft_top5.avg),
            parameter_count_before=parameter_count_before,
            checkpoint_metadata=checkpoint_metadata,
            calibration_loader=calibration_audit,
            validation_metadata=validation_metadata,
            captured_groups=captured_groups,
        )

    base_metrics = {
        'status': 'pruned_only' if args.prune_only else 'training',
        'estimated_actual_sparsity': actual_report['sparsity'],
        'estimated_parameter_reduction': actual_report['sparsity'],
        'pre_ft_top1': pre_ft_top1.avg,
        'pre_ft_top5': pre_ft_top5.avg,
        'removed_heads': removed_heads,
        'removed_neurons': removed_neurons,
        'num_groups': '',
        'singleton_ratio': '',
    }
    if task011_audit is not None:
        base_metrics.update({
            'baseline_top1': baseline_top1_value,
            'baseline_top5': baseline_top5_value,
            'estimated_budget_sparsity': task011_audit[
                'parameter_budget'
            ]['estimated_budget_sparsity'],
            'physical_numel_sparsity': task011_audit[
                'parameter_budget'
            ]['physical_numel_sparsity'],
        })
    if task012_audit is not None:
        base_metrics.update({
            'baseline_top1': baseline_top1_value,
            'baseline_top5': baseline_top5_value,
            'estimated_budget_sparsity': task012_audit[
                'estimated_budget_sparsity'
            ],
            'physical_numel_sparsity': task012_audit[
                'physical_numel_sparsity'
            ],
            'num_groups': task012_audit['num_groups'],
            'singleton_ratio': task012_audit['singleton_ratio'],
        })
    if task013_audit is not None:
        base_metrics.update({
            'baseline_top1': baseline_top1_value,
            'baseline_top5': baseline_top5_value,
            'estimated_budget_sparsity': task013_audit[
                'estimated_budget_sparsity'
            ],
            'physical_numel_sparsity': task013_audit[
                'physical_numel_sparsity'
            ],
            'num_groups': task013_audit['num_groups'],
            'singleton_ratio': task013_audit['singleton_ratio'],
        })
    if task014_audit is not None:
        base_metrics.update({
            'baseline_top1': baseline_top1_value,
            'baseline_top5': baseline_top5_value,
            'estimated_budget_sparsity': task014_audit[
                'estimated_budget_sparsity'
            ],
            'physical_numel_sparsity': task014_audit[
                'physical_numel_sparsity'
            ],
            'num_groups': task014_audit['num_bms_domains'],
            'singleton_ratio': task014_audit['singleton_ratio'],
        })
    if task016_audit is not None:
        base_metrics.update({
            'baseline_top1': baseline_top1_value,
            'baseline_top5': baseline_top5_value,
            'estimated_budget_sparsity': task016_audit[
                'estimated_budget_sparsity'
            ],
            'physical_numel_sparsity': task016_audit[
                'physical_numel_sparsity'
            ],
            'num_groups': task016_audit['num_bms_domains'],
            'singleton_ratio': task016_audit['singleton_ratio'],
        })
    if args.prune_only:
        _write_final_metrics(args, {
            **base_metrics,
            'best_top1': '', 'best_top5': '',
            'final_top1': '', 'final_top5': '', 'best_epoch': '',
        })
        _write_run_metadata(args, gpu_ids, status='pruned_only_complete')
        logger(">>> --prune_only完成；未运行微调，不生成或推断微调结果。")
        raise SystemExit(0)

    # 仅更新未被裁剪且 requires_grad=True 的参数
    optimizer = torch.optim.SGD(
        [p for p in student.parameters() if p.requires_grad], 
        lr=cfg.CONFIG.TRAIN.LR * 0.1, # 微调建议降低学习率
        momentum=0.9,
        weight_decay=cfg.CONFIG.TRAIN.W_DECAY
    )
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    logger(
        f"显存控制: train_batch={args.batch_size}, "
        f"calib_batch={args.calib_batch_size}, accum_steps={args.accum_steps}, "
        f"AMP={use_amp}, checkpoint=True, devices={gpu_ids}"
        , training=True
    )

# 初始化用于记录趋势的容器
    best_acc = float('-inf')
    best_top5 = 0.0
    best_epoch = None
    final_top1 = None
    final_top5 = None
    training_history = [] 

    for epoch in range(1, cfg.CONFIG.TRAIN.EPOCH_NUM + 1):
        # 1. 执行训练
        reset_cuda_peak_memory()
        train_loss, train_acc = run_one_epoch(
            epoch, student, optimizer, train_loader, scaler,
            accum_steps=args.accum_steps, use_amp=use_amp
        )

        # 2. 执行验证
        top1 = AverageMeter()
        top5 = AverageMeter()
        validate_rgb(val_loader, student, top1, top5, use_amp=use_amp)

        # 3. 获取当前稀疏度 (假设 actual_report 是在 prune 函数后生成的)
        current_sparsity = actual_report['sparsity']

        # 4. 增强型日志输出：同时展示 Top1, Top5 和 稀疏度
        logger(f"Epoch [{epoch:02d}] "
               f"| Top1: {top1.avg:6.3f}% "
               f"| Top5: {top5.avg:6.3f}% "
               f"| 稀疏度: {current_sparsity:7.2%}", training=True)
        logger(
            f"Epoch [{epoch:02d}] 峰值显存: {format_cuda_peak_memory()}",
            training=True
        )
        final_top1 = top1.avg
        final_top5 = top5.avg

        # 记录到历史数据（可选，便于导出 CSV 或绘图）
        training_history.append({
            'epoch': epoch,
            'top1': top1.avg,
            'top5': top5.avg,
            'sparsity': current_sparsity
        })

        # 5. 保存逻辑：增加 top5 的持久化
        if top1.avg > best_acc:
            best_acc = top1.avg
            best_top5 = top5.avg
            best_epoch = epoch
            save_checkpoint = {
                'epoch': epoch,
                'state_dict': student.module.state_dict() if isinstance(student, nn.DataParallel) else student.state_dict(),
                'keep_indices': extract_keep_indices(student), 
                'top1': top1.avg,
                'top5': top5.avg,      # 保存最佳 Top5
                'sparsity': current_sparsity, # 保存对应的稀疏度
                'history': training_history   # 保存训练全过程记录
            }
            
            save_path = os.path.join(args.adv_path, 'swin_pruned_best.pth')
            torch.save(save_checkpoint, save_path)
            Path(args.adv_path, 'best_checkpoint_path.txt').write_text(
                str(Path(save_path).resolve()) + '\n', encoding='utf-8'
            )
            logger(
                f">>> 已保存最佳模型! [Epoch {epoch}] "
                f"Top1: {top1.avg:.2f}% | Top5: {top5.avg:.2f}%",
                training=True
            )

    final_metrics = {
        **base_metrics,
        'status': 'complete',
        'best_top1': best_acc,
        'best_top5': best_top5,
        'final_top1': final_top1,
        'final_top5': final_top5,
        'best_epoch': best_epoch,
    }
    _write_final_metrics(args, final_metrics)
    _write_run_metadata(args, gpu_ids, status='complete')
    logger(
        f"任务结束。最高 Top1: {best_acc:.2f}% (epoch {best_epoch}); "
        f"最终 Top1: {final_top1:.2f}%",
        training=True
    )
