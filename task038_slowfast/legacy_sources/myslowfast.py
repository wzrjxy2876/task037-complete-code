import argparse
import os
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
from IPslowfast import slowfast_16x8_resnet101_kinetics400,InteractionPruner
# from wandaslowfast import slowfast_16x8_resnet101_kinetics400,WandaPruner
# from ITslowfast import slowfast_16x8_resnet101_kinetics400,ITPruner
# from Divslowfast import slowfast_16x8_resnet101_kinetics400,DivPruner
# from MDPslowfast import slowfast_16x8_resnet101_kinetics400,MDPPruner
# from plugslowfast import slowfast_16x8_resnet101_kinetics400,PlugPruner
# from Slimgptslowfast import slowfast_16x8_resnet101_kinetics400,SlimGPTPruner


# --- 路径配置 ---
checkpoint_path = '/data/jixinye25/AAAwork1/outlog/swin_transformer/'
train_state_path = '/data/jixinye25/AAAwork1/outlog/swin_transformer/'

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

def validate_rgb(val_loader, net, top1, top5):
    net.eval()
    with torch.no_grad():
        for input, target, _ in tqdm.tqdm(val_loader, desc="Validating", ncols=0):
            input = input.float().cuda()
            target = target.cuda()
            output= net(input)
            prec1, prec5 = accuracy(output, target, topk=(1, 5))
            top1.update(prec1.item(), input.size(0))
            top5.update(prec5.item(), input.size(0))

def run_one_epoch(epoch, net, optimizer, data_loader):
    net.train()
    total_loss = 0.0
    total_correct = 0
    
    with tqdm.tqdm(data_loader, ncols=0) as pbar:
        for n_iter, (input, target, index) in enumerate(pbar):
            input = input.float().cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            
            optimizer.zero_grad()
            logits_student= net(input)
            loss_ce = F.cross_entropy(logits_student, target)
            
            loss_ce.backward()
            optimizer.step()
            
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

def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, default='0,1,2,3')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--model', type=str, required=True, help='e.g. swin_base')
    parser.add_argument('--file_prefix', type=str, default='')
    parser.add_argument('--sparsity', type=float, default=0.6, help='剪枝比例')
    args = parser.parse_args()
    args.adv_path = os.path.join(OPT_PATH, f'UCF-{args.model}_{args.file_prefix}')
    if not os.path.exists(args.adv_path): os.makedirs(args.adv_path)
    return args

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
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

if __name__ == '__main__':
    # 1. 环境初始化
    args = arg_parse()
    
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"#改
    gpu_ids = [0, 1,2,3]
    set_seed(3407)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2. 日志工具
    if not os.path.exists(args.adv_path): os.makedirs(args.adv_path)
    log_file_name = f"prune_report_{args.model}_{time.strftime('%m%d_%H%M')}.txt"
    log_path = os.path.join(args.adv_path, log_file_name)

    def logger(msg):
        print(msg)
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(msg + '\n')

    logger(f"==== 任务启动: {args.model} | Interaction 结构化剪枝 ====")


    student = slowfast_16x8_resnet101_kinetics400(num_classes=101).to(device)

    # 4. 加载权重
    pretrained_path = '/home/jixinye25/jxy_work1/pretrained/slowfast-teacher-ucf101.ckpt'
    if os.path.exists(pretrained_path):
        checkpoint = torch.load(pretrained_path, map_location=device)
        state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
        new_state_dict = {k.replace('module.', '').replace('backbone.', ''): v for k, v in state_dict.items()}
        msg = student.load_state_dict(new_state_dict, strict=False)
        logger(f"成功加载预训练权重。")
    else:
        logger("警告: 未找到预训练权重！")

    # 5. 准备数据集
    cfg_path = CONFIG_PATHS[args.model]
    cfg = get_cfg_custom(cfg_path, args.batch_size)
    train_loader = get_dataset(cfg.CONFIG.DATA.TRAIN_ANNO_PATH, args.batch_size)
    val_loader = get_dataset(cfg.CONFIG.DATA.VAL_ANNO_PATH, args.batch_size)
    #student.get_parameter_distribution()
    # -----------------------------------------------------
    # 6. InteractionPruner 剪枝阶段 (核心步骤)
    # -----------------------------------------------------
    # 必须在单卡模式下进行剪枝计算
    pruner = InteractionPruner(student, target_sparsity=args.sparsity)
    #pruner.analyze_parameter_distribution(student) 
    #pruner.check_model_structure(student)
    student.eval()
    logger("\n>>> 开始校准与特征收集 (Calibration)...")
    
    # 直接调用这个完整方法，它内部会自动处理 hook 的生命周期
    pruner.run_calibration(train_loader, device=device)

    logger(">>> 步骤 B: 执行剪枝算法并更新模型索引...")
    # 注意：此时不需要再手动 remove_hooks，因为 run_calibration 应该已经做过了
    pruner.prune()
    
    pruner.get_stats()
    # 打印剪枝后的报告 (调用你模型里的统计函数)
    actual_report = student.get_detailed_pruning_report()
    logger(f"剪枝完成！理论目标: {args.sparsity}, 实际参数量减少: {actual_report['sparsity']:.2%}")

    # -----------------------------------------------------
    # 7. 微调阶段 (Recovery Fine-tuning)
    # -----------------------------------------------------
    # 剪枝后包装 DataParallel
    student = nn.DataParallel(student, device_ids=gpu_ids)

    # 仅更新未被裁剪且 requires_grad=True 的参数
    optimizer = torch.optim.SGD(
        [p for p in student.parameters() if p.requires_grad], 
        lr=cfg.CONFIG.TRAIN.LR * 0.1, # 微调建议降低学习率
        momentum=0.9,
        weight_decay=cfg.CONFIG.TRAIN.W_DECAY
    )

# 初始化用于记录趋势的容器
    best_acc = 0.0
    training_history = [] 

    for epoch in range(1, cfg.CONFIG.TRAIN.EPOCH_NUM + 1):
        # 1. 执行训练
        train_loss, train_acc = run_one_epoch(epoch, student, optimizer, train_loader)

        # 2. 执行验证
        top1 = AverageMeter()
        top5 = AverageMeter()
        validate_rgb(val_loader, student, top1, top5)

        # 3. 获取当前稀疏度 (假设 actual_report 是在 prune 函数后生成的)
        current_sparsity = actual_report['sparsity']

        # 4. 增强型日志输出：同时展示 Top1, Top5 和 稀疏度
        logger(f"Epoch [{epoch:02d}] "
               f"| Top1: {top1.avg:6.3f}% "
               f"| Top5: {top5.avg:6.3f}% "
               f"| 稀疏度: {current_sparsity:7.2%}"
               )
        #

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
            save_checkpoint = {
                'epoch': epoch,
                'state_dict': student.module.state_dict() if isinstance(student, nn.DataParallel) else student.state_dict(),
                'keep_indices': extract_keep_indices(student), 
                'top1': top1.avg,
                'top5': top5.avg,      # 保存最佳 Top5
                'sparsity': current_sparsity, # 保存对应的稀疏度
                'history': training_history   # 保存训练全过程记录
            }
            
            save_path = os.path.join(args.adv_path, 'slowfast_InteractionPruner_best.pth')
            torch.save(save_checkpoint, save_path)
            logger(f">>> 已保存最佳模型! [Epoch {epoch}] Top1: {top1.avg:.2f}% | Top5: {top5.avg:.2f}%")

    logger(f"任务结束。最高 Top1 准确率: {best_acc:.2f}%")