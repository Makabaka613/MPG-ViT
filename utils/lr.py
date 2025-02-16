# --------------------------------------------------------
# References:
#    mae: https://github.com/facebookresearch/mae/blob/main/util/lr_sched.py
# --------------------------------------------------------

import math

def adjust_learning_rate(optimizer, epoch, args):    #adjust_learning_rate 函数用于根据训练进度动态调整学习率。这种调整策略结合了学习率预热（warmup）和余弦退火（cosine annealing），以便在训练过程中平滑地改变学习率。
    """Decay the learning rate with half-cycle cosine after warmup"""
    if epoch < args.warmup_epochs:
        lr = args.lr * epoch / args.warmup_epochs 
    else:
        lr = args.min_lr + (args.lr - args.min_lr) * 0.5 * \
            (1. + math.cos(math.pi * (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)))
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr
