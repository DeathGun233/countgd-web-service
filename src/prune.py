import torch
import torch.nn as nn
import os
import copy

def load_state_dict(checkpoint_path):
    """加载state_dict"""
    print(f"加载checkpoint: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    if isinstance(checkpoint, dict):
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
            print("  ✓ 从checkpoint['model']加载state_dict")
        elif 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            print("  ✓ 从checkpoint['model_state_dict']加载state_dict")
        else:
            state_dict = checkpoint
            print("  ✓ checkpoint本身是state_dict")
    else:
        raise ValueError("Checkpoint格式不正确")
    
    return state_dict

def get_state_dict_size(state_dict):
    """计算state_dict大小（MB）"""
    # 临时保存以计算大小
    temp_path = 'temp_state_dict.pth'
    torch.save(state_dict, temp_path)
    size_mb = os.path.getsize(temp_path) / (1024 * 1024)
    os.remove(temp_path)
    return size_mb

def count_parameters_from_state_dict(state_dict):
    """从state_dict统计参数量"""
    total = 0
    for key, tensor in state_dict.items():
        if tensor.dtype in [torch.float32, torch.float16, torch.int64, torch.int32]:
            total += tensor.numel()
    return total

def prune_state_dict(state_dict, amount=0.3):
    """
    直接对state_dict进行剪枝
    
    Args:
        state_dict: 模型的state_dict
        amount: 剪枝比例
    
    Returns:
        pruned_state_dict: 剪枝后的state_dict
    """
    print(f"\n开始剪枝（剪枝比例: {amount*100:.1f}%）")
    
    # 复制state_dict
    pruned_state_dict = copy.deepcopy(state_dict)
    
    # 收集所有需要剪枝的权重
    all_weights = []
    weight_keys = []
    
    for key, param in state_dict.items():
        # 只剪枝卷积层和全连接层的权重
        if ('weight' in key and 
            param.dim() >= 2 and  # 至少是2D张量
            'norm' not in key.lower() and  # 排除normalization层
            'bn' not in key.lower()):      # 排除batch norm
            
            all_weights.append(param.abs().flatten())
            weight_keys.append(key)
            
            if len(weight_keys) <= 10:
                print(f"  [{len(weight_keys)}] {key:50s} - shape: {list(param.shape)}")
    
    if len(weight_keys) > 10:
        print(f"  ... 还有 {len(weight_keys) - 10} 层（已省略）")
    
    print(f"\n总共将剪枝 {len(weight_keys)} 个权重张量")
    
    # 全局L1剪枝
    if len(all_weights) > 0:
        # 合并所有权重
        all_weights_concat = torch.cat(all_weights)
        
        # 计算全局阈值
        threshold_index = int(amount * len(all_weights_concat))
        threshold = torch.sort(all_weights_concat)[0][threshold_index]
        
        print(f"全局阈值: {threshold:.6f}")
        
        # 对每个权重应用阈值
        total_zeros = 0
        total_params = 0
        
        for key in weight_keys:
            param = pruned_state_dict[key]
            
            # 创建mask
            mask = (param.abs() > threshold).float()
            
            # 应用mask
            pruned_state_dict[key] = param * mask
            
            # 统计
            zeros = (pruned_state_dict[key] == 0).sum().item()
            total = param.numel()
            total_zeros += zeros
            total_params += total
        
        global_sparsity = 100. * total_zeros / total_params
        print(f"✓ 剪枝完成，全局稀疏度: {global_sparsity:.2f}%")
    
    return pruned_state_dict

def analyze_sparsity(state_dict, verbose=False):
    """分析state_dict的稀疏度"""
    total_zeros = 0
    total_params = 0
    layer_stats = []
    
    for key, param in state_dict.items():
        if 'weight' in key and param.dim() >= 2:
            zeros = (param == 0).sum().item()
            total = param.numel()
            
            total_zeros += zeros
            total_params += total
            
            sparsity = 100. * zeros / total
            layer_stats.append((key, sparsity))
            
            if verbose:
                print(f"  {key:50s} - 稀疏度: {sparsity:.2f}%")
    
    global_sparsity = 100. * total_zeros / total_params if total_params > 0 else 0
    
    if not verbose and len(layer_stats) > 0:
        print("\n前5层稀疏度:")
        for key, sparsity in layer_stats[:5]:
            print(f"  {key:50s} - {sparsity:.2f}%")
        
        if len(layer_stats) > 10:
            print(f"\n  ... 中间 {len(layer_stats)-10} 层已省略 ...")
        
        print("\n后5层稀疏度:")
        for key, sparsity in layer_stats[-5:]:
            print(f"  {key:50s} - {sparsity:.2f}%")
    
    print(f"\n全局稀疏度: {global_sparsity:.2f}%")
    return global_sparsity

def main():
    # ==================== 配置 ====================
    original_checkpoint_path = 'models/checkpoint_best_regular.pth'
    pruned_checkpoint_path = 'models/pruned_checkpoint.pth'
    prune_amount = 0.3
    verbose = False
    # =============================================
    
    print("=" * 70)
    print("CountGD State Dict 剪枝工具".center(70))
    print("=" * 70)
    
    # 1. 检查文件
    if not os.path.exists(original_checkpoint_path):
        print(f"\n✗ 错误: 找不到checkpoint文件")
        print(f"   路径: {original_checkpoint_path}")
        return
    
    print(f"\n✓ 找到checkpoint文件")
    print(f"   路径: {original_checkpoint_path}")
    print(f"   大小: {os.path.getsize(original_checkpoint_path) / 1024**2:.2f} MB")
    
    # 2. 加载state_dict
    print(f"\n{'='*70}")
    print("步骤 1/4: 加载state_dict")
    print('='*70)
    
    try:
        state_dict = load_state_dict(original_checkpoint_path)
        original_size = get_state_dict_size(state_dict)
        total_params = count_parameters_from_state_dict(state_dict)
        
        print(f"\nState Dict信息:")
        print(f"  大小:         {original_size:>10.2f} MB")
        print(f"  参数总量:     {total_params:>15,}")
        print(f"  权重张量数:   {len(state_dict):>15,}")
        
    except Exception as e:
        print(f"\n✗ 加载失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 3. 执行剪枝
    print(f"\n{'='*70}")
    print("步骤 2/4: 执行剪枝")
    print('='*70)
    
    pruned_state_dict = prune_state_dict(state_dict, amount=prune_amount)
    
    # 4. 分析稀疏度
    print(f"\n{'='*70}")
    print("步骤 3/4: 分析稀疏度")
    print('='*70)
    
    sparsity = analyze_sparsity(pruned_state_dict, verbose=verbose)
    
    # 5. 保存
    print(f"\n{'='*70}")
    print("步骤 4/4: 保存剪枝后的checkpoint")
    print('='*70)
    
    # 创建输出目录
    os.makedirs(os.path.dirname(pruned_checkpoint_path), exist_ok=True)
    
    # 加载原始checkpoint以保留其他信息
    original_checkpoint = torch.load(original_checkpoint_path, map_location='cpu')
    
    # 替换model为剪枝后的state_dict
    if isinstance(original_checkpoint, dict):
        if 'model' in original_checkpoint:
            original_checkpoint['model'] = pruned_state_dict
        elif 'model_state_dict' in original_checkpoint:
            original_checkpoint['model_state_dict'] = pruned_state_dict
        else:
            original_checkpoint = pruned_state_dict
    
    # 保存完整checkpoint
    torch.save(original_checkpoint, pruned_checkpoint_path)
    print(f"✓ Checkpoint已保存: {pruned_checkpoint_path}")
    
    # 也单独保存state_dict（便于后续使用）
    state_dict_only_path = pruned_checkpoint_path.replace('.pth', '_state_dict_only.pth')
    torch.save(pruned_state_dict, state_dict_only_path)
    print(f"✓ State dict已保存: {state_dict_only_path}")
    
    # 计算剪枝后大小
    pruned_size = get_state_dict_size(pruned_state_dict)
    total_params_after = count_parameters_from_state_dict(pruned_state_dict)
    
    # 6. 总结
    print(f"\n{'='*70}")
    print("剪枝完成！".center(70))
    print('='*70)
    
    print(f"\n📊 剪枝统计:")
    print(f"  原始:")
    print(f"    - 大小:       {original_size:>10.2f} MB")
    print(f"    - 参数量:     {total_params:>15,}")
    
    print(f"\n  剪枝后:")
    print(f"    - 大小:       {pruned_size:>10.2f} MB")
    print(f"    - 参数量:     {total_params_after:>15,}")
    
    print(f"\n  改进:")
    print(f"    - 大小压缩率:   {(1 - pruned_size/original_size)*100:>6.1f}%")
    print(f"    - 全局稀疏度:   {sparsity:>6.1f}%")
    
    print(f"\n💡 下一步:")
    print(f"  1. 测试剪枝后的模型性能")
    print(f"  2. 运行量化: python src/quantize.py")
    print(f"  3. 如需恢复精度，可以fine-tune剪枝后的模型")
    
    print(f"\n📁 输出文件:")
    print(f"  - {pruned_checkpoint_path}")
    print(f"  - {state_dict_only_path}")

if __name__ == '__main__':
    main()