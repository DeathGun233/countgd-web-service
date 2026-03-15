import torch
import torch.quantization as quantization
import os
import copy

def load_state_dict(checkpoint_path):
    """加载state_dict"""
    print(f"加载checkpoint: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    if isinstance(checkpoint, dict):
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        elif 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint
    
    return state_dict

def quantize_state_dict(state_dict):
    """
    对state_dict进行简单量化
    FP32 → FP16（半精度）
    
    注：完整的INT8量化需要模型对象，这里先做FP16量化
    """
    print("\n开始量化（FP32 → FP16）...")
    
    quantized_state_dict = {}
    
    converted_count = 0
    skipped_count = 0
    
    for key, param in state_dict.items():
        # 只量化浮点数张量
        if param.dtype == torch.float32:
            quantized_state_dict[key] = param.half()  # FP32 → FP16
            converted_count += 1
        else:
            quantized_state_dict[key] = param
            skipped_count += 1
    
    print(f"✓ 量化完成")
    print(f"  转换: {converted_count} 个FP32张量 → FP16")
    print(f"  跳过: {skipped_count} 个非FP32张量")
    
    return quantized_state_dict

def get_state_dict_size(state_dict):
    """计算state_dict大小（MB）"""
    temp_path = 'temp_state_dict.pth'
    torch.save(state_dict, temp_path)
    size_mb = os.path.getsize(temp_path) / (1024 * 1024)
    os.remove(temp_path)
    return size_mb

def main():
    # ==================== 配置 ====================
    input_checkpoint_path = 'models/pruned_checkpoint.pth'
    output_checkpoint_path = 'models/quantized_checkpoint.pth'
    # =============================================
    
    print("=" * 70)
    print("CountGD 量化工具（FP32 → FP16）".center(70))
    print("=" * 70)
    
    # 1. 检查文件
    if not os.path.exists(input_checkpoint_path):
        print(f"\n✗ 错误: 找不到checkpoint文件")
        print(f"   路径: {input_checkpoint_path}")
        print(f"\n请先运行剪枝: python src/prune_state_dict.py")
        return
    
    print(f"\n✓ 找到checkpoint文件")
    print(f"   路径: {input_checkpoint_path}")
    print(f"   大小: {os.path.getsize(input_checkpoint_path) / 1024**2:.2f} MB")
    
    # 2. 加载state_dict
    print(f"\n{'='*70}")
    print("步骤 1/3: 加载state_dict")
    print('='*70)
    
    state_dict = load_state_dict(input_checkpoint_path)
    original_size = get_state_dict_size(state_dict)
    
    print(f"  原始大小: {original_size:.2f} MB")
    
    # 3. 执行量化
    print(f"\n{'='*70}")
    print("步骤 2/3: 执行量化")
    print('='*70)
    
    quantized_state_dict = quantize_state_dict(state_dict)
    
    # 4. 保存
    print(f"\n{'='*70}")
    print("步骤 3/3: 保存量化后的checkpoint")
    print('='*70)
    
    # 创建输出目录
    os.makedirs(os.path.dirname(output_checkpoint_path), exist_ok=True)
    
    # 加载原始checkpoint以保留其他信息
    original_checkpoint = torch.load(input_checkpoint_path, map_location='cpu')
    
    # 替换model
    if isinstance(original_checkpoint, dict):
        if 'model' in original_checkpoint:
            original_checkpoint['model'] = quantized_state_dict
        elif 'model_state_dict' in original_checkpoint:
            original_checkpoint['model_state_dict'] = quantized_state_dict
        else:
            original_checkpoint = quantized_state_dict
    
    # 保存
    torch.save(original_checkpoint, output_checkpoint_path)
    print(f"✓ Checkpoint已保存: {output_checkpoint_path}")
    
    # 单独保存state_dict
    state_dict_only_path = output_checkpoint_path.replace('.pth', '_state_dict_only.pth')
    torch.save(quantized_state_dict, state_dict_only_path)
    print(f"✓ State dict已保存: {state_dict_only_path}")
    
    # 计算量化后大小
    quantized_size = get_state_dict_size(quantized_state_dict)
    
    # 5. 总结
    print(f"\n{'='*70}")
    print("量化完成！".center(70))
    print('='*70)
    
    print(f"\n📊 量化统计:")
    print(f"  原始 (剪枝后FP32):")
    print(f"    - 大小:     {original_size:>10.2f} MB")
    
    print(f"\n  量化后 (FP16):")
    print(f"    - 大小:     {quantized_size:>10.2f} MB")
    
    print(f"\n  改进:")
    print(f"    - 压缩率:   {(1 - quantized_size/original_size)*100:>6.1f}%")
    
    print(f"\n💡 下一步:")
    print(f"  1. 测试量化后的模型性能")
    print(f"  2. 转换为ONNX: python src/convert_onnx.py")
    
    print(f"\n📁 输出文件:")
    print(f"  - {output_checkpoint_path}")
    print(f"  - {state_dict_only_path}")
    
    # 记录指标
    print(f"\n{'='*70}")
    print("📝 简历数据记录")
    print('='*70)
    print(f"模型量化:")
    print(f"  - 量化类型: FP32 → FP16")
    print(f"  - 大小变化: {original_size:.0f}MB → {quantized_size:.0f}MB")
    print(f"  - 压缩率: {(1 - quantized_size/original_size)*100:.1f}%")

if __name__ == '__main__':
    main()