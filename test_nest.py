import torch
import sys
sys.path.insert(0, '.')

from util.misc import nested_tensor_from_tensor_list

# 测试不同的输入
test_cases = [
    # (描述, 输入)
    ("单个4D张量", torch.randn(1, 3, 800, 600)),
    ("张量列表（单个元素）", [torch.randn(1, 3, 800, 600)]),
    ("张量列表（多个元素）", [torch.randn(1, 3, 800, 600), torch.randn(1, 3, 800, 600)]),
    ("3D张量", torch.randn(3, 800, 600)),
    ("3D张量列表", [torch.randn(3, 800, 600)]),
]

for desc, tensor in test_cases:
    print(f"\n测试: {desc}")
    print(f"  输入类型: {type(tensor)}")
    if isinstance(tensor, torch.Tensor):
        print(f"  形状: {tensor.shape}")
    elif isinstance(tensor, list):
        print(f"  列表长度: {len(tensor)}")
        if tensor:
            print(f"  元素形状: {tensor[0].shape}")
    
    try:
        result = nested_tensor_from_tensor_list(tensor)
        print(f"  ✅ 成功")
        print(f"    tensors shape: {result.tensors.shape}")
        print(f"    mask shape: {result.mask.shape}")
    except Exception as e:
        print(f"  ❌ 失败: {e}")