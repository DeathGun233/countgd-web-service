from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Body
from fastapi.responses import JSONResponse, HTMLResponse, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import torch
import numpy as np
import cv2
import os
import time
import json
import argparse
import sys
from PIL import Image, ImageDraw
import io
from typing import List, Optional, Dict, Any
import shutil
from pydantic import BaseModel
from enum import Enum
import matplotlib.pyplot as plt

# --- 1. 添加原始项目路径，导入模型构建代码 ---
# 假设您的项目结构与原始 CountGD 项目一致
# 您需要确保 `models_inference/`, `util/`, `datasets/` 等目录存在
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 导入原始模型构建和工具
from util.slconfig import SLConfig, DictAction
from util.misc import nested_tensor_from_tensor_list
import datasets.transforms as T
from models.registry import MODULE_BUILD_FUNCS

# 创建FastAPI应用
app = FastAPI(
    title="CountGD 鱼群计数API (修正版)",
    description="基于原版CountGD模型的多模态（文本/视觉示例）计数服务",
    version="2.0.0"
)

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有来源
    allow_credentials=True,
    allow_methods=["*"],  # 允许所有方法
    allow_headers=["*"],  # 允许所有头部
    expose_headers=["*"],  # 暴露所有头部
)

# --- 2. 全局变量 ---
model = None
transform = None
device = None
CONF_THRESH = 0.23  # 置信度阈值，与官方一致

# --- 3. 数据模型定义 ---
class BoundingBox(BaseModel):
    """边界框模型"""
    x: float
    y: float
    width: float
    height: float

class VisualPrompts(BaseModel):
    """视觉示例模型"""
    image: Dict[str, Any]  # 图片信息
    points: List[List[float]]  # 边界框点

class PredictionRequest(BaseModel):
    """预测请求模型"""
    text_prompt: Optional[str] = ""
    visual_prompts: Optional[VisualPrompts] = None
    image: UploadFile = File(...)

# --- 4. 模型加载函数 ---
def get_args_parser():
    """获取参数解析器（与官方代码一致）"""
    parser = argparse.ArgumentParser("Set transformer detector", add_help=False)
    parser.add_argument("--device", default="cuda", help="device to use for inference")
    parser.add_argument("--options", nargs="+", action=DictAction)
    parser.add_argument("--remove_difficult", action="store_true")
    parser.add_argument("--fix_size", action="store_true")
    parser.add_argument("--pretrain_model_path", default="checkpoints/checkpoint_fsc147_best.pth")
    return parser

def get_device(gpu_id: int = 1):
    """获取设备，优先使用指定GPU"""
    if torch.cuda.is_available():
        if torch.cuda.device_count() > gpu_id:
            return torch.device(f'cuda:{gpu_id}')
        else:
            return torch.device('cuda:0')
    else:
        return torch.device('cpu')

def build_model_and_transforms(args):
    """构建模型和转换（修正版，允许args参数覆盖配置）"""
    # 数据转换
    normalize = T.Compose(
        [T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
    )
    data_transform = T.Compose([
        T.RandomResize([800], max_size=1333),
        normalize,
    ])
    
    # 加载配置
    cfg_path = "cfg_app.py"  # 需要确保此配置文件存在
    if not os.path.exists(cfg_path):
        # 尝试从config目录查找
        cfg_path = "config/cfg_app.py"
        if not os.path.exists(cfg_path):
            # 如果配置文件不存在，尝试使用一个简单的默认配置
            print(f"⚠ 警告: 未找到配置文件 {cfg_path}，尝试使用内置默认参数。")
            # 这里可以设置一些关键参数的默认值
            # 例如: args.modelname = getattr(args, 'modelname', 'groundingdino')
            # 但更推荐您确保配置文件存在。
            # 为了快速测试，我们可以创建一个最简化的配置字典
            cfg_dict = {}
        else:
            cfg = SLConfig.fromfile(cfg_path)
        #新加东西
            BASE_DIR = os.path.dirname(os.path.abspath(__file__))
            BERT_PATH = os.path.join(BASE_DIR, "checkpoints", "bert-base-uncased")
        #
            cfg.merge_from_dict({"text_encoder_type": BERT_PATH})
            cfg_dict = cfg._cfg_dict.to_dict()
    else:
        cfg = SLConfig.fromfile(cfg_path)
        #新加
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        BERT_PATH = os.path.join(BASE_DIR, "checkpoints", "bert-base-uncased")
        ##
        cfg.merge_from_dict({"text_encoder_type": BERT_PATH})
        cfg_dict = cfg._cfg_dict.to_dict()
    
    # ========== 关键修复：修改配置合并逻辑 ==========
    args_vars = vars(args)
    for k, v in cfg_dict.items():
        if k not in args_vars:
            # 如果args中没有这个键，则添加
            setattr(args, k, v)
        # 否则（即args中已有此键），静默跳过，让args的参数值生效
        # 原代码: raise ValueError("Key {} can used by args only".format(k))
        # 修改为: pass 或 可选地打印一条调试信息
        # print(f"调试: 参数'{k}'已由args提供，跳过配置文件中的值。")
    
    # 固定随机种子
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # 从注册表获取模型构建函数
    # 确保args.modelname已被正确设置（要么来自命令行/代码，要么来自配置文件）
    modelname = getattr(args, 'modelname', None)
    if modelname is None:
        raise ValueError("模型名称 'modelname' 未在参数中指定。")
    
    if modelname not in MODULE_BUILD_FUNCS._module_dict:
        available_models = list(MODULE_BUILD_FUNCS._module_dict.keys())
        raise ValueError(f"未知的模型名称: '{modelname}'。可用模型: {available_models}")
    
    build_func = MODULE_BUILD_FUNCS.get(modelname)
    model, _, _ = build_func(args)
    
    # 加载权重
    checkpoint_path = args.pretrain_model_path
    if not os.path.exists(checkpoint_path):
        # 尝试在checkpoints目录下查找
        checkpoint_path = os.path.join("checkpoints", os.path.basename(checkpoint_path))
    
    print(f"加载模型权重: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    
    if "model" in checkpoint:
        model.load_state_dict(checkpoint["model"], strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)
    
    model.eval()
    return model, data_transform

@app.on_event("startup")
async def load_model():
    """应用启动时加载模型"""
    global model, transform, device
    
    print("=" * 60)
    print("🚀 初始化 CountGD 模型...")
    
    # 1. 选择设备（优先GPU 1）
    device = get_device(gpu_id=1)
    print(f"使用设备: {device}")
    
    # 2. 设置参数
    parser = argparse.ArgumentParser("Counting Application", parents=[get_args_parser()])
    # 这里创建一个简单的命名空间，并手动设置关键参数，而不是解析空列表
    args = argparse.Namespace()  # 创建一个空对象
    args.device = str(device)
    args.pretrain_model_path = "/home/Yjh/fish_count_deploy/models/checkpoint_fsc147_best.pth"  # 根据您的实际情况调整
    args.modelname = "groundingdino"  # 这是关键！必须与配置文件或模型注册表中的名称匹配
    
    # 3. 构建模型和转换
    try:
        model, transform = build_model_and_transforms(args)
        model = model.to(device)
        print("✅ 模型加载成功")
        # ... 后续测试代码 ...
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        raise RuntimeError(f"模型初始化失败: {e}")

# --- 5. 工具函数 ---
def get_box_inputs(prompts_points):
    """从点数据获取边界框输入（与官方代码一致）"""
    box_inputs = []
    if prompts_points:
        for prompt in prompts_points:
            if len(prompt) >= 6 and prompt[2] == 2.0 and prompt[5] == 3.0:
                box_inputs.append([prompt[0], prompt[1], prompt[3], prompt[4]])
    return box_inputs

def get_ind_to_filter(text, word_ids, keywords=""):
    """获取需要过滤的索引（与官方代码一致）"""
    if len(keywords) <= 0:
        return list(range(len(word_ids)))
    
    input_words = text.split()
    keywords = keywords.split(",")
    keywords = [keyword.strip() for keyword in keywords]
    
    word_inds = []
    for keyword in keywords:
        if keyword in input_words:
            if len(word_inds) <= 0:
                ind = input_words.index(keyword)
                word_inds.append(ind)
            else:
                ind = input_words.index(keyword, word_inds[-1])
                word_inds.append(ind)
        else:
            raise Exception("Only specify keywords in the input text!")
    
    inds_to_filter = []
    for ind in range(len(word_ids)):
        word_id = word_ids[ind]
        if word_id in word_inds:
            inds_to_filter.append(ind)
    
    return inds_to_filter

def preprocess_image(image_bytes: bytes):
    """预处理图片为PIL格式"""
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return image

def process_visual_prompts(prompts_data: dict, example_img_bytes: bytes = None):
    """处理视觉提示数据"""
    if not prompts_data or not prompts_data.get("points"):
        return {"image": None, "exemplars": []}
    
    # 如果有示例图片，使用它
    if example_img_bytes:
        example_image = preprocess_image(example_img_bytes)
    elif "image" in prompts_data:
        # 从base64或路径加载图片
        if isinstance(prompts_data["image"], str) and prompts_data["image"].startswith("data:image"):
            # 处理base64图片
            import base64
            header, encoded = prompts_data["image"].split(",", 1)
            example_img_bytes = base64.b64decode(encoded)
            example_image = preprocess_image(example_img_bytes)
        else:
            # 假设是文件路径
            if os.path.exists(prompts_data["image"]):
                with open(prompts_data["image"], "rb") as f:
                    example_image = preprocess_image(f.read())
            else:
                raise ValueError("示例图片文件不存在")
    else:
        raise ValueError("未提供视觉示例图片")
    
    exemplars = get_box_inputs(prompts_data["points"])
    return {"image": example_image, "exemplars": exemplars}

# --- 6. 核心推理函数 ---
async def count_objects(
    input_image_bytes: bytes,
    text_prompt: str = "",
    visual_prompts_data: dict = None
):
    """核心计数函数 - 根据 forward 函数签名修正"""
    global model, transform, device
    
    try:
        print("=" * 60)
        print("🧠 开始计数推理...")
        
        # 1. 预处理输入图片
        input_image = preprocess_image(input_image_bytes)
        print(f"🔸 输入图片: {input_image.size}")
        
        # 2. 初始化变量
        exemplars_list = []
        
        # 3. 处理输入图片
        print("🔸 处理输入图片...")
        empty_exemplars = torch.zeros(0, 4, dtype=torch.float32)
        input_tensor, _ = transform(input_image, {"exemplars": empty_exemplars})
        
        input_tensor = input_tensor.to(device)
        print(f"🔸 输入张量形状: {input_tensor.shape}")
        
        # 4. 准备模型输入
        with torch.no_grad():
            # 创建 NestedTensor
            input_nested = nested_tensor_from_tensor_list([input_tensor])
            print(f"✅ NestedTensor 创建成功")
            
            # 准备 caption
            caption = text_prompt.strip() if text_prompt else "object"
            if not caption.endswith("."):
                caption += " ."
            print(f"🔸 使用 caption: '{caption}'")
            
            # 🔥 根据 forward 函数准备参数
            
            # 参数1: samples (NestedTensor)
            samples = input_nested
            
            # 参数2: exemplars (边界框列表)
            empty_boxes = torch.zeros(0, 4, dtype=torch.float32).to(device)
            exemplars = [empty_boxes]  # 必须放入列表
            
            # 参数3: labels
            # 从代码看，labels 用于标识 exemplar 的类别
            # 当没有视觉示例时，用空列表
            labels = torch.tensor([0]).to(device)  # 或者 []
            
            # 参数4: targets (包含 caption 的字典列表)
            targets = [{
                "image_id": torch.tensor([0]).to(device),
                "caption": caption,
            }]
            
            # 如果不想用 targets，可以用 **kw 传入 captions
            # 但根据代码逻辑，最好用 targets
            
            print(f"🔸 准备调用模型...")
            print(f"  参数1 samples 类型: {type(samples)}")
            print(f"  参数2 exemplars 类型: {type(exemplars)}")
            print(f"  参数3 labels 类型: {type(labels)}")
            print(f"  参数4 targets 类型: {type(targets)}")
            print(f"  targets[0] keys: {targets[0].keys() if targets else '无'}")
            
            # 🔥 关键：调用模型，传入4个参数
            # 根据 forward 函数定义，需要4个参数
            model_output = model(
                samples,     # 参数1: 输入图片
                exemplars,   # 参数2: exemplar 边界框列表
                labels,      # 参数3: 标签
                targets,     # 参数4: 目标字典
            )
        
        # 5. 处理输出...
        print("🔸 处理模型输出...")
        
        if "pred_logits" in model_output:
            logits = model_output["pred_logits"].sigmoid()[0]
            boxes = model_output["pred_boxes"][0]
            
            box_mask = logits.max(dim=-1).values > CONF_THRESH
            boxes = boxes[box_mask, :].cpu().numpy()
            count = len(boxes)
            
            print(f"✅ 检测到 {count} 个目标")
            
            # 可视化
            output_image = visualize_detections(input_image, boxes)
            
            return count, boxes.tolist(), output_image
        else:
            raise ValueError(f"模型输出格式异常: {model_output.keys()}")
        
    except Exception as e:
        print(f"❌ 推理错误: {e}")
        import traceback
        traceback.print_exc()
        raise

def visualize_detections(image, boxes, alpha=0.7):
    """
    可视化检测结果
    
    参数:
        image: PIL图片
        boxes: 检测框列表
        alpha: 透明度
        
    返回:
        output_image_bytes: 可视化图片字节
    """
    w, h = image.size
    
    # 创建热力图
    det_map = np.zeros((h, w))
    for box in boxes:
        x, y = box[0], box[1]
        px = int(x * w)
        py = int(y * h)
        if 0 <= px < w and 0 <= py < h:
            det_map[py, px] = 1
    
    # 高斯滤波
    sigma = max(1, w // 200)
    det_map = cv2.GaussianBlur(det_map, (0, 0), sigmaX=sigma, sigmaY=sigma)
    
    # 创建可视化
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    plt.imshow(det_map, cmap='jet', alpha=alpha)
    plt.axis('off')
    plt.tight_layout()
    
    # 保存到字节流
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    plt.close()
    buf.seek(0)
    
    return buf.getvalue()

# --- 7. API端点 ---
# 挂载静态文件
if not os.path.exists("frontend"):
    os.makedirs("frontend", exist_ok=True)
app.mount("/static", StaticFiles(directory="frontend"), name="static")


@app.get("/")
async def root():
    """根路径返回简单前端"""
    return RedirectResponse(url="/static/index.html")

@app.post("/predict")
async def predict(
    image: UploadFile = File(...),
    text_prompt: str = Form(""),
    mode: str = Form("text_only"),
    visual_example: Optional[UploadFile] = File(None)
):
    """
    预测接口
    
    支持三种模式:
    1. text_only: 仅使用文本描述
    2. visual_only: 仅使用视觉示例
    3. both: 文本+视觉示例
    """
    start_time = time.time()
    
    try:
        # 1. 读取输入图片
        input_image_bytes = await image.read()
        
        # 2. 处理视觉示例
        visual_prompts_data = None
        if mode in ["visual_only", "both"] and visual_example:
            visual_example_bytes = await visual_example.read()
            # 简化处理：将整个示例图片作为视觉提示
            visual_prompts_data = {
                "image": Image.open(io.BytesIO(visual_example_bytes)),
                "points": []  # 实际应用中可添加边界框
            }
        
        # 3. 执行计数
        count, boxes, output_image_bytes = await count_objects(
            input_image_bytes=input_image_bytes,
            text_prompt=text_prompt,
            visual_prompts_data=visual_prompts_data
        )
        
        # 4. 计算耗时
        inference_time = (time.time() - start_time) * 1000
        
        return {
            "success": True,
            "count": count,
            "boxes": boxes,
            "inference_time_ms": round(inference_time, 2),
            "output_image": output_image_bytes.hex(),  # 转换为十六进制字符串传输
            "mode_used": mode
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "inference_time_ms": round((time.time() - start_time) * 1000, 2)
        }

@app.get("/health")
async def health_check():
    """健康检查"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    gpu_info = {}
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            mem_alloc = torch.cuda.memory_allocated(i) / 1024**2
            mem_reserved = torch.cuda.memory_reserved(i) / 1024**2
            gpu_info[f"gpu_{i}"] = {
                "allocated_mb": round(mem_alloc, 1),
                "reserved_mb": round(mem_reserved, 1)
            }
    
    return {
        "status": "healthy",
        "model_loaded": True,
        "device": str(device),
        "gpu_info": gpu_info,
        "timestamp": time.time()
    }

if __name__ == "__main__":
    import uvicorn
    
    print("\n" + "="*60)
    print("CountGD 服务启动中...")
    print("访问 http://localhost:8000 使用前端界面")
    print("API端点:")
    print("  GET  /          - 前端界面")
    print("  POST /predict   - 计数预测")
    print("  GET  /health    - 健康检查")
    print("="*60 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")