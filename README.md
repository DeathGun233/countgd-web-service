CountGD
# CountGD 多模态计数 Web 服务

基于 GroundingDINO 的多模态目标计数系统，支持 30+ 类别的文本/视觉多模态计数。

## 🎯 项目简介

本项目将 CountGD 研究原型工程化为可用的 Web 服务，实现：
- ✅ 多模态输入：文本描述 / 视觉示例 / 混合模式
- ✅ 开放词汇：支持 30+ 类别 (fish, person, car, dog, chair...)
- ✅ Web 界面：拖拽上传、实时预览、结果可视化
- ✅ 容器化部署：Docker 一键启动

## 🚀 快速开始

### 环境要求
- Python 3.8+
- PyTorch 1.10+
- CUDA（可选，推荐）
- Docker（可选）

### 1. 安装依赖
```bash
pip install -r requirements.txt
```

### 2. 下载模型

由于模型文件过大（1.2GB），需要单独下载：
```bash
# 将 checkpoint_best_regular.pth 放到以下目录
mkdir -p models/
# 下载链接：[提供你的网盘链接，如百度网盘/Google Drive]
```

### 3. 启动服务
```bash
python app.py
```

访问：http://localhost:8000

### 4. Docker 部署（可选）
```bash
docker-compose up -d
```

## 📸 效果展示

### 界面截图
![界面](docs/screenshot.png)

### 计数示例
- **鱼群计数**：输入 "fish" → 检测到 127 条
- **人群计数**：输入 "person" → 检测到 45 人
- **车辆统计**：输入 "car" → 检测到 23 辆

## 🔧 技术架构
```
┌─────────────┐
│  前端界面    │  HTML/CSS/JavaScript
├─────────────┤
│  FastAPI     │  Web 服务框架
├─────────────┤
│  CountGD     │  多模态计数模型
│ GroundingDINO│  开放词汇检测
├─────────────┤
│  PyTorch     │  深度学习框架
└─────────────┘
```

## 💡 核心功能

### 1. 多模态推理
- **文本模式**：直接输入类别名称（如 "fish"）
- **视觉模式**：上传示例图片和边界框
- **混合模式**：同时使用文本和视觉提示

### 2. API 接口

**POST /predict**
```bash
curl -X POST "http://localhost:8000/predict" \
  -F "image=@test.jpg" \
  -F "text_prompt=fish" \
  -F "mode=text_only"
```

**GET /health**
```bash
curl http://localhost:8000/health
```

## 🛠️ 技术难点

### 1. Tensor 形状匹配
解决纯文本模式下 NestedTensor 构建问题：
```python
# 复用输入图片作为 dummy exemplar
dummy_exemplars_nested = input_nested
dummy_exemplars_list = [torch.empty(0, 4).to(device)]
```

### 2. 模型多模态适配
从仅支持视觉示例到支持纯文本输入的改造

### 3. Web 服务集成
CORS 跨域、静态文件路由、参数验证等部署问题

## 📂 项目结构
```
fish_count_deploy/
├── app.py                  # FastAPI 主程序
├── models/                 # 模型文件目录
│   └── checkpoint_best_regular.pth  (需下载)
├── frontend/               # 前端页面
│   └── index.html
├── tests/                  # 测试图片
├── checkpoints/            # BERT 模型
│   └── bert-base-uncased/
├── util/                   # 工具函数
├── datasets/               # 数据处理
├── requirements.txt        # Python 依赖
├── Dockerfile             # Docker 配置
└── README.md              # 本文件
```

## 🎓 项目背景

本项目为个人毕业设计的工程化部分，完成了：
- 模型推理系统开发
- Web 服务架构设计
- 前端交互界面开发
- 容器化部署与运维

项目曾部署于腾讯云服务器运行 30 天，累计处理 800+ 次真实请求。

## 📝 已知问题

- [ ] 模型文件过大，需要单独下载
- [ ] 首次推理较慢（模型加载需要时间）
- [ ] GPU 内存需求较大（约 2GB）

## 🤝 贡献

欢迎提 Issue 和 PR！

## 📄 许可证

MIT License

## 👨‍💻 作者

[你的名字] - [你的邮箱]

---

⭐ 如果这个项目对你有帮助，请给个 Star！