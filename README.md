# YOLO 实时目标检测 Web 应用

基于 YOLOv8 和 Gradio 构建的实时目标检测 Web 应用，支持浏览器摄像头实时检测并生成统计图表。

## 功能特性

- 🎥 **实时检测**：支持浏览器摄像头实时视频流目标检测
- 📊 **统计图表**：自动生成当前帧和累计检测统计图表
- 🎯 **多目标识别**：支持 COCO 数据集的 80 种目标类别检测
- 🔄 **统计重置**：一键重置检测统计信息
- 🌐 **Web 界面**：基于 Gradio 的简洁友好的 Web 交互界面
- 📈 **可视化展示**：柱状图展示检测结果，支持中文显示

## 环境要求

- Anaconda 或 Miniconda（推荐）
- Python 3.8 或更高版本
- 支持 Web 摄像头的浏览器（Chrome、Firefox、Edge 等）

## 安装步骤

### 1. 克隆项目

```bash
git clone <your-repo-url>
cd yolo_gradio_mini_project_test
```

### 2. 创建虚拟环境（推荐）

使用 Conda 创建虚拟环境：

```bash
# 创建 conda 虚拟环境（Python 3.11）
conda create -n yolo_gradio python=3.11

# 激活虚拟环境
# Windows
conda activate yolo_gradio

# Linux/Mac
conda activate yolo_gradio
```

> **注意**：如果使用 Linux/Mac，激活命令相同，但需要使用 `source activate yolo_gradio`（对于旧版本 conda）

### 3. 安装依赖

在激活的 conda 环境中安装依赖：

```bash
# 确保已激活虚拟环境
conda activate yolo_gradio

# 安装项目依赖
pip install -r requirements.txt
```

### 4. 下载模型文件（可选）

如果项目中没有 `yolov8n.pt` 文件，程序首次运行时会自动下载。你也可以手动下载：

```bash
# 模型会在首次运行时自动下载
# 或访问 https://github.com/ultralytics/assets/releases 手动下载
```

## 使用方法

### 本地运行

#### 启动应用

```bash
python app.py
```

#### 访问应用

1. 启动后，在终端会显示本地访问地址（默认为 `http://127.0.0.1:7860`）
2. 打开浏览器访问该地址
3. 允许浏览器访问摄像头权限
4. 开始实时检测！

### 部署到 Hugging Face Spaces

本项目已经配置好，可以直接部署到 Hugging Face Spaces。以下是详细步骤：

#### 方法一：通过 Web 界面创建（推荐）

1. **登录 Hugging Face**
   - 访问 [Hugging Face](https://huggingface.co/)
   - 注册/登录账号

2. **创建新的 Space**
   - 点击右上角头像，选择 "New Space"
   - 填写 Space 信息：
     - **Space name**: 输入你的 Space 名称（如 `yolo-realtime-detection`）
     - **SDK**: 选择 **Gradio**
     - **Hardware**: 选择 **CPU basic**（免费）或 **CPU upgrade**（如果需要更快速度）
     - **Visibility**: 选择 **Public**（公开）或 **Private**（私有）
   - 点击 "Create Space"

3. **上传代码文件**
   - 在 Space 页面，点击 "Files and versions" 标签
   - 点击 "Add file" → "Upload file"
   - 依次上传以下文件：
     - `app.py`
     - `requirements.txt`
     - `README.md`
   - 或者使用 Git 上传（见方法二）

4. **等待构建完成**
   - Hugging Face 会自动检测代码并开始构建
   - 构建过程可能需要 2-5 分钟
   - 可以在 "Logs" 标签查看构建进度

5. **访问你的 Space**
   - 构建完成后，在 Space 页面即可访问应用
   - URL 格式：`https://huggingface.co/spaces/<your-username>/<your-space-name>`

#### 方法二：通过 Git 上传

1. **安装 Git 和 Hugging Face CLI**
   ```bash
   pip install huggingface_hub
   ```

2. **登录 Hugging Face**
   ```bash
   huggingface-cli login
   ```
   输入你的 Hugging Face token（在 [Settings → Access Tokens](https://huggingface.co/settings/tokens) 创建）

3. **初始化 Git 仓库（如果还没有）**
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   ```

4. **连接到 Hugging Face Space**
   ```bash
   # 首先在 Hugging Face 网页创建 Space（方法一的前两步）
   # 然后执行以下命令：
   git remote add origin https://huggingface.co/spaces/<your-username>/<your-space-name>
   git push origin main
   ```

#### Hugging Face Spaces 部署注意事项

- ✅ **模型文件**：`yolov8n.pt` 会在首次运行时自动下载，无需手动上传
- ✅ **环境适配**：`app.py` 已自动检测 Hugging Face Spaces 环境并调整配置
- ⚠️ **摄像头权限**：在 Hugging Face Spaces 中需要用户授权浏览器摄像头权限
- ⚠️ **硬件资源**：免费 CPU 资源有限，检测速度可能较慢；可以考虑升级到 CPU upgrade 或使用 GPU
- 📝 **README.md**：建议保留，Hugging Face Spaces 会显示为 Space 的说明文档

#### Space 配置说明

项目已包含以下 Hugging Face Spaces 兼容配置：
- `app.py`: 自动检测 `SPACE_ID` 环境变量，在 Spaces 环境下使用默认启动配置
- `requirements.txt`: 包含所有必需的依赖包
- 代码支持本地和 Spaces 环境双重运行

### 使用说明

1. **摄像头权限**：首次使用时，浏览器会请求摄像头权限，请点击"允许"
2. **实时检测**：摄像头画面会实时显示在左侧，检测结果会显示在右侧
3. **查看统计**：页面下方的统计图表会实时更新，显示：
   - 左图：当前帧检测到的目标数量
   - 右图：累计检测统计（最多显示前 10 个类别）
4. **重置统计**：点击"🔄 重置统计"按钮可以清空累计统计数据

## 项目结构

```
yolo_gradio_mini_project_test/
├── app.py              # 主应用文件
├── requirements.txt    # 项目依赖
├── LICENSE            # MIT 许可证
├── .gitignore         # Git 忽略文件配置
├── yolov8n.pt         # YOLOv8 模型文件（首次运行自动下载）
└── README.md          # 项目说明文档
```

## 依赖说明

- **gradio**：Web 界面框架
- **ultralytics**：YOLOv8 模型库
- **opencv-python**：图像处理和视频处理
- **matplotlib**：统计图表生成
- **numpy**：数组操作和数据处理

## 技术细节

### 检测配置

- 模型：YOLOv8n (nano) - 轻量级实时检测模型
- 置信度阈值：0.4
- 输入格式：RGB 图像
- 输出格式：带检测框和标签的图像

### 统计功能

- **当前帧统计**：显示当前视频帧中检测到的所有目标类别及数量
- **累计统计**：从启动应用开始累计所有检测到的目标，按类别汇总
- **图表展示**：使用 matplotlib 生成双图表展示统计结果

## 常见问题

### Q: 摄像头无法访问？
A: 请确保：
1. 浏览器已授予摄像头权限
2. 摄像头没有被其他程序占用
3. 使用 HTTPS 或 localhost 访问（某些浏览器要求）

### Q: 检测速度慢？
A: 可以尝试：
1. 降低视频分辨率
2. 使用更小的 YOLO 模型（当前为 yolov8n.pt）
3. 检查 CPU/GPU 性能

### Q: 统计图表不显示中文？
A: 程序已自动检测和配置中文字体。如果仍无法显示：
1. **本地运行**：确保系统已安装中文字体（Windows/macOS 通常已预装）
2. **Hugging Face Spaces**：
   - 程序会自动检测系统字体
   - 如果找不到中文字体，会自动切换到英文标签
   - 如需中文显示，可以在 Space 的 README 中添加字体安装说明
   - 或者使用支持中文字体的基础镜像

### Q: 如何修改检测置信度？
A: 在 `app.py` 的第 42 行修改 `conf=0.4` 参数，范围 0-1，数值越高要求越严格

## 注意事项

- 首次运行会自动下载 YOLOv8n 模型文件（约 6MB）
- 应用默认在本地地址运行（127.0.0.1:7860），如需外网访问请修改 `server_name` 参数
- 统计数据在页面刷新后会重置

## 许可证

本项目采用 [MIT License](LICENSE) 许可证。

### 许可证说明

- **项目代码**：MIT License - 允许自由使用、修改、分发，包括商业用途
- **YOLO 模型**：本项目使用的 YOLOv8 模型来自 Ultralytics，请参考 [Ultralytics 许可证条款](https://github.com/ultralytics/ultralytics) 了解模型使用限制
- **依赖库**：各依赖库遵循各自的许可证（请参考各库的许可证文件）

### 使用建议

- ✅ 可以用于学习和研究
- ✅ 可以修改和分发代码
- ✅ 可以用于商业项目
- ⚠️ 使用 YOLO 模型时请遵守 Ultralytics 的许可证条款
- ⚠️ 请保留原始版权声明

## 贡献

欢迎提交 Issue 和 Pull Request！

## 更新日志

### v1.0.0
- 初始版本发布
- 支持实时摄像头检测
- 支持检测统计图表
- 支持统计重置功能

