import os
import gradio as gr
import cv2
from ultralytics import YOLO  # type: ignore[import]
from collections import Counter
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import numpy as np
from matplotlib import font_manager
import platform

# -------------------------
# 配置 matplotlib 中文字体（支持多平台）
# -------------------------
# 全局变量：是否支持中文显示
CHINESE_FONT_AVAILABLE = False
CHINESE_FONT_NAME = None

def setup_chinese_font():
    """配置中文字体，支持 Windows、Linux 和 macOS"""
    global CHINESE_FONT_AVAILABLE, CHINESE_FONT_NAME
    
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
    
    # 获取系统所有可用字体
    try:
        font_list = [f.name for f in font_manager.fontManager.ttflist]
    except Exception:
        font_list = []
    
    # 按优先级排序的中文字体列表
    chinese_fonts = [
        # Linux 常见中文字体（Hugging Face Spaces 通常有这些）
        'Noto Sans CJK SC', 'Noto Sans CJK TC', 'Noto Sans CJK',
        'WenQuanYi Micro Hei', 'WenQuanYi Zen Hei',
        'Source Han Sans CN', 'Source Han Sans SC',
        'Droid Sans Fallback', 'AR PL UMing CN',
        # Windows 中文字体
        'Microsoft YaHei', 'SimHei', 'SimSun', 'KaiTi', 'FangSong',
        # macOS 中文字体
        'PingFang SC', 'Hiragino Sans GB', 'STHeiti',
        # 通用字体
        'Arial Unicode MS'
    ]
    
    # 查找第一个可用的中文字体
    found_font = None
    for font_name in chinese_fonts:
        if font_name in font_list:
            found_font = font_name
            break
    
    if found_font:
        plt.rcParams['font.sans-serif'] = [found_font] + plt.rcParams['font.sans-serif']
        CHINESE_FONT_AVAILABLE = True
        CHINESE_FONT_NAME = found_font
        print(f"✅ 已设置中文字体: {found_font}")
    else:
        # 在 Linux 环境中，尝试从文件路径加载字体
        if platform.system() == 'Linux':
            font_paths = [
                '/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc',
                '/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttf',
                '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc',
                '/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc',
                '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
            ]
            
            for font_path in font_paths:
                if os.path.exists(font_path):
                    try:
                        prop = font_manager.FontProperties(fname=font_path)
                        font_name = prop.get_name()
                        plt.rcParams['font.sans-serif'] = [font_name] + plt.rcParams['font.sans-serif']
                        CHINESE_FONT_AVAILABLE = True
                        CHINESE_FONT_NAME = font_name
                        print(f"✅ 已从路径加载中文字体: {font_path} ({font_name})")
                        break
                    except Exception as e:
                        print(f"⚠️ 加载字体文件失败 {font_path}: {e}")
                        continue
            
            if not CHINESE_FONT_AVAILABLE:
                print("⚠️ 未找到中文字体，将使用英文标签")
                # 使用 DejaVu Sans 作为默认字体
                plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'sans-serif']
        else:
            # Windows/macOS 使用默认配置
            plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'sans-serif']
            CHINESE_FONT_AVAILABLE = True  # 假设 Windows/macOS 有中文字体

# 初始化字体配置
setup_chinese_font()

# -------------------------
# 加载 YOLO 模型（全局一次）
# -------------------------
model = YOLO("yolov8n.pt")

# -------------------------
# 摄像头帧处理函数（带统计）
# -------------------------
# 用于控制图表更新频率的计数器
_frame_counter = 0
_chart_update_interval = 5  # 每5帧更新一次图表（减少计算负担）

def yolo_detect(frame, stats_state):
    """
    frame: numpy.ndarray (RGB)
    stats_state: dict，存储统计信息
    return: (处理后的图片, 统计文本, 更新后的统计状态)
    """
    global _frame_counter
    
    try:
        if frame is None:
            return None, stats_state, stats_state

        # 初始化统计字典（如果不存在）
        if stats_state is None:
            stats_state = {
                'current': Counter(),  # 当前帧检测到的目标
                'total': Counter(),    # 累计检测到的目标总数
                'last_chart': None,    # 缓存的最后一张图表
            }

        # Gradio 给的是 RGB，YOLO / OpenCV 用 BGR
        img = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # YOLO 检测（添加超时保护）
        try:
            results = model(img, conf=0.4, verbose=False)[0]
        except Exception as e:
            print(f"检测错误: {e}")
            # 如果检测失败，返回原图
            result_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return result_img, stats_state.get('last_chart', frame), stats_state

        # 重置当前帧统计
        current_frame_count = Counter()

        if results.boxes is not None:
            for box in results.boxes:
                try:
                    cls_id = int(box.cls[0])
                    label = model.names[cls_id]

                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])  # 置信度

                    # 绘制检测框
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # 显示标签和置信度
                    label_text = f"{label} {conf:.2f}"
                    cv2.putText(
                        img,
                        label_text,
                        (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2,
                    )

                    # 统计当前帧
                    current_frame_count[label] += 1
                except Exception as e:
                    print(f"绘制检测框错误: {e}")
                    continue

        # 更新累计统计（只在有新检测时累计）
        for label, count in current_frame_count.items():
            stats_state['total'][label] += count
        
        stats_state['current'] = current_frame_count

        # 转回 RGB 给 Gradio 显示
        result_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 降低图表更新频率，避免处理过慢导致流中断
        _frame_counter += 1
        if _frame_counter % _chart_update_interval == 0 or not stats_state.get('last_chart'):
            try:
                stats_chart = generate_stats_chart(current_frame_count, stats_state['total'])
                stats_state['last_chart'] = stats_chart
            except Exception as e:
                print(f"生成图表错误: {e}")
                stats_chart = stats_state.get('last_chart', result_img)
        else:
            # 使用缓存的图表
            stats_chart = stats_state.get('last_chart', result_img)
        
        return result_img, stats_chart, stats_state
    
    except Exception as e:
        print(f"处理帧时发生错误: {e}")
        # 返回原图和缓存的图表，确保流不会中断
        if frame is not None:
            result_img = frame
        else:
            result_img = None
        return result_img, stats_state.get('last_chart') if stats_state else None, stats_state

def generate_stats_chart(current_count, total_count):
    """生成统计图表（优化版本，减少计算负担）"""
    global CHINESE_FONT_AVAILABLE
    
    try:
        # 根据字体支持情况选择标签语言
        if CHINESE_FONT_AVAILABLE:
            title_main = '📊 检测统计图表'
            title_current = '当前帧检测'
            title_total = '累计统计'
            label_count = '数量'
            label_category = '目标类别'
            label_detections = '检测次数'
            text_no_detection = '未检测到目标'
            text_no_data = '暂无累计数据'
        else:
            title_main = 'Detection Statistics'
            title_current = 'Current Frame'
            title_total = 'Total Statistics'
            label_count = 'Count'
            label_category = 'Object Category'
            label_detections = 'Detections'
            text_no_detection = 'No Detection'
            text_no_data = 'No Data'
        
        # 创建图表，包含两个子图（减小尺寸以提高速度）
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        fig.suptitle(title_main, fontsize=14, fontweight='bold')
        
        # 左图：当前帧检测统计（柱状图）
        if current_count:
            labels = list(current_count.keys())
            values = list(current_count.values())
            colors = plt.cm.Set3(np.linspace(0, 1, len(labels)))
            
            ax1.bar(labels, values, color=colors, edgecolor='black', linewidth=1.5)
            ax1.set_title(title_current, fontsize=12, fontweight='bold')
            ax1.set_ylabel(label_count, fontsize=10)
            ax1.set_xlabel(label_category, fontsize=10)
            ax1.grid(axis='y', alpha=0.3, linestyle='--')
            
            # 在柱状图上显示数值
            for i, v in enumerate(values):
                ax1.text(i, v + 0.05, str(v), ha='center', va='bottom', fontweight='bold')
            
            # 旋转x轴标签以避免重叠
            ax1.tick_params(axis='x', rotation=45)
        else:
            ax1.text(0.5, 0.5, text_no_detection, 
                    ha='center', va='center', fontsize=14, 
                    transform=ax1.transAxes, color='gray')
            ax1.set_title(title_current, fontsize=12, fontweight='bold')
            ax1.set_xlim(0, 1)
            ax1.set_ylim(0, 1)
        
        # 右图：累计统计（柱状图）
        if total_count:
            # 按数量排序，只显示前10个最多的类别
            sorted_items = sorted(total_count.items(), key=lambda x: -x[1])[:10]
            labels_total = [item[0] for item in sorted_items]
            values_total = [item[1] for item in sorted_items]
            colors_total = plt.cm.viridis(np.linspace(0, 1, len(labels_total)))
            
            total_text = f'{title_total} (Total: {sum(total_count.values())})' if CHINESE_FONT_AVAILABLE else f'{title_total} (Total: {sum(total_count.values())})'
            ax2.barh(labels_total, values_total, color=colors_total, edgecolor='black', linewidth=1.5)
            ax2.set_title(total_text, fontsize=12, fontweight='bold')
            ax2.set_xlabel(label_detections, fontsize=10)
            ax2.grid(axis='x', alpha=0.3, linestyle='--')
            
            # 在柱状图上显示数值
            for i, v in enumerate(values_total):
                ax2.text(v + 0.5, i, str(v), ha='left', va='center', fontweight='bold')
        else:
            ax2.text(0.5, 0.5, text_no_data, 
                    ha='center', va='center', fontsize=14, 
                    transform=ax2.transAxes, color='gray')
            ax2.set_title(title_total, fontsize=12, fontweight='bold')
            ax2.set_xlim(0, 1)
            ax2.set_ylim(0, 1)
        
        plt.tight_layout()
        
        # 将图表转换为numpy数组返回（兼容新版matplotlib）
        fig.canvas.draw()
        # 使用 buffer_rgba() 获取 RGBA 格式的缓冲区
        buf = fig.canvas.buffer_rgba()
        chart_img = np.asarray(buf)
        # 转换为 RGB（去掉 alpha 通道）
        chart_img = cv2.cvtColor(chart_img, cv2.COLOR_RGBA2RGB)
        plt.close(fig)
        
        return chart_img
    except Exception as e:
        print(f"生成图表时发生错误: {e}")
        # 返回一个简单的占位图
        placeholder = np.ones((300, 600, 3), dtype=np.uint8) * 255
        cv2.putText(placeholder, "Chart Error", (200, 150), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        return placeholder

def reset_stats():
    """重置统计信息"""
    empty_state = {
        'current': Counter(),
        'total': Counter(),
        'last_chart': None,
    }
    # 生成空图表
    try:
        empty_chart = generate_stats_chart(Counter(), Counter())
        empty_state['last_chart'] = empty_chart
    except Exception as e:
        print(f"重置时生成图表错误: {e}")
        empty_chart = None
    return empty_state, empty_chart

# -------------------------
# Gradio UI
# -------------------------
with gr.Blocks(title="YOLO 实时目标检测") as demo:
    gr.Markdown("## 🚀 YOLO 实时摄像头目标检测（Gradio）")
    gr.Markdown("允许浏览器摄像头权限后即可实时检测")

    # 用于存储统计信息的状态
    stats_state = gr.State(value={
        'current': Counter(),
        'total': Counter(),
        'last_chart': None,  # 缓存的最后一张图表
    })

    # 视频显示区域（上方）
    with gr.Row():
        webcam = gr.Image(
            sources=["webcam"],   # 使用浏览器摄像头
            streaming=True,       # 开启视频流
            type="numpy",         # 直接拿到 numpy 数组
            label="摄像头输入",
        )
        output = gr.Image(
            type="numpy",         # 输出也是 numpy，方便连续显示
            label="检测结果",
        )
    
    # 统计图表区域（下方）
    with gr.Row():
        with gr.Column():
            # 初始化空图表
            empty_chart = generate_stats_chart(Counter(), Counter())
            stats_display = gr.Image(
                label="📊 检测统计图表",
                type="numpy",
                value=empty_chart,  # 设置初始值
            )
            reset_btn = gr.Button("🔄 重置统计", variant="secondary", size="lg")

    # 注意要显式指定 inputs，否则函数收不到帧
    # 添加 show_progress=False 和 timeout 参数以提高稳定性
    webcam.stream(
        fn=yolo_detect,
        inputs=[webcam, stats_state],
        outputs=[output, stats_display, stats_state],
        show_progress=False,  # 不显示进度条，减少开销
    )
    
    # 重置按钮事件
    reset_btn.click(
        fn=reset_stats,
        outputs=[stats_state, stats_display],
    )

# -------------------------
# 启动
# -------------------------
if __name__ == "__main__":
    # 检查是否在 Hugging Face Spaces 环境
    is_spaces = os.getenv("SPACE_ID") is not None
    
    if not is_spaces:
        # 本地运行配置
        os.environ.setdefault("NO_PROXY", "127.0.0.1,localhost")
        os.environ.setdefault("no_proxy", "127.0.0.1,localhost")
        print("✅ 正在启动 Gradio YOLO 应用，请稍等...")
        demo.launch(
            server_name="127.0.0.1",
            server_port=7860,
            inbrowser=False,
            show_error=True,
        )
    else:
        # Hugging Face Spaces 运行配置
        print("✅ 正在启动 Gradio YOLO 应用（Hugging Face Spaces），请稍等...")
        demo.launch(show_error=True)
