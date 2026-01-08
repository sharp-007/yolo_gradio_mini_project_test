import os
import gradio as gr
import cv2
from ultralytics import YOLO  # type: ignore[import]
from collections import Counter
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import numpy as np

# 配置 matplotlib 中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'Arial Unicode MS']  # 设置中文字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# -------------------------
# 加载 YOLO 模型（全局一次）
# -------------------------
model = YOLO("yolov8n.pt")

# -------------------------
# 摄像头帧处理函数（带统计）
# -------------------------
def yolo_detect(frame, stats_state):
    """
    frame: numpy.ndarray (RGB)
    stats_state: dict，存储统计信息
    return: (处理后的图片, 统计文本, 更新后的统计状态)
    """
    if frame is None:
        return None, stats_state, stats_state

    # 初始化统计字典（如果不存在）
    if stats_state is None:
        stats_state = {
            'current': Counter(),  # 当前帧检测到的目标
            'total': Counter(),    # 累计检测到的目标总数
        }

    # Gradio 给的是 RGB，YOLO / OpenCV 用 BGR
    img = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    results = model(img, conf=0.4, verbose=False)[0]

    # 重置当前帧统计
    current_frame_count = Counter()

    if results.boxes is not None:
        for box in results.boxes:
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

    # 更新累计统计（只在有新检测时累计）
    for label, count in current_frame_count.items():
        stats_state['total'][label] += count
    
    stats_state['current'] = current_frame_count

    # 生成统计图表
    stats_chart = generate_stats_chart(current_frame_count, stats_state['total'])

    # 转回 RGB 给 Gradio 显示
    result_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    return result_img, stats_chart, stats_state

def generate_stats_chart(current_count, total_count):
    """生成统计图表"""
    # 创建图表，包含两个子图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('📊 检测统计图表', fontsize=16, fontweight='bold')
    
    # 左图：当前帧检测统计（柱状图）
    if current_count:
        labels = list(current_count.keys())
        values = list(current_count.values())
        colors = plt.cm.Set3(np.linspace(0, 1, len(labels)))
        
        ax1.bar(labels, values, color=colors, edgecolor='black', linewidth=1.5)
        ax1.set_title('当前帧检测', fontsize=12, fontweight='bold')
        ax1.set_ylabel('数量', fontsize=10)
        ax1.set_xlabel('目标类别', fontsize=10)
        ax1.grid(axis='y', alpha=0.3, linestyle='--')
        
        # 在柱状图上显示数值
        for i, v in enumerate(values):
            ax1.text(i, v + 0.05, str(v), ha='center', va='bottom', fontweight='bold')
        
        # 旋转x轴标签以避免重叠
        ax1.tick_params(axis='x', rotation=45)
    else:
        ax1.text(0.5, 0.5, '未检测到目标', 
                ha='center', va='center', fontsize=14, 
                transform=ax1.transAxes, color='gray')
        ax1.set_title('当前帧检测', fontsize=12, fontweight='bold')
        ax1.set_xlim(0, 1)
        ax1.set_ylim(0, 1)
    
    # 右图：累计统计（柱状图）
    if total_count:
        # 按数量排序，只显示前10个最多的类别
        sorted_items = sorted(total_count.items(), key=lambda x: -x[1])[:10]
        labels_total = [item[0] for item in sorted_items]
        values_total = [item[1] for item in sorted_items]
        colors_total = plt.cm.viridis(np.linspace(0, 1, len(labels_total)))
        
        ax2.barh(labels_total, values_total, color=colors_total, edgecolor='black', linewidth=1.5)
        ax2.set_title(f'累计统计 (总计: {sum(total_count.values())} 个目标)', 
                     fontsize=12, fontweight='bold')
        ax2.set_xlabel('检测次数', fontsize=10)
        ax2.grid(axis='x', alpha=0.3, linestyle='--')
        
        # 在柱状图上显示数值
        for i, v in enumerate(values_total):
            ax2.text(v + 0.5, i, str(v), ha='left', va='center', fontweight='bold')
    else:
        ax2.text(0.5, 0.5, '暂无累计数据', 
                ha='center', va='center', fontsize=14, 
                transform=ax2.transAxes, color='gray')
        ax2.set_title('累计统计', fontsize=12, fontweight='bold')
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

def reset_stats():
    """重置统计信息"""
    empty_state = {
        'current': Counter(),
        'total': Counter(),
    }
    # 生成空图表
    empty_chart = generate_stats_chart(Counter(), Counter())
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
    webcam.stream(
        fn=yolo_detect,
        inputs=[webcam, stats_state],
        outputs=[output, stats_display, stats_state],
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
    # 避免通过系统代理访问本机端口，导致 httpx.RemoteProtocolError
    os.environ.setdefault("NO_PROXY", "127.0.0.1,localhost")
    os.environ.setdefault("no_proxy", "127.0.0.1,localhost")

    print("✅ 正在启动 Gradio YOLO 应用，请稍等...")
    demo.launch(
        server_name="127.0.0.1",   # 只在本机访问，避免 0.0.0.0 经代理转发
        server_port=7860,          # 浏览器访问：http://127.0.0.1:7860
        inbrowser=False,           # 不自动开浏览器，避免触发 httpx 代理访问
        show_error=True,           # 终端里直接看到错误
    )
