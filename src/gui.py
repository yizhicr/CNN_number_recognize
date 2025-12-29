import tkinter as tk
import numpy as np
import torch
import time
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from PIL import Image, ImageDraw
from src.model import ImprovedDualBranchCNN
from src.data import preprocess_for_model
from src.utils import get_device, setup_matplotlib_fonts, setup_ttk_style, get_chinese_font
import config

# 初始化中文配置
setup_matplotlib_fonts()
setup_ttk_style()

class HandwritingDigitGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("手写数字识别")
        self.root.geometry("1000x500")
        
        # 画笔设置
        self.canvas_width, self.canvas_height = 400, 400
        self.brush_size = 8
        self.last_x, self.last_y = None, None
        
        # 预测节流（避免频繁计算）
        self.last_predict_time = 0
        self.predict_interval = 0.05
        
        # 加载模型
        self.device = get_device()
        self.model = ImprovedDualBranchCNN().to(self.device)
        self.model.load_state_dict(torch.load(config.FINAL_MODEL_PATH, map_location=self.device, weights_only=True))
        self.model.eval()
        
        # 左侧画布
        self.canvas = tk.Canvas(
            root, width=self.canvas_width, height=self.canvas_height,
            bg="white", relief="solid", borderwidth=2
        )
        self.canvas.grid(row=0, column=0, padx=20, pady=20)
        
        # 清空按钮
        self.clear_btn = tk.Button(
            root, text="清空画布", command=self.clear_canvas,
            height=2, width=10, font=get_chinese_font()
        )
        self.clear_btn.grid(row=1, column=0, pady=(0, 20))
        
        # 右侧概率柱状图
        self.fig, self.ax = plt.subplots(figsize=(5, 4))
        self.canvas_plot = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas_plot.get_tk_widget().grid(row=0, column=1, padx=20, pady=20)
        
        self.digits = np.arange(10)
        self.prob_bars = self.ax.bar(self.digits, np.zeros(10), color="skyblue")
        self.ax.set_xlabel("数字")
        self.ax.set_ylabel("概率")
        self.ax.set_title("数字概率分布")
        self.ax.set_xticks(self.digits)
        self.ax.set_ylim(0, 1)
        self.prob_texts = [self.ax.text(i, 0.01, "0.000", ha="center", fontsize=8) for i in range(10)]
        
        # 初始化绘图对象
        self.image = Image.new("RGB", (self.canvas_width, self.canvas_height), "white")
        self.draw = ImageDraw.Draw(self.image)
        
        # 绑定鼠标事件
        self.canvas.bind("<B1-Motion>", self.draw_and_predict)
        self.canvas.bind("<Button-1>", self.init_drawing)
        self.canvas.bind("<ButtonRelease-1>", self.reset_last_coords)
    
    def reset_last_coords(self, event):
        self.last_x, self.last_y = None, None
    
    def init_drawing(self, event):
        self.last_x, self.last_y = event.x, event.y
        self.draw_on_canvas(event)
        self.predict_digit_and_update()
    
    def draw_and_predict(self, event):
        self.draw_on_canvas(event)
        current_time = time.time()
        if current_time - self.last_predict_time >= self.predict_interval:
            self.predict_digit_and_update()
            self.last_predict_time = current_time
    
    def draw_on_canvas(self, event):
        x, y = event.x, event.y
        # 绘制点和线
        x1, y1 = (x - self.brush_size), (y - self.brush_size)
        x2, y2 = (x + self.brush_size), (y + self.brush_size)
        self.canvas.create_oval(x1, y1, x2, y2, fill="black", outline="black")
        self.draw.ellipse([x1, y1, x2, y2], fill="black")
        
        if self.last_x is not None:
            self.canvas.create_line(
                self.last_x, self.last_y, x, y, width=self.brush_size*2,
                capstyle=tk.ROUND, joinstyle=tk.ROUND
            )
            self.draw.line(
                [(self.last_x, self.last_y), (x, y)], fill="black", width=self.brush_size*2
            )
        
        self.last_x, self.last_y = x, y
    
    def clear_canvas(self):
        self.canvas.delete("all")
        self.image = Image.new("RGB", (self.canvas_width, self.canvas_height), "white")
        self.draw = ImageDraw.Draw(self.image)
        self.update_prob_plot(np.zeros(10))
    
    def predict_digit_and_update(self):
        prob_np = self.predict_digit()
        self.update_prob_plot(prob_np)
    
    def predict_digit(self):
        try:
            img_tensor = preprocess_for_model(self.image)
            with torch.no_grad():
                logits = self.model(img_tensor.to(self.device))
                prob = torch.softmax(logits, dim=1)
                return prob.cpu().numpy().squeeze()
        except Exception as e:
            print(f"预测错误: {e}")
            return np.zeros(10)
    
    def update_prob_plot(self, prob_np):
        # 更新柱状图和文本
        for i, bar in enumerate(self.prob_bars):
            bar.set_height(prob_np[i])
            bar.set_color("red" if i == np.argmax(prob_np) else "skyblue")
        
        for text in self.prob_texts:
            text.remove()
        self.prob_texts = [
            self.ax.text(i, prob_np[i]+0.01, f"{prob_np[i]:.3f}", ha="center", fontsize=8)
            for i in range(10)
        ]
        
        self.ax.set_ylim(0, min(1.0, max(prob_np)*1.5))
        self.fig.canvas.draw()