import torch
import matplotlib as mpl
from tkinter import ttk

# 设备配置
def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

#  matplotlib中文配置
def setup_matplotlib_fonts():
    mpl.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # Windows
    # mpl.rcParams['font.sans-serif'] = ['PingFang SC']  # Mac
    # mpl.rcParams['font.sans-serif'] = ['DejaVu Sans']  # Linux
    mpl.rcParams['axes.unicode_minus'] = False

# Tkinter中文字体配置
def get_chinese_font():
    return ("微软雅黑", 10)  # Windows
    # return ("PingFang SC", 10)  # Mac
    # return ("WenQuanYi Micro Hei", 10)  # Linux

def setup_ttk_style():
    style = ttk.Style()
    style.configure("Chinese.TButton", font=get_chinese_font())