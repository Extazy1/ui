import sys
import os
import glob
import time
import traceback
from pathlib import Path

# PyQt5界面库
from PyQt5.QtCore import Qt, QTimer, QDir, QRectF
from PyQt5.QtWidgets import (QApplication, QMainWindow, QLabel, QVBoxLayout, 
                            QWidget, QPushButton, QHBoxLayout, QMessageBox, 
                            QFileDialog, QGroupBox, QLineEdit, QFormLayout,
                            QGridLayout, QFrame, QStatusBar, QProgressBar,
                            QSizePolicy, QTextEdit, QScrollArea, QComboBox)
from PyQt5.QtGui import QImage, QPixmap, QIcon, QDoubleValidator, QPainter, QPen, QBrush, QFont, QColor

# 图像处理和模型
import cv2
import numpy as np
import torch
from ultralytics import YOLO

# UI样式常量
class UIStyle:
    # 颜色
    PRIMARY_COLOR = "#6950a1"  # 主题紫色
    SECONDARY_COLOR = "#495057"  # 次要颜色
    BG_COLOR = "#FFFFFF"  # 背景色 - 改为白色
    TEXT_COLOR = "#f8f9fa"  # 文本颜色
    LIGHT_BG = "#f8f9fa"  # 浅色背景
    BORDER_COLOR = "#ced4da"  # 边框颜色
    HOVER_COLOR = "#e9ecef"  # 悬停颜色
    ERROR_BG = "#f8d7da"  # 错误背景
    ERROR_TEXT = "#721c24"  # 错误文本
    WARNING_BG = "#fff3cd"  # 警告背景
    WARNING_TEXT = "#856404"  # 警告文本
    SUCCESS_BG = "#d4edda"  # 成功背景
    SUCCESS_TEXT = "#155724"  # 成功文本
    INFO_COLOR = "#6c757d"  # 信息颜色
    HIGHLIGHT_COLOR = "#ff6b6b"  # 高亮颜色

    # 边框样式
    BORDER_RADIUS = "4px"
    BORDER_WIDTH = "1px"

    # 组件样式
    RESULT_LABEL_STYLE = f"""
        QLabel {{
            border: none;
            border-radius: 5px;
            background-color: {BG_COLOR};
            padding: 5px;
        }}
    """
    
    BUTTON_STYLE = f"""
        QPushButton {{
            background-color: {LIGHT_BG};
            border: {BORDER_WIDTH} solid {BORDER_COLOR};
            border-radius: {BORDER_RADIUS};
            padding: 5px 10px;
            color: {SECONDARY_COLOR};
        }}
        QPushButton:hover {{
            background-color: {HOVER_COLOR};
        }}
    """
    
    ACTION_BUTTON_STYLE = f"""
        QPushButton {{
            background-color: {PRIMARY_COLOR};
            border: {BORDER_WIDTH} solid {PRIMARY_COLOR};
            border-radius: {BORDER_RADIUS};
            padding: 5px 10px;
            color: {TEXT_COLOR};
        }}
        QPushButton:hover {{
            background-color: #5a4380;
        }}
    """
    
    GROUP_BOX_STYLE = f"""
        QGroupBox {{
            border: {BORDER_WIDTH} solid {BORDER_COLOR};
            border-radius: {BORDER_RADIUS};
            margin-top: 15px;
            font-weight: bold;
        }}
        QGroupBox::title {{
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 5px 0 5px;
        }}
    """
    
    HIGHLIGHTED_GROUP_BOX = f"""
        QGroupBox {{
            border: 2px solid {PRIMARY_COLOR};
            border-radius: {BORDER_RADIUS};
            margin-top: 15px;
            font-weight: bold;
        }}
        QGroupBox::title {{
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 5px 0 5px;
        }}
    """
    
    LINE_EDIT_STYLE = f"""
        QLineEdit {{
            border: {BORDER_WIDTH} solid {BORDER_COLOR};
            border-radius: {BORDER_RADIUS};
            padding: 5px;
            background-color: {LIGHT_BG};
            color: {SECONDARY_COLOR};
        }}
        QLineEdit:focus {{
            border: {BORDER_WIDTH} solid {PRIMARY_COLOR};
        }}
    """
    
    COMBO_BOX_STYLE = f"""
        QComboBox {{
            border: {BORDER_WIDTH} solid {BORDER_COLOR};
            border-radius: {BORDER_RADIUS};
            padding: 5px;
            background-color: {LIGHT_BG};
            color: {SECONDARY_COLOR};
        }}
        QComboBox::drop-down {{
            subcontrol-origin: padding;
            subcontrol-position: top right;
            width: 25px;
            border-left: {BORDER_WIDTH} solid {BORDER_COLOR};
        }}
        QComboBox QAbstractItemView {{
            border: {BORDER_WIDTH} solid {BORDER_COLOR};
            selection-background-color: {PRIMARY_COLOR};
            selection-color: {TEXT_COLOR};
        }}
    """
    
    TEXT_EDIT_STYLE = f"""
        QTextEdit {{
            border: {BORDER_WIDTH} solid {BORDER_COLOR};
            border-radius: {BORDER_RADIUS};
            padding: 5px;
            background-color: {LIGHT_BG};
            color: {SECONDARY_COLOR};
            font-family: Arial, sans-serif;
        }}
    """
    
    # HTML模板
    HTML_TEMPLATES = {
        "object_item": """
        <div style='margin-bottom: 10px; padding: 14px; background-color: #f8f9fa; border-radius: 6px; border-left: 4px solid #6950a1;'>
            <div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;'>
                <div style='font-weight: 600; color: #2d3436; font-size: 14px;'>
                    {index}: <span style='color: #6950a1;'>{class_name}</span> 
                    <span style='color: #6c757d; font-size: 13px;'>{conf_text}</span>
                </div>
                <div style='font-size: 12px; color: #adb5bd;'>
                    [尺寸: {width}×{height}]
                </div>
            </div>

        </div>
        """
    }

    # HTML模板
    HTML_TEMPLATES = {
        "object_item": """
        <div style='margin-bottom: 10px; padding: 14px; background-color: #f8f9fa; border-radius: 6px; border-left: 4px solid #6950a1;'>
            <div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;'>
                <div style='font-weight: 600; color: #2d3436; font-size: 14px;'>
                    {index}: <span style='color: #6950a1;'>{class_name}</span> 
                    <span style='color: #6c757d; font-size: 13px;'>{conf_text}</span>
                </div>
                <div style='font-size: 12px; color: #adb5bd;'>
                    [尺寸: {width}×{height}]
                </div>
            </div>
        
        </div>
        """
    }

class Worker:
    def __init__(self):
        self.model = None
        self.video_capture = None
        self.is_video_playing = False
        self.conf_threshold = 0.5
        self.save_dir = "predict/detect"  # 默认保存目录为predict/detect
        self.class_names = {}
        
        # 检测设备
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.imgsz = 640  # 默认推理尺寸
        
        # 如果GPU内存小，使用半精度
        if self.device == 'cuda':
            try:
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
                self.half = gpu_memory < 6  # 如果GPU内存小于6GB则使用半精度
            except:
                self.half = False
        else:
            self.half = False
            
        # 创建保存目录
        os.makedirs(self.save_dir, exist_ok=True)
        
        # 设置默认模型路径
        self.default_model_path = "weights/v8-impbest.pt"

    def load_model(self, model_path=None):
        if model_path is None:
            model_path, _ = QFileDialog.getOpenFileName(None, "选择模型文件", "", "模型文件 (*.pt *.pth *.onnx)")
            if not model_path:
                return False
                
        try:
            # 确保ultralytics库正确初始化
            import os
            import torch
            
            # 检查CUDA是否可用
            cuda_available = torch.cuda.is_available()
            device = 'cuda' if cuda_available else 'cpu'
            
            # 显示加载提示
            QMessageBox.information(None, "提示", f"正在加载模型到{device}设备，请稍候...")
            
            # 使用try-except捕获特定的错误
            try:
                self.model = YOLO(model_path)
                # 预热模型
                dummy_img = torch.zeros((1, 3, 640, 640)).to(device)
                self.model.predict(source=dummy_img, verbose=False)
                
                if cuda_available:
                    msg = f"模型已加载到GPU: {torch.cuda.get_device_name(0)}"
                else:
                    msg = "模型已加载到CPU"
                QMessageBox.information(None, "成功", msg)
                return True
            except RuntimeError as e:
                if 'CUDA out of memory' in str(e):
                    QMessageBox.critical(None, "GPU内存不足", f"GPU内存不足，请尝试使用CPU或较小的模型: {str(e)}")
                else:
                    QMessageBox.critical(None, "运行时错误", f"模型运行出错: {str(e)}")
                return False
        except Exception as e:
            QMessageBox.critical(None, "错误", f"模型加载失败: {str(e)}")
            return False
        return False
 
    def detect_image(self, image, save=False, filename=None, is_batch=False):
        if self.model:
            try:
                # 获取当前置信度阈值
                conf_threshold = self.conf_threshold
                
                # 确保保存目录存在
                os.makedirs(self.save_dir, exist_ok=True)
                
                # 设置保存路径和保存选项
                save_path = None
                # 在批量模式下，禁用YOLO自带的保存以避免重复保存
                yolo_save = save and not is_batch
                
                if save and filename:
                    save_path = os.path.join(self.save_dir, filename)
                
                # 添加更多参数控制
                results = self.model.predict(
                    source=image,
                    conf=conf_threshold,
                    save=yolo_save,  # 只在非批量模式下使用YOLO的保存
                    project=self.save_dir,
                    name="",  # 空名字，这样会保持原始文件名
                    imgsz=self.imgsz,  # 使用类中设置的推理大小
                    augment=False,  # 是否使用数据增强
                    half=self.half,  # 使用类中设置的精度
                    device=self.device,  # 使用类中设置的设备
                    verbose=False  # 不显示处理信息
                )
                
                # 如果是批量模式且需要保存，手动保存检测结果
                if is_batch and save and filename and results:
                    try:
                        annotated_image = results[0].plot()
                        # 在批量模式下直接使用原始文件名
                        save_path = os.path.join(self.save_dir, filename)
                        cv2.imwrite(save_path, annotated_image)
                    except Exception as e:
                        print(f"保存批量检测结果失败: {str(e)}")
                
                return results
            except torch.cuda.OutOfMemoryError:
                QMessageBox.warning(None, "GPU内存不足", "GPU内存不足，自动切换到CPU推理")
                # 切换到CPU
                self.device = 'cpu'
                self.half = False
                try:
                    # 使用CPU重试
                    results = self.model.predict(
                        source=image,
                        conf=self.conf_threshold,
                        save=yolo_save,  # 使用修改后的yolo_save参数
                        project=self.save_dir,
                        name="",
                        device='cpu',
                        verbose=False
                    )
                    
                    # 如果是批量模式且需要保存，手动保存检测结果
                    if is_batch and save and filename and results:
                        try:
                            annotated_image = results[0].plot()
                            # 在批量模式下直接使用原始文件名
                            save_path = os.path.join(self.save_dir, filename)
                            cv2.imwrite(save_path, annotated_image)
                        except Exception as e:
                            print(f"保存批量检测结果失败: {str(e)}")
                    
                    return results
                except Exception as e:
                    QMessageBox.warning(None, "警告", f"CPU推理也失败: {str(e)}")
                    traceback.print_exc()  # 打印详细错误
                    return None
            except Exception as e:
                QMessageBox.warning(None, "警告", f"检测过程中出现错误: {str(e)}")
                traceback.print_exc()  # 打印详细错误
                return None
        else:
            QMessageBox.warning(None, "警告", "请先加载模型")
            return None
        
    def process_folder(self, folder_path):
        """处理文件夹中的图片"""
        image_list = []
        for ext in ['jpg', 'jpeg', 'png']:
            image_list.extend(glob.glob(os.path.join(folder_path, f'*.{ext}')))
            image_list.extend(glob.glob(os.path.join(folder_path, f'*.{ext.upper()}')))
        
        if not image_list:
            return []
            
        return sorted(image_list)  # 排序，以确保处理顺序

    def start_video_capture(self, video_path):
        try:
            self.video_capture = cv2.VideoCapture(video_path)
            return self.video_capture.isOpened()
        except Exception:
            return False

    def stop_video_capture(self):
        if self.video_capture:
            self.video_capture.release()
            self.video_capture = None
            self.is_video_playing = False
            
    def start_camera(self):
        try:
            self.video_capture = cv2.VideoCapture(0)
            return self.video_capture.isOpened()
        except Exception:
            return False

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("道路目标检测")
        self.setGeometry(100, 100, 1200, 700)
        
        # 初始化Worker和其他变量
        self.worker = Worker()
        self.current_results = None
        self.current_image_index = 0
        self.image_list = []
        self.is_folder_mode = False
        
        # 初始化提取的数据字典
        self.extracted_data = {
            'success': False,
            'object_count': 0,
            'classes': [],
            'confidences': [],
            'boxes': [],
            'class_names': {},
            'original_image': None
        }
        
        # 创建保存目录（如果不存在）
        os.makedirs(self.worker.save_dir, exist_ok=True)
 
        # 主布局为水平布局
        main_layout = QHBoxLayout()
        
        # 左侧控制面板 (垂直布局)
        left_panel = QVBoxLayout()
        
        # 创建模型输入源分组
        self.create_input_group()
        left_panel.addWidget(self.input_group)
        
        # 创建参数设置分组
        self.create_params_group()
        left_panel.addWidget(self.params_group)
        
        # 创建检测结果信息区域
        self.create_detection_info_area()
        left_panel.addWidget(self.detection_info_group)
        
        # 创建控制按钮组
        self.create_control_buttons()
        left_panel.addWidget(self.control_group)
        
        # 为左侧面板创建一个容器
        left_widget = QWidget()
        left_widget.setLayout(left_panel)
        left_widget.setFixedWidth(280)  # 稍微减小宽度，使界面更紧凑
        
        # 右侧检测结果显示区域
        self.result_label = QLabel()
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setStyleSheet('''
            QLabel {
                border: none;
                border-radius: 5px;
                background-color: #FFFFFF;
                padding: 5px;
            }
        ''')
        self.result_label.setMinimumSize(640, 480)  # 设置最小尺寸
        self.result_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)  # 允许扩展但保持比例
        
        # 创建初始白色背景
        initial_pixmap = QPixmap(self.result_label.width(), self.result_label.height())
        initial_pixmap.fill(Qt.white)  # 初始时使用白色背景
        self.result_label.setPixmap(initial_pixmap)
        
        # 添加鼠标点击事件
        self.result_label.mousePressEvent = self.handle_image_click
        
        # 添加到主布局
        main_layout.addWidget(left_widget)
        main_layout.addWidget(self.result_label, 1)  # 1表示拉伸比例
        
        # 设置主窗口
        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)
        
        # 创建状态栏
        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)
        self.statusBar.showMessage("就绪")
        
        # 创建进度条
        self.progressBar = QProgressBar()
        self.progressBar.setRange(0, 100)
        self.progressBar.setValue(0)
        self.progressBar.setVisible(False)
        self.statusBar.addPermanentWidget(self.progressBar)
        
        # 创建定时器用于视频播放
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_video_frame)
        
        # 在UI初始化完成后加载默认模型
        QTimer.singleShot(100, self.load_default_model)

    def create_input_group(self):
        self.input_group = QGroupBox("模型输入源")
        input_layout = QGridLayout()
        
        # 图片输入
        img_btn = QPushButton("图片")
        img_btn.clicked.connect(self.detect_single_image)
        img_path = QLineEdit()
        img_path.setReadOnly(True)
        img_path.setPlaceholderText("选择图片文件...")
        
        # 文件夹输入
        folder_btn = QPushButton("文件夹")
        folder_btn.clicked.connect(self.select_folder)
        folder_path = QLineEdit()
        folder_path.setReadOnly(True)
        folder_path.setPlaceholderText("选择文件夹...")
        
        # 视频输入
        video_btn = QPushButton("视频")
        video_btn.clicked.connect(self.select_video)
        video_path = QLineEdit()
        video_path.setReadOnly(True)
        video_path.setPlaceholderText("选择视频文件...")
        
        # 摄像头
        camera_btn = QPushButton("摄像头")
        camera_btn.clicked.connect(self.start_camera)
        
        # 添加到布局
        input_layout.addWidget(img_btn, 0, 0)
        input_layout.addWidget(img_path, 0, 1)
        input_layout.addWidget(folder_btn, 1, 0)
        input_layout.addWidget(folder_path, 1, 1)
        input_layout.addWidget(video_btn, 2, 0)
        input_layout.addWidget(video_path, 2, 1)
        input_layout.addWidget(camera_btn, 3, 0)
        
        self.input_group.setLayout(input_layout)
        
        # 保存引用以便后续更新
        self.img_path = img_path
        self.folder_path = folder_path
        self.video_path = video_path
        
        # 禁用这些按钮，直到模型加载
        img_btn.setEnabled(False)
        folder_btn.setEnabled(False)
        video_btn.setEnabled(False)
        camera_btn.setEnabled(False)
        
        # 保存按钮引用
        self.img_btn = img_btn
        self.folder_btn = folder_btn
        self.video_btn = video_btn
        self.camera_btn = camera_btn
    
    def create_params_group(self):
        self.params_group = QGroupBox("检测参数设置")
        params_layout = QFormLayout()
        
        # 创建参数输入控件
        self.conf_input = QLineEdit("0.5")
        self.conf_input.setValidator(QDoubleValidator(0.0, 1.0, 2))
        self.conf_input.textChanged.connect(self.update_conf)
        
        # 添加保存目录
        save_dir_btn = QPushButton("选择...")
        save_dir_btn.clicked.connect(self.select_save_dir)
        save_dir_path = QLineEdit(self.worker.save_dir)
        save_dir_path.setReadOnly(True)
        
        # 添加到布局
        params_layout.addRow("置信度:", self.conf_input)
        params_layout.addRow("保存目录:", save_dir_path)
        params_layout.addRow("", save_dir_btn)
        
        self.params_group.setLayout(params_layout)
        
        # 保存引用
        self.save_dir_path = save_dir_path

    def create_detection_info_area(self):
        """创建检测信息区域"""
        # 创建检测信息分组
        self.detection_info_group = QGroupBox("检测信息")
        detection_info_layout = QVBoxLayout()
        
        # 添加目标选择下拉框
        target_selection_layout = QHBoxLayout()
        target_selection_layout.addWidget(QLabel("选择目标:"))
        self.target_combobox = QComboBox()
        self.target_combobox.addItem("全部")  
        self.target_combobox.setItemData(0, -1)  # 设置"全部"选项的数据为-1
        self.target_combobox.currentIndexChanged.connect(self.on_target_changed)
        target_selection_layout.addWidget(self.target_combobox)
        detection_info_layout.addLayout(target_selection_layout)
        
        # 创建检测结果文本显示区
        self.detection_text = QTextEdit()
        self.detection_text.setReadOnly(True)
        self.detection_text.setMinimumHeight(200)
        detection_info_layout.addWidget(self.detection_text)
        
        # 添加按钮
        buttons_layout = QHBoxLayout()
        self.clear_info_button = QPushButton("清除信息")
        self.clear_info_button.setStyleSheet("""
            QPushButton {
                background-color: #f8f9fa;
                border: 1px solid #ced4da;
                border-radius: 4px;
                padding: 5px 10px;
                color: #495057;
            }
            QPushButton:hover {
                background-color: #e9ecef;
            }
        """)
        self.clear_info_button.clicked.connect(self.clear_detection_info)
        
        self.copy_info_button = QPushButton("复制信息")
        self.copy_info_button.setStyleSheet("""
            QPushButton {
                background-color: #f8f9fa;
                border: 1px solid #ced4da;
                border-radius: 4px;
                padding: 5px 10px;
                color: #495057;
            }
            QPushButton:hover {
                background-color: #e9ecef;
            }
        """)
        self.copy_info_button.clicked.connect(self.copy_detection_info)
        
        buttons_layout.addWidget(self.clear_info_button)
        buttons_layout.addWidget(self.copy_info_button)
        detection_info_layout.addLayout(buttons_layout)
        
        # 设置QGroupBox的布局
        self.detection_info_group.setLayout(detection_info_layout)
        return self.detection_info_group

    def create_control_buttons(self):
        self.control_group = QGroupBox("系统运行与退出")
        control_layout = QVBoxLayout()
        
        # 添加模型选择按钮
        self.load_model_button = QPushButton("📁 模型选择")
        self.load_model_button.clicked.connect(self.load_model)
        
        # 视频操作按钮布局
        video_controls = QHBoxLayout()
        
        # 添加视频控制按钮
        self.video_control_button = QPushButton("⏯️ 播放")
        self.video_control_button.clicked.connect(self.toggle_video)
        self.video_control_button.setEnabled(False)
        
        # 添加停止检测按钮
        self.stop_detect_button = QPushButton("⏹️ 停止")
        self.stop_detect_button.clicked.connect(self.stop_detection)
        self.stop_detect_button.setEnabled(False)
        
        video_controls.addWidget(self.video_control_button)
        video_controls.addWidget(self.stop_detect_button)
        
        # 添加显示检测物体按钮
        self.display_objects_button = QPushButton("🔍 显示检测结果")
        self.display_objects_button.clicked.connect(self.show_detected_objects)
        self.display_objects_button.setEnabled(False)
        
        # 添加保存结果按钮
        self.save_result_button = QPushButton("💾 保存检测结果")
        self.save_result_button.clicked.connect(self.save_detection_result)
        self.save_result_button.setEnabled(False)
        
        # 添加退出按钮
        self.exit_button = QPushButton("❌ 退出系统")
        self.exit_button.clicked.connect(self.exit_application)
        
        # 添加到布局
        control_layout.addWidget(self.load_model_button)
        control_layout.addLayout(video_controls)
        control_layout.addWidget(self.display_objects_button)
        control_layout.addWidget(self.save_result_button)
        control_layout.addWidget(self.exit_button)
        
        self.control_group.setLayout(control_layout)

    def select_save_dir(self):
        dir_path = QFileDialog.getExistingDirectory(self, "选择保存目录", self.worker.save_dir)
        if dir_path:
            self.worker.save_dir = dir_path
            self.save_dir_path.setText(dir_path)
            self.statusBar.showMessage(f"保存目录设置为: {dir_path}")

    def update_conf(self):
        try:
            value = float(self.conf_input.text())
            if 0 <= value <= 1:
                self.worker.conf_threshold = value
                self.statusBar.showMessage(f"置信度阈值已更新为: {value}")
        except ValueError:
            pass
            
    def select_folder(self):
        folder_path = QFileDialog.getExistingDirectory(self, "选择文件夹")
        if folder_path:
            self.folder_path.setText(folder_path)
            self.statusBar.showMessage("正在搜索文件夹中的图片...")
            
            # 处理文件夹中的图片
            self.image_list = self.worker.process_folder(folder_path)
            if not self.image_list:
                self.show_message_box("提示", "文件夹中没有找到支持的图片格式(jpg, jpeg, png)")
                self.statusBar.showMessage("未找到可处理的图片")
                return
                
            self.is_folder_mode = True
            self.current_image_index = 0
            
            # 为本次批量处理创建唯一的目标文件夹（使用时间戳）
            folder_name = os.path.basename(folder_path)
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            batch_save_dir = os.path.join(self.worker.save_dir, f"batch_{folder_name}_{timestamp}")
            os.makedirs(batch_save_dir, exist_ok=True)
            
            # 保存原始保存路径，稍后恢复
            self.original_save_dir = self.worker.save_dir
            # 设置为批处理专用目录
            self.worker.save_dir = batch_save_dir
            
            total_images = len(self.image_list)
            self.statusBar.showMessage(f"找到 {total_images} 张图片，开始处理...结果将保存到 {batch_save_dir}")
            
            # 设置进度条
            self.progressBar.setRange(0, total_images)
            self.progressBar.setValue(0)
            self.progressBar.setVisible(True)
            
            # 重置之前的结果
            self.display_objects_button.setEnabled(False)
            self.save_result_button.setEnabled(False)
            
            # 开始处理图片，使用QTimer确保界面先更新
            QTimer.singleShot(100, self.process_current_image)
            self.stop_detect_button.setEnabled(True)

    def detect_single_image(self):
        image_path, _ = QFileDialog.getOpenFileName(self, "选择图片", "", "图片文件 (*.jpg *.jpeg *.png)")
        if image_path:
            self.img_path.setText(image_path)
            self.statusBar.showMessage("处理图片中...")
            
            try:
                # 读取图片
                image = cv2.imread(image_path)
                if image is None:
                    self.show_message_box("错误", "无法读取图片文件！")
                    self.statusBar.showMessage("图片读取失败")
                    return
                
                # 保存原始图像以供后续处理使用
                self.current_image = image.copy()
                    
                # 创建图片列表，以便重用处理逻辑
                self.image_list = [image_path]
                self.is_folder_mode = False
                self.current_image_index = 0
                
                # 开始处理
                self.process_current_image()
                self.stop_detect_button.setEnabled(True)
                
            except Exception as e:
                self.show_message_box("错误", f"处理图片时出错: {str(e)}")
                self.statusBar.showMessage("图片处理失败")

    def process_current_image(self):
        # 检查是否有图片待处理
        if not self.image_list or self.current_image_index >= len(self.image_list):
            self.progressBar.setVisible(False)
            self.statusBar.showMessage("没有图片需要处理")
            return
            
        try:
            # 获取当前图片路径
            image_path = self.image_list[self.current_image_index]
            filename = os.path.basename(image_path)
            
            # 更新状态栏和进度条
            self.statusBar.showMessage(f"处理中: {filename} [{self.current_image_index+1}/{len(self.image_list)}]")
            self.progressBar.setValue(self.current_image_index + 1)
            
            # 读取图片
            image = cv2.imread(image_path)
            if image is None:
                self.show_message_box("警告", f"无法读取图片: {filename}")
                # 跳到下一张
                self.current_image_index += 1
                if self.current_image_index < len(self.image_list):
                    QTimer.singleShot(100, self.process_current_image)
                else:
                    self.finalize_processing()
                return
            
            # 保存原始图像用于后续处理
            self.current_image = image.copy()
                
            # 检测图片，并保存结果
            # 在文件夹模式下启用保存，但通过is_batch参数标记这是批量处理
            save_option = True  # 始终启用保存
            self.current_results = self.worker.detect_image(
                image, 
                save=save_option, 
                filename=filename,
                is_batch=self.is_folder_mode  # 标记是否为批量处理
            )
            
            if self.current_results:
                # 立即提取所需数据，避免后续递归访问
                self.extracted_data = self.extract_detection_data(self.current_results)
                
                # 处理检测结果
                annotated_image = self.current_results[0].plot()
                annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
                
                # 转换为QImage和QPixmap
                height, width, channel = annotated_image.shape
                bytesPerLine = 3 * width
                qimage = QImage(annotated_image.data, width, height, bytesPerLine, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(qimage)
                
                # 使用固定大小的缩放显示
                display_pixmap = pixmap.scaled(
                    self.result_label.width(), 
                    self.result_label.height(),
                    Qt.KeepAspectRatio, 
                    Qt.SmoothTransformation
                )
                
                # 更新UI
                self.result_label.setPixmap(display_pixmap)
                self.display_objects_button.setEnabled(True)
                self.save_result_button.setEnabled(True)
                
                # 更新检测信息显示区域 - 使用提取的数据
                self.update_detection_info_safe(skip_combobox_update=False)
                
            # 处理下一张图片
            self.current_image_index += 1
            if self.current_image_index < len(self.image_list):
                # 使用QTimer来保证UI更新和异步处理
                QTimer.singleShot(100, self.process_current_image)
            else:
                self.finalize_processing()
                
        except Exception as e:
            self.show_message_box("错误", f"处理图片时出错: {str(e)}")
            # 继续处理下一张
            self.current_image_index += 1
            if self.current_image_index < len(self.image_list):
                QTimer.singleShot(100, self.process_current_image)
            else:
                self.finalize_processing()
    
    def finalize_processing(self):
        """完成所有图片处理后的收尾工作"""
        if self.is_folder_mode:
            batch_dir = self.worker.save_dir
            # 恢复原始保存路径
            if hasattr(self, 'original_save_dir'):
                self.worker.save_dir = self.original_save_dir
            self.statusBar.showMessage(f"批量处理完成，共检测 {len(self.image_list)} 张图片，结果已保存到 {batch_dir}")
        else:
            self.statusBar.showMessage(f"处理完成，结果已保存到 {self.worker.save_dir}")
        self.progressBar.setVisible(False)

    def select_video(self):
        video_path, _ = QFileDialog.getOpenFileName(self, "选择视频文件", "", "视频文件 (*.mp4 *.avi *.mov)")
        if video_path:
            self.video_path.setText(video_path)
            success = self.worker.start_video_capture(video_path)
            if success:
                self.worker.is_video_playing = True
                self.video_control_button.setEnabled(True)
                self.stop_detect_button.setEnabled(True)
                self.display_objects_button.setEnabled(False)
                self.save_result_button.setEnabled(False)
                
                # 启动定时器更新视频帧
                self.timer = QTimer()
                self.timer.timeout.connect(self.update_video_frame)
                self.timer.start(30)  # 30毫秒更新一次，约33帧/秒
                self.video_control_button.setText("暂停")
                self.statusBar.showMessage("正在播放视频...")
            else:
                self.show_message_box("错误", "无法打开视频文件")

    def update_video_frame(self):
        if self.worker.video_capture and self.worker.is_video_playing:
            ret, frame = self.worker.video_capture.read()
            if ret:
                # 保存当前帧用于后续处理
                self.current_image = frame.copy()
                
                # 检测并显示结果
                self.current_results = self.worker.detect_image(frame)
                if self.current_results:
                    # 立即提取所需数据，避免后续递归访问
                    self.extracted_data = self.extract_detection_data(self.current_results)
                    
                    # 处理检测结果
                    try:
                        annotated_frame = self.current_results[0].plot()
                        annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                        height, width, channel = annotated_frame.shape
                        bytesPerLine = 3 * width
                        qimage = QImage(annotated_frame.data, width, height, bytesPerLine, QImage.Format_RGB888)
                        pixmap = QPixmap.fromImage(qimage)
                        
                        # 使用固定大小的缩放
                        display_pixmap = pixmap.scaled(
                            self.result_label.width(), 
                            self.result_label.height(),
                            Qt.KeepAspectRatio, 
                            Qt.SmoothTransformation
                        )
                        self.result_label.setPixmap(display_pixmap)
                        self.display_objects_button.setEnabled(True)
                        self.save_result_button.setEnabled(True)
                        
                        # 更新状态栏显示目标数量
                        self.statusBar.showMessage(f"视频检测中 - 当前帧检测到 {self.extracted_data['object_count']} 个目标")
                        
                    except Exception as e:
                        # 如果注释图像生成失败，显示原始帧
                        print(f"生成注释帧时出错: {str(e)}")
                        self.statusBar.showMessage(f"生成注释帧时出错: {str(e)}")
                        try:
                            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            height, width, channel = frame_rgb.shape
                            bytesPerLine = 3 * width
                            qimage = QImage(frame_rgb.data, width, height, bytesPerLine, QImage.Format_RGB888)
                            pixmap = QPixmap.fromImage(qimage)
                            display_pixmap = pixmap.scaled(
                                self.result_label.width(), 
                                self.result_label.height(),
                                Qt.KeepAspectRatio, 
                                Qt.SmoothTransformation
                            )
                            self.result_label.setPixmap(display_pixmap)
                        except Exception as e:
                            print(f"显示原始帧也失败: {str(e)}")
                            self.statusBar.showMessage(f"显示原始帧也失败: {str(e)}")
                    
                    # 更新检测信息显示区域 - 视频模式下频率稍低
                    if hasattr(self, 'last_update_time'):
                        current_time = time.time()
                        if current_time - self.last_update_time > 0.5:  # 每0.5秒更新一次
                            # 更新下拉框和检测信息
                            self.update_target_combobox_safe(self.extracted_data)
                            self.update_detection_info_safe(skip_combobox_update=True)
                            self.last_update_time = current_time
                    else:
                        self.last_update_time = time.time()
                        # 更新下拉框和检测信息
                        self.update_target_combobox_safe(self.extracted_data)
                        self.update_detection_info_safe(skip_combobox_update=True)
                else:
                    self.statusBar.showMessage("当前帧未检测到目标")
            else:
                self.worker.stop_video_capture()
                self.video_control_button.setEnabled(False)
                self.stop_detect_button.setEnabled(False)
                self.statusBar.showMessage("视频播放完成")
                self.show_message_box("提示", "视频播放完成")

    def toggle_video(self):
        if self.worker.video_capture:
            if self.worker.is_video_playing:
                self.timer.stop()
                self.worker.is_video_playing = False
                self.video_control_button.setText("继续播放")
                self.statusBar.showMessage("视频已暂停")
            else:
                self.timer.start(30)
                self.worker.is_video_playing = True
                self.video_control_button.setText("暂停")
                self.statusBar.showMessage("继续播放视频...")
                
    def start_camera(self):
        self.statusBar.showMessage("正在打开摄像头...")
        if self.worker.start_camera():
            self.video_control_button.setEnabled(True)
            self.video_control_button.setText("⏯️ 暂停")
            self.timer.start(30)
            self.worker.is_video_playing = True
            self.stop_detect_button.setEnabled(True)
            self.save_result_button.setEnabled(True)
            self.statusBar.showMessage("摄像头检测中...")
        else:
            self.show_message_box("错误", "无法打开摄像头")
            self.statusBar.showMessage("摄像头打开失败")
                
    def stop_detection(self):
        if self.worker.video_capture:
            self.timer.stop()
            self.worker.stop_video_capture()
            
        # 如果是文件夹模式，恢复原始保存路径
        if self.is_folder_mode and hasattr(self, 'original_save_dir'):
            self.worker.save_dir = self.original_save_dir
            
        self.is_folder_mode = False
        self.image_list = []
        self.current_image_index = 0
        
        self.video_control_button.setEnabled(False)
        self.stop_detect_button.setEnabled(False)
        self.display_objects_button.setEnabled(False)
        self.save_result_button.setEnabled(False)
        self.progressBar.setVisible(False)
        self.statusBar.showMessage("检测已停止")

    def show_detected_objects(self):
        """显示检测到的物体信息，现在只是切换到检测信息区域"""
        if hasattr(self, 'extracted_data') and self.extracted_data['success']:
            # 更新检测信息显示
            self.update_detection_info_safe(skip_combobox_update=False)
            # 可以添加一些特效，比如高亮检测信息区域
            self.detection_info_group.setStyleSheet("QGroupBox { border: 2px solid #6950a1; }")
            # 5秒后恢复正常样式
            QTimer.singleShot(5000, lambda: self.detection_info_group.setStyleSheet(""))
        else:
            self.detection_text.setText("未检测到物体")
            self.show_message_box("提示", "未检测到物体或数据提取失败")
 
    def show_message_box(self, title, message):
        msg_box = QMessageBox(self)
        msg_box.setWindowTitle(title)
        msg_box.setText(message)
        msg_box.exec_()
 
    def load_model(self):
        self.statusBar.showMessage("正在准备加载模型...")
        success = self.worker.load_model()
        if success:
            self.img_btn.setEnabled(True)
            self.folder_btn.setEnabled(True)
            self.video_btn.setEnabled(True)
            self.camera_btn.setEnabled(True)
            self.statusBar.showMessage("模型加载成功，可以开始检测")
            
            # 创建保存目录（如果不存在）
            os.makedirs(self.worker.save_dir, exist_ok=True)
            
            # 更新UI提示
            self.params_group.setTitle(f"检测参数设置 - 模型已加载")
            
            return True
        else:
            self.statusBar.showMessage("模型加载失败，请重试")
            return False
 
    def exit_application(self):
        # 确认对话框
        reply = QMessageBox.question(self, '确认退出', '确定要退出应用程序吗？', 
                                    QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        
        if reply == QMessageBox.Yes:
            # 停止所有活动
            self.worker.stop_video_capture()
            if self.timer.isActive():
                self.timer.stop()
            # 直接关闭窗口而不是调用sys.exit()
            self.close()

    def closeEvent(self, event):
        """在关闭窗口前询问用户确认"""
        # 如果是通过exit_application方法关闭的，直接接受事件
        if hasattr(self, '_close_confirmed') and self._close_confirmed:
            event.accept()
            return
            
        reply = QMessageBox.question(
            self, 
            '确认退出', 
            "确定要退出程序吗？", 
            QMessageBox.Yes | QMessageBox.No, 
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            # 停止所有活动
            self.worker.stop_video_capture()
            if self.timer.isActive():
                self.timer.stop()
            # 标记已确认关闭
            self._close_confirmed = True
            event.accept()
        else:
            event.ignore()

    def save_detection_result(self):
        if self.current_results:
            save_path, _ = QFileDialog.getSaveFileName(self, "保存检测结果", "", "图片文件 (*.jpg *.png)")
            if save_path:
                try:
                    # 获取当前显示的检测结果图像
                    current_pixmap = self.result_label.pixmap()
                    if current_pixmap:
                        saved = current_pixmap.save(save_path)
                        if saved:
                            self.statusBar.showMessage(f"检测结果已保存到: {save_path}")
                        else:
                            self.show_message_box("错误", "保存检测结果失败")
                except Exception as e:
                    self.show_message_box("错误", f"保存检测结果时出错: {str(e)}")
        else:
            self.show_message_box("提示", "没有检测结果可保存")

    def load_default_model(self):
        """加载默认模型"""
        if os.path.exists(self.worker.default_model_path):
            self.statusBar.showMessage("正在加载默认模型...")
            try:
                success = self.worker.load_model(self.worker.default_model_path)
                if success:
                    self.img_btn.setEnabled(True)
                    self.folder_btn.setEnabled(True)
                    self.video_btn.setEnabled(True)
                    self.camera_btn.setEnabled(True)
                    self.statusBar.showMessage("默认模型加载成功，可以开始检测")
                    self.params_group.setTitle(f"检测参数设置 - 模型已加载")
                else:
                    self.statusBar.showMessage("默认模型加载失败，请手动选择模型")
            except Exception as e:
                self.statusBar.showMessage(f"默认模型加载失败: {str(e)}")
                print(f"默认模型加载失败: {str(e)}")
        else:
            self.statusBar.showMessage("默认模型路径不存在，请手动选择模型")
            print("默认模型路径不存在，请手动选择模型")

    def clear_detection_info(self):
        """清除检测信息"""
        self.detection_text.clear()
        self.target_combobox.clear()
        self.target_combobox.addItem("全部")
        self.target_combobox.setItemData(0, -1)  # 设置"全部"选项的数据为-1
        
        # 清除图像并设置白色背景
        if self.result_label.pixmap():
            # 创建一个空白pixmap显示在result_label上，使用白色背景
            pixmap = QPixmap(self.result_label.width(), self.result_label.height())
            pixmap.fill(Qt.white)  # 填充白色背景
            self.result_label.setPixmap(pixmap)
        
        self.current_results = None
        self.extracted_data = {
            'success': False,
            'object_count': 0,
            'classes': [],
            'confidences': [],
            'boxes': [],
            'class_names': {},
            'original_image': None
        }
        
        self.statusBar.showMessage("已清除检测信息")

    def copy_detection_info(self):
        """复制检测信息到剪贴板"""
        if self.detection_text.toPlainText():
            clipboard = QApplication.clipboard()
            clipboard.setText(self.detection_text.toPlainText())
            self.statusBar.showMessage("检测信息已复制到剪贴板", 3000)
        else:
            self.statusBar.showMessage("没有检测信息可复制", 3000)

    def handle_image_click(self, event):
        """处理图片点击事件，选择单个物体"""
        self.handle_image_click_safe(event)

    def handle_image_click_safe(self, event):
        """使用提取的数据处理图片点击事件，选择单个物体，避免递归问题"""
        if not hasattr(self, 'extracted_data') or not self.extracted_data['success']:
            return
            
        try:
            data = self.extracted_data
            
            # 获取点击坐标（显示坐标系中的坐标）
            click_x = event.pos().x()
            click_y = event.pos().y()
            
            # 获取图片显示尺寸
            pixmap = self.result_label.pixmap()
            if not pixmap:
                return
                
            img_width = pixmap.width()
            img_height = pixmap.height()
            
            # 获取原始图像尺寸
            if data['original_image'] is None:
                return
                
            orig_img = data['original_image']
            orig_height, orig_width = orig_img.shape[:2]
            
            # 计算图像在显示区域中的实际位置（处理黑边情况）
            # 因为我们使用KeepAspectRatio缩放，图像可能没有填满整个显示区域
            label_width = self.result_label.width()
            label_height = self.result_label.height()
            
            # 计算缩放比例
            scale_factor = min(label_width / orig_width, label_height / orig_height)
            
            # 计算缩放后的图像大小
            scaled_width = int(orig_width * scale_factor)
            scaled_height = int(orig_height * scale_factor)
            
            # 计算图像在标签中的偏移（居中显示）
            offset_x = (label_width - scaled_width) // 2
            offset_y = (label_height - scaled_height) // 2
            
            # 检查点击是否在图像区域内
            if (click_x < offset_x or click_x >= offset_x + scaled_width or
                click_y < offset_y or click_y >= offset_y + scaled_height):
                # 点击在黑边区域，不在图像上
                return
            
            # 调整点击坐标，去除偏移量
            adjusted_click_x = click_x - offset_x
            adjusted_click_y = click_y - offset_y
            
            # 将点击坐标从显示坐标系转换到原始图像坐标系
            orig_click_x = int(adjusted_click_x / scale_factor)
            orig_click_y = int(adjusted_click_y / scale_factor)
            
            # 使用原始坐标直接调用highlight_selected_object_safe
            selected_index = self.highlight_selected_object_safe(orig_click_x, orig_click_y)
            
            # 如果找到点击的物体，更新下拉框选择 - 注意这部分已移到highlight_selected_object_safe中处理
            # 不需要重复处理
        except Exception as e:
            error_msg = f"处理图像点击事件时出错: {str(e)}"
            print(error_msg)
            traceback.print_exc()
            self.statusBar.showMessage(error_msg)

    def highlight_selected_object_safe(self, x, y):
        """安全地高亮显示用户点击的对象所属的类别"""
        try:
            if not hasattr(self, 'extracted_data') or not self.extracted_data['success']:
                return -1
                
            data = self.extracted_data
            boxes = data['boxes']
            
            # 查找包含点击位置的边界框
            selected_index = -1
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = box
                if x >= x1 and x <= x2 and y >= y1 and y <= y2:
                    selected_index = i
                    break
            
            # 如果找到了目标
            if selected_index >= 0:
                # 获取该目标的类别ID
                selected_class_id = data['classes'][selected_index]
                
                # 暂停信号以防止循环调用
                self.target_combobox.blockSignals(True)
                
                # 在下拉框中查找该类别
                combobox_index = -1
                for i in range(self.target_combobox.count()):
                    class_id = self.target_combobox.itemData(i)
                    if class_id == selected_class_id:
                        combobox_index = i
                        # 更新下拉框的选择
                        self.target_combobox.setCurrentIndex(i)
                        break
                
                # 恢复信号
                self.target_combobox.blockSignals(False)
                
                # 如果找到了对应的下拉框项，触发on_target_changed事件
                if combobox_index >= 0:
                    # 手动调用on_target_changed以更新显示和信息
                    self.on_target_changed()
                else:
                    # 如果没有找到对应的下拉框项（不应该发生），直接显示对应的类别
                    self.show_class_targets_safe(data, selected_class_id)
                
            return selected_index
            
        except Exception as e:
            print(f"高亮显示对象时出错: {str(e)}")
            traceback.print_exc()
            return -1

    def extract_detection_data(self, results):
        """从检测结果中提取必要数据，转换为普通的Python数据结构，避免递归问题"""
        try:
            # 初始化结果字典
            extraction = {
                'success': False,
                'object_count': 0,
                'classes': [],
                'confidences': [],
                'boxes': [],
                'class_names': {},
                'original_image': None
            }
            
            if not results or len(results) == 0:
                return extraction
                
            # 尝试安全获取原始图像
            try:
                if hasattr(results[0], 'orig_img'):
                    extraction['original_image'] = results[0].orig_img.copy()
            except Exception as e:
                print(f"提取原始图像失败: {str(e)}")
            
            # 尝试安全获取类别名称字典
            try:
                if hasattr(results[0], 'names'):
                    extraction['class_names'] = dict(results[0].names)
            except Exception as e:
                print(f"提取类别名称失败: {str(e)}")
            
            # 尝试从boxes提取信息
            if hasattr(results[0], 'boxes'):
                boxes = results[0].boxes
                
                # 安全处理：使用索引方式尝试获取长度，避免使用len()
                try:
                    # 使用逐个检查索引是否有效的方式估计长度
                    count = 0
                    max_check = 1000  # 安全限制
                    while count < max_check:
                        try:
                            # 尝试访问索引为count的元素
                            if hasattr(boxes, 'xyxy'):
                                _ = boxes.xyxy[count]
                                count += 1
                            else:
                                break
                        except (IndexError, AttributeError):
                            break
                    
                    extraction['object_count'] = count
                    
                    # 提取各目标的信息
                    for i in range(count):
                        try:
                            # 提取类别
                            if hasattr(boxes, 'cls') and i < len(boxes.cls):
                                cls_val = int(boxes.cls[i].item())
                                extraction['classes'].append(cls_val)
                            else:
                                extraction['classes'].append(-1)  # 未知类别
                                
                            # 提取置信度
                            if hasattr(boxes, 'conf') and i < len(boxes.conf):
                                conf_val = float(boxes.conf[i].item())
                                extraction['confidences'].append(conf_val)
                            else:
                                extraction['confidences'].append(0.0)  # 未知置信度
                                
                            # 提取边界框
                            if hasattr(boxes, 'xyxy') and i < len(boxes.xyxy):
                                bbox = [int(v) for v in boxes.xyxy[i].tolist()]
                                extraction['boxes'].append(bbox)
                            else:
                                extraction['boxes'].append([0, 0, 0, 0])  # 空白边界框
                        except Exception as e:
                            print(f"提取第{i+1}个目标数据失败: {str(e)}")
                            # 添加占位数据保持索引一致
                            if len(extraction['classes']) <= i:
                                extraction['classes'].append(-1)
                            if len(extraction['confidences']) <= i:
                                extraction['confidences'].append(0.0)
                            if len(extraction['boxes']) <= i:
                                extraction['boxes'].append([0, 0, 0, 0])
                
                except Exception as e:
                    print(f"提取检测框数据失败: {str(e)}")
            
            # 标记提取成功
            if extraction['object_count'] > 0:
                extraction['success'] = True
                
            return extraction
            
        except Exception as e:
            print(f"提取检测数据时出错: {str(e)}")
            traceback.print_exc()
            return {
                'success': False,
                'object_count': 0,
                'classes': [],
                'confidences': [],
                'boxes': [],
                'class_names': {},
                'original_image': None
            }

    def update_detection_info_safe(self, skip_combobox_update=False):
        """使用提取的数据更新检测结果信息，避免递归问题"""
        if not hasattr(self, 'extracted_data') or not self.extracted_data['success']:
            self.detection_text.setHtml("""
                <div style='text-align: center; padding: 20px; color: #6c757d;'>
                    <div>未检测到物体或数据提取失败</div>
                </div>
            """)
            return
            
        try:
            # 从提取的数据中获取信息
            data = self.extracted_data
            object_count = data['object_count']
            class_names_dict = data['class_names']
            
            if object_count == 0:
                self.detection_text.setHtml("""
                    <div style='text-align: center; padding: 20px; color: #6c757d;'>
                        <div>未检测到物体</div>
                    </div>
                """)
                return
                
            # 更新下拉框选项，除非明确指示跳过
            if not skip_combobox_update:
                self.update_target_combobox_safe(data)
            
            # 获取当前选择的目标索引
            selected_index = self.target_combobox.currentIndex()
            
            # 准备HTML内容 - 基本信息部分
            html_content = f"""
            <div style='margin-bottom: 10px; padding: 8px; background-color: #f8f9fa; border-radius: 4px;'>
                <div style='font-weight: bold; color: #495057; margin-bottom: 5px;'>基本信息</div>
                <div style='display: flex; justify-content: space-between; margin-bottom: 3px;'>
                    <span>目标数目:</span>
                    <span>{object_count}</span>
                </div>
            </div>
            """
                
            if selected_index == 0:  # "全部目标"
                # 统计各类别的数量
                class_counts = {}
                for cls_id in data['classes']:
                    if cls_id in class_counts:
                        class_counts[cls_id] += 1
                    else:
                        class_counts[cls_id] = 1
                
                # 添加类别统计信息
                html_content += """
                <div style='margin-bottom: 10px; padding: 8px; background-color: #f8f9fa; border-radius: 4px;'>
                    <div style='font-weight: bold; color: #495057; margin-bottom: 5px;'>类别统计</div>
                """
                
                for cls_id, count in class_counts.items():
                    class_name = "未知类别"
                    if cls_id in class_names_dict:
                        class_name = class_names_dict[cls_id]
                    
                    html_content += f"""
                    <div style='display: flex; justify-content: space-between; margin-bottom: 3px;'>
                        <span>{class_name}:</span>
                        <span>{count} </span>
                    </div>
                    """
                
                html_content += """</div>"""
                
                # 目标位置信息
                html_content += """
                <div style='margin-bottom: 10px; padding: 8px; background-color: #f8f9fa; border-radius: 4px;'>
                    <div style='font-weight: bold; color: #495057; margin-bottom: 5px;'>检测结果列表</div>
                """
                    
                for i in range(object_count):
                    try:
                        cls_val = data['classes'][i]
                        class_name = "未知目标"
                        if cls_val in class_names_dict:
                            class_name = class_names_dict[cls_val]
                        
                        # 获取边界框坐标
                        box = data['boxes'][i]
                        x1, y1, x2, y2 = box
                        
                        # 使用更好的格式化方式，确保对齐但不会在数值为0时产生过多间距
                        # 定义一个内部函数来格式化坐标值，确保数字右对齐但不会导致0值有过多空白
                        def format_coord(val):
                            val_str = str(int(val))
                            # 如果是0值或较小的数字，减少前导空格数量
                            if val == 0:
                                return "&nbsp;0"
                            elif val < 10:
                                return f"&nbsp;{val_str}"
                            elif val < 100:
                                return f"&nbsp;{val_str}"
                            elif val < 1000:
                                return f"{val_str}"
                            else:
                                return val_str
                            
                        x1_str = format_coord(x1)
                        y1_str = format_coord(y1)
                        x2_str = format_coord(x2)
                        y2_str = format_coord(y2)
                            
                        # 使用表格结构显示坐标信息，确保对齐，使用更紧凑的布局防止换行
                        html_content += f"""
                        <div style='margin-bottom: 10px; padding: 12px; background-color: #f8f9fa; border-radius: 5px; border-left: 3px solid #6950a1;'>
                            <div style='font-weight: bold; margin-bottom: 8px; color: #495057;'>{i+1}: {class_name}</div>
                            <table style='border-collapse: collapse; width: 100%; font-family: monospace; font-size: 13px;'>
                                <tr>
                                    <td style='color: #6c757d; padding: 3px 8px; white-space: nowrap;'>xmin</td>
                                    <td style='color: #ff6b6b; text-align: right; font-weight: 500; padding: 3px 10px; white-space: nowrap;'>{x1_str}</td>
                                    <td style='color: #6c757d; padding: 3px 8px; white-space: nowrap;'>ymin</td>
                                    <td style='color: #ff6b6b; text-align: right; font-weight: 500; padding: 3px 10px; white-space: nowrap;'>{y1_str}</td>
                                </tr>
                                <tr>
                                    <td style='color: #6c757d; padding: 3px 8px; white-space: nowrap;'>xmax</td>
                                    <td style='color: #ff6b6b; text-align: right; font-weight: 500; padding: 3px 10px; white-space: nowrap;'>{x2_str}</td>
                                    <td style='color: #6c757d; padding: 3px 8px; white-space: nowrap;'>ymax</td>
                                    <td style='color: #ff6b6b; text-align: right; font-weight: 500; padding: 3px 10px; white-space: nowrap;'>{y2_str}</td>
                                </tr>
                            </table>
                            
                        </div>
                        """
                    except Exception as e:
                        print(f"生成目标{i+1}信息时出错: {str(e)}")
                        html_content += f"""
                        <div style='margin-bottom: 8px; padding: 8px; background-color: #f8f9fa; border-radius: 4px; border-left: 3px solid #6950a1;'>
                            <div style='font-weight: bold; margin-bottom: 5px; color: #495057;'>{i+1}: 信息无法显示</div>
                            <div style='color: #721c24;'>无法获取目标详细信息</div>
                        </div>
                        """
                    
                html_content += "</div>"
                
            else:  # 特定类别
                # 获取所选类别ID
                selected_class_id = self.target_combobox.itemData(selected_index)
                if selected_class_id is not None:
                    # 找出所有该类别的目标
                    class_name = "未知类别"
                    if selected_class_id in class_names_dict:
                        class_name = class_names_dict[selected_class_id]
                    
                    # 找出所有该类别的目标索引
                    target_indices = []
                    for i, cls in enumerate(data['classes']):
                        if cls == selected_class_id:
                            target_indices.append(i)
                    
                    target_count = len(target_indices)
                    
                    # 添加类别信息
                    html_content += f"""
                    <div style='margin-bottom: 10px; padding: 8px; background-color: #f8f9fa; border-radius: 4px;'>
                        <div style='font-weight: bold; color: #495057; margin-bottom: 5px;'>类别信息</div>
                        <div style='display: flex; justify-content: space-between; margin-bottom: 3px;'>
                            <span>类别名称:</span>
                            <span>{class_name}</span>
                        </div>
                        <div style='display: flex; justify-content: space-between; margin-bottom: 3px;'>
                            <span>目标数量:</span>
                            <span>{target_count}</span>
                        </div>
                    </div>
                    """
                    
                    # 添加该类别的目标列表
                    html_content += f"""
                    <div style='margin-bottom: 10px; padding: 8px; background-color: #f8f9fa; border-radius: 4px;'>
                        <div style='font-weight: bold; color: #495057; margin-bottom: 5px;'>{class_name}目标列表</div>
                    """
                    
                    for i, target_idx in enumerate(target_indices):
                        try:
                            confidence = data['confidences'][target_idx]
                            box = data['boxes'][target_idx]
                            x1, y1, x2, y2 = box
                            
                            # 使用更好的格式化方式，确保对齐但不会在数值为0时产生过多间距
                            # 定义一个内部函数来格式化坐标值，确保数字右对齐但不会导致0值有过多空白
                            def format_coord(val):
                                val_str = str(int(val))
                                # 如果是0值或较小的数字，减少前导空格数量
                                if val == 0:
                                    return "&nbsp;0"
                                elif val < 10:
                                    return f"&nbsp;{val_str}"
                                elif val < 100:
                                    return f"&nbsp;{val_str}"
                                elif val < 1000:
                                    return f"{val_str}"
                                else:
                                    return val_str
                            
                            x1_str = format_coord(x1)
                            y1_str = format_coord(y1)
                            x2_str = format_coord(x2)
                            y2_str = format_coord(y2)
                            
                            # 使用表格结构显示坐标信息，确保对齐，使用更紧凑的布局防止换行
                            html_content += f"""
                            <div style='margin-bottom: 10px; padding: 12px; background-color: #f8f9fa; border-radius: 5px; border-left: 3px solid #6950a1;'>
                                <div style='font-weight: bold; margin-bottom: 8px; color: #495057;'>{i+1}: {class_name} ({confidence:.2f})</div>
                                <table style='border-collapse: collapse; width: 100%; font-family: monospace; font-size: 13px;'>
                                    <tr>
                                        <td style='color: #6c757d; padding: 3px 8px; white-space: nowrap;'>xmin</td>
                                        <td style='color: #ff6b6b; text-align: right; font-weight: 500; padding: 3px 10px; white-space: nowrap;'>{x1_str}</td>
                                        <td style='color: #6c757d; padding: 3px 8px; white-space: nowrap;'>ymin</td>
                                        <td style='color: #ff6b6b; text-align: right; font-weight: 500; padding: 3px 10px; white-space: nowrap;'>{y1_str}</td>
                                    </tr>
                                    <tr>
                                        <td style='color: #6c757d; padding: 3px 8px; white-space: nowrap;'>xmax</td>
                                        <td style='color: #ff6b6b; text-align: right; font-weight: 500; padding: 3px 10px; white-space: nowrap;'>{x2_str}</td>
                                        <td style='color: #6c757d; padding: 3px 8px; white-space: nowrap;'>ymax</td>
                                        <td style='color: #ff6b6b; text-align: right; font-weight: 500; padding: 3px 10px; white-space: nowrap;'>{y2_str}</td>
                                    </tr>
                                </table>
                            
                            </div>
                            """
                        except Exception as e:
                            print(f"生成类别目标{i+1}信息时出错: {str(e)}")
                            html_content += f"""
                            <div style='margin-bottom: 8px; padding: 8px; background-color: #f8f9fa; border-radius: 4px; border-left: 3px solid #6950a1;'>
                                <div style='font-weight: bold; margin-bottom: 5px; color: #495057;'>目标 {i+1}: 信息无法显示</div>
                                <div style='color: #721c24;'>无法获取目标详细信息</div>
                            </div>
                            """
                    
                    html_content += "</div>"
                else:
                    html_content += """
                    <div style='padding: 15px; background-color: #f8d7da; border-radius: 4px; color: #721c24;'>
                        <div>无法获取所选类别的信息</div>
                    </div>
                    """
                
            # 设置HTML内容
            self.detection_text.setHtml(html_content)
                    
        except Exception as e:
            error_msg = f"显示检测信息时出错: {str(e)}"
            print(error_msg)
            traceback.print_exc()
            self.statusBar.showMessage(error_msg)
            self.detection_text.setHtml(f"""
                <div style='padding: 15px; background-color: #f8d7da; border-radius: 4px; color: #721c24;'>
                    <div style='font-weight: bold; margin-bottom: 5px;'>显示检测信息时出错:</div>
                    <div>{error_msg}</div>
                </div>
            """)
    
    def update_target_combobox_safe(self, data):
        """安全地更新目标选择下拉框 - 按类别分组"""
        try:
            # 保存当前选择的类别ID（如果有）
            current_selection = -1
            if self.target_combobox.currentIndex() >= 0:
                current_selection = self.target_combobox.itemData(self.target_combobox.currentIndex())
            
            # 首先暂停combobox的信号连接，防止触发change事件
            self.target_combobox.blockSignals(True)
            
            # 清空当前选项
            self.target_combobox.clear()
            
            # 如果没有有效数据，则只添加默认选项并返回
            if not data['success'] or data['object_count'] == 0:
                self.target_combobox.addItem("全部")
                self.target_combobox.setItemData(0, -1)  # 设置"全部"选项的数据为-1
                self.target_combobox.setEnabled(False)
                self.target_combobox.blockSignals(False)
                return
                
            # 添加"全部"选项 (与MainProgram.py保持一致)
            self.target_combobox.addItem("全部")
            self.target_combobox.setItemData(0, -1)  # 设置"全部"选项的数据为-1
            
            # 获取类别名称字典和类别列表
            class_names_dict = data['class_names']
            classes = data['classes']
            
            # 统计每个类别的目标数量
            class_counts = {}
            for cls_id in classes:
                if cls_id in class_counts:
                    class_counts[cls_id] += 1
                else:
                    class_counts[cls_id] = 1
            
            # 记录新添加的类别及其索引，用于恢复选择
            new_indices = {-1: 0}  # "全部"选项的类别ID为-1，索引为0
            
            # 添加每个类别到下拉框
            for cls_id, count in class_counts.items():
                try:
                    class_name = "未知类别"
                    if cls_id in class_names_dict:
                        class_name = class_names_dict[cls_id]
                    
                    # 添加类别及其计数
                    self.target_combobox.addItem(f"{class_name} ({count})")
                    # 将类别ID存储为用户数据，便于后续处理
                    current_index = self.target_combobox.count() - 1
                    self.target_combobox.setItemData(current_index, cls_id)
                    
                    # 记录这个类别ID对应的新索引
                    new_indices[cls_id] = current_index
                    
                except Exception as e:
                    print(f"添加类别 {cls_id} 到下拉框时出错: {str(e)}")
                    self.target_combobox.addItem(f"类别 {cls_id} ({count})")
                    current_index = self.target_combobox.count() - 1
                    self.target_combobox.setItemData(current_index, cls_id)
                    new_indices[cls_id] = current_index
            
            # 尝试恢复之前的选择
            if current_selection in new_indices:
                self.target_combobox.setCurrentIndex(new_indices[current_selection])
            else:
                # 如果之前的选择不在新的列表中，则默认选择"全部"
                self.target_combobox.setCurrentIndex(0)
            
            # 启用下拉框
            self.target_combobox.setEnabled(True)
            
            # 恢复combobox的信号连接
            self.target_combobox.blockSignals(False)
            
        except Exception as e:
            print(f"更新目标下拉框时出错: {str(e)}")
            traceback.print_exc()
            # 出错时只保留默认选项
            self.target_combobox.clear()
            self.target_combobox.addItem("全部")
            self.target_combobox.setItemData(0, -1)  # 设置"全部"选项的数据为-1
            self.target_combobox.setEnabled(False)
            self.target_combobox.blockSignals(False)

    def show_all_targets_safe(self, data):
        """安全地显示所有检测目标，使用预测的注释图像"""
        try:
            if not data['success'] or data['object_count'] == 0:
                self.statusBar.showMessage("没有检测到物体或数据无效")
                return
                
            # 直接使用当前结果中的注释图像，而不是手动绘制
            if hasattr(self, 'current_results') and self.current_results:
                try:
                    # 使用模型自带的plot方法生成注释图像
                    annotated_image = self.current_results[0].plot()
                    annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
                    
                    # 转换为QImage和QPixmap
                    height, width, channel = annotated_image.shape
                    bytesPerLine = 3 * width
                    qimage = QImage(annotated_image.data, width, height, bytesPerLine, QImage.Format_RGB888)
                    pixmap = QPixmap.fromImage(qimage)
                    
                    # 计算Label尺寸
                    label_width = self.result_label.width()
                    label_height = self.result_label.height()
                    
                    # 计算缩放比例和偏移量
                    scale_factor = min(label_width / width, label_height / height)
                    scaled_width = int(width * scale_factor)
                    scaled_height = int(height * scale_factor)
                    offset_x = (label_width - scaled_width) // 2
                    offset_y = (label_height - scaled_height) // 2
                    
                    # 缩放图像以适应显示区域，保持纵横比
                    display_pixmap = pixmap.scaled(
                        scaled_width, 
                        scaled_height,
                        Qt.KeepAspectRatio, 
                        Qt.SmoothTransformation
                    )
                    
                    # 创建一个新的pixmap，包括可能的边距区域
                    final_pixmap = QPixmap(label_width, label_height)
                    final_pixmap.fill(Qt.white)  # 填充白色背景
                    
                    
                    # 创建绘图对象
                    painter = QPainter(final_pixmap)
                    
                    # 在正确位置绘制缩放后的图像
                    painter.drawPixmap(offset_x, offset_y, display_pixmap)
                    painter.end()
                    
                    # 显示绘制后的图像
                    self.result_label.setPixmap(final_pixmap)
                    self.statusBar.showMessage(f"显示全部目标 - 共 {data['object_count']} ")
                    
                except Exception as e:
                    print(f"显示注释图像时出错: {str(e)}")
                    traceback.print_exc()
                    # 尝试使用原始图像作为备选
                    if data['original_image'] is not None:
                        try:
                            original_image = data['original_image'].copy()
                            img_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
                            height, width, channel = img_rgb.shape
                            bytesPerLine = 3 * width
                            qimage = QImage(img_rgb.data, width, height, bytesPerLine, QImage.Format_RGB888)
                            pixmap = QPixmap.fromImage(qimage)
                            
                            # 使用固定大小的缩放显示
                            display_pixmap = pixmap.scaled(
                                self.result_label.width(), 
                                self.result_label.height(),
                                Qt.KeepAspectRatio, 
                                Qt.SmoothTransformation
                            )
                            self.result_label.setPixmap(display_pixmap)
                            self.statusBar.showMessage(f"使用原始图像显示 - 无法生成带注释的图像")
                        except Exception as e:
                            print(f"显示原始图像也失败: {str(e)}")
                            self.statusBar.showMessage(f"无法显示图像: {str(e)}")
            else:
                self.statusBar.showMessage("没有当前检测结果可显示")
                print("没有当前检测结果可显示")
            
        except Exception as e:
            print(f"显示所有目标时出错: {str(e)}")
            traceback.print_exc()
            self.statusBar.showMessage(f"显示所有目标时出错: {str(e)}")

    def show_class_targets_safe(self, data, class_id):
        """安全地显示特定类别的所有目标，仅使用YOLO绘图风格"""
        try:
            if not data['success'] or data['object_count'] == 0:
                return
                
            if hasattr(self, 'current_results') and self.current_results and hasattr(self, 'current_image'):
                # 使用YOLO的API进行筛选和绘制
                result = self.current_results[0]
                
                # 复制原始图像用于绘制
                if result.orig_img is not None:
                    orig_img = result.orig_img.copy()
                else:
                    orig_img = self.current_image.copy()
                
                # 获取类别名称
                class_name = "未知类别"
                if class_id in data['class_names']:
                    class_name = data['class_names'][class_id]
                
                # 找出所有该类别的目标索引
                target_indices = []
                for i, cls in enumerate(data['classes']):
                    if cls == class_id:
                        target_indices.append(i)
                
                if not target_indices:
                    self.statusBar.showMessage(f"未找到{class_name}类别的目标")
                    # 如果没有找到该类别的目标，显示所有目标
                    self.show_all_targets_safe(data)
                    return
                
                # 使用YOLO的boxes绘制特定类别
                if hasattr(result, 'boxes'):
                    try:
                        # 创建一个过滤后的boxes副本
                        from ultralytics.engine.results import Results, Boxes
                        import torch
                        
                        # 1. 提取要保留的boxes
                        boxes_data = result.boxes.data.cpu()
                        mask = boxes_data[:, 5] == class_id  # 类别ID在第5列
                        filtered_boxes = boxes_data[mask]
                        
                        if len(filtered_boxes) == 0:
                            self.statusBar.showMessage(f"未找到{class_name}类别的目标")
                            self.show_all_targets_safe(data)
                            return
                        
                        # 2. 创建新的Results对象
                        # 检查result是否有必要的属性
                        path = getattr(result, 'path', '')  # 如果没有path属性，使用空字符串
                        names = result.names if hasattr(result, 'names') else data['class_names']
                        
                        # 正确创建Results对象，提供所有必需的参数
                        filtered_result = Results(
                            orig_img=orig_img.copy(),
                            path=path,
                            names=names
                        )
                        
                        # 3. 创建新的Boxes对象并添加到filtered_result
                        filtered_boxes_obj = Boxes(filtered_boxes, filtered_result.orig_img.shape)
                        filtered_result.boxes = filtered_boxes_obj
                        
                        # 4. 使用YOLO的plot方法绘制结果
                        filtered_img = filtered_result.plot()
                        
                        # 显示图像
                        img_rgb = cv2.cvtColor(filtered_img, cv2.COLOR_BGR2RGB)
                        height, width, channel = img_rgb.shape
                        bytesPerLine = 3 * width
                        qimage = QImage(img_rgb.data, width, height, bytesPerLine, QImage.Format_RGB888)
                        pixmap = QPixmap.fromImage(qimage)
                        
                        # 计算显示尺寸
                        label_width = self.result_label.width()
                        label_height = self.result_label.height()
                        scale_factor = min(label_width / width, label_height / height)
                        scaled_width = int(width * scale_factor)
                        scaled_height = int(height * scale_factor)
                        
                        # 缩放图像
                        display_pixmap = pixmap.scaled(
                            scaled_width, 
                            scaled_height,
                            Qt.KeepAspectRatio, 
                            Qt.SmoothTransformation
                        )
                        
                        # 创建一个新的pixmap，包括可能的边距区域
                        final_pixmap = QPixmap(label_width, label_height)
                        final_pixmap.fill(Qt.white)  # 填充白色背景
                        
                        # 创建绘图对象
                        painter = QPainter(final_pixmap)
                        
                        # 在正确位置绘制缩放后的图像
                        offset_x = (label_width - scaled_width) // 2
                        offset_y = (label_height - scaled_height) // 2
                        painter.drawPixmap(offset_x, offset_y, display_pixmap)
                        painter.end()
                        
                        # 显示绘制后的图像
                        self.result_label.setPixmap(final_pixmap)
                        
                        # 更新状态栏，显示该类别的目标数量
                        self.statusBar.showMessage(f"类别: {class_name}, 目标数量: {len(target_indices)}")
                    except Exception as e:
                        print(f"使用YOLO API绘制类别时出错: {str(e)}")
                        traceback.print_exc()
                        # 无法使用YOLO API，显示全部目标
                        self.statusBar.showMessage(f"使用YOLO API绘制类别时出错: {str(e)}")
                        self.show_all_targets_safe(data)
                else:
                    # 如果无法获取boxes，显示所有目标
                    self.statusBar.showMessage(f"无法筛选显示{class_name}类别，显示全部目标")
                    self.show_all_targets_safe(data)
            else:
                # 如果当前结果不可用，显示所有目标
                self.statusBar.showMessage(f"当前检测结果不可用，显示全部目标")
                self.show_all_targets_safe(data)
                
        except Exception as e:
            print(f"显示类别目标时出错: {str(e)}")
            traceback.print_exc()
            # 出错时显示所有目标
            self.statusBar.showMessage(f"显示类别目标时出错，显示全部目标")
            self.show_all_targets_safe(data)

    def show_single_target_safe(self, data, target_index):
        """安全地显示单个目标，仅使用YOLO的plot方法"""
        try:
            if not data['success'] or data['object_count'] == 0 or target_index < 0 or target_index >= data['object_count']:
                return
                
            if hasattr(self, 'current_results') and self.current_results:
                # 使用YOLO结果的索引绘制
                result = self.current_results[0]
                
                # 创建一个过滤后的Results对象，只包含指定目标
                from ultralytics.engine.results import Results, Boxes
                import torch
                
                # 获取目标信息
                cls_id = data['classes'][target_index]
                
                # 复制原始图像
                if result.orig_img is not None:
                    orig_img = result.orig_img.copy()
                elif hasattr(self, 'current_image'):
                    orig_img = self.current_image.copy()
                else:
                    self.statusBar.showMessage("无法获取原始图像，显示全部目标")
                    self.show_all_targets_safe(data)
                    return
                
                # 提取单个目标的边界框数据
                try:
                    # 获取所有边界框数据
                    boxes_data = result.boxes.data.cpu()
                    
                    # 创建单个目标的掩码
                    if len(boxes_data) > target_index:
                        # 创建只包含单个目标的掩码
                        single_box = boxes_data[target_index:target_index+1]
                        
                        # 检查result是否有必要的属性
                        path = getattr(result, 'path', '')  # 如果没有path属性，使用空字符串
                        names = result.names if hasattr(result, 'names') else data['class_names']
                        
                        # 创建新的Results对象，提供所有必需的参数
                        filtered_result = Results(
                            orig_img=orig_img.copy(),
                            path=path,
                            names=names
                        )
                        
                        # 创建新的Boxes对象并添加到filtered_result
                        filtered_boxes_obj = Boxes(single_box, filtered_result.orig_img.shape)
                        filtered_result.boxes = filtered_boxes_obj
                        
                        # 使用YOLO的plot方法绘制结果
                        single_target_img = filtered_result.plot()
                        
                        # 转换为RGB格式用于Qt显示
                        img_rgb = cv2.cvtColor(single_target_img, cv2.COLOR_BGR2RGB)
                        
                        # 显示图像
                        height, width, channel = img_rgb.shape
                        bytesPerLine = 3 * width
                        qimage = QImage(img_rgb.data, width, height, bytesPerLine, QImage.Format_RGB888)
                        pixmap = QPixmap.fromImage(qimage)
                        
                        # 计算显示尺寸
                        label_width = self.result_label.width()
                        label_height = self.result_label.height()
                        scale_factor = min(label_width / width, label_height / height)
                        scaled_width = int(width * scale_factor)
                        scaled_height = int(height * scale_factor)
                        
                        # 缩放图像
                        display_pixmap = pixmap.scaled(
                            scaled_width, 
                            scaled_height,
                            Qt.KeepAspectRatio, 
                            Qt.SmoothTransformation
                        )
                        
                        # 创建一个新的pixmap，包括可能的边距区域
                        final_pixmap = QPixmap(label_width, label_height)
                        final_pixmap.fill(Qt.white)  # 填充白色背景
                        
                        # 创建绘图对象
                        painter = QPainter(final_pixmap)
                        
                        # 在正确位置绘制缩放后的图像
                        offset_x = (label_width - scaled_width) // 2
                        offset_y = (label_height - scaled_height) // 2
                        painter.drawPixmap(offset_x, offset_y, display_pixmap)
                        painter.end()
                        
                        # 显示绘制后的图像
                        self.result_label.setPixmap(final_pixmap)
                        
                        # 更新状态栏
                        if cls_id in data['class_names']:
                            class_name = data['class_names'][cls_id]
                        else:
                            class_name = "未知类别"
                        conf = data['confidences'][target_index]
                        self.statusBar.showMessage(f"显示目标: {class_name}, 置信度: {conf:.2f}")
                    else:
                        # 索引超出范围，显示所有目标
                        self.statusBar.showMessage(f"目标索引{target_index}超出范围，显示全部目标")
                        self.show_all_targets_safe(data)
                except Exception as e:
                    print(f"创建单个目标过滤结果时出错: {str(e)}")
                    traceback.print_exc()
                    self.statusBar.showMessage("显示单个目标失败，显示全部目标")
                    self.show_all_targets_safe(data)
            else:
                # 无法获取当前结果，回退到显示所有目标
                self.statusBar.showMessage("无法获取当前检测结果，显示全部目标")
                self.show_all_targets_safe(data)
                    
        except Exception as e:
            print(f"显示单个目标时出错: {str(e)}")
            traceback.print_exc()
            self.statusBar.showMessage(f"显示单个目标时出错，显示全部目标")
            self.show_all_targets_safe(data)

    def on_target_changed(self):
        """处理目标选择下拉框的变化"""
        if not hasattr(self, 'extracted_data') or not self.extracted_data['success']:
            return
            
        try:
            selected_index = self.target_combobox.currentIndex()
            data = self.extracted_data
            
            # 获取选项的数据值
            selected_data = self.target_combobox.itemData(selected_index)
            
            # 更新状态栏提示
            if selected_index == 0 or selected_data == -1:  # 判断是否为"全部"选项
                self.statusBar.showMessage("显示全部目标")
                # 显示所有目标
                self.show_all_targets_safe(data)
                # 确保检测信息也更新为全部目标，但跳过更新下拉框以避免循环调用
                self.update_detection_info_safe(skip_combobox_update=True)
            else:
                # 获取所选类别ID
                selected_class_id = selected_data
                class_name = "未知类别"
                if selected_class_id in data['class_names']:
                    class_name = data['class_names'][selected_class_id]
                self.statusBar.showMessage(f"筛选显示类别: {class_name}")
                
                # 显示所选类别的所有目标
                self.show_class_targets_safe(data, selected_class_id)
                
                # 确保检测信息也更新为所选类别，但跳过更新下拉框以避免循环调用
                self.update_detection_info_safe(skip_combobox_update=True)
            
        except Exception as e:
            error_msg = f"切换目标视图时出错: {str(e)}"
            print(error_msg)
            traceback.print_exc()
            self.statusBar.showMessage(error_msg)
            self.detection_text.setHtml(f"""
                <div style='padding: 15px; background-color: #f8d7da; border-radius: 4px; color: #721c24;'>
                    <div style='font-weight: bold; margin-bottom: 5px;'>切换目标视图时出错:</div>
                    <div>{error_msg}</div>
                </div>
            """)
            
    def update_display_info(self, class_name, conf, box):
        """更新显示选中目标的信息"""
        try:
            # 将坐标值转为字符串，并使用非断空格填充，确保对齐
            x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            
            # 使用更好的格式化方式，确保对齐但不会在数值为0时产生过多间距
            # 定义一个内部函数来格式化坐标值，确保数字右对齐但不会导致0值有过多空白
            def format_coord(val):
                val_str = str(int(val))
                # 如果是0值或较小的数字，减少前导空格数量
                if val == 0:
                    return "&nbsp;0"
                elif val < 10:
                    return f"&nbsp;{val_str}"
                elif val < 100:
                    return f"&nbsp;{val_str}"
                elif val < 1000:
                    return f"{val_str}"
                else:
                    return val_str
                
            x1_str = format_coord(x1)
            y1_str = format_coord(y1)
            x2_str = format_coord(x2)
            y2_str = format_coord(y2)
            
            # 更新检测信息文本区域，使用表格结构显示坐标，使用更紧凑的布局防止换行
            html_content = f"""
            <div style='margin-bottom: 10px; padding: 12px; background-color: #f8f9fa; border-radius: 5px;'>
                <div style='font-weight: bold; color: #495057; margin-bottom: 8px;'>目标信息</div>
                <div style='display: flex; justify-content: space-between; margin-bottom: 5px;'>
                    <span>类别:</span>
                    <span style='font-weight: 500;'>{class_name}</span>
                </div>
                <div style='display: flex; justify-content: space-between; margin-bottom: 8px;'>
                    <span>置信度:</span>
                    <span style='font-weight: 500;'>{conf:.2f}</span>
                </div>
                <table style='border-collapse: collapse; width: 100%; font-family: monospace; font-size: 13px; margin-top: 10px;'>
                    <tr>
                        <td style='color: #6c757d; padding: 3px 8px; white-space: nowrap;'>xmin</td>
                        <td style='color: #ff6b6b; text-align: right; font-weight: 500; padding: 3px 10px; white-space: nowrap;'>{x1_str}</td>
                        <td style='color: #6c757d; padding: 3px 8px; white-space: nowrap;'>ymin</td>
                        <td style='color: #ff6b6b; text-align: right; font-weight: 500; padding: 3px 10px; white-space: nowrap;'>{y1_str}</td>
                    </tr>
                    <tr>
                        <td style='color: #6c757d; padding: 3px 8px; white-space: nowrap;'>xmax</td>
                        <td style='color: #ff6b6b; text-align: right; font-weight: 500; padding: 3px 10px; white-space: nowrap;'>{x2_str}</td>
                        <td style='color: #6c757d; padding: 3px 8px; white-space: nowrap;'>ymax</td>
                        <td style='color: #ff6b6b; text-align: right; font-weight: 500; padding: 3px 10px; white-space: nowrap;'>{y2_str}</td>
                    </tr>
                </table>
                
            </div>
            """
            self.detection_text.setHtml(html_content)
        except Exception as e:
            print(f"更新目标信息显示时出错: {str(e)}")
            traceback.print_exc()

if __name__ == '__main__':
    try:
        # 检查必要的库是否可用
        import torch
        import cv2
        from ultralytics import YOLO
        
        # 显示系统信息
        print("系统环境信息:")
        print(f"PyTorch 版本: {torch.__version__}")
        print(f"CUDA 是否可用: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"GPU型号: {torch.cuda.get_device_name(0)}")
            print(f"可用GPU数量: {torch.cuda.device_count()}")
        print(f"OpenCV 版本: {cv2.__version__}")
        
        # 启动应用
        app = QApplication(sys.argv)
        # 设置应用样式，使界面更美观
        app.setStyle('Fusion')
        window = MainWindow()
        window.show()
        app.exec_()  # 移除sys.exit()包装，直接使用app.exec_()
    except Exception as e:
        # 如果在启动时发生错误，显示错误消息
        app = QApplication(sys.argv)
        error_msg = QMessageBox()
        error_msg.setIcon(QMessageBox.Critical)
        error_msg.setWindowTitle("启动错误")
        error_msg.setText(f"程序启动失败:\n{str(e)}")
        error_msg.setDetailedText(f"错误详情:\n{str(e)}")
        error_msg.exec_()
        sys.exit(1)

