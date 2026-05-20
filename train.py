
import sys
import os
import torch
import numpy as np
from PIL import Image

# 在Windows环境中需要添加freeze_support来支持多进程
if sys.platform.startswith('win'):
    from multiprocessing import freeze_support
    freeze_support()

# 添加RTDETR-20251122目录到Python路径
sys.path.insert(0, '/root/.trae-cn/RTDETR-20251122/RTDETR-main')

# 首先创建必要的assets目录和示例图像
print("准备创建测试图像文件...")
assets_dir = '/root/.trae-cn/RTDETR-20251122/RTDETR-main/ultralytics/assets'
os.makedirs(assets_dir, exist_ok=True)
# 创建一个简单的测试图像
img_path = os.path.join(assets_dir, 'bus.jpg')
if not os.path.exists(img_path):
    # 创建一个640x640的随机图像
    img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    Image.fromarray(img).save(img_path)
    print(f"已创建测试图像: {img_path}")
else:
    print(f"测试图像已存在: {img_path}")

# 验证图像是否成功创建
if os.path.exists(img_path):
    print(f"图像文件验证成功: {os.path.getsize(img_path)} bytes")
else:
    print(f"警告: 图像文件创建失败: {img_path}")

# 导入必要的模块
from ultralytics import RTDETR
from ultralytics.nn.tasks import RTDETRDetectionModel
from ultralytics.models.utils.loss import RTDETRDetectionLoss
from ultralytics.utils.metrics import WiseIouLoss

# 定义新的init_criterion方法
def patched_init_criterion(self):
    """初始化损失函数，结合使用MPDIoU和NWD损失函数（特别适合焊接缺陷检测）"""
    # 创建RTDETRDetectionLoss实例
    criterion = RTDETRDetectionLoss(
        nc=self.nc, 
        use_vfl=True, 
        use_sl=False, 
        use_emasl=False, 
        use_svfl=False, 
        use_emasvfl=False, 
        use_mal=False
    )
    
    # 设置IoU和NWD损失的权重各占0.5
    criterion.iou_ratio = 0.5
    
    # 启用WiseIoU并使用MPDIoU变体
    criterion.use_wiseiou = True
    # 初始化WiseIoU损失，使用MPDIoU变体
    # MPDIoU考虑了预测框和真实框的重叠部分形状，对焊接缺陷这种形状复杂的目标检测效果更好
    criterion.wiou_loss = WiseIouLoss(ltype='MPDIoU', monotonous=False, inner_iou=False, focaler_iou=False)
    
    # 启用NWD损失
    criterion.nwd_loss = True
    
    return criterion

# 应用猴子补丁
RTDETRDetectionModel.init_criterion = patched_init_criterion

# 设置数据集和模型配置路径
base_dir = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(os.path.dirname(base_dir), "weldqualityinspectionv9", "fixed_data.yaml")
# 使用指定的配置文件（包含特征融合模块）
model_config = "/root/.trae-cn/RTDETR-20251122/RTDETR-main/ultralytics/cfg/models/rt-detr/rtdetr-C2f-MSMHSA-DASI.yaml"
# 从头开始训练，不使用恢复训练的权重文件

if __name__ == '__main__':
    # 确保路径存在
    if not os.path.exists(dataset_path):
        print(f"错误: 找不到数据集配置文件: {dataset_path}")
        print("请修改脚本中的dataset_path变量，指向正确的data.yaml文件路径")
        exit(1)

    if not os.path.exists(model_config):
        print(f"错误: 找不到模型配置文件: {model_config}")
        exit(1)

    # 初始化RT-DETR模型 - 使用rtdetr-C2f-MSMHSA-CGLU配置
    print("正在加载RT-DETR模型 - 使用rtdetr-C2f-MSMHSA-CGLU配置（包含特征融合模块）...")
    print(f"使用数据集配置: {dataset_path}")
    print(f"使用模型配置: {model_config}")
    
    # 初始化模型
    try:
        model = RTDETR(model_config)
    except Exception as e:
        print(f"模型初始化失败: {e}")
        exit(1)
  
    # 显示模型信息
    print("\n模型结构信息:")
    model.info()
    
    # 打印损失函数配置信息
    print("\n损失函数配置:")
    print("- 结合使用MPDIoU和NWD损失函数，权重各占0.5")
    print("- MPDIoU通过WiseIoU实现，考虑了预测框和真实框的重叠部分形状")
    print("- NWD损失（Normalized Wasserstein Distance）对小目标和边界框不精确的情况有更好的鲁棒性")
    print("- 两者结合可以同时兼顾边界框精确性和对复杂形状目标的检测效果")
    
    # 配置训练参数 - 只保留最基础参数
    train_params = {
        'data': dataset_path,       # 数据集配置路径
        'epochs': 200,              # 训练轮数
        'imgsz': 640,               # 图像大小
        'batch': 8,                 # 批次大小
        'device': '0',               # 使用GPU进行计算
        'workers': 8,               # 数据加载线程数
        'project': 'rtdetr_weld_quality',  # 项目名称
        'name': 'rtdetr-C2f-MSMHSA-CGLU_CA_HSFPN3+nwd损失函数',     # 训练名称（包含特征融合模块）
        'exist_ok': True,           # 允许覆盖现有结果
        'amp': True,                # 启用混合精度训练
        'resume': False  # 不恢复训练，开始新训练
    }

    # 显示训练配置
    print("\n训练配置:")
    for key, value in train_params.items():
        print(f"  {key}: {value}")

    # 开始训练
    print("\n开始训练模型...")
    try:
        results = model.train(**train_params)
        print("训练完成!")
        
        # 训练完成后进行验证并生成评估图片
        print("\n开始验证模型并生成评估图片...")
        # 确保plots参数为True以生成评估图片
        val_results = model.val(data=dataset_path, plots=True, save_json=False)
        print("验证和图片生成完成!")
        
    except Exception as e:
        print(f"训练过程中发生错误: {e}")
        print("请检查错误信息并修复问题后重新运行")