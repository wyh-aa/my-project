# RT-DETR 焊接质量检测模型训练配置文档

## 一、项目概述

本项目基于 **RT-DETR（Real-Time Detection Transformer）** 框架，实现焊接缺陷的实时检测。项目针对焊接缺陷检测任务，引入了改进的特征融合模块和混合损失函数（MPDIoU + NWD），以提升复杂形状目标的检测精度。

**项目新增文件：**
- `weights/` - 训练好的模型权重目录（含 `best.pt` 和 `last.pt`）
- `rtdetr-l.yaml` - RT-DETR Large 模型配置文件

---

## 二、数据集配置

### 2.1 数据集结构

数据集位于 `/root/.trae-cn/my-project/weldqualityinspectionv9/` 目录下：

| 目录 | 用途 |
|------|------|
| `train/` | 训练集 |
| `valid/` | 验证集 |
| `test/` | 测试集 |

### 2.2 数据配置文件

数据配置文件：[data.yaml](file:///root/.trae-cn/my-project/weldqualityinspectionv9/data.yaml)

```yaml
names:
- adj    # 气孔类 (Adjacent)
- int    # 夹杂类 (Inclusion)
- geo    # 几何缺陷 (Geometric)
- pro    # 裂纹类 (Propagation)
- non    # 未焊透 (Non-fusion)
nc: 5    # 类别数量
test: /root/.trae-cn/weldqualityinspectionv9/train
train: /root/.trae-cn/weldqualityinspectionv9/train
val: /root/.trae-cn/weldqualityinspectionv9/valid
```

### 2.3 类别定义

| 类别ID | 类别名称 | 中文含义 | 缺陷描述 |
|--------|----------|----------|----------|
| 0 | adj | 气孔类 | 焊接过程中气体未逸出形成的孔洞 |
| 1 | int | 夹杂类 | 焊接材料中的杂质或熔渣残留 |
| 2 | geo | 几何缺陷 | 焊缝形状不规则、尺寸偏差 |
| 3 | pro | 裂纹类 | 焊缝或热影响区产生的裂纹 |
| 4 | non | 未焊透 | 母材与焊缝金属未完全熔合 |

---

## 三、模型配置

项目提供两种模型配置文件：

### 3.1 模型一：rtdetr-C2f-MSMHSA-DASI（改进版）

配置文件：[rtdetr-C2f-MSMHSA-DASI.yaml](file:///root/.trae-cn/my-project/rtdetr-C2f-MSMHSA-DASI.yaml)

**核心结构特点：**

| 组件 | 模块名称 | 作用 |
|------|----------|------|
| Backbone | C2f_MSMHSA | 融合C2f模块与多头自注意力机制 |
| Neck | DASI | 深度自适应特征融合模块 |
| Decoder | RTDETRDecoder | 检测解码器 |

```yaml
nc: 5
scales:
  l: [1.00, 1.00, 1024]

backbone:
  - [-1, 1, Conv, [64, 3, 2]]       # P1/2
  - [-1, 1, Conv, [128, 3, 2]]      # P2/4
  - [-1, 1, C2f_MSMHSA, [128]]      # 融合MSMHSA注意力
  - [-1, 1, Conv, [256, 3, 2]]      # P3/8
  - [-1, 1, C2f_MSMHSA, [256]]
  - [-1, 1, Conv, [384, 3, 2]]      # P4/16
  - [-1, 1, C2f_MSMHSA, [384]]
  - [-1, 1, Conv, [384, 3, 2]]      # P5/32
  - [-1, 3, C2f_MSMHSA, [384]]
```

### 3.2 模型二：rtdetr-l（Large版本）

配置文件：[rtdetr-l.yaml](file:///root/.trae-cn/my-project/rtdetr-l.yaml)

**核心结构特点：**

| 组件 | 模块名称 | 作用 |
|------|----------|------|
| Stem | HGStem | 高效分组卷积主干 |
| Backbone | HGBlock | 分层分组卷积块 |
| Attention | AIFI | 自适应特征融合 |
| Neck | FPN+PAN | 特征金字塔网络 |

**配置详情：**

```yaml
nc: 5  # 类别数量
scales:
  l: [1.00, 1.00, 1024]

backbone:
  - [-1, 1, HGStem, [32, 48]]        # 0-P2/4
  - [-1, 6, HGBlock, [48, 128, 3]]   # stage 1
  - [-1, 1, DWConv, [128, 3, 2, 1, False]]  # 2-P3/8
  - [-1, 6, HGBlock, [96, 512, 3]]   # stage 2
  - [-1, 1, DWConv, [512, 3, 2, 1, False]]  # 4-P4/16
  - [-1, 6, HGBlock, [192, 1024, 5, True, False]]  # stage 3
  - [-1, 6, HGBlock, [192, 1024, 5, True, True]]
  - [-1, 6, HGBlock, [192, 1024, 5, True, True]]
  - [-1, 1, DWConv, [1024, 3, 2, 1, False]]  # 8-P5/32
  - [-1, 6, HGBlock, [384, 2048, 5, True, False]]  # stage 4

head:
  - [-1, 1, Conv, [256, 1, 1]]
  - [-1, 1, AIFI, [1024, 8]]         # AIFI注意力融合
  - [-1, 1, Conv, [256, 1, 1]]       # Y5
  # FPN上采样路径...
  # PAN下采样路径...
  - [[21, 24, 27], 1, RTDETRDecoder, [nc]]  # Detect(P3, P4, P5)
```

### 3.3 模型对比

| 特性 | rtdetr-C2f-MSMHSA-DASI | rtdetr-l |
|------|------------------------|----------|
| 主干结构 | C2f + MSMHSA | HGStem + HGBlock |
| 特征融合 | DASI模块 | FPN+PAN |
| 模型大小 | 中等 | 较大 |
| 推理速度 | 较快 | 适中 |
| 检测精度 | 高 | 更高 |

---

## 四、训练参数配置

### 4.1 训练配置文件

配置文件：[args.yaml](file:///root/.trae-cn/my-project/args.yaml)

### 4.2 核心训练参数

| 参数 | 值 | 说明 |
|------|-----|------|
| `task` | detect | 任务类型：目标检测 |
| `mode` | train | 运行模式：训练 |
| `epochs` | **200** | 训练轮数 |
| `batch` | **8** | 批次大小 |
| `imgsz` | **640** | 输入图像尺寸 |
| `device` | '0' | 使用GPU设备 |
| `workers` | **8** | 数据加载线程数 |
| `optimizer` | **AdamW** | 优化器 |
| `pretrained` | true | 使用预训练权重 |
| `resume` | false | 不恢复训练 |
| `amp` | true | 启用混合精度训练 |

### 4.3 学习率配置

| 参数 | 值 |
|------|-----|
| `lr0` | 0.0001 |
| `lrf` | 1.0 |
| `warmup_epochs` | 2000 |
| `warmup_bias_lr` | 0.1 |

### 4.4 优化器参数

| 参数 | 值 |
|------|-----|
| `momentum` | 0.9 |
| `weight_decay` | 0.0001 |

### 4.5 损失函数配置

**混合损失策略：**

| 损失类型 | 权重 | 说明 |
|----------|------|------|
| MPDIoU | 0.5 | 考虑重叠形状的IoU变体 |
| NWD | 0.5 | 归一化Wasserstein距离 |

**损失函数代码实现**（来自 [train.py](file:///root/.trae-cn/my-project/train.py)）：

```python
def patched_init_criterion(self):
    criterion = RTDETRDetectionLoss(
        nc=self.nc, 
        use_vfl=True, 
        use_sl=False, 
        use_emasl=False, 
        use_svfl=False, 
        use_emasvfl=False, 
        use_mal=False
    )
    criterion.iou_ratio = 0.5
    criterion.use_wiseiou = True
    criterion.wiou_loss = WiseIouLoss(ltype='MPDIoU', monotonous=False, inner_iou=False, focaler_iou=False)
    criterion.nwd_loss = True
    return criterion

RTDETRDetectionModel.init_criterion = patched_init_criterion
```

### 4.6 数据增强参数

| 参数 | 值 |
|------|-----|
| `hsv_h` | 0.015 |
| `hsv_s` | 0.7 |
| `hsv_v` | 0.4 |
| `fliplr` | 0.5 |
| `scale` | 0.5 |

---

## 五、训练执行

### 5.1 训练脚本

训练入口文件：[train.py](file:///root/.trae-cn/my-project/train.py)

**执行命令：**

```bash
cd /root/.trae-cn/my-project
python train.py
```

**训练流程：**

1. **环境准备**：添加RTDETR路径，创建测试图像
2. **模型初始化**：加载模型配置，应用损失函数补丁
3. **参数配置**：设置训练参数（epochs=200, batch=8, imgsz=640）
4. **训练阶段**：使用混合精度训练，每轮进行验证
5. **验证阶段**：生成混淆矩阵、PR曲线等评估图表

### 5.2 训练参数配置代码

```python
train_params = {
    'data': dataset_path,
    'epochs': 200,
    'imgsz': 640,
    'batch': 8,
    'device': '0',
    'workers': 8,
    'project': 'rtdetr_weld_quality',
    'name': 'rtdetr-C2f-MSMHSA-CGLU_CA_HSFPN3+nwd损失函数',
    'exist_ok': True,
    'amp': True,
    'resume': False
}
```

---

## 六、预训练模型权重

### 6.1 权重文件

训练好的模型权重位于 [weights/](file:///root/.trae-cn/my-project/weights/) 目录：

| 文件 | 路径 | 说明 |
|------|------|------|
| `best.pt` | `/root/.trae-cn/my-project/weights/best.pt` | 验证集性能最佳的权重 |
| `last.pt` | `/root/.trae-cn/my-project/weights/last.pt` | 最后一轮训练的权重 |

### 6.2 权重使用方法

**使用最佳权重进行推理：**

```python
from ultralytics import RTDETR

model = RTDETR('/root/.trae-cn/my-project/weights/best.pt')
results = model.predict('/path/to/image.jpg')
results[0].show()
```

**使用最佳权重继续训练：**

```python
model = RTDETR('/root/.trae-cn/my-project/weights/best.pt')
model.train(data='data.yaml', epochs=100, resume=True)
```

### 6.3 预训练配置

```yaml
pretrained: true
model: /root/.trae-cn/my-project/rtdetr-C2f-MSMHSA-DASI.yaml
```

---

## 七、输出结果

### 7.1 结果目录结构

```
rtdetr_weld_quality/
└── rtdetr-C2f-MSMHSA-DASI+mpdIoU和NWD损失函数/
    ├── weights/
    │   ├── best.pt
    │   └── last.pt
    ├── confusion_matrix.png
    ├── PR_curve.png
    ├── results.csv
    └── args.yaml
```

### 7.2 评估指标

| 指标 | 目标值 |
|------|--------|
| mAP@0.5 | > 0.85 |
| mAP@0.5:0.95 | > 0.60 |
| 推理速度 | > 30 FPS |

---

## 八、关键技术改进

### 8.1 损失函数改进

**问题**：焊接缺陷形状不规则，传统IoU损失对边界框位置变化敏感。

**解决方案**：

1. **MPDIoU**：考虑预测框和真实框的重叠部分形状
2. **NWD**：基于最优传输理论，对小目标鲁棒

**权重分配**：MPDIoU : NWD = 0.5 : 0.5

### 8.2 特征融合改进

**DASI模块优势**：
- 多尺度特征整合（P3、P4、P5）
- 自适应深度交互
- 增强小目标检测能力

### 8.3 rtdetr-l 模型优势

**HGBlock特点**：
- 分组卷积，减少计算量
- 轻量级设计，保持精度的同时提升速度
- 支持多尺度特征提取

---

## 九、硬件要求与性能预估

### 9.1 硬件要求

| 组件 | 推荐配置 |
|------|----------|
| GPU | NVIDIA RTX 3090/4090（24GB+显存） |
| 内存 | 32GB+ |
| 存储 | 至少50GB可用空间 |

### 9.2 训练时间预估

| 配置 | 单轮时间 | 总时间（200轮） |
|------|----------|----------------|
| RTX 3090, batch=8 | ~2-3分钟 | ~6-10小时 |
| RTX 4090, batch=8 | ~1-2分钟 | ~3-6小时 |

---

## 十、故障排除

### 10.1 常见问题

| 问题 | 原因 | 解决方案 |
|------|------|----------|
| 模型初始化失败 | 配置文件路径错误 | 检查 `model_config` 路径 |
| 数据加载失败 | 数据集路径错误 | 检查 `dataset_path` 路径 |
| CUDA out of memory | 批次过大 | 减小 `batch` 参数 |
| 训练不收敛 | 学习率问题 | 调整 `lr0` 参数 |

### 10.2 日志查看

训练日志保存在 `rtdetr_weld_quality/` 目录下：
- `results.csv`：每轮训练的损失和指标
- `train.log`：完整训练日志

---

## 十一、附录

### 11.1 参数汇总表

| 类别 | 参数 | 值 |
|------|------|-----|
| 训练设置 | epochs | 200 |
| | batch | 8 |
| | imgsz | 640 |
| 优化器 | optimizer | AdamW |
| | lr0 | 0.0001 |
| | weight_decay | 0.0001 |
| 损失函数 | iou_ratio | 0.5 |
| | wiou_ltype | MPDIoU |
| | nwd_loss | true |

### 11.2 项目文件结构

```
/root/.trae-cn/my-project/
├── train.py                    # 训练入口脚本
├── args.yaml                   # 训练参数配置
├── rtdetr-C2f-MSMHSA-DASI.yaml # 改进版模型配置
├── rtdetr-l.yaml               # Large版本模型配置
├── weights/
│   ├── best.pt                 # 最佳权重
│   └── last.pt                 # 最后权重
├── confusion_matrix.png        # 混淆矩阵
├── PR_curve.png                # PR曲线
├── weldqualityinspectionv9/
│   ├── data.yaml               # 数据集配置
│   ├── train/                  # 训练集
│   ├── valid/                  # 验证集
│   └── test/                   # 测试集
└── TRAINING_GUIDE.md           # 本文档
```

---

**文档版本**: v2.0  
**创建日期**: 2026-05-21  
**适用项目**: RT-DETR焊接质量检测