# Fundus Image Restoration Model Based on Vessel Prior and Channel-Gated UNet-Mamba

> 基于血管先验和通道门控 UNet-Mamba 的眼底图像复原模型

---

## 📁 项目目录结构

```
 ├── baseline_code/              # 项目主要内容
│   ├── losses/                 # 损失函数设计
│   ├── models/                 # 模型定义（本项目主要使用的 model）
│   ├── work_fundus_mamba/      # 训练时的训练集和验证集
│   ├── infer.py                # 推理代码
│   ├── train.py                # 训练代码
│   ├── train.yml               # 训练配置文件
│   ├── requirement.txt         # 环境依赖
│   └── ...                     # 其他 debug 及测试过程文件（暂未系统整理）
│
├── data/                       # 数据预处理、读取与数据增强工具
│   ├── data_check/             # 数据获取策略测试与数据质量检查
│   └── dataset.py              # 训练过程中的数据采样与获取方法
│
├── notebook_excat_project/     # 教师模型（Teacher Model）相关内容
│   ├── model.py                # 教师模型定义
│   ├── train.py                # 教师模型训练代码
│   ├── train.yml               # 教师模型训练配置文件
│   ├── requirement.txt         # 教师模型环境依赖
│   ├── dataset.py              # 教师模型的数据增强与预处理
│   ├── predict_one.py          # 单张图的血管分割推理
│   └── infer.py                # 批量推理、数值指标计算及 debug 残留文件 
|   
│
├── README.md   
└── ...                       
```

---

## 🔧 主要代码文件说明

### baseline_code/

| 文件/目录                          | 说明                                                                                             |
| ---------------------------------- | ------------------------------------------------------------------------------------------------ |
| `models/model.py`                | 项目核心模型定义，主要使用其中的 model，其余的为项目初期smoking和pipleline测试的等工作的残留文件 |
| `losses/`                        | 损失函数设计                                                                                     |
| `train.py`                       | 学生模型训练入口                                                                                 |
| `train.yml`                      | 学生模型训练超参数与配置                                                                         |
| `infer.py`                       | 学生模型推理代码                                                                                 |
| `requirement.txt`                | 学生模型运行环境依赖                                                                             |
| `work_fundus_mamba/split`        | 训练集与验证集数据                                                                               |
| `work_fundus_mamba/checkpoints/` | 训练过程中保存的模型权重                                                                         |

### data/

| 文件/目录      | 说明                           |
| -------------- | ------------------------------ |
| `dataset.py` | 训练过程中的数据采样与获取方法 |
| `data_check` | 数据获取策略测试与数据质量检查 |

### notebook_excat_project/（教师模型）

| 文件                     | 说明                                |
| ------------------------ | ----------------------------------- |
| `model.py`             | 教师模型定义                        |
| `train.py`             | 教师模型训练代码                    |
| `train.yml`            | 教师模型训练配置文件                |
| `requirement.txt`      | 教师模型环境依赖                    |
| `dataset.py`           | 教师模型的数据增强与预处理          |
| `predict_one.py`       | 单张图血管分割推理                  |
| `infer_*.py`           | 批量推理、指标计算及 debug 残留文件 |
| `files/checkpoint.pth` | 教师模型权重                        |

---

## 📦 补充资源

以下文件/目录包含模型权重、测试数据、推理前后对比结果及测试视频等，因文件较大，已通过 `.gitignore` 排除，可按需单独获取：

- **checkpoints/** — 模型权重文件（`.pt`）
- **推理结果图片** — 推理前后对比图（`.png`）
- **测试数据** — 部分测试样本及数值指标（`.csv`）
- **测试视频** — 模型效果演示

---

## ⚠️ 说明

项目目前仍在持续拓展中，部分 debug 和测试过程的残留文件暂未进行系统整理。以上仅介绍核心代码文件，后续将逐步完善项目结构。
