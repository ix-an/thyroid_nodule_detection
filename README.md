# thyroid_nodule_detection
学习深度学习的项目，甲状腺结节良性/恶性的检测

文件结构：
thyroid_nodule_project/
├── data/                        # 原始数据目录
│   └── Thyroid_nodule_Dataset/  # 你的数据集文件夹
│       ├── train-images/        # 训练图片
│       ├── test-images/         # 测试图片
│       ├── label4train.csv      # 训练标签
│       └── label4test.csv       # 测试标签
├── models/                      # 保存训练好的模型
├── results/                     # 保存评估结果
├── scripts/                     # 主运行脚本
│   ├── train.py                 # 训练脚本
│   └── evaluate.py              # 评估脚本
├── utils/                       # 工具函数
│   └── dataset.py               # 数据集加载器
└── requirements.txt             # 依赖库
