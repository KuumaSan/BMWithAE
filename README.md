# BMWithAE - Bias Mitigation with Accuracy Enhancement

<div align="center">

![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)
![Flask Version](https://img.shields.io/badge/flask-3.0.0-green)
![License](https://img.shields.io/badge/license-MIT-orange)

一个用于机器学习模型偏见缓解和准确性提升的交互式可视化系统

[功能特性](#功能特性) • [快速开始](#快速开始) • [文档](#文档) • [截图](#截图)

</div>

---

## 📖 项目简介

BMWithAE 是一个专注于机器学习公平性的可视化分析系统。它提供了直观的Web界面，帮助数据科学家和研究人员：

- 🔍 **探索数据偏见** - 交互式分析多维度偏见来源
- ⚖️ **缓解算法偏见** - 迭代式偏见缓解过程
- 📈 **提升模型准确性** - 在保证公平性的同时优化性能
- 📊 **实时可视化** - 动态追踪指标演化过程

## ✨ 功能特性

### 🎯 核心功能

- **多维度偏见分析**
  - Statistical Parity (统计平等性)
  - Equal Opportunity (机会均等)
  - Equalized Odds (均衡赔率)
  - Disparate Impact (差异影响)
  
- **交互式数据探索**
  - 动态特征分布可视化
  - 子组分析和详细偏见指标
  - 多受保护属性选择
  - 实时偏见评分

- **迭代式去偏过程**
  - Bias Mitigation (偏见缓解)
  - Accuracy Enhancement (准确性提升)
  - 实时进度监控
  - 历史记录追踪

- **丰富的可视化**
  - 双图表实时更新
  - Max Epsilon 收敛追踪
  - 准确率演化曲线
  - 模态弹窗详细展示

### 🛠️ 技术特性

- **后端**: Python + Flask + Pandas + Scikit-learn
- **前端**: 原生 HTML/CSS/JavaScript (无需构建)
- **机器学习**: 支持多种分类器 (XGBoost, LightGBM, CatBoost等)
- **数据格式**: Excel (.xlsx), CSV
- **API设计**: RESTful 架构

## 🚀 快速开始

### 前置要求

- Python 3.8+
- pip 包管理器

### 安装步骤

1. **克隆仓库**

```bash
git clone https://github.com/yourusername/BMWithAE.git
cd BMWithAE
```

2. **创建虚拟环境**

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

3. **安装依赖**

```bash
pip install -r requirements.txt
```

4. **启动后端服务**

```bash
cd backend
python app.py
```

后端将在 `http://localhost:5000` 启动

5. **打开前端界面**

在浏览器中打开 `frontend/index.html`

或使用本地服务器（推荐）：

```bash
cd frontend
python -m http.server 8000
# 访问 http://localhost:8000
```

## 📚 文档

### 项目结构

```
BMWithAE/
├── backend/              # 后端Python代码
│   ├── app.py           # Flask主应用
│   ├── core_config.py   # 核心配置
│   ├── module_BM.py     # 偏见缓解模块
│   ├── module_AE.py     # 准确性提升模块
│   ├── eval.py          # 评估模块
│   └── ...
├── frontend/            # 前端代码
│   ├── index.html       # 主界面
│   ├── styles.css       # 样式
│   └── api.js           # API客户端
├── data/                # 数据文件
├── requirements.txt     # Python依赖
└── README.md           # 本文件
```

### 使用指南

#### 1. 加载数据

- 点击 "Load Demo" 加载示例数据集
- 或点击 "Upload Data" 上传自己的数据

#### 2. 数据探索

- 选择受保护属性（可多选）
- 查看偏见评分和详细指标
- 点击特征查看分布和子组分析

#### 3. 配置参数

- 点击 "Configuration" 设置去偏参数
- 选择分类器、迭代次数、阈值等

#### 4. 执行去偏

- "Run All Steps": 自动执行所有迭代
- "Step by Step": 单步执行，便于观察

#### 5. 查看结果

- 左图: Max Epsilon / 公平性指标演化
- 右图: 准确率演化
- 下方: 迭代历史记录

### API文档

详细API文档请参考 [PROJECT_SETUP.md](PROJECT_SETUP.md)

主要端点：

- `POST /api/upload` - 上传数据
- `POST /api/debias/init` - 初始化去偏任务
- `POST /api/debias/<job_id>/step` - 执行迭代步骤
- `GET /api/debias/<job_id>/status` - 获取任务状态
- `POST /api/data/<dataset_id>/bias-metrics` - 计算偏见指标

## 📸 截图

### 数据探索界面
*交互式偏见分析和特征分布*

### 去偏过程可视化
*实时追踪Max Epsilon和准确率演化*

### 子组分析
*详细的偏见来源定位*

## ⚙️ 配置

主要配置参数在 `backend/core_config.py`:

```python
PARAMS_MAIN_MAX_ITERATION = 20        # 最大迭代次数
PARAMS_MAIN_THRESHOLD_EPSILON = 0.9   # Epsilon阈值
PARAMS_MAIN_CLASSIFIER = 'XGB'        # 分类器选择
PARAMS_MAIN_TRAINING_RATE = 0.7       # 训练集比例
```

## 🐛 故障排除

### 端口冲突

```bash
# 修改 backend/backend_config.py 中的端口
PORT = 5001  # 改为其他端口
```

### 依赖安装失败

```bash
# 升级pip
pip install --upgrade pip

# 单独安装问题包
pip install xgboost --no-cache-dir
```

### CORS错误

确保后端正确启动并且 Flask-CORS 已配置

## 🤝 贡献

欢迎贡献！请遵循以下步骤：

1. Fork 本仓库
2. 创建功能分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

## 📄 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件

## 📬 联系方式

如有问题或建议，请通过以下方式联系：

- 提交 Issue: [GitHub Issues](https://github.com/yourusername/BMWithAE/issues)
- Email: your.email@example.com

## 🙏 致谢

- 参考了 FairSight 和 FairVis 的可视化设计
- 基于公平机器学习的前沿研究
- 感谢开源社区的支持

---

<div align="center">

**[⬆ 回到顶部](#bmwithae---bias-mitigation-with-accuracy-enhancement)**

Made with ❤️ for Fair ML

</div>
