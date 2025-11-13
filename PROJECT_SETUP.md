# BMWithAE - Bias Mitigation with Accuracy Enhancement

## 项目概述

一个用于机器学习模型偏见缓解和准确性提升的可视化系统。该系统提供交互式Web界面，支持数据探索、偏见分析和迭代式去偏过程。

## 技术栈

### 后端
- **Python 3.8+**
- **Flask 3.0.0** - Web框架
- **pandas 2.0+** - 数据处理
- **scikit-learn 1.3+** - 机器学习算法
- **XGBoost, LightGBM, CatBoost** - 高级boosting算法

### 前端
- **纯HTML/CSS/JavaScript** - 无需构建工具
- **SVG图表** - 动态可视化
- **Fetch API** - 与后端通信

## 项目结构

```
BMWithAE/
├── backend/                    # 后端Python代码
│   ├── app.py                 # Flask主应用
│   ├── core_config.py         # 核心配置参数
│   ├── backend_config.py      # 后端配置
│   ├── datasets_info.py       # 数据集信息
│   ├── module_BM.py          # 偏见缓解模块
│   ├── module_AE.py          # 准确性提升模块
│   ├── module_load.py        # 数据加载模块
│   ├── module_transform.py   # 数据转换模块
│   ├── eval.py               # 评估模块
│   ├── uploads/              # 上传文件目录
│   ├── results/              # 实验结果
│   └── logs/                 # 日志文件
│
├── frontend/                   # 前端代码
│   ├── index.html            # 主页面
│   ├── styles.css            # 样式文件
│   ├── api.js                # API客户端
│   ├── explorer.html         # 数据浏览器
│   └── explorer.css          # 浏览器样式
│
├── data/                      # 数据文件
│   ├── credit.xlsx           # 示例数据集
│   └── data_compas.csv       # COMPAS数据集
│
├── requirements.txt          # Python依赖
└── README.md                 # 项目说明
```

## 快速开始

### 1. 环境准备

```bash
# 克隆项目（如果需要）
cd BMWithAE

# 创建虚拟环境（推荐）
python -m venv venv

# 激活虚拟环境
# macOS/Linux:
source venv/bin/activate
# Windows:
venv\Scripts\activate
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 启动后端服务

```bash
cd backend
python app.py
```

后端将在 `http://localhost:5000` 启动

### 4. 打开前端

**选项1：直接打开**
```bash
open frontend/index.html
```

**选项2：使用本地服务器（推荐）**
```bash
cd frontend
python -m http.server 8000
# 然后在浏览器访问 http://localhost:8000
```

## 主要功能

### 1. 数据加载与探索
- 支持Excel (.xlsx) 和CSV文件
- 自动检测特征类型（分类/连续）
- 交互式数据分布可视化
- 受保护属性选择

### 2. 偏见分析
- **Statistical Parity** - 统计平等性
- **Equal Opportunity** - 机会均等
- **Equalized Odds** - 均衡赔率
- **Disparate Impact** - 差异影响
- **Subgroup Analysis** - 子组分析

### 3. 去偏过程
- 迭代式偏见缓解（Bias Mitigation）
- 准确性提升（Accuracy Enhancement）
- 实时进度监控
- Max Epsilon收敛跟踪
- 历史迭代记录

### 4. 可视化
- 双图表实时更新
  - 左侧：指标演化（Max Epsilon / 公平性指标）
  - 右侧：准确率演化
- 模糊背景弹窗展示详细偏见指标
- 子组正例率条形图

## API端点

### 数据管理
- `GET /api/datasets` - 获取可用数据集列表
- `POST /api/upload` - 上传新数据集
- `GET /api/data/<dataset_id>/info` - 获取数据集详细信息
- `POST /api/data/<dataset_id>/bias-metrics` - 计算偏见指标

### 去偏过程
- `POST /api/debias/init` - 初始化去偏任务
- `POST /api/debias/<job_id>/step` - 执行一步迭代
- `GET /api/debias/<job_id>/status` - 获取任务状态
- `POST /api/debias/<job_id>/stop` - 停止任务

### 配置
- `GET /api/config` - 获取当前配置
- `POST /api/config` - 更新配置

## 配置参数

主要配置参数在 `backend/core_config.py` 中：

- `PARAMS_MAIN_MAX_ITERATION`: 最大迭代次数（默认20）
- `PARAMS_MAIN_THRESHOLD_EPSILON`: Epsilon阈值（默认0.9）
- `PARAMS_MAIN_CLASSIFIER`: 分类器选择
- `PARAMS_MAIN_TRAINING_RATE`: 训练集比例
- `SEED`: 随机种子

## 开发说明

### 后端开发
1. 修改代码后重启 `app.py`
2. 查看终端日志调试
3. 使用 `VERBOSE=True` 获取详细输出

### 前端开发
1. 修改代码后刷新浏览器
2. 使用浏览器开发者工具（F12）调试
3. 查看Console日志

### 添加新的偏见指标
1. 在 `backend/eval.py` 中添加计算函数
2. 在 `backend/app.py` 的API中暴露
3. 在 `frontend/index.html` 中添加显示逻辑

## 故障排除

### 端口冲突
如果5000端口被占用：
```bash
# 查找占用端口的进程
lsof -i :5000
# 或在backend_config.py中修改端口
```

### CORS错误
确保 Flask-CORS 已安装并在 `app.py` 中正确配置

### 数据加载失败
- 检查文件格式（.xlsx或.csv）
- 确保有目标列（target）
- 验证数据中没有过多缺失值

### 模型训练错误
- 检查是否安装了所有boosting库（xgboost, lightgbm, catboost）
- 确保数据预处理正确
- 查看后端日志获取详细错误信息

## 性能优化

### 大数据集
- 考虑减少 `PARAMS_MAIN_MAX_ITERATION`
- 使用采样方法减少数据量
- 选择更快的分类器（如LogisticRegression）

### 前端性能
- 子组分析限制为前10个（已实现）
- 使用虚拟滚动处理大量历史记录
- 考虑使用Web Workers处理复杂计算

## 贡献指南

1. Fork项目
2. 创建功能分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启Pull Request

## 许可证

[添加你的许可证信息]

## 联系方式

[添加你的联系信息]

## 致谢

- 基于公平机器学习的研究
- 参考FairSight和FairVis的可视化设计
- 使用多种开源机器学习库

---

**最后更新**: 2025年1月
**版本**: 1.0.0

