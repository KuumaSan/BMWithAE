# BMWithAE 后端API设计文档

## 概述

本文档说明如何将现有的Python代码转换为Flask后端API，以便与前端界面集成。

---

## 技术栈

- **框架**: Flask
- **跨域支持**: Flask-CORS
- **异步任务**: Flask-SocketIO (用于实时进度更新)
- **文件上传**: werkzeug

---

## 需要创建的API端点

### 1. 数据管理相关

#### 1.1 上传数据集
**端点**: `POST /api/data/upload`

**功能**: 接收用户上传的CSV文件

**对应函数**: 
- `module_load.py` 中的 `DataLoader.load_data()`

**请求参数**:
```json
{
  "file": "multipart/form-data",
  "target_column": "string",      // 目标变量列名
  "protected_columns": ["array"]  // 受保护属性列名数组
}
```

**响应**:
```json
{
  "status": "success",
  "data": {
    "dataset_id": "uuid",
    "rows": 30000,
    "columns": 24,
    "features": ["list", "of", "columns"],
    "categorical": ["list"],
    "numerical": ["list"],
    "preview": [/* 前10行数据 */]
  }
}
```

**作用**: 让用户上传自己的数据集，系统自动分析数据结构

---

#### 1.2 加载Demo数据集
**端点**: `POST /api/data/demo`

**功能**: 加载预设的演示数据集（COMPAS, Adult, Credit）

**对应函数**:
- `module_load.py` 中的 `DataLoader.load_data()`
- 使用 `config.py` 中的 `DATASET_INFO`

**请求参数**:
```json
{
  "dataset_name": "credit"  // "compas" | "adult" | "credit"
}
```

**响应**: 同上传接口

**作用**: 方便用户快速体验系统，无需准备数据

---

#### 1.3 获取数据统计信息
**端点**: `GET /api/data/{dataset_id}/stats`

**功能**: 获取数据集的详细统计信息

**对应函数**:
- `eval.py` 中的 `Evaluator.calculate_epsilon()`

**响应**:
```json
{
  "status": "success",
  "data": {
    "total_samples": 30000,
    "feature_stats": {
      "AGE": {
        "type": "numerical",
        "min": 18,
        "max": 96,
        "mean": 35.5,
        "distribution": [/* 分布数据 */]
      }
    },
    "epsilon_values": {
      "SEX": {
        "AGE": 0.000489,
        "EDUCATION": 0.000393
      }
    }
  }
}
```

**作用**: 为前端的Data Explorer提供数据可视化支持

---

### 2. 参数配置相关

#### 2.1 获取当前配置
**端点**: `GET /api/config`

**功能**: 获取当前的所有配置参数

**对应函数**:
- 读取 `config.py` 中的所有参数

**响应**:
```json
{
  "status": "success",
  "config": {
    "PARAMS_NUM_TO_CAT_METHOD": "quartile",
    "PARAMS_NUM_TO_CAT_CUTS": 4,
    "SEED": 0,
    "USE_BIAS_MITIGATION": true,
    "USE_ACCURACY_ENHANCEMENT": false,
    "PARAMS_MAIN_CLASSIFIER": "LR",
    "PARAMS_MAIN_MAX_ITERATION": 2,
    "PARAMS_MAIN_THRESHOLD_EPSILON": 0.9,
    // ... 所有其他参数
  }
}
```

**作用**: 让前端参数配置弹窗显示当前设置

---

#### 2.2 更新配置
**端点**: `POST /api/config`

**功能**: 更新配置参数

**对应函数**:
- 动态修改 `config.py` 中的参数值（运行时）

**请求参数**:
```json
{
  "PARAMS_MAIN_THRESHOLD_EPSILON": 0.85,
  "PARAMS_MAIN_MAX_ITERATION": 5,
  "USE_ACCURACY_ENHANCEMENT": true
  // ... 任意参数
}
```

**响应**:
```json
{
  "status": "success",
  "message": "Configuration updated successfully"
}
```

**作用**: 允许用户自定义算法参数

---

### 3. 去偏过程控制

#### 3.1 启动去偏过程
**端点**: `POST /api/debias/start`

**功能**: 开始执行偏差消除和准确性增强流程

**对应函数**:
- `main.py` 中的 `run_test()` 函数
- 需要重构为异步任务

**请求参数**:
```json
{
  "dataset_id": "uuid",
  "mode": "all",  // "all" | "step"
  "config": {/* 可选的临时配置覆盖 */}
}
```

**响应**:
```json
{
  "status": "success",
  "job_id": "uuid",
  "message": "Debiasing process started"
}
```

**作用**: 启动主要的去偏流程

---

#### 3.2 执行单步操作
**端点**: `POST /api/debias/{job_id}/next`

**功能**: 在step-by-step模式下执行下一步

**对应函数**:
- `main.py` 中的迭代逻辑（需要拆分）
- `module_BM.py` 中的 `BiasMitigation.mitigate()`
- `module_AE.py` 中的 `AccuracyEnhancement.enhance()`

**响应**:
```json
{
  "status": "success",
  "data": {
    "step": 1,
    "total_steps": 7,
    "step_name": "Bias Mitigation",
    "description": "Applying debiasing techniques",
    "selected_attribute": "AGE",
    "selected_label": "MARRIAGE",
    "metrics": {
      "fairness_score": 0.966,
      "accuracy": 0.894,
      "epsilon": 0.000277
    },
    "visualization_data": {/* 图表数据 */}
  }
}
```

**作用**: 支持前端的分步执行模式

---

#### 3.3 获取进度
**端点**: `GET /api/debias/{job_id}/status`

**功能**: 查询当前任务的执行状态

**对应函数**:
- 需要在运行时维护任务状态

**响应**:
```json
{
  "status": "success",
  "data": {
    "job_id": "uuid",
    "state": "running",  // "pending" | "running" | "completed" | "failed"
    "progress": 0.57,     // 0-1
    "current_iteration": 2,
    "max_iteration": 5,
    "current_step": "Bias Detection",
    "elapsed_time": 15.3  // seconds
  }
}
```

**作用**: 前端显示进度条和当前状态

---

#### 3.4 停止/重置
**端点**: `POST /api/debias/{job_id}/stop`

**功能**: 停止当前运行的任务

**响应**:
```json
{
  "status": "success",
  "message": "Job stopped successfully"
}
```

**作用**: 允许用户中断长时间运行的任务

---

### 4. 结果获取

#### 4.1 获取最终结果
**端点**: `GET /api/results/{job_id}`

**功能**: 获取完整的去偏结果

**对应函数**:
- 读取 `results/all_results.json`
- `main.py` 中的 `save_results()` 保存的数据

**响应**:
```json
{
  "status": "success",
  "data": {
    "job_id": "uuid",
    "config_parameters": {/* 使用的配置 */},
    "initial_metrics": {
      "ACC": 0.7778,
      "fairness_scores": {/* ... */}
    },
    "iterations": [
      {
        "iteration": 1,
        "metrics": {/* ... */},
        "changed_dict": {/* ... */},
        "selected_attributes": {/* ... */}
      }
    ],
    "final_results": {
      "metrics": {/* ... */},
      "changed_dict": {
        "AGE": "dropped",
        "EDUCATION": {3: 1}
      }
    }
  }
}
```

**作用**: 前端展示最终结果和对比

---

#### 4.2 获取迭代历史
**端点**: `GET /api/results/{job_id}/history`

**功能**: 获取每次迭代的详细历史记录

**对应函数**:
- 从 `results_history` 提取数据

**响应**:
```json
{
  "status": "success",
  "data": {
    "iterations": [
      {
        "iteration": 1,
        "fairness_evolution": [/* 时间序列数据 */],
        "accuracy_evolution": [/* 时间序列数据 */],
        "epsilon_values": {/* ... */}
      }
    ]
  }
}
```

**作用**: 为前端的Fairness Metric Evolution图表提供数据

---

#### 4.3 导出结果
**端点**: `GET /api/results/{job_id}/export`

**功能**: 导出完整结果为JSON或CSV

**查询参数**: `?format=json|csv`

**响应**: 文件下载

**作用**: 允许用户下载和保存结果

---

### 5. 评估和指标

#### 5.1 计算公平性指标
**端点**: `POST /api/evaluate/fairness`

**功能**: 计算指定数据集的公平性指标

**对应函数**:
- `eval.py` 中的 `Evaluator.evaluate()`
- `eval.py` 中的 `Evaluator.calculate_epsilon()`

**请求参数**:
```json
{
  "dataset_id": "uuid",
  "metrics": ["BNC", "EOpp", "SP"]  // 选择要计算的指标
}
```

**响应**:
```json
{
  "status": "success",
  "metrics": {
    "BNC": {"SEX": 0.0078, "MARRIAGE": 0.0386},
    "EOpp": {"SEX": 0.0, "MARRIAGE": 0.0},
    "SP": {"SEX": 0.0, "MARRIAGE": 0.0}
  }
}
```

**作用**: 实时计算和显示公平性指标

---

#### 5.2 获取支持的分类器列表
**端点**: `GET /api/classifiers`

**功能**: 返回所有支持的分类器类型

**对应函数**:
- `eval.py` 中的分类器初始化逻辑

**响应**:
```json
{
  "status": "success",
  "classifiers": [
    {"code": "LR", "name": "Logistic Regression"},
    {"code": "DT", "name": "Decision Tree"},
    {"code": "KNN", "name": "K-Nearest Neighbors"},
    {"code": "GBDT", "name": "Gradient Boosting Decision Tree"},
    {"code": "XGBoost", "name": "eXtreme Gradient Boosting"},
    {"code": "RF", "name": "Random Forest"},
    {"code": "LGBM", "name": "LightGBM"},
    {"code": "CatBoost", "name": "CatBoost"}
  ]
}
```

**作用**: 动态生成前端的分类器选择下拉框

---

### 6. WebSocket接口（实时更新）

#### 6.1 连接
**端点**: `ws://localhost:5000/socket.io`

**事件**: 
- `connect`: 连接建立
- `disconnect`: 连接断开
- `progress_update`: 进度更新
- `step_completed`: 步骤完成
- `job_completed`: 任务完成
- `error`: 错误信息

**数据格式**:
```json
{
  "event": "progress_update",
  "data": {
    "job_id": "uuid",
    "progress": 0.45,
    "current_step": "Bias Mitigation",
    "message": "Processing iteration 2..."
  }
}
```

**作用**: 实时推送进度更新到前端

---

## 代码重构建议

### 1. 主函数重构

将 `main.py` 中的 `run_test()` 函数重构为：

```python
class DebiasJob:
    def __init__(self, dataset_id, config):
        self.job_id = str(uuid.uuid4())
        self.dataset_id = dataset_id
        self.config = config
        self.state = "pending"
        self.progress = 0
        self.results = None
        
    def run(self, mode="all", callback=None):
        """执行去偏过程，支持回调函数报告进度"""
        # 原 run_test() 的逻辑
        # 每个步骤后调用 callback(progress, step_info)
        
    def run_next_step(self):
        """执行下一步（step模式）"""
        # 单步执行逻辑
```

### 2. 任务管理

创建任务管理器：

```python
class JobManager:
    def __init__(self):
        self.jobs = {}  # job_id -> DebiasJob
        
    def create_job(self, dataset_id, config):
        job = DebiasJob(dataset_id, config)
        self.jobs[job.job_id] = job
        return job
        
    def get_job(self, job_id):
        return self.jobs.get(job_id)
        
    def delete_job(self, job_id):
        if job_id in self.jobs:
            del self.jobs[job_id]
```

### 3. 数据存储

使用临时文件或数据库存储：

```python
# 数据集存储
datasets = {}  # dataset_id -> {"X": df, "Y": series, "O": df, ...}

# 结果存储
results = {}   # job_id -> results_dict
```

---

## Flask应用结构

```
backend/
├── app.py                  # Flask应用入口
├── config.py              # 配置管理（已有）
├── requirements.txt       # 依赖（已有）
├── api/
│   ├── __init__.py
│   ├── data.py           # 数据相关API
│   ├── config.py         # 配置相关API
│   ├── debias.py         # 去偏过程API
│   ├── results.py        # 结果相关API
│   └── evaluate.py       # 评估相关API
├── core/
│   ├── __init__.py
│   ├── job_manager.py    # 任务管理器
│   ├── debias_job.py     # 去偏任务类
│   └── data_manager.py   # 数据管理器
├── module_load.py        # 现有模块
├── module_BM.py          # 现有模块
├── module_AE.py          # 现有模块
├── module_transform.py   # 现有模块
├── eval.py               # 现有模块
├── uploads/              # 上传文件目录
└── results/              # 结果文件目录
```

---

## 最小可行示例（app.py）

```python
from flask import Flask, request, jsonify
from flask_cors import CORS
import uuid

app = Flask(__name__)
CORS(app)

# 简单的数据存储
datasets = {}
jobs = {}

@app.route('/api/data/upload', methods=['POST'])
def upload_data():
    file = request.files.get('file')
    target = request.form.get('target_column')
    protected = request.form.getlist('protected_columns')
    
    # 处理数据...
    dataset_id = str(uuid.uuid4())
    
    return jsonify({
        'status': 'success',
        'data': {
            'dataset_id': dataset_id,
            'rows': 30000,
            'columns': 24
        }
    })

@app.route('/api/debias/start', methods=['POST'])
def start_debias():
    data = request.json
    dataset_id = data.get('dataset_id')
    
    # 创建任务...
    job_id = str(uuid.uuid4())
    
    # 异步执行...
    
    return jsonify({
        'status': 'success',
        'job_id': job_id
    })

@app.route('/api/results/<job_id>', methods=['GET'])
def get_results(job_id):
    results = jobs.get(job_id, {}).get('results')
    
    return jsonify({
        'status': 'success',
        'data': results
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)
```

---

## 前后端交互流程

```
1. 用户上传数据
   Frontend -> POST /api/data/upload
   Backend  -> 返回 dataset_id

2. 用户配置参数
   Frontend -> GET /api/config (获取默认值)
   Frontend -> 用户修改参数
   Frontend -> POST /api/config (保存配置)

3. 启动去偏过程
   Frontend -> POST /api/debias/start
   Backend  -> 返回 job_id
   
4. 实时更新（WebSocket）
   Backend  -> 推送进度更新
   Frontend -> 更新UI显示

5. 获取结果
   Frontend -> GET /api/results/{job_id}
   Frontend -> 展示可视化结果
```

---

## 需要安装的额外依赖

```bash
pip install flask flask-cors flask-socketio python-socketio
```

---

## 开发优先级建议

**Phase 1 - 核心功能**:
1. 数据上传和Demo加载 API
2. 启动去偏过程 API
3. 获取结果 API

**Phase 2 - 参数控制**:
4. 配置管理 API
5. 单步执行支持

**Phase 3 - 增强功能**:
6. 实时进度更新（WebSocket）
7. 数据统计和可视化 API
8. 结果导出功能

---

## 注意事项

1. **安全性**: 
   - 文件上传大小限制
   - 文件类型验证
   - 路径遍历攻击防护

2. **性能**:
   - 使用后台任务队列（Celery）处理长时间运行的任务
   - 考虑使用Redis缓存中间结果

3. **错误处理**:
   - 统一的错误响应格式
   - 详细的错误日志

4. **资源管理**:
   - 定期清理过期的数据集和任务
   - 限制并发任务数量

---

## 总结

将Python代码转换为Flask API的关键是：
1. **分离逻辑**: 将UI逻辑和核心算法分离
2. **异步处理**: 长时间任务使用后台队列
3. **状态管理**: 维护任务和数据的状态
4. **实时通信**: 使用WebSocket推送进度
5. **RESTful设计**: 遵循REST API最佳实践

这样设计后，前端可以灵活地调用后端API，实现一个功能完整的去偏系统。


