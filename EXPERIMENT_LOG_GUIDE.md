# 实验日志功能说明

## 功能概述

系统运行完毕后会自动生成一个JSON格式的实验日志文件，包含所有参数、中间态metrics和最终结果，方便复现实验。

## 日志文件位置

日志文件保存在：`backend/logs/`

文件命名格式：`experiment_{dataset_name}_{timestamp}_{job_id}.json`

例如：`experiment_credit_20241028_153045_a1b2c3d4.json`

## 日志内容

### 1. 实验信息 (experiment_info)
- job_id: 任务唯一标识
- timestamp: 实验时间戳
- dataset_name: 数据集名称
- dataset_shape: 数据集行列数
- target_column: 目标列
- protected_columns: 受保护属性
- duration_seconds: 总运行时长

### 2. 配置参数 (configuration)
所有实验配置参数，包括：
- 分类器类型
- 最大迭代次数
- Epsilon阈值
- Accuracy阈值
- 公平性指标列表
- 准确性指标列表
- 其他算法参数

### 3. 初始状态 (initial_state)
- metrics: 初始公平性和准确性指标
- epsilon: 初始bias concentration
- epsilon_threshold: Epsilon终止阈值
- accuracy_threshold: Accuracy终止阈值

### 4. 迭代历史 (iterations)
每次迭代的详细记录：
- iteration: 迭代编号
- metrics: 该轮的所有评估指标
- epsilon: 该轮的bias concentration
- selected_attribute: 选中的属性
- selected_label_O: 选中的保护属性标签
- current_max_epsilon: 当前最大epsilon值
- changed_dict: 数据变换字典

### 5. 最终状态 (final_state)
- terminated: 是否提前终止
- termination_reason: 终止原因
- total_iterations: 总迭代次数
- final_metrics: 最终评估指标

## 使用方式

### 前端界面
1. 运行实验完成后，系统会在右上角显示一个文档图标按钮
2. 点击该按钮可以查看日志文件路径
3. 点击"Reset"按钮后，该按钮会隐藏

### 后端API
日志路径会包含在 `/api/debias/<job_id>/status` 的响应中：
```json
{
  "status": "success",
  "data": {
    "state": "completed",
    "log_path": "/path/to/backend/logs/experiment_credit_20241028_153045_a1b2c3d4.json",
    ...
  }
}
```

## 日志文件示例结构

```json
{
  "experiment_info": {
    "job_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
    "timestamp": "20241028_153045",
    "dataset_name": "credit",
    "dataset_shape": {
      "rows": 30000,
      "columns": 22
    },
    "target_column": "default payment next month",
    "protected_columns": ["SEX", "MARRIAGE"],
    "duration_seconds": 45.23
  },
  "configuration": {
    "PARAMS_MAIN_CLASSIFIER": "LR",
    "PARAMS_MAIN_MAX_ITERATION": 20,
    "PARAMS_MAIN_THRESHOLD_EPSILON": 0.05,
    "USE_BIAS_MITIGATION": true,
    "USE_ACCURACY_ENHANCEMENT": true,
    ...
  },
  "initial_state": {
    "metrics": {
      "ACC": 0.7850,
      "F1": 0.4234,
      "Overall_Fairness": 0.8123,
      "BNC": 0.0156,
      ...
    },
    "epsilon": {
      "SEX": {
        "AGE": 0.1234,
        "LIMIT_BAL": 0.0987,
        ...
      }
    },
    "epsilon_threshold": 0.05,
    "accuracy_threshold": 0.95
  },
  "iterations": [
    {
      "iteration": 1,
      "metrics": { ... },
      "epsilon": { ... },
      "selected_attribute": "AGE",
      "selected_label_O": "SEX",
      "current_max_epsilon": 0.1234,
      "changed_dict": { ... }
    },
    ...
  ],
  "final_state": {
    "terminated": true,
    "termination_reason": "Epsilon threshold reached: 0.0489 <= 0.0500",
    "total_iterations": 8,
    "final_metrics": {
      "ACC": 0.7823,
      "F1": 0.4201,
      "Overall_Fairness": 0.9456,
      ...
    }
  }
}
```

## 注意事项

1. 日志文件会在**任务完成或失败时**自动生成
2. 即使任务失败，也会生成日志文件用于调试
3. 日志文件采用UTF-8编码，支持中文
4. JSON格式便于程序读取和二次分析
5. 所有NumPy类型会自动转换为Python原生类型以确保JSON兼容性



