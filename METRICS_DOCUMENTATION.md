# Fairness Metrics 实现文档

## ✅ 完成的功能

### 1. Overall Fairness Score（综合公平性分数）

#### 后端实现 (`backend/app.py`)

**函数位置**: `calculate_overall_fairness_score(metrics)` (第85-126行)

```python
def calculate_overall_fairness_score(metrics):
    """
    Calculate Overall Fairness Score from individual fairness metrics.
    综合公平性分数 = 所有fairness metrics的平均值（越小越公平）
    
    Args:
        metrics (dict): Dictionary containing fairness metrics
    
    Returns:
        float: Overall fairness score (lower is better, 0 is perfect fairness)
    """
    fairness_metric_names = [
        'BNC', 'BPC', 'CUAE', 'EOpp', 'EO', 
        'FDRP', 'FORP', 'FNRB', 'FPRB', 
        'NPVP', 'OAE', 'PPVP', 'SP'
    ]
    
    fairness_values = []
    
    for metric_name in fairness_metric_names:
        if metric_name in metrics:
            metric_value = metrics[metric_name]
            
            # Handle different metric formats
            if isinstance(metric_value, dict):
                # Nested dict (e.g., {'SEX': 0.001, 'MARRIAGE': 0.002})
                for v in metric_value.values():
                    if isinstance(v, (int, float, np.number)):
                        fairness_values.append(float(v))
            elif isinstance(metric_value, (int, float, np.number)):
                # Direct value
                fairness_values.append(float(metric_value))
    
    # Calculate mean of all fairness values
    if fairness_values:
        overall_score = np.mean(fairness_values)
    else:
        overall_score = 0.0
    
    return float(overall_score)
```

#### 计算逻辑

1. **提取所有 fairness metrics 的值**
   - 支持嵌套字典格式（如 `{'SEX': 0.001, 'MARRIAGE': 0.002}`）
   - 支持直接数值格式

2. **计算平均值**
   - 使用 `np.mean()` 计算所有 fairness 值的平均
   - **越小越公平**，0 表示完美公平

3. **自动添加到 metrics**
   - 每次调用 `evaluator.evaluate()` 后自动添加
   - 作为 `metrics['Overall_Fairness']` 返回

#### 调用位置

后端在以下4处调用 `evaluate()` 后都会添加 Overall_Fairness：

1. **初始化时** (`api/debias/init`)
   ```python
   init_metrics = evaluator.evaluate(...)
   init_metrics['Overall_Fairness'] = calculate_overall_fairness_score(init_metrics)
   ```

2. **Step by Step** (`api/debias/{id}/step`)
   ```python
   metrics = job['evaluator'].evaluate(...)
   metrics['Overall_Fairness'] = calculate_overall_fairness_score(metrics)
   ```

3. **Run All Steps - 每轮** (`_run_full_process_thread`)
   ```python
   metrics = job['evaluator'].evaluate(...)
   metrics['Overall_Fairness'] = calculate_overall_fairness_score(metrics)
   ```

4. **Run All Steps - 最终结果** (`_run_full_process_thread`)
   ```python
   final_metrics = job['evaluator'].evaluate(...)
   final_metrics['Overall_Fairness'] = calculate_overall_fairness_score(final_metrics)
   ```

---

### 2. 前端图表显示

#### 默认显示

**Overall Fairness Score** 作为默认选项显示在图表中：

```html
<select class="metric-selector" id="metricSelector">
  <option value="Overall_Fairness" selected>
    Overall Fairness Score (综合公平性)
  </option>
  <optgroup label="Individual Metrics">
    <option value="BNC">BNC - Between Negative Classes</option>
    <option value="BPC">BPC - Between Positive Classes</option>
    <!-- ... 其他 metrics ... -->
  </optgroup>
</select>
```

#### 用户交互

1. **默认状态**
   - 图表显示 **Overall Fairness Score** 随 iteration 的变化
   - Y轴：Overall Fairness 值
   - X轴：Iteration 轮数

2. **切换 Metric**
   - 用户通过下拉选择框选择其他 metric
   - 图表立即更新显示选中的 metric
   - 每个 metric 都有完整的描述

3. **动态 Y轴**
   - Y轴范围根据数据自动调整
   - 适应小数值的 fairness metrics
   - 使用科学计数法显示（如 `1.2e-4`）

---

## 📊 Fairness Metrics 说明

### 支持的 Metrics

| Metric | 全称 | 说明 |
|--------|------|------|
| **Overall_Fairness** | Overall Fairness Score | **综合公平性分数**（所有 metrics 平均值） |
| BNC | Between Negative Classes | 负类别间差异 |
| BPC | Between Positive Classes | 正类别间差异 |
| CUAE | Conditional Use Accuracy Equality | 条件使用准确性平等 |
| EOpp | Equal Opportunity | 机会平等 |
| EO | Equalized Odds | 平衡赔率 |
| FDRP | False Discovery Rate Parity | 错误发现率均等 |
| FORP | False Omission Rate Parity | 错误遗漏率均等 |
| FNRB | False Negative Rate Balance | 假阴性率平衡 |
| FPRB | False Positive Rate Balance | 假阳性率平衡 |
| NPVP | Negative Predictive Value Parity | 负预测值均等 |
| OAE | Overall Accuracy Equality | 整体准确性平等 |
| PPVP | Positive Predictive Value Parity | 正预测值均等 |
| SP | Statistical Parity | 统计均等 |

### Metric 解释

- **所有 fairness metrics 的值越小越好**
- **0 表示完美公平**（两个群体完全相同）
- **值越大表示不公平程度越高**

---

## 🎯 数据流程

### 完整流程

```
1. 用户选择数据 & 运行
   ↓
2. 后端 init_debias
   ├─ evaluator.evaluate()
   ├─ 计算所有 individual metrics (BNC, BPC, ...)
   ├─ calculate_overall_fairness_score()
   └─ 添加 metrics['Overall_Fairness']
   ↓
3. 后端 run_full / step
   每轮 iteration:
   ├─ BM (if enabled)
   ├─ AE (if enabled)
   ├─ evaluator.evaluate()
   ├─ calculate_overall_fairness_score()
   └─ metrics['Overall_Fairness'] 添加到 history
   ↓
4. 前端轮询 /api/debias/{id}/status
   ├─ 获取 history (包含每轮的 Overall_Fairness)
   ├─ 用户选择要显示的 metric
   └─ 图表实时更新
```

### API 返回数据结构

```json
{
  "status": "success",
  "data": {
    "history": [
      {
        "iteration": 1,
        "metrics": {
          "ACC": 0.7845,
          "F1": 0.6234,
          "BNC": 0.000123,
          "BPC": 0.000234,
          "EOpp": 0.000345,
          "Overall_Fairness": 0.000267,  // ← 综合分数
          // ... 其他 metrics
        }
      },
      {
        "iteration": 2,
        "metrics": {
          "ACC": 0.7892,
          "Overall_Fairness": 0.000145,  // ← 越来越小，越来越公平
          // ...
        }
      }
    ]
  }
}
```

---

## 🔍 代码位置总结

### 后端 (`backend/app.py`)

| 功能 | 位置 | 说明 |
|------|------|------|
| `calculate_overall_fairness_score()` | 第85-126行 | 计算综合分数 |
| 初始化添加 | 第403行 | `init_metrics['Overall_Fairness'] = ...` |
| Step添加 | 第575行 | `metrics['Overall_Fairness'] = ...` |
| Run All添加 | 第686行 | `metrics['Overall_Fairness'] = ...` |
| Final添加 | 第745行 | `final_metrics['Overall_Fairness'] = ...` |

### 前端 (`frontend/index.html`)

| 功能 | 位置 | 说明 |
|------|------|------|
| 默认选中 | 第883行 | `const currentSelectedMetric = state.selectedMetric \|\| 'Overall_Fairness'` |
| 下拉选项 | 第958-964行 | `<option value="Overall_Fairness" selected>` |
| 图表渲染 | 第889-894行 | 从 `h.metrics[currentSelectedMetric]` 获取值 |
| 事件监听 | 第1034-1039行 | metric selector 变化时重新渲染 |

---

## ✅ 验证清单

- [x] 后端计算 Overall Fairness Score
- [x] 后端在所有 evaluate() 后添加 Overall_Fairness
- [x] 前端默认显示 Overall Fairness
- [x] 用户可以切换显示其他 individual metrics
- [x] 图表动态更新
- [x] Y轴自动缩放
- [x] 所有计算逻辑在后端（不是前端随便算的）
- [ ] 实际运行测试（需要用户执行）

---

## 🚀 使用方式

### 启动测试

1. **重启后端**
   ```bash
   cd backend
   conda activate bmwithae
   python app.py
   ```

2. **刷新前端**
   - 重新打开 `frontend/index.html`

3. **运行测试**
   - Load Credit 数据
   - 点击 "Run All Steps"
   - 观察图表：
     - 默认显示 **Overall Fairness Score**
     - 值应该随 iteration 递减（越来越公平）
     - 可以切换到其他 metrics 查看

4. **验证数据来源**
   - 打开浏览器开发者工具
   - Network标签 → 查看 `/api/debias/{id}/status`
   - 响应中应包含 `metrics['Overall_Fairness']`

---

## 📝 注意事项

1. **所有 fairness metrics 都是越小越好**
   - Overall_Fairness = 0：完美公平
   - Overall_Fairness > 0：存在不公平

2. **计算完全在后端**
   - 前端只负责显示
   - 不会出现前后端计算不一致的问题

3. **实时更新**
   - Run All Steps 模式：每轮完成后立即更新
   - Step by Step 模式：每次点击立即更新

4. **扩展性**
   - 如需添加新的 fairness metric，只需：
     1. 在 `code_v_0_1/eval.py` 中实现
     2. 在 `backend/app.py` 的 `fairness_metric_names` 列表中添加
     3. 在前端 `metricDescriptions` 中添加描述

---

**实现完成！所有 metrics 计算都在后端，确保数据准确性！** ✨



