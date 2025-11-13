# 配置参数流程说明

## 参数传递逻辑

### 1. 配置更新流程

```
用户修改参数 (前端)
    ↓
POST /api/config
    ↓
更新 core_config (code_v_0_1/config.py)
    ↓
参数生效 (下次 init_debias 时使用)
```

### 2. 任务初始化流程

```
POST /api/debias/init
    ↓
读取当前 core_config
    ↓
(可选) 应用自定义配置
    ↓
更新已导入模块的全局变量
    ├─ eval.SEED
    ├─ eval.PARAMS_MAIN_CLASSIFIER
    ├─ module_BM.USE_BIAS_MITIGATION
    └─ ...
    ↓
创建 Evaluator/BM/AE 实例
    ↓
实例从当前配置读取参数
    ↓
完成后恢复原始配置
```

## 关键机制

### 问题：from config import 引用问题

`code_v_0_1` 模块使用：
```python
from config import SEED, PARAMS_MAIN_CLASSIFIER
```

这会创建**静态引用**，修改 `config.SEED` 不会影响已导入的 `SEED` 变量。

### 解决方案：双重更新

1. **更新 config 模块**
   ```python
   setattr(core_config, 'SEED', new_value)
   ```

2. **更新导入模块的全局变量**
   ```python
   import sys
   eval_module = sys.modules['eval']
   setattr(eval_module, 'SEED', new_value)
   ```

## API 使用示例

### 方式一：全局配置（推荐）

```javascript
// 1. 更新配置
await api.updateConfig({
  PARAMS_MAIN_MAX_ITERATION: 5,
  PARAMS_MAIN_THRESHOLD_EPSILON: 0.85,
  USE_BIAS_MITIGATION: true
});

// 2. 初始化任务（使用全局配置）
const result = await api.initDebias(datasetId);
```

### 方式二：任务专属配置

```javascript
// 直接在 init 时传入配置（覆盖全局配置）
const result = await api.initDebias(datasetId, {
  PARAMS_MAIN_MAX_ITERATION: 3,
  USE_ACCURACY_ENHANCEMENT: true
});
```

## 配置文件说明

### backend/config.py
- 后端服务器配置
- Demo 数据集路径
- 上传/结果目录

### code_v_0_1/config.py
- 算法参数
- 评估指标配置
- 转换参数

## 参数列表

可通过 `GET /api/config` 查看所有可配置参数：

```json
{
  "SEED": 0,
  "USE_BIAS_MITIGATION": true,
  "USE_ACCURACY_ENHANCEMENT": false,
  "PARAMS_MAIN_CLASSIFIER": "LR",
  "PARAMS_MAIN_MAX_ITERATION": 2,
  "PARAMS_MAIN_THRESHOLD_EPSILON": 0.9,
  ...
}
```



