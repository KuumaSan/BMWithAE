# 图表显示修复文档

## ✅ 已修复的问题

### 问题1: 图表显示不美观，文字重叠

#### 修复内容

1. **移除中文标签**
   - 下拉选项：`Overall Fairness Score (综合公平性)` → `Overall Fairness Score`

2. **移除数据点上的文字标签**
   - **之前**：每个数据点上都显示具体值（如 `1.4e-1`），导致标签重叠
   - **现在**：只显示数据点（圆圈），鼠标悬停时显示值
   - 使用SVG `<title>` 标签实现tooltip效果

3. **优化Y轴标签**
   - 调整位置：`x="35"` → `x="32"`（更靠左，避免与图表重叠）
   - 减小字体：`font-size="10"` → `font-size="9"`
   - 减少精度：`.toExponential(2)` → `.toExponential(1)`
   - 增加网格线透明度：`opacity="0.5"`

4. **增强数据点视觉效果**
   - 增大半径：`r="4"` → `r="5"`
   - 添加透明度：`opacity="0.8"`
   - 鼠标悬停显示完整值（3位小数）

#### 修改文件
- `frontend/index.html` 第941行：移除中文
- `frontend/index.html` 第952-962行：优化Y轴标签
- `frontend/index.html` 第975-983行：移除数据点标签

---

### 问题2: 切换 Metric 后图表不更新

#### 根本原因

1. **事件监听器重复绑定**
   - 每次 `showProcessStep()` 重新渲染HTML，创建新的 `<select>` 元素
   - 使用 `addEventListener()` 会重复绑定多个监听器

2. **数据格式问题**
   - 某些 fairness metrics 返回嵌套字典（如 `{'SEX': 0.001, 'MARRIAGE': 0.002}`）
   - 需要特殊处理，计算平均值

#### 修复方案

1. **改用 `onchange` 属性**
   ```javascript
   // 之前
   metricSelector.addEventListener('change', function() {...});
   
   // 现在
   metricSelector.onchange = function() {...};
   ```
   - `onchange` 会覆盖之前的监听器，避免重复

2. **添加 `setTimeout` 确保DOM渲染完成**
   ```javascript
   setTimeout(() => {
     const metricSelector = $('#metricSelector');
     // 绑定事件...
   }, 0);
   ```

3. **处理嵌套字典格式的 metrics**
   ```javascript
   if (typeof metricValue === 'object' && metricValue !== null) {
     // 取所有数值的平均值
     const values = Object.values(metricValue).filter(v => typeof v === 'number');
     value = values.reduce((a, b) => a + b, 0) / values.length;
   } else if (typeof metricValue === 'number') {
     value = metricValue;
   }
   ```

4. **添加调试日志**
   ```javascript
   console.log(`[DEBUG] Rendering chart for metric: ${currentSelectedMetric}`);
   console.log(`[DEBUG] Metric changed to: ${state.selectedMetric}`);
   ```

#### 修改文件
- `frontend/index.html` 第1037-1058行：重构事件监听器
- `frontend/index.html` 第872-902行：处理嵌套数据格式
- `frontend/index.html` 第867行：添加调试日志

---

## 🎨 改进后的效果

### 视觉效果

**之前**:
- ❌ 数据点标签密集重叠
- ❌ Y轴标签与图表线重叠
- ❌ 包含中文标签
- ❌ 信息过载

**现在**:
- ✅ 数据点清晰，无文字重叠
- ✅ Y轴标签位置合理
- ✅ 纯英文，国际化
- ✅ 鼠标悬停显示详细信息

### 交互效果

**之前**:
- ❌ 切换 metric 后图表不变
- ❌ 无反馈信息

**现在**:
- ✅ 切换 metric 立即更新图表
- ✅ 控制台显示调试信息
- ✅ 支持嵌套字典格式的 metrics
- ✅ 自动计算平均值

---

## 📊 图表数据处理流程

### 数据获取

```javascript
// 1. 从 state.history 获取所有 iterations 的数据
state.history = [
  { iteration: 0, metrics: { Overall_Fairness: 0.001, BNC: {...}, ...} },
  { iteration: 1, metrics: { Overall_Fairness: 0.0008, BNC: {...}, ...} },
  ...
]

// 2. 根据用户选择的 metric 提取数据
currentSelectedMetric = 'Overall_Fairness' // 或 'BNC', 'EOpp', etc.

// 3. 处理不同格式的 metric 值
if (typeof metricValue === 'object') {
  // 嵌套字典 → 计算平均值
  value = avg(Object.values(metricValue))
} else {
  // 直接数值 → 直接使用
  value = metricValue
}
```

### 图表渲染

```javascript
// 1. 动态Y轴范围
yMax = max(values) + 10% padding
yMin = max(0, min(values) - 10% padding)
yRange = yMax - yMin

// 2. 缩放函数
scaleY = (value) => 200 - ((value - yMin) / yRange) * 160

// 3. 渲染数据点和线条
fairnessData.map(d => {
  x = 40 + (index / total) * 340
  y = scaleY(d.value)
  return <circle cx={x} cy={y} r={5}>
           <title>Iteration {d.iteration}: {d.value.toExponential(3)}</title>
         </circle>
})
```

---

## 🔍 测试步骤

### 1. 测试图表显示

**步骤**:
1. 刷新前端页面
2. Load Credit 数据
3. 点击 "Run All Steps"
4. 等待完成

**预期结果**:
- ✅ Y轴标签清晰，不重叠
- ✅ 数据点上没有文字
- ✅ 鼠标悬停数据点显示 tooltip
- ✅ 图表线条流畅

### 2. 测试 Metric 切换

**步骤**:
1. 完成上述运行
2. 点击下拉选择框
3. 选择 "BNC - Between Negative Classes"
4. 观察图表变化

**预期结果**:
- ✅ 图表立即更新，显示 BNC 的值
- ✅ Y轴范围自动调整
- ✅ 控制台显示：`[DEBUG] Metric changed to: BNC`
- ✅ 控制台显示：`[DEBUG] Rendering chart for metric: BNC`

### 3. 测试多次切换

**步骤**:
1. 切换到 "EOpp"
2. 切换到 "Overall_Fairness"
3. 切换到 "SP"
4. 重复多次

**预期结果**:
- ✅ 每次切换都正确更新
- ✅ 无错误信息
- ✅ 性能流畅

---

## 🐛 调试技巧

### 打开浏览器控制台

1. **Chrome/Edge**: 按 `F12` 或 `Ctrl+Shift+I`
2. **切换到 Console 标签**
3. **观察调试信息**:
   ```
   [DEBUG] Rendering chart for metric: Overall_Fairness, step: 1, realData: true
   [DEBUG] Metric changed to: BNC
   [DEBUG] Rendering chart for metric: BNC, step: 1, realData: true
   ```

### 检查数据

**在控制台输入**:
```javascript
// 查看当前选中的 metric
state.selectedMetric

// 查看 history 数据
state.history

// 查看某个 metric 的值
state.history[0].metrics.Overall_Fairness
state.history[0].metrics.BNC

// 查看所有 metrics
Object.keys(state.history[0].metrics)
```

---

## 📝 注意事项

1. **嵌套字典的 metrics**
   - 某些 metrics（如 BNC, BPC）可能返回 `{'SEX': 0.001, 'MARRIAGE': 0.002}`
   - 图表显示这些值的**平均值**
   - 如需显示每个保护属性的值，可以扩展为多条线

2. **Overall_Fairness 始终是直接数值**
   - 后端计算的综合分数
   - 不会是嵌套字典

3. **Tooltip 显示格式**
   - 使用 `toExponential(3)` 显示3位小数的科学计数法
   - 例如：`1.234e-4`

4. **Y轴自动缩放**
   - 根据实际数据范围动态调整
   - 添加10%的padding，避免数据点贴边
   - 最小值不会小于0

---

## ✅ 修复完成清单

- [x] 移除中文标签
- [x] 移除数据点上的文字（避免重叠）
- [x] 优化Y轴标签位置和大小
- [x] 添加鼠标悬停tooltip
- [x] 修复 metric 切换不生效的问题
- [x] 处理嵌套字典格式的 metrics
- [x] 添加调试日志
- [x] 优化事件监听器绑定
- [ ] 实际测试（需要用户运行）

---

**修复完成！请刷新前端页面测试！** 🎉



