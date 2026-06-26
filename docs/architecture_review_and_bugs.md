# 多模型并行推理与 PPE 检测 - 架构与 Bug 审查需求文档（v2）

> 审查日期：2026-06-25
> 文档版本：v2（根据用户反馈修订）
> 审查依据：`docs/final_implementation_summary.md` + 实际代码

---

## 一、用户确认的设计意图（非 bug）

以下 3 项是**有意的设计**，不需要修复：

### 1. PPE model_id="3099" 当前未在 model_mappings 中注册
- **用户说明**：3099 是后续会配置的，当前空缺是临时状态
- **结论**：保留现有 `_load_ppe_to_pipeline` 的"路径为空就跳过 + 打 WARNING"逻辑，无需修改

### 2. multi 违规只触发 ppe_multi 告警，不触发 helmet/vest
- **用户说明**：同时未戴安全帽和未穿反光衣的人，业务期望只报一条 multi 告警，helmet 和 vest 单项告警不应该响
- **结论**：当前 `get_violation_type()` 的互斥分类逻辑就是对的，保留

### 3. PPE 告警等级使用与机械检测一致的空间分级
- **用户说明**：PPE 工人违规也要根据所在区域（红/黄/橙边界）决定告警等级，与机械检测完全一致；边界外的工人不告警是预期行为
- **结论**：`_resolve_overlay_alarm_level` 不需要为 PPE 加特殊分支

---

## 一.5 用户提出的架构改进：统一用 model_id 替代 "ppe" 算法 ID

### 改进背景

当前代码中 PPE 模块的算法 ID 是 `"ppe"` 字符串（硬编码），而其他检测模型用的是数字 model_id（如 `"3001"`、`"3099"`）。这造成两套命名体系并存：

| 维度 | 当前状态 | 问题 |
|------|---------|------|
| 配置文件 | `model_intervals["3099"]=5` | 用 model_id |
| PPEWorker | `algo_id="ppe"` | 用 "ppe" 字符串 |
| ResultStore | 结果存在 `"ppe"` key 下 | 用 "ppe" 字符串 |
| camera.py | `if aid == 'ppe': ppe_result = res` | 用 "ppe" 字符串 |
| 告警 overlay | `algo_id="ppe"` | 用 "ppe" 字符串 |
| 告警去重键 | `(track_id, "ppe", class_name)` | 用 "ppe" 字符串 |

**用户决定**：不再用 `"ppe"`，全部统一为 `model_id`（如 `"3099"`）。

### 改造范围

#### 1. `inference/parallel_scheduler.py`
```python
# 改造前：
self._pipeline.add_ppe_model(algo_id="ppe", ...)

# 改造后：
self._pipeline.add_ppe_model(algo_id=ppe_model_id, ...)
# 同时传入 inference_interval（从 model_intervals[ppe_model_id] 读取）
```

#### 2. `pipeline/ppe_worker.py`
```python
# 改造前：
self.algo_id = algo_id  # = "ppe"
self._inference_interval = max(1, int(model_intervals.get('ppe', 1)))

# 改造后：
self.algo_id = algo_id  # = "3099" 或用户配置的 PPE model_id
# inference_interval 由构造函数参数传入（来自 parallel_scheduler 读 model_intervals[model_id]）
self._inference_interval = max(1, int(inference_interval))
```

#### 3. `inference/ppe/ppe_result_types.py`
```python
# 改造前：
def to_overlay(self, algo_id: str = "ppe", color=...):
    return {"algo_id": algo_id, ...}

def get_violation_overlays(self, algo_id: str = "ppe", color=...):
    ...

# 改造后：
def to_overlay(self, algo_id: str, color=...):  # 无默认值，必须传入
    return {"algo_id": algo_id, ...}

def get_violation_overlays(self, algo_id: str, color=...):  # 无默认值
    ...
```

#### 4. `inference/ppe/ppe_detector.py`
```python
# 改造前：
def get_violation_overlays(self, ppe_result, algo_id="ppe"):
    return ppe_result.get_violation_overlays(algo_id=algo_id, ...)

# 改造后：增加 self._algo_id 字段（从 parallel_scheduler 传入）
def __init__(self, config, model_path='', device='cpu', shared_model=None, algo_id=None):
    ...
    self._algo_id = algo_id or config.get('detection', {}).get('model_id', '3099')

def get_violation_overlays(self, ppe_result, algo_id=None):
    return ppe_result.get_violation_overlays(
        algo_id=algo_id or self._algo_id, ...
    )
```

#### 5. `camera.py:_process_infer_results`
```python
# 改造前：用字符串匹配识别 PPE 结果
for aid, res in results.items():
    if aid == 'ppe':
        ppe_result = res
        continue
    ...

# 改造后：用 PPEResult 类型判断（更稳健）
from inference.ppe import PPEResult

for aid, res in results.items():
    if isinstance(res, PPEResult):
        ppe_result = res
        continue
    # 普通模型结果处理 ...
```

#### 6. `camera.py:_convert_ppe_to_overlays`（PPE overlay 生成处）
```python
# 改造前：
ppe_overlays = ppe_result.get_violation_overlays(algo_id="ppe", color=...)

# 改造后：从配置或上下文获取 PPE 的 model_id
ppe_model_id = self.config.ppe_config.get('detection', {}).get('model_id', '3099')
ppe_overlays = ppe_result.get_violation_overlays(algo_id=ppe_model_id, color=...)
```

### 保持不变的部分

以下名称是**业务语义**，不是算法 ID，保持原样：

| 名称 | 用途 | 是否改 |
|------|------|--------|
| `ppe_helmet_violation` | 告警规则 ID | ❌ 不改 |
| `ppe_vest_violation` | 告警规则 ID | ❌ 不改 |
| `ppe_multi_violation` | 告警规则 ID | ❌ 不改 |
| `ppe_helmet` | overlay 的 class_name | ❌ 不改 |
| `ppe_vest` | overlay 的 class_name | ❌ 不改 |
| `ppe_multi` | overlay 的 class_name | ❌ 不改 |
| `ppe.enabled` | 配置字段 | ❌ 不改 |
| `ppe.detection.model_id` | 配置字段 | ❌ 不改 |

**理由**：违规类型（helmet/vest/multi）是业务语义，需要在告警平台上可读。算法 ID 只是技术 ID，统一为 model_id 即可。

### 改造后的执行流程

```
配置：
  ppe.detection.model_id = "3099"
  model_intervals["3099"] = 5

启动：
1. InferenceEngine 加载 3001、3099
2. pipeline 给 3001 创建 AlgoWorker, interval=3
3. pipeline 给 3099 创建 AlgoWorker, interval=5  # 后面会被移除
4. _load_ppe_to_pipeline:
   - ppe_model_id = "3099"
   - shared_model = engine._models["3099"]
   - pipeline.remove_model("3099")  # 移除原 AlgoWorker
   - ppe_interval = model_intervals["3099"] = 5  # ✅ 读到配置
   - 创建 PPEDetector(algo_id="3099", shared_model=...)
   - pipeline.add_ppe_model(algo_id="3099", ppe_detector=..., inference_interval=5)
   - PPEWorker 内 algo_id="3099", interval=5  # ✅ 配置生效

运行：
1. PPEWorker 每 5 帧推理一次
2. 结果存入 ResultStore["3099"]
3. camera.py 用 isinstance(res, PPEResult) 识别 PPE 结果
4. PPE overlay 的 algo_id="3099"
5. 告警去重键：(track_id, "3099", "ppe_helmet")
```

### 改造收益

| 收益 | 说明 |
|------|------|
| 配置自动生效 | `model_intervals["3099"]=5` 直接生效 |
| 命名统一 | 全代码用 model_id，无双轨制 |
| 类型识别更稳健 | `isinstance(res, PPEResult)` 比字符串匹配可靠 |
| 配置文件不需要改 | 用户当前的 config.json 不需要新增任何字段 |

---

## 二、确认需要修复的 Bug 清单

### CRITICAL 级别

#### C1：tracker 隔离方案核心实现错误 🔴

**问题位置**：
- `pipeline/algo_worker.py:107-123` (`_restore_tracker_state`)
- `inference/ppe/ppe_detector.py:129-145` (`_restore_tracker_state`)

**问题描述**：

当前的"save/restore tracker"方案核心代码：
```python
def _restore_tracker_state(self, stream_key):
    cached = self._stream_trackers.get(stream_key)
    if cached is None:
        # 首次访问该流：将 predictor.trackers 设为空列表
        predictor = getattr(self.model, 'predictor', None)
        if predictor is not None:
            predictor.trackers = []  # ❌ 错误
        return
```

**根本原因**：

ultralytics 的 `on_predict_start` 回调（`ultralytics/trackers/track.py`）：
```python
if hasattr(predictor, "trackers") and persist:
    return  # 已有 trackers 且 persist=True，不重新创建
```

执行流程：
1. 流 A 首次 `model.track()` → trackers 不存在 → 创建新 trackers → save 缓存
2. 流 B 首次进来 → `_restore_tracker_state` 设 `predictor.trackers = []`
3. 流 B 调用 `model.track(persist=True)` → `hasattr(predictor, 'trackers')` 为 True → **直接 return，不重建**
4. 推理后 `on_predict_postprocess_end` 访问 `predictor.trackers[0]` → **IndexError**
5. 异常被 worker 的 `except Exception` 吞掉，流 B 推理失败

**实际症状**：
- 第一个流可以工作
- 第二个流及之后所有流的推理都被 IndexError 吞掉
- 日志显示"推理失败"，但没有 stack trace（被 `logging.error` 简化）
- **11 路摄像头只能跑通第一路**

**修复方案**：

把 `predictor.trackers = []` 改成 `delattr(predictor, 'trackers')`，让 ultralytics 重新创建：

```python
def _restore_tracker_state(self, stream_key):
    cached = self._stream_trackers.get(stream_key)
    if cached is None:
        predictor = getattr(self.model, 'predictor', None)
        if predictor is not None and hasattr(predictor, 'trackers'):
            try:
                delattr(predictor, 'trackers')
            except AttributeError:
                pass
        return
    predictor = getattr(self.model, 'predictor', None)
    if predictor is not None:
        predictor.trackers = cached
```

---

### HIGH 级别

#### H1：PPE 跳帧配置失效 — 默认每帧推理 🟡

**问题位置**：`pipeline/ppe_worker.py:55-60`

**问题描述**：

用户在 `config.json` 这样配（和其他模型保持一致的语义，用 model_id 做 key）：
```json
"model_intervals": {
    "3001": 3,    // 机械检测每 3 帧推理一次
    "3099": 5     // PPE 人体检测每 5 帧推理一次
}
```

**用户预期**：PPE 每 5 帧推理一次。

**实际执行流程**：
```
1. InferenceEngine 加载 3001、3099 两个模型
2. pipeline 给 3001 创建 AlgoWorker → 读 model_intervals["3001"]=3 ✅
3. pipeline 给 3099 创建 AlgoWorker → 读 model_intervals["3099"]=5 ✅
4. 发现 ppe.enabled=true，进入 _load_ppe_to_pipeline
5. shared_model = self._engine._models["3099"]   # 复用 3099 模型
6. self._pipeline.remove_model("3099")           # ❌ 删除 3099 的 AlgoWorker
7. 创建 PPEDetector(shared_model=shared_model)
8. add_ppe_model(algo_id="ppe", ...)             # PPEWorker 算法 ID = "ppe"
9. PPEWorker 读取 model_intervals.get("ppe", 1) → 1（默认值）❌
```

**根本原因**：
- 配置文件用 `model_id`（"3099"）做 key
- PPEWorker 内部用算法 ID（"ppe"）做 key
- 两个 key 不同，配置读不到

**实际症状**：
- 11 路流 × 25 FPS × 每帧 PPE 推理 = GPU 直接打满
- 用户配置的 `model_intervals["3099"]=5` 完全无效
- PPE 默认每帧推理一次

**修复方案**：

**与"一.5 统一用 model_id 替代 ppe"的改造一并完成**。统一 algo_id 后：

1. `PPEWorker` 的 `algo_id` 变为 `"3099"`
2. `parallel_scheduler._load_ppe_to_pipeline` 读 `model_intervals["3099"]=5`，作为 `inference_interval` 传给 PPEWorker
3. PPEWorker 直接使用传入的 interval，不再自己读 `model_intervals['ppe']`

```python
# inference/parallel_scheduler.py:_load_ppe_to_pipeline
ppe_model_id = ppe_detection.get('model_id', '3099')
model_intervals = getattr(self.config, 'model_intervals', {}) or {}
ppe_interval = int(model_intervals.get(ppe_model_id, 1))  # 读 model_intervals["3099"]

self._pipeline.add_ppe_model(
    algo_id=ppe_model_id,                   # ← 用 "3099"，不用 "ppe"
    ppe_detector=ppe_detector,
    inference_interval=ppe_interval,        # ← 新增
)
```

```python
# pipeline/ppe_worker.py:__init__
def __init__(self, algo_id, ppe_detector, ..., inference_interval=1):
    self.algo_id = algo_id                  # = "3099"
    self._inference_interval = max(1, int(inference_interval))  # 直接用参数
```

**修复后的效果**：
- 用户配置 `model_intervals["3099"]=5` 自动对 PPE 生效
- 不需要在配置文件中加 `"ppe": 5`
- 配置语义统一（所有模型用 model_id）

---

#### H2：track_id == -1 缓存串扰 🟡

**问题位置**：
- `inference/ppe/ppe_detector.py:194-199` (track_id 读取)
- `inference/ppe/ppe_detector.py:_classify_with_cache` (缓存查询)

**问题描述**：

ByteTrack 没分配 track_id 时：
```python
if track_ids_tensor is not None and i < len(track_ids_tensor):
    track_id = int(track_ids_tensor[i].item())
elif box.id is not None:
    track_id = int(box.id[0])
else:
    track_id = -1  # ⚠️ 所有未跟踪的人共享 -1
```

`_classify_with_cache` 用 track_id 做 key：
```python
if track_id in stream_cache:
    cached = stream_cache[track_id]
    # 所有 track_id=-1 的人共用同一个缓存项
```

**实际症状**：
- 同一帧多个未被跟踪的人（新进入画面、ByteTrack 还没确认）
- 他们的 helmet/vest 状态互相覆盖
- 第一个人的属性结果被第二个人覆盖

**修复方案**：

```python
def _classify_with_cache(self, frame, crop_box, track_id, stream_key, frame_count):
    # track_id=-1 时跳过缓存，每次都做属性分类
    if track_id < 0:
        return self._do_classification(frame, crop_box)
    # 正常 track_id 走缓存
    ...
```

同时在告警去重时也处理：

```python
# alert_system.py:651-658
for tid in list(rule_track_ids or []):
    if tid in (None, '') or (isinstance(tid, int) and tid < 0):
        # 未跟踪的目标不参与组合键去重，但要让它告警
        continue
```

---

### MEDIUM 级别

#### M2：多 algo frame_id 不对齐导致画面错位 🟠

**问题位置**：`inference/parallel_scheduler.py:get_latest_result` + `camera.py:1538`

**问题描述**：

PPE 推理慢（两阶段 + 多人），单帧耗时 200ms+。
机械检测推理快，单帧 30ms。

结果到达 ResultStore：
- 机械检测 frame_id=105 先到
- PPE frame_id=100 后到

`get_latest_result` 返回：
```python
{
    "frame_id": max(105, 100) = 105,  # 误以为 PPE 也是 105 的结果
    "results": {"3001": ..., "ppe": ...}
}
```

camera.py 按 fid=105 找参考帧画 overlay，但 PPE 的检测框对应的是 5 帧前的画面 → **PPE 检测框画错位置**

**修复方案**：

让 `get_latest_result` 返回每个 algo 各自的 frame_id：
```python
{
    "results": {
        "3001": {"data": ..., "frame_id": 105, "ts": ...},
        "ppe":  {"data": ..., "frame_id": 100, "ts": ...},
    }
}
```

camera.py 按每个 algo 自己的 frame_id 找参考帧。

---

#### M3：PPE overlay 的 confidence 字段语义混淆 🟠

**问题位置**：`inference/ppe/ppe_result_types.py:88`、`camera.py:1770`

**问题描述**：

```python
return {
    "xyxy": self.det_box,
    "text": text,
    "confidence": self.person_conf,  # ❌ 是人体检测置信度
    "class_name": violation_type,
    ...
}
```

用户看到 `"ppe_helmet 0.95"`，0.95 实际是**人体检测置信度**，不是"未戴安全帽的确信度"。

**业务后果**：
- 上报到平台后，0.95 被理解为"95% 确信未戴安全帽"
- 实际可能是 0.95 检测到人体，但属性分类只有 0.4（不确定）

**修复方案**：

```python
if violation_type == "ppe_helmet":
    confidence = 1 - self.helmet_prob  # 越接近 1 越确信没戴
elif violation_type == "ppe_vest":
    confidence = 1 - self.vest_prob
else:  # ppe_multi
    confidence = ((1 - self.helmet_prob) + (1 - self.vest_prob)) / 2
```

---

### LOW 级别

#### L1：stream 移除时残留缓存（内存泄漏）⚪

**问题位置**：`pipeline/multi_model_pipeline.py:333-341` 的 `remove_stream`

**问题描述**：

```python
def remove_stream(self, stream_key):
    self._frame_hub.remove_stream(stream_key)
    self._result_store.remove_stream(stream_key)
    # ❌ 未清理 worker 内部缓存
```

未清理：
- `AlgoWorker._stream_trackers[stream_key]`
- `AlgoWorker._stream_frame_counters[stream_key]`
- `PPEDetector._stream_trackers[stream_key]`
- `PPEDetector._attr_cache[stream_key]`
- `PPEWorker._stream_frame_counts[stream_key]`

**修复方案**：

每个 worker 加 `remove_stream` 方法，pipeline.remove_stream 时调用所有 worker 的对应方法。

---

#### L2：健康检查重启 Worker 时丢失 stream_keys ⚪

**问题位置**：`pipeline/multi_model_pipeline.py:210`

**问题描述**：

```python
def health_check(self, auto_restart=True):
    if not is_alive and self._started:
        worker.start()  # ❌ 没传 stream_keys
```

当前 `pipeline.start()` 不传 stream_keys（默认处理所有流），影响不暴露。但未来用子集启动时会出问题。

**修复方案**：

worker 自己缓存第一次启动的 stream_keys，重启时复用。

---

#### L3：cleanup 后 detect 不报错，PPE 静默失效 ⚪

**问题位置**：`inference/ppe/ppe_detector.py:359-365`

**问题描述**：

cleanup 后再调用 `detect()`：
- `_person_model` 还在（共享），人体检测继续工作
- `_attr_model` 为 None
- `_classify_with_cache` 进入兜底分支，返回 `(0.5, 0.5)`
- 所有人 state="unknown"，不告警
- **PPE 静默失效**，没有任何日志提示

**修复方案**：

加 `_alive` 标志位：
```python
def cleanup(self):
    self._alive = False
    ...

def detect(self, frame, stream_key, frame_count):
    if not getattr(self, '_alive', True):
        raise RuntimeError("PPEDetector 已 cleanup，不能再使用")
```

---

## 三、文档与代码不一致

### 单元测试覆盖率不足

`tests/test_multi_model_ppe.py` 当前测试都是不依赖 ultralytics 的：
- 配置加载、FrameHub、ResultStore、PPE 结果类型、PPE 渲染器

**缺失的关键测试**：
- save/restore tracker 跨流隔离（需要 ultralytics）
- PPE shared_model 不重复加载
- PPE 告警规则触发
- frame_id 多 algo 对齐

**实际影响**：在 C1 修复前，文档声称的"6 测试通过"无法证明 tracker 隔离真的工作。

---

## 四、修复优先级

| 优先级 | Bug | 修复工作量 | 预期效果 |
|--------|-----|----------|---------|
| **P0** | C1：tracker 空列表 | 改 4 行（algo_worker + ppe_detector） | 流隔离真正生效 |
| **P1** | H1：PPE interval 失效 | 改 PPEWorker + scheduler 传参 | GPU 占用降低 |
| **P1** | H2：track_id=-1 串扰 | 加缓存跳过逻辑 + 告警去重跳过 | 多人场景准确 |
| **P2** | M2：frame_id 不对齐 | 改 result store 返回结构 | 画面对齐 |
| **P2** | M3：confidence 语义 | 改 overlay 字段 | 上报数据正确 |
| **P3** | L1-L3 | 各自处理 | 健壮性提升 |

---

## 五、必须先验证的事项

修复 C1 之后必须实测：

1. **环境检查**：`pip show ultralytics`，记录版本号
2. **流隔离测试**：用两路 mock 流（同一模型）实际跑 100 帧，记录每路流的 track_id
3. **PPE 端到端**：开启 PPE，模拟"未戴安全帽"+"未穿反光衣"+"全不戴"三种场景，验证各自规则触发
4. **frame_id 对齐**：模拟 PPE 慢推理，验证检测框位置是否正确

---

## 六、修复路线图

### 阶段一：P0 修复（必做）

修复 C1：
- 修改 `pipeline/algo_worker.py:_restore_tracker_state`
- 修改 `inference/ppe/ppe_detector.py:_restore_tracker_state`
- 把 `predictor.trackers = []` 改为 `delattr(predictor, 'trackers')`
- 不修复就只有第一路流能工作

### 阶段二：P1 修复（强烈建议）

修复 H1：
- 修改 `inference/parallel_scheduler.py:_load_ppe_to_pipeline`，读取 PPE model_id 对应的 interval
- 修改 `pipeline/ppe_worker.py` 接收 inference_interval 参数

修复 H2：
- 修改 `inference/ppe/ppe_detector.py:_classify_with_cache`，track_id<0 时跳过缓存
- 修改 `alert/alert_system.py:process_frame_alerts`，track_id<0 时不进入组合键去重

### 阶段三：P2 修复（建议）

修复 M2：
- 修改 `pipeline/result_store.py` 和 `inference/parallel_scheduler.py`，结果按 algo 独立返回 frame_id
- 修改 `camera.py:_process_infer_results`，按 algo 独立找参考帧

修复 M3：
- 修改 `inference/ppe/ppe_result_types.py:to_overlay`，confidence 字段用属性概率

### 阶段四：P3 修复（可选）

修复 L1-L3：
- 加 worker.remove_stream() 方法
- worker 缓存 stream_keys
- PPEDetector 加 _alive 标志

---

## 七、结论

### 用户澄清后的真实状态

| 之前认为是 bug | 实际情况 |
|---------------|---------|
| C2：PPE 配置错误 | ❌ 不是 bug，3099 后续会配 |
| H3：multi 吞 helmet/vest | ❌ 不是 bug，业务期望如此 |
| M1：空间分级吞 PPE 告警 | ❌ 不是 bug，PPE 就要走空间分级 |

### 真正必须修复的问题

1. **C1**：tracker 隔离实现错误（不修就只有 1 路流能用）
2. **H1**：PPE 默认每帧推理（GPU 资源浪费）
3. **H2**：track_id=-1 缓存串扰（多人场景属性错乱）
4. **M2**：frame_id 不对齐（PPE 慢时画面错位）
5. **M3**：confidence 语义混淆（上报数据有歧义）

**核心问题只有 1 个：C1。** 不修 C1，所有部署测试都没意义。
