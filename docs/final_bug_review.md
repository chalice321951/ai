# 多模型并行推理与 PPE 检测 - 最终 Bug 审查需求文档（v3）

> 审查时间：2026-06-25
> 审查范围：全部已修改文件（8 个）
> 审查结论：发现 1 个 HIGH + 4 个 MEDIUM + 4 个 LOW

---

## 一、修复清单

### HIGH 级别（必须修复）

#### Bug #6：camera.py 未使用 per_algo_frame_ids，导致检测结果被误丢弃

**严重程度**：HIGH | **文件**：`camera.py:1537-1555`

**问题**：

`parallel_scheduler.get_latest_result()` 返回了 `per_algo_frame_ids`，但 camera.py 从未使用它。

当前代码：
```python
latest_result = self.inference_scheduler.get_latest_result(self.stream_tracking_key)
if latest_result:
    result_fid = int(latest_result.get('frame_id', 0) or 0)  # 取所有 algo 的 max
    result_age = time.time() - result_ts
    frame_lag = max(0, fid - result_fid)
    if (result_fid > self._last_applied_result_frame_id
        and result_age <= max_result_age
        and frame_lag <= max_frame_lag):
        # 应用结果
```

`result_fid` 是所有模型的最大 frame_id。当 PPE 推理间隔=5 帧时：
- 机械检测 frame_id=105（最新）
- PPE frame_id=100（5帧前）
- result_fid = max(105, 100) = 105

这本身没问题。**但反过来**：如果机械检测结果先过期（result_fid 来自 PPE），frame_lag 会异常大，导致整个结果被误丢弃。

**修复方案**：

```python
# 改为按 algo 独立判断新鲜度
per_algo_fids = latest_result.get('per_algo_frame_ids', {})
# 用机械检测的 frame_id 做主判断
detection_fid = per_algo_fids.get('3001', result_fid)
frame_lag = max(0, fid - max(detection_fid, result_fid))
```

---

### MEDIUM 级别（建议修复）

#### Bug #1：PPEDetector cleanup 与 detect 线程竞争

**严重程度**：MEDIUM | **文件**：`inference/ppe/ppe_detector.py:56-58, 171-172, 386-388`

**问题**：

`cleanup()` 在主线程执行，`detect()` 在 PPEWorker 线程执行：
```python
def cleanup(self):
    self._alive = False
    self._attr_model = None  # 主线程置空
    ...

def detect(self, frame, stream_key, frame_count):
    if not self._alive:  # PPEWorker 线程检查
        raise RuntimeError("...")
    # ... 使用 self._attr_model ...  # 此时 _attr_model 可能已被 cleanup 置空
```

**可能症状**：cleanup 后 detect 仍执行时 `AttributeError`。

**修复方案**：

```python
def detect(self, frame, stream_key, frame_count):
    if not self._alive:
        raise RuntimeError("PPEDetector 已 cleanup，不能再使用")

    # 捕获到局部变量，避免中途被 cleanup 置空
    attr_model = self._attr_model
    person_model = self._person_model
    if person_model is None:
        return PPEResult(inference_time_ms=0, frame_id=frame_count)

    # 用局部变量 attr_model 做后续操作
    ...
```

---

#### Bug #2：_attr_cache 字典跨线程并发修改

**严重程度**：MEDIUM | **文件**：`inference/ppe/ppe_detector.py:395-398, 323, 345`

**问题**：

- `detect()` 在 PPEWorker 线程中迭代 `_attr_cache[stream_key]`
- `remove_stream()` 在主线程中 `pop` 同一个 key
- CPython GIL 保护单条字节码原子性，但迭代跨越多条字节码

**可能症状**：`RuntimeError: dictionary changed size during iteration`

**修复方案**：

```python
def _cleanup_cache(self, stream_key, frame_count, max_age=100):
    if stream_key not in self._attr_cache:
        return
    stream_cache = self._attr_cache.get(stream_key, {})
    # 先收集过期 key，再删除（避免迭代时修改）
    expired_keys = [k for k, v in list(stream_cache.items()) if frame_count - v[2] > max_age]
    for k in expired_keys:
        stream_cache.pop(k, None)
```

---

#### Bug #4：model_intervals 值转换无错误处理

**严重程度**：MEDIUM | **文件**：`inference/parallel_scheduler.py:97, 151`

**问题**：

```python
ppe_interval = int(model_intervals.get(ppe_model_id, 1))
```

如果用户配置 `"model_intervals": {"3001": "abc"}`，`int("abc")` 会抛 `ValueError`，整个 pipeline 初始化失败。

**修复方案**：

```python
def _safe_int_interval(value, default=1):
    try:
        return max(1, int(value))
    except (ValueError, TypeError):
        logging.warning(f"[ParallelScheduler] model_intervals 值无效: {value}，使用默认值 {default}")
        return default

ppe_interval = _safe_int_interval(model_intervals.get(ppe_model_id, 1))
```

---

#### Bug #7：共享模型上 remove_stream 与 detect 竞争

**严重程度**：MEDIUM | **文件**：`inference/ppe/ppe_detector.py:135-154`

**问题**：

PPEDetector 和可能的其他组件共享同一个 model 实例。`remove_stream()` 时 `_restore_tracker_state` 会 `delattr(predictor, 'trackers')`，如果此时另一个流正在使用该 model 的 tracker，会破坏其状态。

**实际影响**：在 PPE 场景下，PPEWorker 独占 shared_model，其他流用各自独立的 AlgoWorker，所以**当前架构下此竞争不太可能触发**。但设计上存在隐患。

**修复方案**：暂不修复，在文档中记录为已知限制。后续如有多 PPEDetector 实例场景，需加 per-model 锁。

---

### LOW 级别（可选修复）

#### Bug #3：alert_system.py PPE 去重 algo_id 硬编码为 'ppe'

**严重程度**：LOW | **文件**：`alert/alert_system.py:643`

**问题**：

```python
algo_id = 'ppe'  # 硬编码，与实际 model_id 不一致
```

PPE 规则的去重键中 algo_id='ppe'，但 camera.py overlay 的 algo_id='3099'。当前不影响功能（PPE 走独立的 `ppe_track_ids_key` 分支），但语义不一致。

**修复方案**：从 `detection_dict` 传递实际的 model_id，或在 PPE 规则分支中读取配置的 model_id。

---

#### Bug #5：model_id 默认值 '3099' 静默降级

**严重程度**：LOW | **文件**：`inference/parallel_scheduler.py:127`, `inference/ppe/ppe_detector.py:56`, `camera.py:1643`

**问题**：

三处都有 `get('model_id', '3099')` 的默认值。如果用户配置中缺少 `detection.model_id`，会静默使用 '3099'，可能与实际注册的模型不一致。

**修复方案**：在 `validate()` 中已检查 model_id 是否注册，此处可保持现状或加 warning。

---

#### Bug #8：_cached_stream_keys 未在 __init__ 初始化

**严重程度**：LOW | **文件**：`pipeline/algo_worker.py:69-74`, `pipeline/ppe_worker.py:56`

**问题**：

`_cached_stream_keys` 没有在 `__init__` 中初始化，依赖 `getattr` 的默认值。健康检查重启时如果没传 stream_keys，会回退到动态获取所有流。

**修复方案**：在 `__init__` 中初始化 `self._cached_stream_keys = None`。

---

#### Bug #9：非数值 track_id 去重不正确

**严重程度**：LOW | **文件**：`alert/alert_system.py:651-663`

**问题**：

如果 track_id 是无法转为 int 的字符串，`int(tid)` 抛异常后用原字符串做去重键，跨帧可能不一致。

**实际影响**：当前 ByteTrack 分配的 track_id 都是整数，此场景几乎不会触发。

**修复方案**：在 camera.py 中统一将 track_id 转为 int。

---

## 二、修复优先级

| 优先级 | Bug | 说明 | 工作量 |
|--------|-----|------|--------|
| **P0** | Bug #6 | camera.py 不用 per_algo_frame_ids，检测结果可能被误丢 | 小 |
| **P1** | Bug #1 | cleanup 与 detect 竞争 | 小 |
| **P1** | Bug #2 | _attr_cache 并发修改 | 小 |
| **P1** | Bug #4 | model_intervals int 转换无异常处理 | 小 |
| **P2** | Bug #7 | 共享模型 remove_stream 竞争 | 暂不修 |
| **P2** | Bug #3 | alert algo_id 硬编码 | 小 |
| **P2** | Bug #5 | model_id 默认值静默降级 | 小 |
| **P2** | Bug #8 | _cached_stream_keys 未初始化 | 小 |
| **P2** | Bug #9 | 非数值 track_id 去重 | 小 |

---

## 三、修复方案汇总

### Bug #6 修复方案

`camera.py` 中 `get_latest_result` 的结果处理：

```python
latest_result = self.inference_scheduler.get_latest_result(self.stream_tracking_key)
if latest_result:
    result_fid = int(latest_result.get('frame_id', 0) or 0)
    per_algo_fids = latest_result.get('per_algo_frame_ids', {})

    # 优先使用非 PPE 模型的 frame_id 判断新鲜度
    # PPE 推理慢，其 frame_id 可能较旧，不能作为丢弃整个结果的依据
    non_ppe_fids = {k: v for k, v in per_algo_fids.items()
                    if k not in self.config.ppe_config.get('detection', {}).get('model_id', '')}
    if non_ppe_fids:
        effective_fid = max(non_ppe_fids.values())
    else:
        effective_fid = result_fid

    result_ts = float(latest_result.get('result_ts', 0.0) or 0.0)
    result_age = time.time() - result_ts if result_ts > 0 else 0.0
    frame_lag = max(0, fid - effective_fid)
    # 后续判断用 effective_fid 代替 result_fid
```

### Bug #1 修复方案

`ppe_detector.py` 的 `detect()` 方法开头：

```python
def detect(self, frame, stream_key, frame_count):
    if not self._alive:
        raise RuntimeError("PPEDetector 已 cleanup，不能再使用")

    # 捕获到局部变量，防止 cleanup 中途置空
    person_model = self._person_model
    attr_model = self._attr_model
    if person_model is None:
        return PPEResult(inference_time_ms=0, frame_id=frame_count)

    # 后续代码用 person_model / attr_model 局部变量，不用 self._person_model
```

### Bug #2 修复方案

`ppe_detector.py` 的 `_cleanup_cache`：

```python
def _cleanup_cache(self, stream_key, frame_count, max_age=100):
    if stream_key not in self._attr_cache:
        return
    stream_cache = self._attr_cache.get(stream_key)
    if stream_cache is None:
        return
    # snapshot keys first, then delete (avoid iteration-during-mutation)
    expired = [k for k, v in list(stream_cache.items()) if frame_count - v[2] > max_age]
    for k in expired:
        stream_cache.pop(k, None)
```

### Bug #4 修复方案

`parallel_scheduler.py` 添加辅助函数：

```python
def _safe_int_interval(value, default=1):
    try:
        return max(1, int(value))
    except (ValueError, TypeError):
        logging.warning(f"model_intervals 值无效: {value}，使用默认值 {default}")
        return default
```

在 `__init__` 和 `_load_ppe_to_pipeline` 中使用：

```python
# __init__ 中
algo_interval = _safe_int_interval(model_intervals.get(algo_id, 1))

# _load_ppe_to_pipeline 中
ppe_interval = _safe_int_interval(model_intervals.get(ppe_model_id, 1))
```

---

## 四、已知限制（不需要修复）

| 项 | 说明 |
|----|------|
| 共享模型的 tracker 状态 | PPEWorker 独占 shared_model，当前无竞争；多 PPEDetector 场景需加锁 |
| CPython GIL 保护 | 单字节码原子性已保证，但复杂操作（迭代 + 修改）仍需手动同步 |
| cleanup 后 detect | 通过 `_alive` 标志防护，cleanup 期间正在执行的 detect 用局部变量快照 |

---

## 五、结论

**需要立即修复的只有 1 个 HIGH**：Bug #6（camera.py 不用 per_algo_frame_ids）。这个 bug 会导致 PPE 推理间隔 > 1 时，有效的机械检测结果被误判为"过期"而丢弃，表现为间歇性漏检。

**4 个 MEDIUM 级别问题**都是线程安全相关的边界条件，实际触发概率低，但修复工作量也很小（每个改 3-5 行代码）。

**4 个 LOW 级别问题**属于代码健壮性和语义一致性问题，可后续版本处理。

---

**建议修复顺序**：
1. Bug #6（HIGH）→ 必须修
2. Bug #4（MEDIUM）→ 建议修（配置校验防崩溃）
3. Bug #1 + Bug #2（MEDIUM）→ 建议修（线程安全）
4. 其余 LOW → 可以排到下个版本
