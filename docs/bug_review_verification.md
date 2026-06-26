# final_bug_review.md 的 Bug 准确性分析

> 分析时间：2026-06-25
> 分析方法：逐个核对代码，验证文档中的 9 个 bug 是否真实存在

---

## 总结

| Bug | 文档严重度 | 实际情况 | 修正后严重度 |
|-----|----------|---------|------------|
| #6 | HIGH | ⚠️ 误判（场景描述错误） | LOW（语义不严谨，无实际影响）|
| #1 | MEDIUM | ✅ 真实存在 | MEDIUM |
| #2 | MEDIUM | ⚠️ 部分准确 | LOW（实际是 _cleanup_cache 的问题）|
| #4 | MEDIUM | ✅ 真实存在 | MEDIUM |
| #7 | MEDIUM | ✅ 真实存在 | LOW（架构上 PPE 独占）|
| #3 | LOW | ✅ 真实存在 | LOW |
| #5 | LOW | ✅ 真实存在 | LOW |
| #8 | LOW | ✅ 真实存在 | LOW |
| #9 | LOW | ✅ 真实存在 | LOW |

**结论**：9 个 bug 中 7 个真实存在但严重度被夸大，Bug #6 是误判，Bug #2 描述不准确。

---

## 逐个核对

### Bug #6 - HIGH ⚠️ **误判**

**文档断言**：camera.py 未用 per_algo_frame_ids，导致检测结果被误丢弃。

**核对代码**（camera.py:1537-1556）：
```python
result_fid = int(latest_result.get('frame_id', 0) or 0)
frame_lag = max(0, fid - result_fid)
if (result_fid > self._last_applied_result_frame_id
    and result_age <= max_result_age
    and frame_lag <= max_frame_lag):
    # 应用结果
```

**实际分析**：

文档说"如果机械检测结果先过期（result_fid 来自 PPE），frame_lag 会异常大"。

这个推理**不对**：
1. `result_fid = max(各algo的frame_id)` 是取**最新**的
2. `frame_lag = fid - result_fid` 用最新 frame_id 计算，**lag 最小**
3. 不存在"误判过期"的场景

**真实场景**：
- 机械检测 frame_id=105，PPE frame_id=100，`result_fid = max(105,100) = 105`
- `frame_lag = fid - 105`，是最小可能的 lag
- 检测结果**不会**被误判过期

**实际问题**（被文档忽略了）：

`result_fid > self._last_applied_result_frame_id` 这个条件下，PPE 的旧结果会随机械检测的新结果"搭便车"被应用。但 PPE 结果在 _process_infer_results 中是按"最新"应用的，会用 5 帧前的 PPE 检测框画在当前帧上，**画面错位**。

**修正后的严重度**：LOW（画面错位）+ 文档说的"漏检"是错的。

**修正后的修复方案**：
- 不应该改 `frame_lag` 计算
- 应该改 _process_infer_results 中 PPE overlay 绘制时按 PPE 自己的 frame_id 找参考帧

---

### Bug #1 - MEDIUM ✅ **真实存在**

**核对代码**（ppe_detector.py:171-177）：
```python
if not self._alive:
    raise RuntimeError(...)

start_time = time.time()
if self._person_model is None:
    return PPEResult(...)
# ... 后续使用 self._person_model, self._attr_model
```

**确认**：cleanup 在主线程置 `_alive=False` 和 `_attr_model=None`，detect 在 PPEWorker 线程读取，确实存在竞争。

但实际触发概率极低：
- cleanup 一般在停止管线时调用
- 此时 PPEWorker 也会被 stop()
- 同时调用的窗口很短

**修正后的严重度**：LOW（罕见竞争）。

---

### Bug #2 - MEDIUM ⚠️ **描述不准确**

**文档断言**：detect 迭代 `_attr_cache[stream_key]`，remove_stream 同时 pop 同一个 key。

**核对代码**（ppe_detector.py:353-363）：
```python
def _cleanup_cache(self, stream_key, frame_count, max_age=100):
    if stream_key not in self._attr_cache:
        return
    stream_cache = self._attr_cache[stream_key]
    expired_keys = [k for k, v in stream_cache.items() if ...]  # 迭代
    for k in expired_keys:
        del stream_cache[k]
```

**实际情况**：
- `_cleanup_cache` 自己确实有"迭代时收集 + 之后删除"的两步操作，本身就是安全的
- 但**真正的并发问题**是：`_cleanup_cache` 迭代 `stream_cache.items()` 时，`remove_stream` 在主线程 `pop(stream_key)`，把整个 `stream_cache` dict 都拿走了
- 这样 `_cleanup_cache` 还在迭代旧的 dict，没问题（Python 引用计数）
- 但 `_classify_with_cache` 里 `stream_cache[track_id] = ...` 写入的可能是已经被弹出的孤儿 dict

**修正后的严重度**：LOW（数据丢失但不崩溃）。

---

### Bug #4 - MEDIUM ✅ **真实存在**

**核对代码**（parallel_scheduler.py:97）：
```python
inference_interval = int(model_intervals.get(algo_id, 1))
```

**确认**：如果用户配置 `"3001": "abc"`，会直接抛 ValueError，整个 pipeline 初始化失败。

**修正后的严重度**：MEDIUM（用户输入错误导致服务挂掉）。

---

### Bug #7 - MEDIUM ✅ **真实存在但不暴露**

**文档承认**："当前架构下此竞争不太可能触发"。

**确认**：PPE 占用 shared_model 后，原 AlgoWorker 已被移除，没有其他 worker 用同一个 model 实例。文档自己也建议"暂不修复"。

**修正后的严重度**：LOW（架构上已规避）。

---

### Bug #3 - LOW ✅ **真实存在**

**核对代码**（alert_system.py:643）：
```python
algo_id = 'ppe'  # 硬编码
```

**确认**：与 camera.py 中 overlay 的 `algo_id='3099'` 不一致。但因为 PPE 规则走的是 `ppe_track_ids_key` 分支，组合键内部一致即可，不会跨分支匹配。

**真实影响**：仅语义不一致，不影响功能。

---

### Bug #5 - LOW ✅ **真实存在**

**核对代码**：parallel_scheduler.py:127、ppe_detector.py:56、camera.py:1643

```python
ppe_model_id = ppe_config.get('detection', {}).get('model_id', '3099')
```

**确认**：三处都硬编码 `'3099'` 默认值。如果用户配置缺失，会静默用 3099。

**真实影响**：`validate()` 已经检查过 model_id 是否注册，所以基本被前置检查兜底。

---

### Bug #8 - LOW ✅ **真实存在**

**核对代码**（algo_worker.py:69-77）：
```python
self._stop_event = threading.Event()
self._thread: Optional[threading.Thread] = None
self._wake_event = threading.Event()
# 缺：self._cached_stream_keys = None
```

**确认**：依赖 `getattr(self, '_cached_stream_keys', None)`。

**真实影响**：
- 首次 start 传 `stream_keys=None` → cache 未设置
- 健康检查 restart 时 `actual_stream_keys = None or None = None`
- worker 处理所有流（默认行为）

这正是当前用例的预期行为，所以**实际无影响**。但代码风格不严谨。

---

### Bug #9 - LOW ✅ **真实存在**

**核对代码**（alert_system.py:651-663）：
```python
for tid in list(rule_track_ids or []):
    if tid in (None, ''):
        continue
    try:
        tid_int = int(tid)
        ...
    except Exception:
        track_key = (tid, algo_id, class_name)  # 用原值做 key
```

**确认**：非数值 track_id 会用字符串做 key，跨帧可能不一致。

**真实影响**：ByteTrack 永远给整数，**不会触发**。

---

## 修正后的真实问题清单

按真实严重度排序：

| 优先级 | Bug | 真实严重度 | 实际影响 |
|--------|-----|----------|---------|
| **P1** | Bug #4 | MEDIUM | 用户配置错误时整个服务挂掉 |
| **P2** | Bug #6（修正版） | LOW | PPE 慢时 overlay 画面错位（不是漏检） |
| **P2** | Bug #1 | LOW | cleanup 与 detect 罕见竞争 |
| **P3** | Bug #2 | LOW | 并发数据丢失（不崩溃） |
| **P3** | Bug #3, #5, #8, #9 | LOW | 语义不严谨/防御性代码缺失 |
| **N/A** | Bug #7 | LOW | 当前架构已规避 |

---

## 关键修正

### 关于 Bug #6 的真正问题

文档说的"机械检测被误丢弃导致漏检"是**错的**。`result_fid = max()` 取的是最新，frame_lag 是最小，不会误判过期。

**真正的问题是画面错位**：

```
fid=105 当前帧
机械检测在 fid=105 推理完，frame_id=105
PPE 在 fid=100 推理完，frame_id=100（间隔 5 帧）

ResultStore:
  3001 → frame_id=105
  3099 → frame_id=100

get_latest_result 返回：
  frame_id = max(105, 100) = 105
  results = {3001: ..., 3099: ...}

camera.py:
  result_fid = 105
  调用 _find_frame_in_buffer(105) 找参考帧
  画机械检测框（对应 105 帧）✓ 正确
  画 PPE 检测框（对应 100 帧）❌ 错位 5 帧
```

**修复方向**应该是：PPE overlay 绘制时按 PPE 自己的 frame_id 找参考帧，不是改 frame_lag 判断。

### 文档需要修正

1. Bug #6 的严重度从 HIGH 降到 LOW
2. Bug #6 的"漏检"描述改为"画面错位"
3. Bug #6 的修复方向重写
4. Bug #1、#2 的严重度从 MEDIUM 降到 LOW

---

## 真正应该优先修的

1. **Bug #4**（MEDIUM）：配置错误导致服务挂掉，建议加 `_safe_int_interval` 防御
2. **Bug #6 修正版**（LOW）：PPE 画面错位，影响视觉体验
3. 其他都是 LOW，不影响实际功能

**建议**：文档夸大了严重度，9 个 bug 中只有 1 个真正值得立即修（Bug #4），其他可以排到下个迭代。
