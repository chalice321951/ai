# AI Camera

多路 RTMP/RTSP 视频流 AI 检测与推流项目。

当前项目的主入口是 [camera.py](/d:/AI_code/ai_camera/camera.py)，核心能力包括：

- 多路输入流并发拉流
- 统一调度的共享模型推理
- 每路独立的轻量跟踪与画框
- AI 结果视频推流
- 告警截图、告警视频、MinIO 上传、平台上报

## 1. 项目主要逻辑

### 1.1 整体处理链路

当前实现的主链路可以理解成：

`多路采集 -> 各路最新帧缓存 -> 统一推理调度器 -> 推理结果回写各路 -> 各路 tracker 连续跟踪 -> 各路独立渲染/推流 -> 告警处理`

其中：

- 拉流是每一路独立运行的
- 推理是多路共享模型、统一调度的
- 跟踪和画框是每一路各自处理的
- 推流和告警也是每一路独立处理的

### 1.2 运行入口

入口文件：

- [camera.py](/d:/AI_code/ai_camera/camera.py)

程序启动后主要做这些事：

1. 初始化日志、崩溃跟踪、配置
2. 加载模型和统一推理调度器
3. 为每一路流创建一个 `StreamProcessor`
4. 每路启动拉流线程、处理线程、推流线程
5. 主循环定期打印活跃流状态
6. 收到退出信号后停止所有流和子进程

### 1.3 多路流是怎么处理的

每一路流对应一个 `StreamProcessor`，但不是每一路都各自持有一份 YOLO 模型。

当前方案是：

- 每路只负责取最新帧
- 最新帧提交给统一推理调度器
- 调度器把不同流的帧攒成一个小批次
- 共享模型一次性做批量推理
- 再把结果按 `stream_key + frame_id` 回写给对应流

对应模块：

- [camera.py](/d:/AI_code/ai_camera/camera.py)
- [inference/unified_scheduler.py](/d:/AI_code/ai_camera/inference/unified_scheduler.py)
- [inference/inference_engine.py](/d:/AI_code/ai_camera/inference/inference_engine.py)

这样做的目的：

- 避免 10 路/11 路流时每路都起一份模型实例
- 减少显存浪费和 GPU 上下文切换
- 让多路流更适合动态批处理

### 1.4 拉流是怎么做的

拉流由子进程负责，核心模块：

- [stream/capture_process.py](/d:/AI_code/ai_camera/stream/capture_process.py)

基本逻辑：

1. 每一路启动一个 FFmpeg 拉流子进程
2. FFmpeg 把输入流解码成原始 BGR 帧
3. 通过共享内存把最新帧传给主进程
4. 主进程只保留“最新帧”，不堆积整条帧队列

这样做的目的：

- 降低 Python 主线程被拉流阻塞的风险
- 防止积压过多旧帧
- 更适合实时视频场景

### 1.5 推理是怎么做的

推理由统一调度器和推理引擎共同完成：

- [inference/unified_scheduler.py](/d:/AI_code/ai_camera/inference/unified_scheduler.py)
- [inference/inference_engine.py](/d:/AI_code/ai_camera/inference/inference_engine.py)

基本逻辑：

1. 每一路按自己的推理间隔提交最新帧
2. 调度器收集多个流的待推理帧
3. 调度器按 `inference_batch_size` 做微批处理
4. 共享 YOLO 模型一次推理多张图
5. 结果按输入顺序回写给对应流

当前推理不是“每帧必推”，而是“按间隔推理 + 中间帧跟踪”。

### 1.6 跟踪和画框是怎么做的

当前每路都使用一个轻量跟踪器：

- [tracking/simple_tracker.py](/d:/AI_code/ai_camera/tracking/simple_tracker.py)

当前跟踪策略是：

- 检测帧：用检测结果更新 tracker
- 非检测帧：tracker 按上一帧状态继续跟踪
- 如果两帧间隔太大，跳过光流预测，防止框被推偏
- 如果推理结果过期或落后太多帧，直接丢弃，防止旧框覆盖新位置

所以当前模式可以理解成：

`低频检测 + 高频跟踪`

默认情况下：

- `push_fps = 10`
- `detection_inference_interval = 3`

也就是大约每 3 帧检测一次，中间帧由 tracker 补跟。

### 1.7 推流是怎么做的

推流同样由 FFmpeg 负责，主要逻辑在：

- [camera.py](/d:/AI_code/ai_camera/camera.py)

基本逻辑：

1. 处理线程拿到最新可渲染结果
2. 在原始帧上画框、文字、AI 标识
3. 推流线程按 `push_fps` 从渲染结果队列取帧
4. 把原始 BGR 帧送给 FFmpeg
5. FFmpeg 编码后推到 RTMP 或 RTSP 输出地址

支持：

- `libx264` 软编码
- `h264_nvenc` 硬编码

同时带有：

- FFmpeg 健康检查
- Broken pipe 检测
- 自动重启推流
- NVENC 槽位限制和回退

### 1.8 告警是怎么做的

告警模块位于：

- [alert/alert_system.py](/d:/AI_code/ai_camera/alert/alert_system.py)

当前项目支持：

- 按规则触发告警
- 保存告警截图
- 生成告警视频片段
- 上传 MinIO
- 上报平台接口

当前默认规则是：

- `alarm_any_detection`
- 只要检测到目标数量达到阈值，就触发告警

### 1.9 旧版本和当前版本的主要区别

旧版本更接近：

`一路流 -> 一路推理 -> 一路跟踪状态`

尤其在 `tracking_only` 模式下，会直接调用 `YOLO.track()`，并为每路维持自己的模型跟踪状态。

当前版本改成了：

`多路共用模型实例 -> 统一调度批量推理 -> 每路自己轻量 tracker 跟踪`

主要收益：

- 更省显存
- 更适合 10 路以上流
- 更方便统一调度和批处理

## 2. 主要模块说明

### 2.1 配置

- [config/config.json](/d:/AI_code/ai_camera/config/config.json)
- [config/algorithm_config.py](/d:/AI_code/ai_camera/config/algorithm_config.py)
- [config/config_manager.py](/d:/AI_code/ai_camera/config/config_manager.py)

### 2.2 主流程

- [camera.py](/d:/AI_code/ai_camera/camera.py)

### 2.3 推理

- [inference/unified_scheduler.py](/d:/AI_code/ai_camera/inference/unified_scheduler.py)
- [inference/inference_engine.py](/d:/AI_code/ai_camera/inference/inference_engine.py)
- [inference/inference_process.py](/d:/AI_code/ai_camera/inference/inference_process.py)

### 2.4 拉流/推流

- [stream/capture_process.py](/d:/AI_code/ai_camera/stream/capture_process.py)
- [stream/enhanced_video_processor.py](/d:/AI_code/ai_camera/stream/enhanced_video_processor.py)
- [stream/stream_health_monitor.py](/d:/AI_code/ai_camera/stream/stream_health_monitor.py)

### 2.5 跟踪

- [tracking/simple_tracker.py](/d:/AI_code/ai_camera/tracking/simple_tracker.py)

### 2.6 告警与上报

- [alert/alert_system.py](/d:/AI_code/ai_camera/alert/alert_system.py)
- [nan/camera_server.py](/d:/AI_code/ai_camera/nan/camera_server.py)
- [nan/minio_update.py](/d:/AI_code/ai_camera/nan/minio_update.py)
- [nan/post_request.py](/d:/AI_code/ai_camera/nan/post_request.py)

## 3. 启动方式

安装依赖：

```bash
pip install -r requirements.txt
```

启动：

```bash
python camera.py
```

## 4. config.json 参数说明

下面按区块说明 [config/config.json](/d:/AI_code/ai_camera/config/config.json) 中每个字段的作用。

### 4.1 `algorithm`

#### `algorithm.mode`

算法运行模式。

可选值：

- `realtime_multi`
- `tracking_only`
- `segmentation_only`
- `hybrid`

当前项目主要使用：

- `tracking_only`

表示走“检测 + 跟踪”的实时处理链路。

#### `algorithm.inference_engine_type`

推理引擎类型。

当前主要使用：

- `optimized`

用于选择当前使用的推理引擎实现。

### 4.2 `inference`

#### `inference.device`

推理设备。

常用值：

- `cpu`
- `gpu`
- `auto`

当前 YOLO 模型通常建议放在 `gpu`。

### 4.3 `stream`

#### `stream.pull_device`

拉流设备模式。

含义：

- `cpu`：CPU 解码
- `gpu`：GPU 解码
- `auto`：自动选择

常见建议：

- 如果先求稳，优先 `cpu`

#### `stream.push_device`

推流编码设备模式。

含义：

- `cpu`：走 `libx264`
- `gpu`：走 `h264_nvenc`
- `auto`：自动选择

常见建议：

- 当前环境如果 NVENC 不稳，优先 `cpu`

### 4.4 `tracking`

#### `tracking.persist`

是否保持 tracker 的跨帧状态。

- `true`：保持连续跟踪状态
- `false`：不保持

#### `tracking.tracker`

跟踪器配置名。

当前值一般为：

- `bytetrack.yaml`

这里主要是兼容旧逻辑和配置语义。

#### `tracking.conf_threshold`

跟踪相关检测置信度阈值。

目标太低置信度时，不建议进入跟踪。

#### `tracking.match_iou`

tracker 匹配时使用的 IoU 阈值。

作用：

- 控制当前检测框是否可以匹配到已有轨迹

值越大：

- 匹配更严格
- 更不容易串目标
- 但更容易产生新 ID

值越小：

- 更容易延续已有 ID
- 但也更容易误匹配

#### `tracking.max_predict_gap_ms`

两帧时间间隔超过该值时，跳过光流预测。

作用：

- 防止卡顿、丢帧后框被错误推偏

建议：

- `10fps` 场景可先从 `150~250` ms 试起

### 4.5 `models`

#### `models.default_conf_threshold`

模型默认检测置信度阈值。

如果单个模型没有单独配置阈值，就使用这个值。

#### `models.model_mappings`

模型映射表。

结构一般是：

```json
"model_mappings": {
  "detection": {
    "3001": "/path/to/model.pt"
  }
}
```

含义：

- `detection`：任务类型
- `3001`：模型或算法 ID
- 值：模型文件路径

### 4.6 `streams`

这是输入流列表，每个元素代表一路视频流。

#### `streams[].name`

流名称，用于日志、输出目录、展示。

#### `streams[].input_url`

输入流地址。

例如：

- `rtmp://.../live/camera-1002`
- `rtsp://...`

#### `streams[].output_url`

AI 结果输出流地址。

例如：

- `rtmp://.../ai/camera-1002`

#### `streams[].enabled`

是否启用该路流。

- `true`：启用
- `false`：禁用

#### `streams[].taskId`

业务任务 ID。

通常用于平台上报或资源命名。

#### `streams[].gatewayDeviceSn`

网关设备编号。

#### `streams[].droneDeviceSn`

无人机或前端设备编号。

#### `streams[].monitorEq`

监控点或监控设备标识。

#### `streams[].area`

区域名称。

#### `streams[].rootName`

根节点名称或业务树节点名称。

#### `streams[].alarmAccuratePosition`

告警定位坐标。

通常是：

- `经度,纬度`

### 4.7 `classes`

#### `classes.filtered_classes.detection_filtered_classes_by_id`

按模型 ID 配置过滤类别。

作用：

- 指定某个模型只保留某些类别
- 或过滤掉某些类别

如果为空，表示不过滤。

### 4.8 `performance`

#### `performance.detection_inference_interval`

检测推理间隔。

含义：

- 每多少帧做一次检测

例如：

- `3` 表示每 3 帧检测一次

当前项目中非常重要，因为它决定了：

- GPU 压力
- 检测频率
- 跟踪校正频率

#### `performance.result_max_back_frames`

结果或帧缓存允许保留的最大回看帧数。

作用：

- 控制缓存长度
- 防止缓存无限增长

#### `performance.max_infer_result_age`

推理结果允许的最大时间延迟，单位秒。

作用：

- 结果回来太晚时直接丢弃
- 防止旧结果覆盖当前目标位置

#### `performance.max_infer_frame_lag`

推理结果相对当前处理帧允许落后的最大帧数。

作用：

- 结果虽然刚回来，但如果对应帧已经落后太多，也直接丢弃

这个配置和 `max_infer_result_age` 配合使用更稳。

### 4.9 `alarm`

#### `alarm.target_threshold`

告警目标数量阈值。

达到这个数量后才触发告警。

#### `alarm.interval_seconds`

同类告警的冷却时间。

作用：

- 防止短时间重复频繁告警

#### `alarm.video_clip_seconds`

告警视频总时长。

#### `alarm.video_buffer_seconds`

告警视频缓冲区时长。

#### `alarm.video_pre_alert_seconds`

告警前保留的视频时长。

#### `alarm.video_post_alert_seconds`

告警后保留的视频时长。

### 4.10 `application`

#### `application.output_directory`

输出目录。

用于保存：

- 告警截图
- 告警视频
- 其他本地结果

#### `application.error_retry_interval`

错误重试间隔，单位秒。

### 4.11 `video`

#### `video.auto_detect_resolution`

是否自动检测输入流分辨率。

#### `video.target_width`

目标宽度。

#### `video.target_height`

目标高度。

#### `video.default_width`

默认宽度。

启动初期、尚未探测到真实分辨率时使用。

#### `video.default_height`

默认高度。

#### `video.fps`

输入处理基准帧率。

#### `video.push_fps`

输出推流帧率。

这个值会直接影响：

- 推流流畅度
- CPU/GPU 压力
- tracker 更新频率

#### `video.bitrate`

推流编码码率。

#### `video.max_bitrate`

最大码率。

#### `video.buffer_size`

编码缓冲区大小。

#### `video.gop_size`

GOP 大小。

影响关键帧间隔。

#### `video.push_enabled`

是否启用 AI 结果推流。

#### `video.push_codec`

推流编码器模式。

可选值：

- `auto`
- `libx264`
- `h264_nvenc`

#### `video.encoding_preset`

编码预设。

对于 `h264_nvenc` 或其他编码器会影响：

- 编码速度
- 压缩率

### 4.12 `minio`

#### `minio.endpoint`

MinIO 服务地址。

#### `minio.access_key`

MinIO 访问密钥。

#### `minio.secret_key`

MinIO 密码。

#### `minio.secure`

是否启用 HTTPS。

#### `minio.bucket_name`

上传使用的桶名。

### 4.13 `platform_api`

#### `platform_api.base_url`

业务平台接口地址。

#### `platform_api.username`

平台用户名。

#### `platform_api.password`

平台密码。

#### `platform_api.captcha`

平台登录验证码字段。

#### `platform_api.checkKey`

平台登录或请求校验字段。

#### `platform_api.vendor_id`

厂商 ID。

#### `platform_api.device_type`

设备类型。

#### `platform_api.task_type`

任务类型。

#### `platform_api.report_enabled`

是否启用平台告警上报。

#### `platform_api.login_timeout`

登录超时时间。

#### `platform_api.report_timeout`

上报超时时间。

## 5. 当前推荐调参方向

如果当前主要问题是“框跟不上目标”，优先调这些：

- `performance.detection_inference_interval`
- `tracking.match_iou`
- `tracking.max_predict_gap_ms`
- `performance.max_infer_result_age`
- `performance.max_infer_frame_lag`
- `video.push_fps`

常见思路：

- 想让框更稳：降低 `max_predict_gap_ms`，减小过期结果容忍度
- 想让框更跟手：减小 `detection_inference_interval`
- 想减少串目标：提高 `tracking.match_iou`
- 想降低算力压力：提高 `detection_inference_interval`

## 6. 注意事项

- 当前 `config.json` 中部分流名称和业务字段仍存在乱码历史数据，这不影响程序主流程，但会影响日志可读性和 README 中的展示效果。
- 如果 GPU 推流不稳定，优先改成 `stream.push_device = cpu`。
- 如果输入和输出都在同一局域网内，优先使用内网地址，避免外网映射带来的 Broken pipe 和抖动问题。
