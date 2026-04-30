# 摄像头保护区界线投影部署说明

## 文件放置

将本包内容复制到算法接口项目根目录：

```text
项目根目录/
├─camera.py
├─config/
│  ├─camera_projector_config.json
│  └─borders/
│     └─NanchangLine2MPZ_20260403.json
└─utils/
   └─zone_projector.py
```

`zone_projector.py` 是原 `CameraProjectorAPI_20251113.py` 的正式精简命名版本。

## 摄像头匹配规则

`camera.py` 会按以下顺序为每一路视频流查找投影配置：

1. `streams[].camera_id` / `cameraId`
2. `streams[].stream_id` / `streamId`
3. 从 `input_url` / `output_url` 中自动提取 `camera-1002` 这类 ID
4. `monitorEq`、`taskId`、`name`
5. `camera_projector_config.json` 中每个摄像头配置的 `aliases`

当前 `config/config.json` 没有显式 `camera_id` 字段，但输入流 URL 中包含 `camera-1002`、`camera-1004` 等，因此无需修改 `config/config.json` 也能匹配。

## 当前配置状态

`config/camera_projector_config.json` 已从本次提供的 `config.json` 生成：

- `camera-1002`（罗家集）已启用，并使用此前确认的 `LuoJiaJi_Camera_P1_marked-2.jpg` 的 info 参数 1。
- 其他摄像头已生成占位配置，但 `enabled=false`，避免参数未标定时画出错误投影线。
- 后续补齐某个摄像头的 `height`、`gimbal_yaw`、`gimbal_pitch`、`gimbal_roll`、`zoom_factor` 后，将该摄像头的 `enabled` 改为 `true` 即可。

## 投影缓存逻辑

每路摄像头会缓存一张“只包含保护区界线的透明效果叠加层”。

当以下内容都没有变化时，下一帧直接复用上一次的线层，不重复计算投影：

- 摄像头 ID
- 当前视频帧分辨率
- `info` 中的相机位置、姿态、变焦参数
- `ground_alt`
- `line_thickness`
- `border_json` 文件路径、修改时间、文件大小
- 传感器/裁切参数

只要其中任意一项变化，缓存会自动失效并重新投影。

这版虽然使用静态 `camera_projector_config.json`，但缓存键已经按“未来相机参数会动态变化”的方式设计，后续接入实时云台角、变焦倍数时，只要更新对应路的 `boundary_projector_info`，缓存就会自动刷新。

## 启动

仍然按原项目方式启动：

```bash
python camera.py
```

日志中看到类似下面信息说明投影模块启用成功：

```text
[罗家集] 保护区界线投影已启用: camera_identity=camera-1002 ...
[罗家集] 保护区界线投影缓存已更新 ...
```
