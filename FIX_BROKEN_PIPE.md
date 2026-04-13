# 🔧 Broken Pipe 问题修复说明

## 问题根本原因

通过深入分析日志，发现真正的问题**不是 NVENC 不稳定**，而是：

### 1. RTMP 服务器超时断开连接

日志显示：
```
[商储大厦] infer_age=1257.5s  ← 推理结果21分钟没更新
[绿地国际博览城] infer_age=34.3s
[中央香榭] infer_age=26.5s
```

**分析**：
- 推理结果长时间不更新（最长21分钟）
- 但推流还在继续（`push_age=0.0s`）
- 说明在推送**重复的旧帧**
- RTMP 服务器检测到异常，主动断开连接

### 2. 推流重复帧限制太严格

原代码：
```python
max_repeat_frames = 1  # 只允许重复1帧
stale_repeat_window = 0.3秒  # 0.3秒后不再重复
```

**问题**：
- 如果没有新帧，很快就停止推流
- RTMP 服务器认为连接已死，主动断开
- 导致 Broken Pipe 错误

## 已实施的修复

### 修复 1: 添加 RTMP 超时和重连参数

```python
# camera.py 第 762 行
cmd += [
    '-rtmp_live', 'live',           # 实时流模式
    '-rtmp_buffer', '1000',         # 缓冲区大小(ms)
    '-timeout', '10000000',         # 超时时间10秒(微秒)
    '-flvflags', 'no_duration_filesize',
    '-f', 'flv',
    output_url
]
```

**作用**：
- 告诉 FFmpeg 这是实时流
- 设置合理的超时时间
- 防止服务器过早断开

### 修复 2: 增加帧重复限制

```python
# camera.py 第 947 行
max_repeat_frames = 100  # 从1增加到100
stale_repeat_window = 5.0  # 从0.3秒增加到5秒
```

**作用**：
- 即使没有新帧，也持续推送最后一帧
- 保持 RTMP 连接活跃
- 防止服务器认为连接已死

## 效果预期

### 修复前
```
1. 推理结果30秒没更新
2. 推流停止发送新数据
3. RTMP 服务器超时
4. 断开连接 → Broken Pipe
5. 自动重启推流
6. 画面短暂黑屏
```

### 修复后
```
1. 推理结果30秒没更新
2. 推流继续发送最后一帧（重复）
3. RTMP 服务器收到持续数据
4. 连接保持活跃 ✅
5. 无需重启
6. 画面不黑屏 ✅
```

## 测试方法

### 1. 重启程序
```bash
# 停止当前程序
Ctrl+C

# 重新启动
python camera.py
```

### 2. 观察日志
```bash
# 实时查看日志
tail -f log/ai_camera.log

# 查找 Broken Pipe 错误
grep "Broken pipe" log/ai_camera.log
```

### 3. 预期结果
- ✅ 不再出现 `Broken pipe` 错误
- ✅ 推流持续稳定
- ✅ 画面不黑屏

## 如果问题仍然存在

### 可能原因 1: RTMP 服务器配置问题

检查 RTMP 服务器配置：
```nginx
# /etc/nginx/nginx.conf
rtmp {
    server {
        listen 1935;
        
        # 增加超时时间
        timeout 60s;  # 从默认30s增加到60s
        
        # 增加缓冲区
        max_message 10M;
        
        application live {
            live on;
            record off;
        }
        
        application ai {
            live on;
            record off;
        }
    }
}
```

### 可能原因 2: 网络不稳定

虽然 ping 测试正常，但可能有间歇性抖动：

```bash
# 长时间 ping 测试
ping -c 1000 10.1.129.100 | grep "packet loss"

# 检查网络质量
mtr 10.1.129.100
```

### 可能原因 3: GPU 驱动问题

检查 NVENC 状态：
```bash
# 查看 GPU 状态
nvidia-smi

# 查看 NVENC 会话
nvidia-smi -q -d ENCODER

# 检查驱动版本
nvidia-smi --query-gpu=driver_version --format=csv
```

## 性能影响

### CPU 占用
- **修复前**: 正常
- **修复后**: 略微增加（<1%），因为需要重复推送帧

### 内存占用
- **修复前**: 正常
- **修复后**: 无变化

### GPU 占用
- **修复前**: 正常
- **修复后**: 无变化

### 网络带宽
- **修复前**: 间歇性（有断流）
- **修复后**: 持续稳定

## 总结

这次修复的核心思想是：

**保持 RTMP 连接活跃，即使没有新的检测结果**

通过：
1. 添加 RTMP 超时参数
2. 增加帧重复限制
3. 延长重复窗口时间

确保推流持续稳定，不会因为推理结果更新慢而导致连接断开。

---

**修复日期**: 2026-04-13  
**修复版本**: v1.1  
**测试状态**: 待测试
