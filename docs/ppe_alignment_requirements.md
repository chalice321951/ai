# PPE 检测模块对齐需求文档

> 目标：将 ai_camera 项目的 PPE 实现完全对齐 `D:\AI_code\安全帽反光衣属性识别` 项目
> 日期：2026-06-29

---

## 一、差异总览

| 模块 | 文件 | 差异数 | 严重程度 |
|------|------|--------|---------|
| 模型定义 | `inference/ppe/ppe_attr_model.py` | 5 处 | 🔴 高（模型无法加载） |
| 模型加载 | `inference/ppe/ppe_attr_model.py` | 2 处 | 🔴 高（模型无法加载） |
| 预处理 | `inference/ppe/ppe_attr_model.py` | 1 处 | 🟡 中（逻辑等价但实现不同） |
| 推理方式 | `inference/ppe/ppe_attr_model.py` | 1 处 | 🟡 中（输出格式不同） |
| 阈值配置 | `config/config.json` | 2 处 | 🟡 中（影响精度） |
| 检测框扩展 | `inference/ppe/ppe_detector.py` | 1 处 | 🟡 中（影响检测质量） |
| 人体检测参数 | `config/config.json` | 2 处 | 🟡 中（影响检测质量） |

---

## 二、逐项修改需求

### 2.1 模型定义对齐（ppe_attr_model.py）

**文件**：`D:/AI_code/ai_camera/inference/ppe/ppe_attr_model.py`

**当前代码**（第 33-81 行）：

```python
class PPEAttrModel(nn.Module):
    def __init__(self, pretrained: bool = False):
        super().__init__()
        backbone = mobilenet_v3_small(pretrained=pretrained)
        self.features = backbone.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        # ❌ 问题1: 硬编码 576、128
        self.projector = nn.Sequential(
            nn.Linear(576, 128),
            nn.ReLU(inplace=True),       # ❌ 问题2: 激活函数错误
        )
        # ❌ 问题3: 无 Dropout
        self.helmet_head = nn.Linear(128, 1)
        self.vest_head = nn.Linear(128, 1)

    # ❌ 问题4: 返回 Tuple
    def forward(self, x) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.features(x)
        x = self.pool(x).flatten(1)
        x = self.projector(x)
        return self.helmet_head(x).squeeze(-1), self.vest_head(x).squeeze(-1)
```

**参考项目代码**（`D:/AI_code/安全帽反光衣属性识别/infer/ppe_attr_model.py`）：

```python
class PPEAttrModel(nn.Module):
    def __init__(self, pretrained: bool = True, dropout: float = 0.2):
        super().__init__()
        weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        backbone = mobilenet_v3_small(weights=weights)
        self.features = backbone.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        # ✅ 动态获取维度
        in_features = backbone.classifier[0].in_features    # = 576
        hidden_dim = backbone.classifier[0].out_features     # = 1024
        # ✅ 正确的投影层
        self.proj = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.Hardswish(),              # ✅ Hardswish 激活
            nn.Dropout(p=dropout),       # ✅ Dropout
        )
        self.helmet_head = nn.Linear(hidden_dim, 1)
        self.vest_head = nn.Linear(hidden_dim, 1)

    # ✅ 返回 Dict
    def forward(self, x) -> Dict[str, torch.Tensor]:
        x = self.features(x)
        x = self.pool(x).flatten(1)
        x = self.proj(x)
        return {
            "helmet_logits": self.helmet_head(x).squeeze(1),
            "vest_logits": self.vest_head(x).squeeze(1),
        }
```

**修改清单**：

| 编号 | 改动 | 说明 |
|------|------|------|
| M-1 | 投影层 `Linear(576, 128)` → `Linear(576, 1024)` | hidden_dim 从 128 改为 1024 |
| M-2 | 激活函数 `ReLU` → `Hardswish` | 与训练代码一致 |
| M-3 | 增加 `Dropout(p=0.2)` | 与训练代码一致 |
| M-4 | `projector` 重命名为 `proj` | 与参考项目一致（避免权重 key 不匹配） |
| M-5 | `forward()` 返回 `Dict` 而非 `Tuple` | 与参考项目接口一致 |

**新增 import**：
```python
from typing import Dict  # 替换 Tuple
from torchvision.models import MobileNet_V3_Small_Weights  # 新增
```

---

### 2.2 模型加载对齐（ppe_attr_model.py）

**文件**：`D:/AI_code/ai_camera/inference/ppe/ppe_attr_model.py`

**当前代码**（第 84-111 行）：

```python
def load_ppe_attr_model(model_path, device='cpu'):
    model = PPEAttrModel(pretrained=False)
    state_dict = torch.load(model_path, map_location='cpu')  # ❌ 整个 checkpoint 当 state_dict
    model.load_state_dict(state_dict)                        # ❌ 报错：key 不匹配
    ...
```

**参考项目代码**（`D:/AI_code/安全帽反光衣属性识别/infer/run_ppe_pipeline.py` 第 35-43 行）：

```python
def load_attr_model(checkpoint_path, device):
    model = PPEAttrModel(pretrained=False).to(device)
    try:
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except TypeError:
        ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])  # ✅ 从 checkpoint 提取
    model.eval()
    return model
```

**修改清单**：

| 编号 | 改动 | 说明 |
|------|------|------|
| M-6 | 加载时先判断格式 | checkpoint 格式提取 `ckpt["model_state"]` |
| M-7 | 支持 `weights_only=True` | 安全加载，老版本 torch 兼容 |

---

### 2.3 预处理对齐（ppe_attr_model.py）

**文件**：`D:/AI_code/ai_camera/inference/ppe/ppe_attr_model.py`

**当前代码**（`preprocess_crop` 函数）：

```python
def preprocess_crop(crop, image_size=160):
    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    crop_resized = cv2.resize(crop_rgb, (image_size, image_size))
    crop_float = crop_resized.astype(np.float32) / 255.0
    crop_normalized = (crop_float - mean) / std
    crop_chw = np.transpose(crop_normalized, (2, 0, 1))
    tensor = torch.from_numpy(crop_chw).unsqueeze(0).float()
    return tensor
```

**参考项目代码**（`run_ppe_pipeline.py` 第 20-28 行）：

```python
def build_eval_transforms(image_size):
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
```

**分析**：两者逻辑等价（BGR→RGB→resize→normalize→CHW），但实现方式不同。

**修改需求**：

| 编号 | 改动 | 说明 |
|------|------|------|
| M-8 | 保持现有实现，但验证数值一致性 | 两种方式应该输出相同的 tensor |

**不需要改**：预处理逻辑已经等价，保持 numpy + cv2 实现即可（避免引入 PIL 依赖）。

---

### 2.4 推理方式对齐（ppe_attr_model.py）

**文件**：`D:/AI_code/ai_camera/inference/ppe/ppe_attr_model.py`

**当前代码**（`classify_attributes` 函数）：

```python
def classify_attributes(model, crop, device, image_size):
    ...
    helmet_logits, vest_logits = model(tensor)      # ❌ Tuple 解包
    helmet_prob = torch.sigmoid(helmet_logits).item()
    vest_prob = torch.sigmoid(vest_logits).item()
    return helmet_prob, vest_prob
```

**参考项目代码**（`run_ppe_pipeline.py` 第 106-109 行）：

```python
outputs = attr_model(batch)
helmet_probs = torch.sigmoid(outputs["helmet_logits"]).cpu().tolist()
vest_probs = torch.sigmoid(outputs["vest_logits"]).cpu().tolist()
```

**修改需求**：

| 编号 | 改动 | 说明 |
|------|------|------|
| M-9 | `classify_attributes` 改为 dict 解包 | `outputs = model(tensor)` → `outputs["helmet_logits"]` |

---

### 2.5 阈值配置对齐（config.json）

**文件**：`D:/AI_code/ai_camera/config/config.json`

| 参数 | 当前值 | 参考项目值 | 修改 |
|------|--------|----------|------|
| `helmet_pos_threshold` | `0.6` | `0.65` | → `0.65` |
| `helmet_neg_threshold` | `0.3` | `0.30` | 不变 |
| `vest_pos_threshold` | `0.6` | `0.90` | → `0.90` |
| `vest_neg_threshold` | `0.3` | `0.30` | 不变 |

**注意**：`vest_pos_threshold` 差异最大（0.6 vs 0.9），参考项目对反光衣判定更严格。

---

### 2.6 检测框扩展对齐（ppe_detector.py）

**文件**：`D:/AI_code/ai_camera/inference/ppe/ppe_detector.py`

**当前代码**（`_expand_box` 方法）：

```python
def _expand_box(self, x1, y1, x2, y2, frame_shape):
    expand_h = int(box_h * self._box_expand_ratio)    # = 0.15
    expand_w = int(box_w * self._box_expand_ratio)
    new_y1 = max(0, y1 - expand_h)                    # 顶部扩展 = expand_h
    ...
```

**参考项目代码**（`run_ppe_pipeline.py` 第 56-76 行）：

```python
def expand_box(x1, y1, x2, y2, width, height, expand_ratio, top_extra_ratio):
    dx = box_w * expand_ratio        # = 0.05
    dy = box_h * expand_ratio        # = 0.05
    top_extra = box_h * top_extra_ratio  # = 0.05
    nx1 = max(0, int(round(x1 - dx)))
    ny1 = max(0, int(round(y1 - dy - top_extra)))  # 顶部多扩展 top_extra
    nx2 = min(width - 1, int(round(x2 + dx)))
    ny2 = min(height - 1, int(round(y2 + dy)))
    ...
```

**差异**：

| 参数 | 当前值 | 参考项目值 | 说明 |
|------|--------|----------|------|
| `box_expand_ratio` | `0.15` | `0.05` | 当前扩展过大 |
| `top_extra_ratio` | 无 | `0.05` | 参考项目独立控制头顶扩展 |

**修改需求**：

| 编号 | 改动 | 说明 |
|------|------|------|
| M-10 | 增加 `top_extra_ratio` 参数 | 独立控制头顶区域扩展 |
| M-11 | 默认 `box_expand_ratio` 改为 `0.05` | 与参考项目一致 |
| M-12 | `_expand_box` 增加 `top_extra_ratio` 逻辑 | `new_y1 = max(0, y1 - expand_h - top_extra)` |

---

### 2.7 人体检测参数对齐（config.json）

**文件**：`D:/AI_code/ai_camera/config/config.json`

| 参数 | 当前值 | 参考项目值 | 修改 |
|------|--------|----------|------|
| `person_conf_threshold` | `0.5` | `0.25` | → `0.25` |
| `person_class_names` | `["person", "pedestrian", "people"]` | `["pedestrian", "people", "person"]` | 顺序不同但等价 |

**注意**：`person_conf_threshold` 从 0.5 降到 0.25 会检测出更多人体（包括低置信度的），可能增加误检但减少漏检。

---

## 三、修改优先级

| 优先级 | 编号 | 说明 | 原因 |
|--------|------|------|------|
| **P0** | M-1 ~ M-7 | 模型定义和加载 | 不改就无法加载模型 |
| **P1** | M-9 | 推理输出格式 | 不改就无法正确读取结果 |
| **P1** | M-5 | forward 返回 Dict | 与 M-9 配套 |
| **P2** | M-10 ~ M-12 | 检测框扩展 | 影响属性分类质量 |
| **P2** | 阈值配置 | vest_pos_threshold=0.9 | 影响反光衣检测精度 |
| **P3** | person_conf_threshold | 0.5 → 0.25 | 影响人体检测数量 |
| **P3** | M-8 | 预处理验证 | 逻辑已等价，只需确认 |

---

## 四、验证方法

### 4.1 模型加载验证

```python
# 加载模型
model = load_ppe_attr_model("best_model.pt", device="cuda")
assert model is not None, "模型加载失败"

# 验证权重 key
state_dict = model.state_dict()
assert "proj.0.weight" in state_dict, "投影层权重 key 不匹配"
assert state_dict["proj.0.weight"].shape == (1024, 576), "投影层 shape 不匹配"
```

### 4.2 推理输出验证

```python
# 单帧推理
tensor = preprocess_crop(crop, image_size=160)
outputs = model(tensor)
assert isinstance(outputs, dict), "输出应为 dict"
assert "helmet_logits" in outputs, "缺少 helmet_logits"
assert "vest_logits" in outputs, "缺少 vest_logits"
```

### 4.3 端到端验证

```bash
# 使用参考项目的测试图片
python inference/ppe/ppe_detector.py --image test.jpg --output result.jpg
```

---

## 五、配置迁移指南

### 5.1 config.json 修改

```json
{
    "ppe": {
        "enabled": true,
        "detection": {
            "model_id": "3099",
            "person_class_names": ["pedestrian", "people", "person"],
            "person_conf_threshold": 0.25,  // 从 0.5 改为 0.25
            "box_expand_ratio": 0.05,       // 从 0.15 改为 0.05
            "top_extra_ratio": 0.05         // 新增：头顶扩展比例
        },
        "attribute": {
            "model_path": "/home/admin123/wjj/models/best_model.pt",
            "image_size": 160,
            "inference_interval": 3,
            "helmet_pos_threshold": 0.65,   // 从 0.6 改为 0.65
            "helmet_neg_threshold": 0.30,   // 不变
            "vest_pos_threshold": 0.90,     // 从 0.6 改为 0.90
            "vest_neg_threshold": 0.30      // 不变
        },
        "rendering": { ... },
        "alarm": { ... }
    }
}
```

---

## 六、风险与注意事项

| 风险 | 说明 | 缓解措施 |
|------|------|---------|
| **阈值变化影响** | `vest_pos_threshold` 从 0.6→0.9，反光衣检测会更严格 | 需要在现场测试调优 |
| **检测框变化** | `box_expand_ratio` 从 0.15→0.05，裁剪区域变小 | 可能漏掉边缘的安全帽/反光衣 |
| **置信度阈值** | `person_conf_threshold` 从 0.5→0.25，检测出更多人体 | 可能增加误检，需现场验证 |
| **模型文件兼容** | 只有参考项目训练的 `best_model.pt` 能加载 | 需要确认模型文件来源 |

---

## 七、测试清单

- [ ] 模型加载成功（`load_ppe_attr_model` 返回非 None）
- [ ] forward 输出为 Dict 格式
- [ ] 推理结果包含 helmet_logits 和 vest_logits
- [ ] sigmoid 后概率在 [0, 1] 范围内
- [ ] 三态判定逻辑正确（yes/no/unknown）
- [ ] 检测框扩展逻辑正确（包含头顶区域）
- [ ] 配置文件解析正确
- [ ] 端到端测试：检测人体 + 分类属性 + 渲染结果
