# -*- coding: utf-8 -*-
"""
PPE 属性分类模型定义。

使用 MobileNet V3 Small 作为骨干网络，进行安全帽和反光衣的属性分类。

模型规格（对齐参考项目 D:\AI_code\安全帽反光衣属性识别）：
- 输入: (batch_size, 3, 160, 160) RGB 图像，归一化到 [0, 1]
- 输出: Dict{"helmet_logits": (batch_size,), "vest_logits": (batch_size,)}
- 后处理: sigmoid(logits) -> [0, 1] 概率

三态判定：
- prob >= pos_threshold -> "yes"
- prob <= neg_threshold -> "no"
- 否则 -> "unknown"
"""
import logging
from typing import Dict, Tuple, Optional

import numpy as np

try:
    import torch
    import torch.nn as nn
    from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logging.warning("torch/torchvision 未安装，PPE 属性分类模型不可用")


if TORCH_AVAILABLE:
    class PPEAttrModel(nn.Module):
        """
        PPE 属性分类模型（MobileNet V3 Small）。

        用于判断人体是否佩戴安全帽和反光衣。
        模型定义对齐参考项目 D:\AI_code\安全帽反光衣属性识别\infer\ppe_attr_model.py。
        """

        def __init__(self, pretrained: bool = False, dropout: float = 0.2):
            """
            初始化模型。

            Args:
                pretrained: 是否使用预训练权重
                dropout: Dropout 概率
            """
            super().__init__()

            # 加载 MobileNet V3 Small 骨干网络
            weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
            backbone = mobilenet_v3_small(weights=weights)
            self.features = backbone.features
            self.pool = nn.AdaptiveAvgPool2d(1)

            # 动态获取维度（与参考项目一致）
            in_features = backbone.classifier[0].in_features   # = 576
            hidden_dim = backbone.classifier[0].out_features    # = 1024

            # 投影层（对齐参考项目：Linear + Hardswish + Dropout）
            self.proj = nn.Sequential(
                nn.Linear(in_features, hidden_dim),
                nn.Hardswish(),
                nn.Dropout(p=dropout),
            )

            # 双头分类
            self.helmet_head = nn.Linear(hidden_dim, 1)
            self.vest_head = nn.Linear(hidden_dim, 1)

        def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
            """
            前向传播。

            Args:
                x: 输入张量 (batch_size, 3, 160, 160)

            Returns:
                Dict 包含 "helmet_logits" 和 "vest_logits"
            """
            x = self.features(x)
            x = self.pool(x).flatten(1)
            x = self.proj(x)
            return {
                "helmet_logits": self.helmet_head(x).squeeze(1),
                "vest_logits": self.vest_head(x).squeeze(1),
            }


    def load_ppe_attr_model(
        model_path: str,
        device: str = 'cpu',
    ) -> Optional[PPEAttrModel]:
        """
        加载 PPE 属性分类模型。

        对齐参考项目 D:\AI_code\安全帽反光衣属性识别\infer\run_ppe_pipeline.py 的加载方式：
        - 支持 checkpoint 格式（提取 ckpt["model_state"]）
        - 支持纯 state_dict 格式
        - 支持 weights_only=True（安全加载）

        Args:
            model_path: 模型文件路径
            device: 设备 ('cpu' / 'cuda:0')

        Returns:
            模型实例，如果加载失败返回 None
        """
        try:
            model = PPEAttrModel(pretrained=False).to(device)

            # 加载权重（对齐参考项目的加载方式）
            try:
                ckpt = torch.load(model_path, map_location=device, weights_only=True)
            except TypeError:
                ckpt = torch.load(model_path, map_location=device)

            # 判断格式：checkpoint（含 "model_state" key）还是纯 state_dict
            if isinstance(ckpt, dict) and "model_state" in ckpt:
                model.load_state_dict(ckpt["model_state"])
            else:
                model.load_state_dict(ckpt)

            model.eval()
            logging.info(f"[PPEAttrModel] 模型加载成功: {model_path}")
            return model
        except Exception as e:
            logging.error(f"[PPEAttrModel] 模型加载失败: {e}")
            return None


    def preprocess_crop(
        crop: np.ndarray,
        image_size: int = 160,
    ) -> Optional['torch.Tensor']:
        """
        预处理裁剪图像。

        预处理逻辑与参考项目等价（BGR→RGB→resize→normalize→CHW），
        但使用 numpy + cv2 实现（避免引入 PIL 依赖）。

        Args:
            crop: 裁剪图像 (H, W, 3) BGR
            image_size: 目标尺寸

        Returns:
            预处理后的张量 (1, 3, image_size, image_size)
        """
        import cv2

        if crop.size == 0:
            return None

        # BGR -> RGB
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

        # 缩放
        crop_resized = cv2.resize(crop_rgb, (image_size, image_size))

        # 归一化到 [0, 1]
        crop_float = crop_resized.astype(np.float32) / 255.0

        # ImageNet 标准化
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        crop_normalized = (crop_float - mean) / std

        # HWC -> CHW
        crop_chw = np.transpose(crop_normalized, (2, 0, 1))

        # 转为张量
        tensor = torch.from_numpy(crop_chw).unsqueeze(0).float()

        return tensor


    def classify_attributes(
        model: PPEAttrModel,
        crop: np.ndarray,
        device: str = 'cpu',
        image_size: int = 160,
    ) -> Tuple[float, float]:
        """
        执行属性分类。

        Args:
            model: PPE 属性分类模型
            crop: 裁剪图像 (H, W, 3) BGR
            device: 设备
            image_size: 目标尺寸

        Returns:
            (helmet_prob, vest_prob) 元组
        """
        # 预处理
        tensor = preprocess_crop(crop, image_size)
        if tensor is None:
            return 0.5, 0.5

        # 移到设备
        if device.startswith('cuda') and torch.cuda.is_available():
            tensor = tensor.to(device)

        # 推理（对齐参考项目的 dict 输出格式）
        with torch.no_grad():
            outputs = model(tensor)
            helmet_prob = torch.sigmoid(outputs["helmet_logits"]).item()
            vest_prob = torch.sigmoid(outputs["vest_logits"]).item()

        return helmet_prob, vest_prob


    def prob_to_state(
        prob: float,
        pos_threshold: float = 0.6,
        neg_threshold: float = 0.3,
    ) -> str:
        """
        概率转三态。

        Args:
            prob: 概率值 [0, 1]
            pos_threshold: 正类阈值
            neg_threshold: 负类阈值

        Returns:
            "yes" / "no" / "unknown"
        """
        if prob >= pos_threshold:
            return "yes"
        elif prob <= neg_threshold:
            return "no"
        return "unknown"

else:
    # torch 不可用时的占位符
    class PPEAttrModel:
        """占位符：torch 不可用"""
        pass

    def load_ppe_attr_model(*args, **kwargs):
        logging.error("torch 未安装，无法加载 PPE 属性分类模型")
        return None

    def preprocess_crop(*args, **kwargs):
        logging.error("torch 未安装，无法预处理图像")
        return None

    def classify_attributes(*args, **kwargs):
        logging.error("torch 未安装，无法执行属性分类")
        return 0.5, 0.5

    def prob_to_state(prob, pos_threshold=0.6, neg_threshold=0.3):
        return "unknown"
