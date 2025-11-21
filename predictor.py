import json
from pathlib import Path
import numpy as np
import torch
import librosa  # 我们使用 librosa 来确保音频以正确的采样率加载
from models.AASIST import Model as AASIST_L_Model

# --- 从您的 data_utils.py 复制 'pad' 函数 ---
# 将它放在这里可以使 predictor.py 自给自足
def pad(x, max_len=64600):
    x_len = x.shape[0]
    if x_len >= max_len:
        return x[:max_len]
    # need to pad
    num_repeats = int(max_len / x_len) + 1
    padded_x = np.tile(x, (1, num_repeats))[:, :max_len][0]
    return padded_x
# ----------------------------------------------

class AASISTPredictor:
    def __init__(self, model_path: str, config_path: str, threshold: float = 0.5):
        """
        初始化预测器。
        :param model_path: 指向 .pth 权重文件的路径
        :param config_path: 指向模型 .json 配置文件的路径
        :param threshold: 用于区分 真实/伪造 的阈值
        """
        print("--- 正在初始化AASIST预测器 ---")
        
        # 1. 设置设备
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            print("正在使用 NVIDIA CUDA GPU")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
            print("正在使用 Apple Metal (MPS) GPU")
        else:
            self.device = torch.device("cpu")
            print("正在使用 CPU")

        # 2. 加载模型配置
        with open(config_path, "r") as f:
            self.config = json.load(f)
        model_config = self.config["model_config"]
        
        # 3. 初始化模型
        self.model = AASIST_L_Model(model_config).to(self.device)
        
        # 4. 加载权重
        print(f"正在从以下路径加载权重: {model_path}")
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        
        # 5. 设置评估模式
        self.model.eval()
        print("模型已成功加载并设置到评估模式。")

        # 6. 设置参数
        self.cut = 64600  # 固定音频长度 (来自 evaluate_custom.py)
        self.target_sr = 16000  # AASIST 期望的采样率
        self.threshold = threshold # 判定阈值

    def predict(self, file_path: str) -> dict:
        """
        对单个 .wav 文件执行预测。
        :param file_path: 音频文件的路径
        :return: 包含结果和分数的字典
        """
        try:
            # 1. 加载音频
            #    - 使用 librosa 确保:
            #      a) 音频被重新采样到 self.target_sr (16000 Hz)
            #      b) 音频被转换为单声道 (mono=True)
            X, sr = librosa.load(file_path, sr=self.target_sr, mono=True)

            # 2. 填充/裁剪
            X_pad = pad(X, self.cut)
            
            # 3. 转换为张量
            x_inp = torch.Tensor(X_pad).to(self.device)
            
            # 4. 添加批次维度 (batch dimension)
            x_inp = x_inp.unsqueeze(0)

            # 5. 执行推理
            with torch.no_grad():
                # _, batch_out = model(batch_x)
                # 您的 evaluate_custom.py 指出 batch_out[:, 1] 是 bonafide 分数
                _, batch_out = self.model(x_inp)
            
            # 提取 bonafide (真实) 分数
            # batch_out 形状为 [1, 2], 我们需要 [0, 1]
            score = batch_out[0, 1].item() 

            # 6. 判定结果
            if score > self.threshold:
                result_label = "真实"
            else:
                result_label = "伪造"

            return {
                "label": result_label,
                "score": score,
                "threshold": self.threshold
            }

        except Exception as e:
            print(f"处理文件 {file_path} 时出错: {e}")
            return {"error": str(e)}