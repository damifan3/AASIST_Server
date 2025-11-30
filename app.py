import uvicorn
import os
import tempfile
from typing import List
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse
from pydub import AudioSegment  # 用于格式转换

# 引入预测器
from predictor import AASISTPredictor

# ==========================================================
# 配置路径
# ==========================================================
MODEL_PATH = "epoch_45_0.441.pth" 
CONFIG_PATH = "config_standalone_eval.json"
THRESHOLD = 1.510585 
# ==========================================================

# --- [关键] 配置 Pydub 使用本地 ffmpeg ---
# 检查当前目录下是否有 ffmpeg.exe，如果有，指定给 pydub
current_dir = os.path.dirname(os.path.abspath(__file__))
local_ffmpeg = os.path.join(current_dir, "ffmpeg.exe")
local_ffprobe = os.path.join(current_dir, "ffprobe.exe")

if os.path.exists(local_ffmpeg):
    AudioSegment.converter = local_ffmpeg
    AudioSegment.ffprobe = local_ffprobe
    print(f"检测到本地 FFmpeg，已启用: {local_ffmpeg}")
else:
    print("警告: 未在当前目录检测到 ffmpeg.exe。如果系统未安装 FFmpeg，.m4a 格式将无法处理。")
# ----------------------------------------

if not os.path.exists(MODEL_PATH) or not os.path.exists(CONFIG_PATH):
    print("错误: 找不到模型或配置文件。")
    exit()

app = FastAPI(title="AASIST 语音伪造检测 (批量版)")

print("正在加载模型...")
predictor = AASISTPredictor(
    model_path=MODEL_PATH,
    config_path=CONFIG_PATH,
    threshold=THRESHOLD
)
print("模型加载完毕。")

@app.post("/predict/")
async def predict_audio_batch(files: List[UploadFile] = File(...)):
    results = []

    for file in files:
        temp_input_path = None
        temp_wav_path = None
        
        try:
            # 1. 保存用户上传的原始文件
            file_ext = os.path.splitext(file.filename)[1].lower()
            if not file_ext:
                file_ext = ".temp" 

            with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as temp_file:
                content = await file.read()
                temp_file.write(content)
                temp_input_path = temp_file.name
            
            # 2. 格式转换逻辑
            # 模型和librosa对WAV支持最好。如果不是wav，用pydub转成wav
            target_file_path = temp_input_path
            
            if file_ext != ".wav":
                print(f"正在转换格式: {file.filename} -> wav")
                # 创建一个临时的 wav 文件名
                temp_wav_path = temp_input_path + ".converted.wav"
                
                # 使用 pydub 加载并导出为 wav
                audio = AudioSegment.from_file(temp_input_path)
                audio.export(temp_wav_path, format="wav")
                
                # 将目标路径指向这个新的 wav 文件
                target_file_path = temp_wav_path

            # 3. 运行预测
            print(f"正在处理: {file.filename}")
            pred_result = predictor.predict(target_file_path)
            
            results.append({
                "filename": file.filename,
                "result_label": pred_result.get("label", "错误"),
                "score": pred_result.get("score", 0),
                "is_bonafide": pred_result.get("label") == "真实",
                "error": pred_result.get("error", None)
            })

        except Exception as e:
            print(f"文件 {file.filename} 处理出错: {e}")
            results.append({
                "filename": file.filename,
                "result_label": "错误",
                "score": 0,
                "is_bonafide": False,
                "error": f"处理失败: {str(e)} (请检查 ffmpeg.exe 是否在目录下)"
            })
            
        finally:
            # 4. 清理所有临时文件
            try:
                if temp_input_path and os.path.exists(temp_input_path):
                    os.remove(temp_input_path)
                if temp_wav_path and os.path.exists(temp_wav_path):
                    os.remove(temp_wav_path)
            except:
                pass

    return results

@app.get("/", response_class=HTMLResponse)
async def get_frontend():
    try:
        with open("index.html", "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return "错误: 找不到 index.html。"

if __name__ == "__main__":
    print("启动服务器，访问 http://127.0.0.1:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)