import uvicorn
import os
import tempfile
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse
from predictor import AASISTPredictor

# ==========================================================
# ⚠️ 重要: 更新这些路径！
# ==========================================================
MODEL_PATH = "epoch_45_0.441.pth"   # 您的 .pth 权重文件路径
CONFIG_PATH = "config_standalone_eval.json"   # 您的 .json 配置文件路径
# 
# ℹ️ 关于阈值 (THRESHOLD):
# 0.5 是一个常见的起始点，但您应该使用验证集来找到最佳阈值。
# 0.5 意味着 "如果模型有超过50%的置信度认为是真实的，那就是真实的"
THRESHOLD = 0.5
# ==========================================================

# 检查文件是否存在
if not os.path.exists(MODEL_PATH) or not os.path.exists(CONFIG_PATH):
    print("错误: 找不到模型或配置文件。")
    print(f"请确保 '{MODEL_PATH}' 和 '{CONFIG_PATH}' 存在于当前目录。")
    exit()

# ---- 初始化应用和预测器 ----
app = FastAPI(title="AASIST 语音伪造检测")

# 在全局范围内加载模型，这样它只会被加载一次
print("正在加载模型... 这可能需要几秒钟。")
predictor = AASISTPredictor(
    model_path=MODEL_PATH,
    config_path=CONFIG_PATH,
    threshold=THRESHOLD
)
print("模型加载完毕，服务器已准备就绪。")


# ---- API 端点 ----

@app.post("/predict/")
async def predict_audio(file: UploadFile = File(...)):
    """
    接收一个音频文件并返回检测结果。
    """
    # FastAPI 的 UploadFile 是一个 "spooled" 文件，
    # 我们需要将其保存到一个临时的 .wav 文件中，以便 librosa 可以读取它。
    
    # 使用 tempfile 安全地创建临时文件
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            # 将上传的文件内容写入临时文件
            temp_file.write(await file.read())
            temp_file_path = temp_file.name
        
        # --- 核心: 在临时文件上运行预测 ---
        print(f"正在处理文件: {file.filename}")
        result = predictor.predict(temp_file_path)
        print(f"预测结果: {result}")
        
    except Exception as e:
        print(f"处理中发生错误: {e}")
        raise HTTPException(status_code=500, detail=f"处理文件时发生错误: {str(e)}")
        
    finally:
        # 确保我们总是清理临时文件
        if 'temp_file_path' in locals() and os.path.exists(temp_file_path):
            os.remove(temp_file_path)
            
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])

    # 返回 JSON 结果
    return {
        "filename": file.filename,
        "result_label": result["label"],
        "bonafide_score": result["score"],
        "is_bonafide": result["label"] == "真实"
    }

@app.get("/", response_class=HTMLResponse)
async def get_frontend():
    """
    提供 HTML 前端页面。
    """
    try:
        with open("index.html", "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return "错误: 找不到 index.html。请确保它与 app.py 位于同一目录中。"

@app.get("/docs")
async def get_docs():
    # 重定向到 FastAPI 自动生成的文档
    return RedirectResponse(url="/docs")


# ---- 运行服务器 ----
if __name__ == "__main__":
    # 运行服务器，监听所有接口(0.0.0.0)的8000端口
    # uvicorn.run(app, host="127.0.0.1", port=8000)
    print("启动服务器，访问 http://127.0.0.1:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)