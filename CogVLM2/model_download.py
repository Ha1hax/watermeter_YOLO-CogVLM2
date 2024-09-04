#模型下载
from modelscope import snapshot_download
model_dir = snapshot_download('ZhipuAI/cogvlm2-llama3-chinese-chat-19B-int4')
print("Model downloaded to:", model_dir)
