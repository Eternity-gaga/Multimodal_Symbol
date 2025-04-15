from openai import OpenAI
import requests
import json
import re
import random
import evaluate
import argparse
from tqdm import tqdm
import torch
import os
from transformers import AutoTokenizer
import base64

API_URL = "https://api.chatanywhere.tech/v1/chat/completions"
API_KEY = "sk-r3ivtcnPlnNBlLsHYEhPj748yp59ObtiOb5JFFx8afsOyUP6"

headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {API_KEY}"
}

image_folder = "/data/kuangjy/symbol/math_data/output_images_multimath"

system_prompt_zh = """你正在执行多模态复杂符号理解任务，请一步步思考"""
system_prompt_en = """You are performing a multimodal complex symbol understanding task. Please think step by step"""

prompt_template_zh = """请仔细分析图像并一步步思考题目内容，{condition}，{question}，最终答案放在\\boxed{{}}中。"""
prompt_template_en = """Please analyze the image and Question step by step : {condition}, {question}, put your final answer with \\boxed{{}}."""

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def translate_question(input_data, language='zh', output_file='gpt-4o-multimath-zh.jsonl'):
    with open(output_file, 'a', encoding='utf-8') as f:
        for item in tqdm(input_data, desc="Processing items", unit="item"):
            # 获取图像ID
            image_id = item.get("image_id")
            if not image_id:
                print(f"No image_id found for item: {item}")
                continue
            
            # 构建图像路径
            image_path = os.path.join(image_folder, f"{image_id}")
            if not os.path.exists(image_path):
                print(f"Image file not found: {image_path}")
                continue
            
            # 编码图像
            base64_image = encode_image(image_path)

            # 遍历 QA_pair 中的每个问题
            for qa_pair in item.get("QA_pair", []):
                # 根据语言选择输入问题与 prompt
                if language == 'zh':
                    question = qa_pair.get("question_zh", "")
                    condition = qa_pair.get("condition_zh", "")
                    system_prompt = system_prompt_zh
                    prompt_template = prompt_template_zh
                elif language == 'en':
                    question = qa_pair.get("question_en", "")
                    condition = qa_pair.get("condition_en", "")
                    system_prompt = system_prompt_en
                    prompt_template = prompt_template_en
                else:
                    raise ValueError("Unsupported language. Please choose 'zh' or 'en'.")

                # 创建完整的提示
                prompt = prompt_template.format(condition=condition, question=question)

                # 发送请求到 API
                response = requests.post(API_URL, headers=headers, json={
                    "model": "gpt-4o-ca",
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}]
                        }
                    ],
                    "max_tokens": 2048,  # 根据需要调整
                    "temperature": 0.7,
                })

                # 检查响应
                if response.status_code == 200:
                    result = response.json()
                    response_content = result['choices'][0]['message']['content'].strip()  # 获取纯文本结果并去除首尾空白字符
                    
                    qa_pair["model_output"] = response_content

                    # 立即保存更新后的数据到文件中
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
                else:
                    print(f"Error: {response.status_code}, {response.text}")

# 读取原始 JSON 数据
with open('adjusted_multimath_symbol copy.json', 'r', encoding='utf-8') as f:
    input_data = json.load(f)

# 调用处理函数
translate_question(input_data, language='zh')  # 选择语言。默认为zh，英文为en