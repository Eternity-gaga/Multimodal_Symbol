from tqdm import tqdm  # 导入 tqdm 库
import base64
import requests
import json
import re
import random
import evaluate
import argparse
import torch
import os
from transformers import AutoTokenizer
from collections import defaultdict

API_URL = "https://api.chatanywhere.tech/v1/chat/completions"
API_KEY = "sk-r3ivtcnPlnNBlLsHYEhPj748yp59ObtiOb5JFFx8afsOyUP6"

headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {API_KEY}"
}

### chat prompts
instr = "请根据参考答案判断预测的答案是否正确，不需要考虑单位，格式等是否一致，只看数学含义是否正确。只输出'True' or 'False'"

def classify_math_tasks(input_data, output_file='/data/kuangjy/symbol/experimental/results-multimath/Qwen-2.5-vl-instruct-2048/predictions-score.jsonl'):
    level_task_dict = defaultdict(list)
    
    # 将数据按level和task分类
    for item in input_data:
        level = item['level']
        task_type = item['task_type']
        level_task_dict[(level, task_type)].append(item)
    
    # 分层采样
    sampled_data = []
    for (level, task_type), items in level_task_dict.items():
        sample_size = min(100, len(items))  # 每个level-task组合最多采样100条
        sampled_items = random.sample(items, sample_size)
        sampled_data.extend(sampled_items)
    
    # 确保总共采样300条数据
    if len(sampled_data) > 300:
        sampled_data = random.sample(sampled_data, 300)
    elif len(sampled_data) < 300:
        raise ValueError("无法从现有数据中分层采样出300条数据")
    
    # 初始化统计变量
    total_correct = 0
    level_correct = defaultdict(int)
    task_correct = defaultdict(int)
    level_count = defaultdict(int)
    task_count = defaultdict(int)
    
    # 使用 tqdm 包装 sampled_data 以显示进度条
    with open(output_file, 'w', encoding='utf-8') as f:
        for idx, item in enumerate(tqdm(sampled_data, desc="Processing", unit="item")):
            # 假设 QA_pair 是一个列表，我们取第一个元素
            qa_item = item['QA_pair'][0] if item['QA_pair'] else {}
            answer_zh = qa_item.get('answer_zh', '')
            
            prompt = (
                "参考答案："
                f"{answer_zh}"
                "预测答案："
                f"{item['model_answer']}"
            )
            
            # 发送请求到API
            response = requests.post(API_URL, headers=headers, json={
                "model": "gpt-4o-ca",
                "messages": [{"role": "system", "content": instr}, {"role": "user", "content": prompt}],
                "max_tokens": 2048,  # 根据需要调整
                "temperature": 0.7,
            })
            
            # 检查响应
            if response.status_code == 200:
                result = response.json()
                response_content = result['choices'][0]['message']['content'].strip()  # 获取纯文本结果并去除首尾空白字符
                
                # 构建要保存的JSON条目
                output_entry = {
                    "question": item['QA_pair'],
                    "answer": answer_zh,
                    "prediction": item['model_answer'],
                    "level": item['level'],
                    "task_type": item['task_type'],
                    "judgement": response_content
                }
                
                # 写入文件
                f.write(json.dumps(output_entry, ensure_ascii=False))
                f.write('\n')
                
                # 更新统计信息
                correct = response_content.lower() == 'true'
                total_correct += correct
                level_correct[item['level']] += correct
                task_correct[item['task_type']] += correct
                level_count[item['level']] += 1
                task_count[item['task_type']] += 1
                
                # 每评估10条数据打印一次结果
                if (idx + 1) % 10 == 0:
                    print(f"已处理 {idx + 1} 条，总准确率: {total_correct / (idx + 1):.4f}")
            
            else:
                print(f"Error: {response.status_code}, {response.text}")
    
    # 计算最终的准确率
    final_results = {
        "overall_accuracy": total_correct / len(sampled_data),
        "level_accuracies": {level: level_correct[level] / level_count[level] for level in level_correct},
        "task_accuracies": {task: task_correct[task] / task_count[task] for task in task_correct}
    }
    
    # 保存最终的结果到文件
    with open('/data/kuangjy/symbol/experimental/results-multimath/Qwen-2.5-vl-instruct-2048/final_score.json', 'w', encoding='utf-8') as f:
        json.dump(final_results, f, ensure_ascii=False, indent=4)

# 读取输入数据
with open('/data/kuangjy/symbol/experimental/results-multimath/Qwen-2.5-vl-instruct-2048/output.jsonl', 'r') as f:
    lines = f.readlines()
    input_data = [json.loads(line) for line in lines]

# 调用处理函数
classify_math_tasks(input_data)
