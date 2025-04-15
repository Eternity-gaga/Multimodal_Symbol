import random
import json
import vllm
import evaluate
import argparse
from tqdm import tqdm
import torch
import os
import re
from transformers import AutoTokenizer
import base64

# exact_match = evaluate.load("./evaluation/metrics/exact_match")
# f1 = evaluate.load('./evaluation/metrics/f1')

image_folder = "/data/kuangjy/symbol/math_data/output_images_multimath"

system_prompt_zh = """你正在执行多模态复杂符号理解任务，请一步步思考"""
system_prompt_en = """You are performing a multimodal complex symbol understanding task. Please think step by step"""

prompt_template_zh = """请仔细分析图像并一步步思考题目内容，{condition}，{question}，最终答案放在\\boxed{{}}中。"""
prompt_template_en = """Please analyze the image and Question step by step : {condition}, {question}, put your final answer with \\boxed{{}}."""

def extract_answer(output):
    # Placeholder function to extract the answer from the model output
    # You can customize this function based on how the model outputs answers
    match = re.search(r'\\boxed\{(.*?)\}', output)
    if match:
        return match.group(1).strip()
    return ""

def image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    return encoded_string

def main(args):
    random.seed(42)

    print("Loading data...")
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir, exist_ok=True)

    with open('/data/kuangjy/symbol/math_data/adjusted_multimath_symbol.json', 'r') as f:
        full_data = json.load(f)
    
    if args.num_instances is not None and len(full_data) >= args.num_instances:
        full_data = random.sample(full_data, args.num_instances)

    # Load model and tokenizer
    model = vllm.LLM(
        model=args.model_name,
        tensor_parallel_size=torch.cuda.device_count(),
        trust_remote_code=True,
        # max_model_len=32960, # for internlm2-math-plus-mixtral8x22b
        # gpu_memory_utilization=.75
    )

    stop_strings = args.additional_stop_sequence
    if args.newline_stop:
        if args.stop_at_double_newline:
            stop_strings += ["\n\n"] 
        elif args.stop_at_triple_newline:
            stop_strings += ["\n\n\n"]
        else:
            stop_strings += ["\n"]

    sampling_params = vllm.SamplingParams(
        temperature=0,
        max_tokens=args.clm_max_length,
        stop=stop_strings,
        skip_special_tokens=True,
    )

    prompts = []
    targets = []
    data_items = []

    print("Processing items...")
    for item in tqdm(full_data, desc="Processing items", unit="item"):
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

        # 将图像转换为Base64编码
        base64_image = image_to_base64(image_path)

        for qa_pair in item.get("QA_pair", []):
            # 根据语言选择输入问题与 prompt
            if args.eval_lang == 'chn':
                question = qa_pair.get("question_zh", "")
                condition = qa_pair.get("condition_zh", "")
                system_prompt = system_prompt_zh
                prompt_template = prompt_template_zh
            elif args.eval_lang == 'eng':
                question = qa_pair.get("question_en", "")
                condition = qa_pair.get("condition_en", "")
                system_prompt = system_prompt_en
                prompt_template = prompt_template_en
            else:
                raise ValueError("Unsupported language. Please choose 'chn' or 'eng'.")

            # 创建完整的提示
            prompt = prompt_template.format(condition=condition, question=question)

            # 将图像Base64编码和ID加入提示
            # full_prompt = {
            #     "messages": [
            #         {"role": "system", "content": system_prompt},
            #         {
            #             "role": "user",
            #             "content": [
            #                 {"type": "text", "text": prompt},
            #                 {
            #                     "type": "image_url",
            #                     "image_url": {
            #                         "url": f"data:image/jpeg;base64,{base64_image}",
            #                     },
            #                 }
            #                 ]
            #             }
            #     ]
            # }

            full_prompt = f"{system_prompt}\n{prompt}\n![image](data:image/jpeg;base64,{base64_image})"

            # 添加到 prompts 列表
            prompts.append(full_prompt)
            targets.append(qa_pair.get("answer_zh", ""))
            data_items.append((item, qa_pair))

    # 生成模型输出
    generations = model.generate(prompts, sampling_params)
    prompt_to_output = {
        str(g.prompt): g.outputs[0].text for g in generations
    }
    outputs = [prompt_to_output[str(prompt)] if str(prompt) in prompt_to_output else "" for prompt in prompts]

    print("Calculating accuracy...")
    predictions = []
    for output, target, (item, qa_pair) in zip(outputs, targets, data_items):
        answer = extract_answer(output)
        if answer:
            predictions.append(answer)
        else:
            predictions.append("")
        
        qa_pair["model_output"] = output

        prediction_entry = {
            "image_id": item.get('image_id', ''),
            "data_type": item.get('data_type', ''),
            "question_type": item.get('question_type', ''),
            "level": item.get('level', ''),
            "task_type": item.get('task_type', ''),
            "prompt": prompt,
            "answer": target,
            "prediction": answer,
            "model_output": output,
            "QA_pair": item.get('QA_pair', ''),
        }
        predictions.append(prediction_entry)

    with open(os.path.join(args.save_dir, f"predictions.jsonl"), "w") as fout:
        for prediction in predictions:
            fout.write(json.dumps(prediction) + "\n") 

    # em_score = exact_match.compute(predictions=predictions, references=targets, ignore_case=True, ignore_punctuation=True)["exact_match"]
    # f1_score = f1.compute(predictions=f1_answer_mapping(predictions), references=f1_answer_mapping(targets), average='macro')['f1']
    # print(f"Exact match : {em_score}")
    # print(f"F1 Score : {f1_score}")

    # Save metrics (if needed)
    # with open(os.path.join(args.save_dir, "metrics.json"), "w") as fout:
    #     json.dump({
    #         "exact_match": em_score,
    #         "f1_score": f1_score
    #     }, fout, indent=4)


parser = argparse.ArgumentParser()
parser.add_argument(
    '--model_name', 
    type=str, 
    help="The HuggingFace model to be evaluated."
)
parser.add_argument(
    '--num_instances', 
    type=int, 
    default=None,
    help="Num of sampled instances for evaluation"
)
parser.add_argument(
    "--newline_stop",
    action="store_true",
    help="If given, we will use stop token (usually newline or double newline) to stop generation."
)
parser.add_argument(
    "--stop_at_double_newline",
    action="store_true",
    help="If given, will stop generation at double newline instead of single."
)
parser.add_argument(
    "--stop_at_triple_newline",
    action="store_true",
    help="If given, will stop generation at triple newline instead of single."
)
parser.add_argument(
    '--additional_stop_sequence',
    type=str,
    nargs="+",
    default=[],
    help="Additional stop sequences to use when generating completions. Useful for e.g. llama-3-instruct."
)
parser.add_argument(
    "--clm_max_length",
    type=int,
    default=256
)
parser.add_argument(
    "--eval_lang",
    type=str,
    choices=['chn', 'eng'],
    default='eng'
)
parser.add_argument(
    "--save_dir", 
    type=str
)

args = parser.parse_args()
main(args)



