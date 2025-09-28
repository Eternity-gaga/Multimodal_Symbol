import json
import re
from collections import defaultdict

def extract_model_answer(model_output):
    match = re.search(r'\\boxed\{(.*?)\}', model_output)
    if match:
        return match.group(1)
    return ""

def extract_standard_answer(QA_pairs):
    for qa in QA_pairs:
        answer_zh = qa.get('answer_zh', '')
        
        # Convert to string if it's not already
        if isinstance(answer_zh, list):
            answer_zh = ''.join(map(str, answer_zh))  # Convert list elements to strings and join them
        elif not isinstance(answer_zh, str):
            answer_zh = str(answer_zh)  # Convert other types to string
        
        answer_zh = answer_zh.strip()
        
        if not answer_zh:
            solution_zh = qa.get('solution_zh', '')
            # Extract the first \boxed{} content from solution_zh
            match = re.search(r'\\boxed\{(.*?)\}', solution_zh)
            if match:
                answer_zh = match.group(1)
        
        if answer_zh:
            return f"\\boxed{{{answer_zh}}}"
    
    return ""

def calculate_accuracy(data):
    total_correct = 0
    total_entries = 0
    
    level_task_acc = defaultdict(lambda: {'correct': 0, 'total': 0})
    
    processed_data = []
    
    for entry in data:
        model_output = entry.get('model_output', '')
        if not model_output:
            print(f"Warning: Missing 'model_output' field in entry with image_id: {entry.get('image_id', 'unknown')}")
            continue
        
        standard_answer = extract_standard_answer(entry['QA_pair'])
        
        model_answer = extract_model_answer(model_output)
        
        entry['model_answer'] = model_answer
        
        if model_answer:
            total_entries += 1
            level_task_key = (entry['level'], entry['task_type'])
            
            level_task_acc[level_task_key]['total'] += 1
            
            if model_answer == standard_answer:
                total_correct += 1
                level_task_acc[level_task_key]['correct'] += 1
                
        processed_data.append(entry)
    
    overall_accuracy = total_correct / total_entries if total_entries > 0 else 0
    
    level_task_accuracy = {}
    for key, counts in level_task_acc.items():
        level_task_accuracy[f"{key[0]}_{key[1]}"] = counts['correct'] / counts['total']
    
    return processed_data, overall_accuracy, level_task_accuracy

def main(input_file='input_PATH', output_file='output.jsonl', result_file='results.json'):
    data = []
    with open(input_file, 'r', encoding='utf-8') as infile:
        for line_number, line in enumerate(infile, start=1):
            try:
                entry = json.loads(line.strip())
                if isinstance(entry, dict):  # Ensure that the entry is a dictionary
                    data.append(entry)
                else:
                    print(f"Error: Entry on line {line_number} is not a valid JSON object.")
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON on line {line_number}: {e}")
    
    processed_data, overall_accuracy, level_task_accuracy = calculate_accuracy(data)
    
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for entry in processed_data:
            outfile.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    results = {
        'overall_accuracy': overall_accuracy,
        'level_task_accuracy': level_task_accuracy
    }
    
    with open(result_file, 'w', encoding='utf-8') as resfile:
        json.dump(results, resfile, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    main()
