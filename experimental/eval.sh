MODEL_NAME=Qwen-2.5-vl-instruct

python baseline_test.py \
    --save_dir results-multimath/$MODEL_NAME-2048 \
    --model_name /home/kjy/.cache/modelscope/hub/Qwen/Qwen2.5-VL-7B-Instruct \
    --clm_max_length 2048 \
    --additional_stop_sequence "<|im_end|>"