model="mplug1"

categories=("privacy" "bias" "toxicity" "non-exisitent" "position-swapping" "noise-injection" "legality")

for category in "${categories[@]}"
do
    save_path="results/${category}_${model}.jsonl"

    # 检查 save_path 是否已存在
    if [ -f "$save_path" ]; then
        echo "Skipping $category. File $save_path already exists."
        continue
    fi

    python evaluate.py --model {YOUR MODEL PATH} \
                    --save_path "$save_path" \
                    --data_path "data/${category}" \
                    --log_file "logs/evaluate-${category}_${model}.log" \
done