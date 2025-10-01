cd /research/projects/trans_llm/Zeru_Shi

python -m Mless_release.arm \
    --task Math-500 \
    --model_name Qwen/Qwen2.5-Math-7B \
    --batch_size 1 \