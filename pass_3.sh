cd /research/projects/trans_llm/Zeru_Shi

python -m Mless_release.pass_3 \
    --task Math-500 \
    --model_name google/gemma-3-4b-it \
    --batch_size 1 \
    --choose False