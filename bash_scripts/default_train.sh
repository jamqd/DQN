python ./main.py \
    --learning_rate 0.001 \
    --discount_factor 0.99 \
    --env_name LunarLander-v2 \
    --iterations 50000 \
    --episodes 64 \
    --batch_size 32 \
    --n_threads 0 \
    --max_replay 4 \
    --epsilon 0.9
