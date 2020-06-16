python ./main.py \
    --n_threads 0 \
    --decay 0.995 \
    --gd_optimizer ADAM \
    --max_replay 250000 \
    --batch_size 128 \
    --learning_rate 0.001 \
    --epsilon 0.995 \
    --discount_factor 0.99 \
    --save_model_every 15
