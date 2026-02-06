
python ../train.py \
    --model bertweet-base \
    --dataset ../dataset/twitter2017 \
    --log_dir ../logs \
    --device cuda:1\
    --seed 115 \
    --epochs 8
