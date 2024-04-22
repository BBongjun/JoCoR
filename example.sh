#!/usr/bin/env bash
for noise_rate in 0.4; do
    log_file="./log/log_noise_rate${noise_rate}_epoch50.txt"
    
    # Python 스크립트 실행 및 결과를 기존 파일에 추가
    python main.py --dataset ssd --model TCN --n_epoch 50 --batch_size 256 --lr 0.001 --co_lambda 0.95 --noise_rate $noise_rate | tee $log_file
done 

# python main.py --dataset mnist --noise_type symmetric --noise_rate 0.2 --n_epoch 200 --co_lambda 0.95