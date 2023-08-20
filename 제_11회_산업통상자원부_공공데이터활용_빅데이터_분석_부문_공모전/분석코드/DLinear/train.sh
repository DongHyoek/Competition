if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi

model_name=DLinear

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path data2015.csv \
  --model_id 2015_multi_hum_6048'_'72 \
  --model $model_name \
  --data Train \
  --features 'M' \
  --seq_len 6048 \
  --pred_len 72 \
  --loss 'L1'\
  --des 'Exp' \
  --moving_avg 11\
  --enc_in 1\
  --use_gpu\
  --logdir './tensorboard/2015_multi_hum_6048_72'\
  --variable 5\
  --batch_size 16\
  --learning_rate 0.001 >logs/LongForecasting/$model_name'_'Train_6048'_'72.log 


