if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi

model_name=DLinear

python -u run.py \
  --is_training 0 \
  --root_path ./dataset/ \
  --data_path pred.csv \
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
  --variable 5\
  --batch_size 1\
  --learning_rate 0.001 >logs/LongForecasting/$model_name'_'Pred_6048'_'72.log 
