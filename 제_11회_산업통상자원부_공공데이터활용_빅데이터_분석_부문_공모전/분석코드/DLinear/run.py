import argparse
import os
import torch
from exp.exp_main import Exp_Main
import random
import numpy as np

if __name__ == "__main__":
    # seed fix
    fix_seed = 2021
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    parser = argparse.ArgumentParser(description='Autoformer & Transformer family for Time Series Forecasting')

    # basic config
    parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
    parser.add_argument('--train_only', type=bool, required=False, default=False, help='perform training on full input dataset without validation and testing')
    parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
    parser.add_argument('--model', type=str, required=True, default='Autoformer',
                        help='model name, options: [Autoformer, Informer, Transformer]')

    # data loader
    parser.add_argument('--data', type=str, required=True, default='Train', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./dataset', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='data2015.csv', help='data file')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='MW', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='5min',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
    parser.add_argument('--logdir', type=str, default='None', help='location of model checkpoints')
    parser.add_argument('--variable', type=int, default = 3 , help=' 기온: 3, 풍속 : 4, 습도 : 5, 일사: 6, 일조:7  use indexing multi variable')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=2016, help='start token length')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')


    # DLinear
    parser.add_argument('--individual', action='store_true', default=False, help='DLinear: a linear layer for each variate(channel) individually')
    parser.add_argument('--enc_in', type=int, default=1, help='encoder input size') # DLinear with --individual, use this hyperparameter as the number of channels
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')


    # optimization
    parser.add_argument('--num_workers', type=int, default=2, help='data loader num workers')
    parser.add_argument('--train_epochs', type=int, default=30, help='train epochs') # 30 epoch doing
    parser.add_argument('--batch_size', type=int, default=16, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, default='mse', help='loss function')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

    # GPU
    parser.add_argument('--use_gpu', action = 'store_true', default=False, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')
    parser.add_argument('--test_flop', action='store_true', default=False, help='See utils/tools for usage')

    args = parser.parse_args()

    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    if args.use_gpu or args.use_multi_gpu:
        args.dvices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]
    
    print('Args in experiment:')
    print(args)
    print(f'Usage Mutlivariate : {args.variable - 1}')


    Exp = Exp_Main

    if args.is_training:
        # setting record of experiments
        setting = '{}_{}_{}_{}'.format(args.model_id,args.model,args.data,args.loss)   

        exp = Exp(args)  # set experiments
        print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))

        # Doing Train, Valid, Test
        exp.train(setting)

        torch.cuda.empty_cache()

    else:
        setting = '{}_{}_{}_{}'.format(args.model_id,args.model,args.data,args.loss)

        exp = Exp(args)  # set experiments

        exp.predict(setting)

        torch.cuda.empty_cache()
