from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from models import DLinear
from utils.tools import EarlyStopping, adjust_learning_rate, visual, test_params_flop
from utils.metrics import metric

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import optim

import os
import time

import warnings
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

warnings.filterwarnings('ignore')

class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)
        if self.args.logdir != 'None':
            self.writer = SummaryWriter(self.args.logdir)

    def _build_model(self):
        model_dict = {
            'DLinear': DLinear,
        }
        model = model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        if self.args.loss == 'L1':
            criterion = nn.L1Loss()
        else:
            criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion,epoch,setting,flag):
        total_loss = []
        total_mape = []
        preds=[]
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(tqdm(vali_loader)):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x)

                else:
                    outputs = self.model(batch_x)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()
                preds.append(pred.numpy())
                loss = criterion(pred, true)

                if self.args.features == "M" or self.args.features == "MS":
                    _, _, _, mape, _, _, _ = metric(pred.numpy()[:,:,0], true.numpy()[:,:,0])
                else:
                    _, _, _, mape, _, _, _ = metric(pred.numpy(), true.numpy())
                
                total_mape.append(mape)                                   
                total_loss.append(loss)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(folder_path + flag + '_real_prediction.npy', np.array(preds))

        total_loss = np.average(total_loss)
        total_mape = np.average(total_mape)
        self.model.train()
        return total_loss,total_mape

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        
        if not self.args.train_only:
            vali_data, vali_loader = self._get_data(flag='val')
            test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            total_mape = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(tqdm(train_loader)):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # use_amp : mixed precision learning
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x) 

                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                        _, _, _, mape, _, _, _ = metric(outputs.detach().cpu().numpy()[:,:,0], batch_y.detach().cpu().numpy()[:,:,0])
                        total_mape.append(mape)

                else:
                    outputs = self.model(batch_x) 

                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())

                if self.args.features == "M" or self.args.features == "MS":
                    _, _, _, mape, _, _, _ = metric(outputs.detach().cpu().numpy()[:,:,0],  batch_y.detach().cpu().numpy()[:,:,0])
                else:
                    _, _, _, mape, _, _, _ = metric(outputs.detach().cpu().numpy(),  batch_y.detach().cpu().numpy())
                    
                    total_mape.append(mape)
                    
                if (i + 1) % 500 == 0:                    
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    self.writer.add_scalar('Train_loss_iter',loss,epoch*len(train_loader) + i)
                    self.writer.add_scalar('Train_mape_iter',mape,epoch*len(train_loader) + i)
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            total_mape = np.average(total_mape)
            self.writer.add_scalar('Train_loss_epoch',train_loss,epoch)
            self.writer.add_scalar('Train_mape_epoch',total_mape,epoch)

            if not self.args.train_only:
                vali_loss,vali_mape = self.vali(vali_data, vali_loader, criterion, epoch, setting, 'valid')
                test_loss,test_mape = self.vali(test_data, test_loader, criterion, epoch, setting, 'test')

                self.writer.add_scalar('Valid_loss_epoch',vali_loss,epoch)
                self.writer.add_scalar('Valid_mape_epoch',vali_mape,epoch)
                self.writer.add_scalar('Test_loss_epoch',test_loss,epoch)
                self.writer.add_scalar('Test_mape_epoch',test_mape,epoch)

                print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss : {3:.7f}".format(
                    epoch + 1, train_steps, train_loss, vali_loss, test_loss))

            else:
                print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f}".format(
                    epoch + 1, train_steps, train_loss, vali_loss))

            adjust_learning_rate(model_optim, epoch + 1, self.args)
            torch.save(self.model.state_dict(), path + '/' +f'checkpoint_{str(epoch+1).zfill(2)}.pth')
            
        self.writer.close()
        best_model_path = path + '/' + f'checkpoint_{str(30).zfill(2)}.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def predict(self, setting, load=True):
        pred_data, pred_loader = self._get_data(flag='pred')
        
        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + f'checkpoint_{str(30).zfill(2)}.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        preds = []
        start = time.time()

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(tqdm(pred_loader)):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
              

                # use_amp : mixed precision learning
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x)
                else:
                    outputs = self.model(batch_x)

                if self.args.features == "M" or self.args.features == "MS":
                    pred = outputs.detach().cpu().numpy()[:,:,0]
                else:
                    pred = outputs.detach().cpu().numpy()

                print(f"Inference Speed (Latency) : {(time.time() - start):.5f} sec")

                preds.append(pred)



        preds = np.array(preds)
        preds = np.concatenate(preds, axis=0)
        
        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(folder_path + 'real_prediction.npy', preds)
        pd.DataFrame({'date':pred_data.future_dates,'MW':preds[0]}).to_csv(folder_path + 'real_prediction.csv', index=False)


        return
