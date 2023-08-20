import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')


class Dataset_Train(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='data2015.csv',
                 target='MW', timeenc=0, freq='5min', train_only=False, variable = 3):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 12 * 24 * 14
            self.label_len = 12 *24 * 7
            self.pred_len = 12 * 6
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]

        assert flag in ['train']

        self.features = features
        self.target = target
        self.timeenc = timeenc
        self.freq = freq
        self.train_only = train_only

        self.root_path = root_path
        self.data_path = data_path
        self.variable = variable
        self.__read_data__()

    def __read_data__(self):
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        cols = list(df_raw.columns)
        if self.features == 'S':
            cols.remove(self.target)
        cols.remove('date')
        print(cols)

        # Univariate | Multivariate
        if self.features == 'M' or self.features == 'MS':
            df_raw = df_raw[['date'] + cols]
            cols_data = df_raw.columns[1:self.variable]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_raw = df_raw[['date'] + cols + [self.target]] 
            df_data = df_raw[[self.target]] 

        data = df_data.values
        
  
        df_stamp = df_raw[['date']].iloc[: (-2016 * 3) - self.seq_len]
        df_stamp['date'] = pd.to_datetime(df_stamp.date) 

        
        df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
        df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
        df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
        df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
        df_stamp['minute'] = df_stamp.date.apply(lambda row:row.minute,1)
        df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 5)
        data_stamp = df_stamp.drop(['date'], 1).values

        self.data_x = data[:(-2016 * 3) -self.seq_len]
        self.data_y = data[:(-2016 * 3) -self.seq_len]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
       
        s_begin = index 
        s_end = s_begin + self.seq_len 

        r_begin = s_end 
        r_end = r_begin + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

class Dataset_Valid(Dataset):
    def __init__(self, root_path,flag = 'valid' ,size=None,
                 features='S', data_path='data2015.csv',
                 target='MW', timeenc=0, freq='5min', cols=None, train_only=False, variable = 3):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 12 * 24 * 14
            self.label_len = 12 *24 * 7
            self.pred_len = 12 * 6
        else:
            self.seq_len = size[0]
            self.label_len = 12 * 24 * 14
            self.pred_len = size[2]

        assert flag in ['val']

        self.features = features
        self.target = target

        self.timeenc = timeenc
        self.freq = freq
        self.cols = cols
        self.root_path = root_path
        self.data_path = data_path
        self.flag = flag
        self.variable = variable
        self.__read_data__()

    def __read_data__(self):
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        if self.cols:
            cols = self.cols.copy()
        else:
            cols = list(df_raw.columns)
            self.cols = cols.copy()
            cols.remove('date')
        if self.features == 'S':
            cols.remove(self.target)
        
        border1 = len(df_raw) - self.seq_len
        border2 = len(df_raw)

        # Univariate | Multivariate
        if self.features == 'M' or self.features == 'MS':
            df_raw = df_raw[['date'] + cols]
            cols_data = df_raw.columns[1:self.variable]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_raw = df_raw[['date'] + cols + [self.target]]
            df_data = df_raw[[self.target]]

        data = df_data.values

        tmp_stamp = df_raw[['date']][(-2016 * 3) - self.seq_len : -2016]
        tmp_stamp['date'] = pd.to_datetime(tmp_stamp.date)

        pred_dates = pd.date_range(tmp_stamp.date.values[self.seq_len], periods=tmp_stamp.shape[0]- self.seq_len, freq=self.freq)

        df_stamp = pd.DataFrame(columns=['date'])
        df_stamp.date = list(tmp_stamp.date.values)
        self.future_dates = list(pred_dates)
        
        df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
        df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
        df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
        df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
        df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
        df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 5)
        data_stamp = df_stamp.drop(['date'], 1).values



        self.data_x = data[(-2016 * 3) - self.seq_len : -2016]
        self.data_y = data[(-2016 * 3) - self.seq_len : -2016]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index * self.pred_len 
        s_end = s_begin + self.seq_len
        r_begin = s_end
        r_end = r_begin + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return self.label_len // self.pred_len

class Dataset_Test(Dataset):
    def __init__(self, root_path,flag = 'test' ,size=None,
                 features='S', data_path='data2015.csv',
                 target='MW', timeenc=0, freq='5min', cols=None, train_only=False, variable = 3):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 12 * 24 * 14
            self.label_len = 12 *24 * 7
            self.pred_len = 12 * 6
        else:
            self.seq_len = size[0]
            self.label_len = 12 *24 * 7
            self.pred_len = size[2]

        assert flag in ['test']

        self.features = features
        self.target = target

        self.timeenc = timeenc
        self.freq = freq
        self.cols = cols
        self.root_path = root_path
        self.data_path = data_path
        self.flag = flag
        self.variable = variable
        self.__read_data__()

    def __read_data__(self):
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        if self.cols:
            cols = self.cols.copy()
        else:
            cols = list(df_raw.columns)
            self.cols = cols.copy()
            cols.remove('date')
        if self.features == 'S':
            cols.remove(self.target)
        
        border1 = len(df_raw) - self.seq_len
        border2 = len(df_raw)

        # Univariate | Multivariate
        if self.features == 'M' or self.features == 'MS':
            df_raw = df_raw[['date'] + cols]
            cols_data = df_raw.columns[1:self.variable]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_raw = df_raw[['date'] + cols + [self.target]]
            df_data = df_raw[[self.target]]

        data = df_data.values

        tmp_stamp = df_raw[['date']][-2016 - self.seq_len:]
        tmp_stamp['date'] = pd.to_datetime(tmp_stamp.date)

        pred_dates = pd.date_range(tmp_stamp.date.values[self.seq_len], periods=tmp_stamp.shape[0]- self.seq_len, freq=self.freq)

        df_stamp = pd.DataFrame(columns=['date'])
        df_stamp.date = list(tmp_stamp.date.values)
        self.future_dates = list(pred_dates)
        
        df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
        df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
        df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
        df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
        df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
        df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 5)
        data_stamp = df_stamp.drop(['date'], 1).values



        self.data_x = data[-2016 - self.seq_len:]
        self.data_y = data[-2016 - self.seq_len:]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index * self.pred_len 
        s_end = s_begin + self.seq_len
        r_begin = s_end
        r_end = r_begin + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return self.label_len // self.pred_len

class Dataset_Pred(Dataset):
    def __init__(self, root_path, flag='pred', size=None,
                 features='S', data_path='pred.csv',
                 target='MW', timeenc=0, freq='5min', cols=None, train_only=False,variable = 3):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]

        assert flag in ['pred']

        self.features = features
        self.target = target
        self.timeenc = timeenc
        self.freq = freq
        self.cols = cols
        self.root_path = root_path
        self.data_path = data_path
        self.variable = variable
        self.__read_data__()

    def __read_data__(self):
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        if self.cols:
            cols = self.cols.copy()
        else:
            cols = list(df_raw.columns)
            self.cols = cols.copy()
            cols.remove('date')
        if self.features == 'S':
            cols.remove(self.target)
        border1 = len(df_raw) - self.seq_len
        border2 = len(df_raw)

        # Univariate | Multivariate
        if self.features == 'M' or self.features == 'MS':
            df_raw = df_raw[['date'] + cols]
            cols_data = df_raw.columns[1:self.variable]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_raw = df_raw[['date'] + cols + [self.target]]
            df_data = df_raw[[self.target]]


        data = df_data.values

        tmp_stamp = df_raw[['date']][border1:border2]
        tmp_stamp['date'] = pd.to_datetime(tmp_stamp.date)
        pred_dates = pd.date_range(tmp_stamp.date.values[-1], periods=self.pred_len + 1, freq=self.freq)

        df_stamp = pd.DataFrame(columns=['date'])
        df_stamp.date = list(tmp_stamp.date.values) + list(pred_dates[1:])
        self.future_dates = list(pred_dates[1:])

        df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
        df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
        df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
        df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
        df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
        df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
        data_stamp = df_stamp.drop(['date'], 1).values

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_begin + self.label_len]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len + 1
