from data_provider.data_loader import Dataset_Train,Dataset_Valid ,Dataset_Test,Dataset_Pred
from torch.utils.data import DataLoader

data_dict = {
    'Train': Dataset_Train,
    'Val' : Dataset_Valid,
    'Test' : Dataset_Test,
    'Pred' : Dataset_Pred
}


def data_provider(args, flag):
    Data = data_dict[args.data]
    train_only = args.train_only

    if flag == 'val':
        shuffle_flag = False
        drop_last = False
        batch_size = 1
        freq = args.freq
        Data = Dataset_Valid
    elif flag == 'pred':
        shuffle_flag = False
        drop_last = False
        batch_size = 1
        freq = args.freq
        Data = Dataset_Pred

    elif flag == 'test':
        shuffle_flag = False
        drop_last = False
        batch_size = 1
        freq = args.freq
        Data = Dataset_Test
    else:
        shuffle_flag = True
        drop_last = False
        batch_size = args.batch_size
        freq = args.freq
        Data = Dataset_Train
    

    data_set = Data(
        root_path=args.root_path,
        data_path=args.data_path,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        target=args.target,
        freq=freq,
        train_only=args.train_only,
        variable = args.variable
    )

    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last)
    return data_set, data_loader
