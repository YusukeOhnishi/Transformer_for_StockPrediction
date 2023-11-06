import numpy as np
import torch


def data_Normalization(df):
    mean_list = df.mean()
    std_list = df.std()
    df = (df-mean_list)/std_list
    return df, mean_list, std_list


def create_dataset(data_norm, observation_period_num, predict_period_num, train_rate, device):
    inout_data = []
    for i in range(len(data_norm)-observation_period_num-predict_period_num):
        data = data_norm[i:i+observation_period_num]
        label = data_norm[i+predict_period_num:i +
                          observation_period_num+predict_period_num]
        inout_data.append((data, label))
    inout_data = torch.FloatTensor(inout_data)

    train_data = inout_data[:int(
        np.shape(inout_data)[0]*train_rate)].to(device)
    valid_data = inout_data[int(
        np.shape(inout_data)[0]*train_rate):].to(device)
    return train_data, valid_data


def get_batch(source, i, batch_size, observation_period_num):
    seq_len = min(batch_size, len(source)-1-i)
    data = source[i:i+seq_len]
    input = torch.stack(torch.stack(
        [item[0] for item in data]).chunk(observation_period_num, 1))
    target = torch.stack(torch.stack(
        [item[1] for item in data]).chunk(observation_period_num, 1))

    return input, target
