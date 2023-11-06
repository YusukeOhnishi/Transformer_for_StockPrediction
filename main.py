import warnings
from src.train import train
from src.model import TransformerModel, EarlyStopping
from src.data_create import data_Normalization, create_dataset
from sidebar import sidebar_content
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import yfinance as yfin
from pandas_datareader import data as wb
yfin.pdr_override()


warnings.simplefilter('ignore')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# dataset_parameters
observation_period_num = 30
predict_period_num = 5
train_rate = 0.7
# train_parameters
lr = 0.00005
epochs=10
# epochs = 2
patience = 3
batch_size = 64


st.title('Stock Prediction App')
stock_code, start_date, end_date = sidebar_content()

try:
    df = wb.DataReader(stock_code, start=start_date,
                       end=end_date, progress=False)
    if len(df) == 0:
        raise Exception
    st.dataframe(df, height=200)
    st.line_chart(df.iloc[:, -2])
    data = df.iloc[:, -2]
    data_norm, data_mean, data_std = data_Normalization(data)

    if st.button("predict"):
        train_data, valid_data = create_dataset(data_norm=data_norm,
                                                observation_period_num=observation_period_num,
                                                predict_period_num=predict_period_num,
                                                train_rate=train_rate,
                                                device=device)
        model = TransformerModel().to(device)
        model.load_state_dict(torch.load(
            './data/model_weight.pth', map_location="cpu"))
        criterion = nn.MSELoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)
        earlystopping = EarlyStopping(patience)

        progress_text = "Train"
        my_bar = st.progress(0, text=progress_text)

        for percent_complete in range(epochs):
            model, total_loss_valid = train(
                model, train_data, valid_data, optimizer, criterion, scheduler, batch_size, observation_period_num)
            earlystopping((total_loss_valid), model)
            if earlystopping.early_stop:
                my_bar.progress(100, text=progress_text)
                break
            my_bar.progress((percent_complete + 1) *
                            (100//epochs), text=progress_text)

        model.eval()
        with torch.no_grad():
            data_norm = data_norm[-30:].values
            data_norm = data_norm.reshape(-1, 1)
            result = model(torch.FloatTensor(data_norm))[-1].view(-1)
            result = result*data_std+data_mean
        old_data_length = 20
        show_data = data[-old_data_length:].values
        show_result = result[-1*predict_period_num:].to(
            'cpu').detach().numpy().copy().astype(np.int64)
        show_result = np.concatenate([show_data[-1:], show_result])

        show_index_tmp = data[-old_data_length:].index.values
        show_index_1 = []
        for i in range(len(show_index_tmp)):
            show_index_1.append(str(show_index_tmp[i])[:10])
        show_index_2 = [show_index_1[-1]]
        for i in range(1, predict_period_num+1):
            show_index_2.append("after "+str(i)+" days")

        plt.figure(figsize=(10, 5))
        plt.grid(color='b', linestyle=':', linewidth=0.3)
        plt.xticks(rotation=90)
        plt.xlabel('Date')
        plt.ylabel('Stock Price')
        plt.plot(show_index_1, show_data, color='black', label="data")
        plt.plot(show_index_2, show_result, "--", color='red', label="pred")
        plt.legend()

        st.pyplot(plt)

except:
    st.write('no data!')
    st.write('Plese modeify the sidebar.')
