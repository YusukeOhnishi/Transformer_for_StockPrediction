import streamlit as st
import datetime


def sidebar_content():
    stock_code = st.sidebar.text_input('Stock Code', '9984.T')
    start_date = st.sidebar.date_input(
        'Start Date', datetime.datetime.now()-datetime.timedelta(days=1000))
    end_date = st.sidebar.date_input('End Date', datetime.datetime.now())

    return stock_code, start_date, end_date
