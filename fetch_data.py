import psycopg2
import pandas as pd
import streamlit as st
import joblib

@st.cache_data
def load_data(sql_txt):
    conn = psycopg2.connect(
        dbname='apartments_for_sale',
        user='apartments_for_sale_owner',
        password='Nf5uFWt2bqrT',
        host='ep-flat-lake-a2hd1hdw.eu-central-1.aws.neon.tech',
        port=5432
    )
    cur = conn.cursor()
    cur.execute(sql_txt)
    rows = cur.fetchall()
    df = pd.DataFrame(rows, columns=[desc[0] for desc in cur.description])
    cur.close()
    conn.close()
    return df

@st.cache_resource
def load_model(path):
    return joblib.load('./model.pkl')