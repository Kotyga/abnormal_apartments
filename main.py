import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import matplotlib.pyplot as plt
from phik import phik_matrix
import joblib
import streamlit.components.v1 as components
import time

df = pd.read_csv('./data/rooms.csv')
data = df[:]
data = data.rename(columns={'Cost': 'Стоимость',
                            'Longitude': 'Долгота',
                            'Latitude': 'Широта',
                            'TotalArea': 'Общая площадь',
                            'Floor': 'Этаж',
                            'room_type': 'Тип комнаты',
                            'room_count': 'Количество комнат',
                            'metro_min': 'Расстояние до метро',
                            'admin_name': 'Название района',
                            'cost_type': 'Тип стоимости'})
pipeline_loaded = joblib.load('./model.pkl')

st.title('Aномальные предложения: изучение рынка недвижимости')
st.text('''
    Этот проект использует математические методы машинного обучения для обнаружения 
необычных объявлений о продаже квартир в Санкт-Петербурге. 
Проведен анализ данных, использованы различные методы анализа, 
такой подход улучшит процесс выявления аномалий в объявлениях 
и поможет предотвратить мошенничество на рынке недвижимости.

Посмотрим на пример данных.
''')

with st.expander('Показать данные'):
    st.dataframe(data.head(10))

st.header('Визуальный анализ')
st.text('На графике показано распределение объявлений по районам Санкт-Петербурга')
# -------------------------------------------------------------
labels = df['admin_name'].value_counts().index.tolist()
values = df['admin_name'].value_counts().values.tolist()

fig = px.bar(x=labels, y=values)
fig.update_xaxes(tickangle=90)

st.plotly_chart(fig)
# -------------------------------------------------------------
st.text('''
Взглянем на то, что предлагает рынок: какой тип недвижимости?
Заметим, что больше всего квартир, а как квартира распределена по комнатам?''')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 5))
fig.subplots_adjust(wspace=0)

overall_ratios = df['room_type'].value_counts() / df['room_type'].value_counts().sum()
labels = ['Квартира', 'Квартира-студия', 'Апартаменты-студия']
explode = [0.1, 0, 0]
angle = -180 * overall_ratios[0]
wedges, *_ = ax1.pie(overall_ratios, autopct='%1.1f%%', startangle=angle,
                     labels=labels, explode=explode)

age_ratios = df[df['room_type']==1].groupby('room_count').count()['Cost'] / df[df['room_type']==1].groupby('room_count').count()['Cost'].sum()
age_labels = ['1 комната', '2 комната', '3 комната', '4 комната']
bottom = 1
width = .2

for j, (height, label) in enumerate(reversed([*zip(age_ratios, age_labels)])):
    bottom -= height
    bc = ax2.bar(0, height, width, bottom=bottom, color='C0', label=label,
                 alpha=0.1 + 0.25 * j)
    ax2.bar_label(bc, labels=[f"{height:.0%}"], label_type='center')

ax2.set_title('Количество комнат в квартире')
ax2.legend()
ax2.axis('off')
ax2.set_xlim(- 2.5 * width, 2.5 * width)

st.pyplot(fig)
# -------------------------------------------------------------
st.text('''
У нас есть информация по ценовому сегменту. 
Больше всего высокого. Посмотрим, что в него входит по типу жилья.''')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
fig.subplots_adjust(wspace=0)

overall_ratios = df['cost_type'].value_counts()[:-1] / df['cost_type'].value_counts().sum()
labels = df['cost_type'].value_counts()[:-1].index
explode = [0.1, 0, 0, 0]

angle = -180 * overall_ratios[0]
wedges, *_ = ax1.pie(overall_ratios, autopct='%1.1f%%', startangle=angle,
                     labels=labels, explode=explode)

age_ratios = df[df['cost_type']=="Высокий сегмент"].groupby('room_count').count()['Cost'] / df[df['cost_type']=="Высокий сегмент"].groupby('room_count').count()['Cost'].sum()
age_labels = ['Квартира', 'Квартира-студия', 'Апартаменты-студия']
bottom = 2
width = .9

for j, (height, label) in enumerate(reversed([*zip(age_ratios, age_labels)])):
    bottom -= height
    bc = ax2.bar(0, height, width, bottom=bottom, color='C0', label=label,
                 alpha=0.5 + 0.25 * j)
    ax2.bar_label(bc, labels=[f"{height:.0%}"], label_type='center')

ax2.set_title('Тип жилья')
ax2.legend(loc='upper right')
ax2.axis('off')
ax2.set_xlim(- 2.5 * width, 2.5 * width)

st.pyplot(fig)
# -------------------------------------------------------------
st.text('''
Интересно, как по районам распределены ценовые сегменты''')
pivot_tab = df.groupby(['admin_name', 'cost_type']).size().reset_index(name='Количество')
pivot_tab = pivot_tab.pivot(index='admin_name', columns='cost_type', values='Количество')
pivot_tab = pivot_tab.fillna(0)
df_norm = pivot_tab.div(pivot_tab.sum(axis=1), axis=0)
df_norm = df_norm.reset_index()
df_norm = df_norm.sort_values(['Высокий сегмент', 'Премиум сегмент','Средний сегмент','Элитный сегмент','Низкий сегмент'], ascending=False)

fig = px.bar(df_norm, x='admin_name', y=['Высокий сегмент', 'Премиум сегмент', 'Средний сегмент', 'Элитный сегмент', 'Низкий сегмент'],
            barmode='stack')
fig.update_layout(
    xaxis=dict(tickangle=90),
    yaxis=dict(range=[0, 1.5]),
    xaxis_title='Районы',
    yaxis_title='Нормализованное количество квартир'
)

st.plotly_chart(fig)
# -------------------------------------------------------------
st.text('''
Интересно, есть ли взаимосвязь между данными?''')
corr_mat = data.phik_matrix().stack().reset_index(name="correlation")
fig = px.scatter(corr_mat, x="level_0", y="level_1", color="correlation",
                 size="correlation", color_continuous_scale="RdBu",
                 width=500, height=400)
fig.update_layout(xaxis=dict(tickangle=90),
                  yaxis=dict(tickangle=0),
                  xaxis_title="", yaxis_title="",
                  showlegend=False)
st.plotly_chart(fig)
# -------------------------------------------------------------
st.text('''
Посмотрим на квартиры на карте. 
При наведении на координату вывеедтся флаг с данными:''')

st.json(
    {
        'Outlier': {
            1: 'Проблем не обнаружено',
           -1: 'Подозрительная недвижимость'
        },
        'room_type': {
            -1:'Апартаменты-студия',
             0:'Квартира-студия',
             1:'Квартира'
        }
    }
)

with open("map_city.html", "r") as f:
    map_html = f.read()

# Отображение карты
components.html(map_html, height=600)
# -------------------------------------------------------------
with st.sidebar:
    st.title('Окно предсказаний')
    Cost=st.number_input(
    "Стоимость недвижимости:", min_value=df['Cost'].min().astype(int),
    max_value=df['Cost'].max().astype(int), value=df['Cost'].mean().astype(int), step=1_000_000,
    )
    TotalArea=st.slider(
    "Общая площадь:", min_value=df['TotalArea'].min().astype(int),
    max_value=df['TotalArea'].max().astype(int), value=df['TotalArea'].mean().astype(int), step=10,
    )
    Floor=st.slider(
    "Этаж:", min_value=df['Floor'].min().astype(int),
    max_value=df['Floor'].max().astype(int), value=df['Floor'].mean().astype(int), step=1,
    )
    room_type=st.radio(
    'Тип недвижимости:',
    sorted(df['room_type'].unique()))

    room_count=st.radio(
    'Количество комнат:',
    sorted(df['room_count'].unique()))

    metro_min=st.radio(
    'Сколько минут до метро?',
    sorted(df['metro_min'].unique()))
    if st.button('Показать прогноз'):
        with st.spinner('Ожидайте...'):
            time.sleep(1)
        p = pipeline_loaded.predict(pd.DataFrame({'Cost':[Cost], 
                                                'TotalArea':[TotalArea],
                                                'Floor':[Floor], 
                                                'room_type':[room_type], 
                                                'room_count':[room_count], 
                                                'metro_min':[metro_min]}))[0]
        if p == 1:
            st.success('Странностей не обнаружено')
        else:
            st.error('Подозрительное предложение!')
