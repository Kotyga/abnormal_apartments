# Базовый образ с Python и pip
FROM python:3.8-slim

# Создаем директорию "app" внутри контейнера
RUN mkdir /app

# Копируем requirements.txt в контейнер
COPY requirements.txt /app/requirements.txt

# Устанавливаем зависимости из requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Копируем все файлы приложения в папку "app"
COPY . /app

# Указываем рабочую директорию
WORKDIR /app

# Объявляем порт, который будет использоваться приложением Streamlit
EXPOSE 8501

# Запускаем приложение Streamlit
CMD ["streamlit", "run", "main.py"]