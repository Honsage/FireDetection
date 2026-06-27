#!/bin/bash

set -e  # останавливаемся при любой ошибке

echo "=== Обновляем систему ==="
sudo apt-get update -q
sudo apt-get install -y git tmux htop tree python3-pip python3-venv

echo "=== Клонируем репозиторий ==="
git clone https://github.com/Honsage/FireDetection.git
cd FireDetection

echo "=== Создаём виртуальное окружение ==="
python3 -m venv venv
source venv/bin/activate

echo "=== Устанавливаем зависимости ==="
pip install --upgrade pip
pip install -r requirements.txt

echo "=== Готово ==="
echo "Следующий шаг: загрузка TFRecord файлов и обучение"