#!/bin/bash
# Показывает текущий статус обучения

LOG_FILE="checkpoints/training.log"
CSV_FILE="checkpoints/training_log.csv"

echo "=== Статус процесса ==="
if tmux has-session -t training 2>/dev/null; then
    echo "Обучение: ЗАПУЩЕНО (tmux сессия 'training' активна)"
else
    echo "Обучение: НЕ ЗАПУЩЕНО"
fi

echo ""
echo "=== Использование ресурсов ==="
echo "CPU и RAM:"
top -bn1 | grep -E "Cpu|Mem" | head -2

echo ""
echo "=== Последние строки лога ==="
if [ -f "$LOG_FILE" ]; then
    tail -20 "$LOG_FILE"
else
    echo "Лог не найден: $LOG_FILE"
fi

echo ""
echo "=== Лучшие метрики ==="
if [ -f "$CSV_FILE" ]; then
    echo "Последние 5 эпох:"
    tail -6 "$CSV_FILE" | column -t -s ','
else
    echo "CSV лог не найден: $CSV_FILE"
fi