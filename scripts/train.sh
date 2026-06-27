#!/bin/bash
# Запускает обучение в tmux сессии
# Логи пишутся в файл — можно смотреть в реальном времени

SESSION="training"
LOG_FILE="checkpoints/training.log"

mkdir -p checkpoints

# Если сессия уже есть — убиваем
tmux kill-session -t $SESSION 2>/dev/null || true

echo "=== Запускаем обучение в tmux сессии '$SESSION' ==="

tmux new-session -d -s $SESSION \
    "source venv/bin/activate && \
     python train.py 2>&1 | tee $LOG_FILE"

echo ""
echo "Обучение запущено в фоне."
echo ""
echo "Следить за логами:"
echo "  tail -f $LOG_FILE"
echo ""
echo "Подключиться к сессии:"
echo "  tmux attach -t $SESSION"
echo ""
echo "Отключиться от сессии (обучение продолжится):"
echo "  Ctrl+B, затем D"