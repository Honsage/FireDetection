#!/bin/bash
# Загружает TFRecord файлы на ВМ по SCP

VM_USER=""
VM_IP=""
VM_PATH="~/FireDetection"

echo "=== Загружаем TFRecord на ВМ ==="
echo "Это займёт несколько минут..."

scp train.tfrecord $VM_USER@$VM_IP:$VM_PATH/
scp valid.tfrecord $VM_USER@$VM_IP:$VM_PATH/
scp test.tfrecord  $VM_USER@$VM_IP:$VM_PATH/

echo "=== Готово ==="