#!/bin/bash
# Installation des dépendances pour l'étape 1 (classification YOLO)
# Lancer depuis le venv : source ~/IA_labels/env/bin/activate && bash install.sh

set -e

pip install ultralytics
# ultralytics installe automatiquement : torch, torchvision, opencv-python, etc.

echo ""
echo "Installation terminée. Vérification :"
python -c "from ultralytics import YOLO; print('ultralytics OK')"
python -c "import torch; print(f'torch {torch.__version__}, CUDA={torch.cuda.is_available()}')"
