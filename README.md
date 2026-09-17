# bird-watcher

Два задания на фотографиях птиц:

1. [ResNet](notebooks/01_resnet.ipynb): классификатор и цикл обучения.
2. [VLM + LoRA](notebooks/02_vlm.ipynb): LoRA-слой, токенизация и вопросы к изображению.

Заполните TODO и выполните ноутбук целиком. Решения — в [solutions](solutions/).

## Запуск

Python 3.12 и PyTorch:

```bash
pip install -e '.[dev,vlm]'
python scripts/prepare_data.py
```

Либо распакуйте подготовленный набор в `data/cub8`.
В нём 8 видов и 478 фото: 190 train, 48 validation, 240 test.

Для Kaggle включите Internet и GPU и откройте
[ноутбук запуска](notebooks/00_kaggle_validation.ipynb).
Готовый набор можно подключить через Upload input.

[Как добавить свои фотографии](docs/data.md).

Код — MIT. Условия использования изображений — в [DATA_LICENSE](DATA_LICENSE).
