# Свои фотографии

Добавьте JPEG в набор и строку в `manifest.jsonl`:

```json
{"image_id":"camera-001","image":"train/house_sparrow/001.jpg","label":4,"species":"house_sparrow","split":"train","attributes":{}}
```

`label` — индекс вида в `classes.json`, `split` — `train`, `val` или `test`.
Кадры одной серии оставляйте в одной части. Новый вид добавляйте в конец
`classes.json`; при изменении числа видов переобучите классификатор.

Для VLM добавьте известные ответы в `attributes`, например:

```json
{"has_bill_shape::cone":"yes","has_breast_color::white":"no"}
```

Неизвестные ответы пропускайте. Все вопросы — в `birdlab/data.py:QUESTIONS`.
Путь к другому набору задаётся через `BIRD_DATA` в ноутбуке или `--data` в скрипте.
