"""Generate exercises and solutions from the same cells. No notebook tooling needed."""
import json
from pathlib import Path
import textwrap

ROOT = Path(__file__).resolve().parents[1]


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": textwrap.dedent(text).strip()}


def code(text):
    return {"cell_type": "code", "metadata": {}, "source": textwrap.dedent(text).strip(),
            "execution_count": None, "outputs": []}


def exercise(text, solution, stub):
    return (md(text), code(solution), code(stub))


SETUP = '''
from pathlib import Path
import os, sys
# Start in repository root or its notebooks/solutions directory.
ROOT = Path.cwd()
if ROOT.name in {"notebooks", "solutions"}:
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))
DATA = Path(os.environ.get("BIRD_DATA", str(ROOT / "data/cub8")))
assert (DATA / "manifest.jsonl").exists(), "Run scripts/prepare_data.py first"
import torch
from torch import nn
torch.manual_seed(42)
torch.set_num_threads(4)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(DEVICE)
'''

model_source = (ROOT / "birdlab/models.py").read_text()
classifier = model_source[model_source.index("class BirdClassifier"):model_source.index("class LoRALinear")]
lora = model_source[model_source.index("class LoRALinear"):]
training = (ROOT / "birdlab/training.py").read_text().split("def run_epoch", 1)[1]
vlm_source = (ROOT / "birdlab/vlm.py").read_text()
mask = "def answer_labels" + vlm_source.split("def answer_labels", 1)[1].split("class QACollator", 1)[0]

resnet = [
md('''# 1. От фиксированных признаков к обучению ResNet
Цель: почувствовать, что модель — обычный `nn.Module`, состояние и градиенты которого
можно контролировать. Метрики и разбиение считаем знакомыми.

Пройдите TODO, затем Restart & Run All. Проверки диагностируют ошибки кода;
accuracy не обязана совпасть до последнего знака. Подсказки — в `birdlab/`,
полное решение — в `solutions/01_resnet.ipynb`.
'''), code(SETUP),
code('''import json
from torch.utils.data import DataLoader
from birdlab.data import Birds, transforms, open_rgb
import matplotlib.pyplot as plt
classes = json.loads((DATA / "classes.json").read_text())
train_data = Birds(DATA, "train", transforms(True))
val_data = Birds(DATA, "val")
fig, axes = plt.subplots(2, 4, figsize=(12, 6))
for ax, row in zip(axes.flat, train_data.rows[::max(1, len(train_data)//8)]):
    ax.imshow(open_rgb(DATA / row["image"]))
    ax.set_title(row["species"]); ax.axis("off")
plt.show()
'''),
exercise('''## TODO 1 — свой модуль
Создайте `BirdClassifier`: ResNet-18 без финального fc и отдельная `Linear(512, K)`.
`set_stage('head')` замораживает backbone; `last_block` размораживает layer4.
Переопределите `train`: у замороженных BatchNorm running statistics не меняются.

Математика: $z=f_\\theta(x)$, $\\ell=Wz+b$. Какие параметры меняются в каждом режиме?
Чем `requires_grad=False` отличается от `eval()`?
''', 'from torchvision.models import resnet18, ResNet18_Weights\n' + classifier,
'''from torchvision.models import resnet18, ResNet18_Weights
class BirdClassifier(nn.Module):
    def __init__(self, n_classes, pretrained=True):
        super().__init__()
        raise NotImplementedError("TODO: backbone, head, initial stage")
    def set_stage(self, stage):
        raise NotImplementedError("TODO: select trainable parameters")
    def train(self, mode=True):
        raise NotImplementedError("TODO: preserve frozen BatchNorm state")
    def forward(self, images):
        raise NotImplementedError("TODO: features -> logits")
'''),
code('''probe = BirdClassifier(len(classes), pretrained=False)
probe.train()
old_mean = probe.backbone.bn1.running_mean.clone()
logits = probe(torch.randn(2, 3, 64, 64))
assert logits.shape == (2, len(classes))
logits.sum().backward()
assert probe.head.weight.grad is not None
assert probe.backbone.conv1.weight.grad is None
torch.testing.assert_close(old_mean, probe.backbone.bn1.running_mean)
probe.set_stage("last_block")
assert probe.backbone.layer4[0].conv1.weight.requires_grad
del probe
'''),
exercise('''## TODO 2 — одна эпоха
Реализуйте `run_epoch(model, loader, device, optimizer=None)`. При наличии optimizer
обучаем, иначе оцениваем. Возвращаем средний по изображениям loss и accuracy.
Перед `cross_entropy` softmax не нужен. Почему?
''', 'def run_epoch' + training,
'''def run_epoch(model, loader, device, optimizer=None):
    raise NotImplementedError("TODO: forward, loss, backward, step, weighted metrics")
'''),
code('''tiny = nn.Linear(4, 2)
loader = [(torch.randn(8, 4), torch.zeros(8, dtype=torch.long))]
opt = torch.optim.SGD(tiny.parameters(), lr=.1)
before = run_epoch(tiny, loader, "cpu")["loss"]
for _ in range(20):
    run_epoch(tiny, loader, "cpu", opt)
assert run_epoch(tiny, loader, "cpu")["loss"] < before
'''),
md('''## Эксперимент: голова и последний блок
Сначала 3 эпохи головы, затем 5 эпох последнего блока. Сохраняем лучший validation.
Сравните кривые; при желании замените голову на MLP и повторите с тем же seed.
'''),
code('''model = BirdClassifier(len(classes)).to(DEVICE)
loaders = {"train": DataLoader(train_data, batch_size=32, shuffle=True),
           "val": DataLoader(val_data, batch_size=32)}
optimizer = torch.optim.AdamW(model.head.parameters(), lr=1e-3)
history, best = [], -1
RUN = ROOT / "runs/notebook_resnet"
RUN.mkdir(parents=True, exist_ok=True)
for epoch in range(8):
    if epoch == 3:
        model.set_stage("last_block")
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
    train_score = run_epoch(model, loaders["train"], DEVICE, optimizer)
    val_score = run_epoch(model, loaders["val"], DEVICE)
    history.append((train_score["accuracy"], val_score["accuracy"]))
    print(epoch, train_score, val_score)
    if val_score["accuracy"] > best:
        best = val_score["accuracy"]
        torch.save({"state_dict": model.state_dict(), "classes": classes}, RUN / "model.pt")
plt.plot(history); plt.legend(["train", "validation"]); plt.show()
'''),
md('''## Сохранение и применение
Загрузите лучший checkpoint в новый экземпляр. Объясните, почему вместе с весами
нужно сохранять порядок классов. После всех решений оцените на test один раз.
'''), code('''checkpoint = torch.load(RUN / "model.pt", map_location=DEVICE, weights_only=True)
restored = BirdClassifier(len(classes), pretrained=False).to(DEVICE)
restored.load_state_dict(checkpoint["state_dict"])
test_loader = DataLoader(Birds(DATA, "test"), batch_size=32)
print(run_epoch(restored, test_loader, DEVICE))
'''),
md('''## Бонус: кэш признаков
В режиме `head` вычислите и сохраните $z=f(x)$ через `torch.no_grad()`. Обучите на них
линейную голову и MLP. Измерьте скорость. Почему новый случайный crop на каждой эпохе
уже нельзя получить из одного вектора? Как это меняет сравнение с online-обучением?
''')]

vlm = [md('''# 2. LoRA и вопросы к изображению
Задача: ответить yes/no на разные вопросы о конкретной фотографии. Название вида не
подаём во вход. Атрибуты вида не подставляем вместо атрибутов изображения.

Сначала собственный модуль и математика на CPU, затем processor, маска loss и LoRA
на Qwen3.5-0.8B. Для GPU запуска используйте одну карту: CUDA_VISIBLE_DEVICES=0.
'''), code(SETUP),
exercise('''## TODO 1 — LoRALinear
$y=W_0x+b+(\\alpha/r)BAx$, где $A\\in R^{r\\times d_{in}}$, $B\\in R^{d_{out}\\times r}$.
Заморозьте base, инициализируйте A случайно, B нулями. Напишите forward и merged().

До запуска предскажите: какой градиент на первом шаге равен нулю? Почему обе матрицы
нельзя занулить? Сколько параметров при d_in=d_out=1024 и r=8?
''', lora,
'''class LoRALinear(nn.Module):
    def __init__(self, base, rank=4, alpha=4):
        super().__init__()
        raise NotImplementedError("TODO: frozen base, A, B, scaling")
    def forward(self, x):
        raise NotImplementedError("TODO: low-rank update")
    def merged(self):
        raise NotImplementedError("TODO: return equivalent nn.Linear")
'''), code('''base = nn.Linear(5, 3)
adapter = LoRALinear(base, rank=2)
x = torch.randn(4, 5)
torch.testing.assert_close(adapter(x), base(x))
adapter(x).square().sum().backward()
assert adapter.A.grad.abs().sum() == 0
assert adapter.B.grad.abs().sum() > 0
assert base.weight.grad is None
with torch.no_grad():
    adapter.B.add_(.1)
torch.testing.assert_close(adapter(x), adapter.merged()(x))
'''),
code('''from birdlab.data import make_qa
from birdlab.vlm import load_model, QACollator, evaluate, messages
train_qa = make_qa(DATA, "train", balance=True)
val_qa = make_qa(DATA, "val", heldout_wording=True)
print(len(train_qa), len(val_qa), train_qa[0])
model, processor = load_model()
'''),
md('''## Processor и токенизация
Рассмотрите IDs, токены и обратное декодирование. Совпадает ли число токенов с числом слов?
Посмотрите `pixel_values` и `image_grid_thw`. Почему картинку не обрабатывает tokenizer?
'''), code('''for text in ["yes", "no", "house sparrow", "домовый воробей"]:
    ids = processor.tokenizer.encode(text, add_special_tokens=False)
    print(text, ids, processor.tokenizer.convert_ids_to_tokens(ids))
example = train_qa[0]
print(processor.apply_chat_template(messages(example, True), tokenize=False, enable_thinking=False))
'''),
exercise('''## TODO 2 — маска loss
$L=-\\sum_t m_t\\log p(y_t|I,q,y_{<t})/\\sum_t m_t$.
Верните копию input_ids; prompt и padding замените на -100. EOS ответа сохраняется.
Используйте attention_mask, а не равенство pad_token_id: pad и EOS могут совпадать.
Collator проверит границы prompt на настоящем chat template; не задавайте длину вручную.
''', mask,
'''def answer_labels(input_ids, attention_mask, prompt_lengths):
    raise NotImplementedError("TODO: mask prompt and padding, preserve answer/EOS")
'''),
code('''ids = torch.tensor([[1, 2, 7, 9, 9], [1, 2, 3, 8, 9]])
attention = torch.tensor([[1, 1, 1, 1, 0], [1, 1, 1, 1, 1]])
assert answer_labels(ids, attention, [2, 3]).tolist() == [[-100,-100,7,9,-100],[-100,-100,-100,8,9]]
collator = QACollator(processor, label_function=answer_labels)
batch = collator(train_qa[:2])
for key, value in batch.items():
    print(key, tuple(value.shape))
for labels in batch["labels"]:
    print("Loss on:", processor.decode(labels[labels != -100]))
loss = model(**{k:v.to(DEVICE) for k,v in batch.items()}).loss
loss.backward()
assert any(p.grad is not None and p.grad.abs().sum() > 0 for n,p in model.named_parameters() if "lora_B" in n)
model.zero_grad(set_to_none=True)
del batch, loss
'''),
md('''## До и после LoRA
Используем новые формулировки вопросов на validation. Сначала исходная модель (нулевые
LoRA-обновления), потом 100 шагов обучения. Ограниченный evaluation — только быстрый пилот.
Для окончательной оценки используйте все вопросы и отдельный test.
'''), code('''import random
random.Random(42).shuffle(val_qa)
evaluation = val_qa[:int(os.environ.get("BIRD_EVAL_SIZE", "24"))]
before = evaluate(model, processor, evaluation)
print(before["tasks"])
from transformers import Trainer, TrainingArguments
RUN = ROOT / "runs/notebook_vlm"
args = TrainingArguments(output_dir=str(RUN), max_steps=int(os.environ.get("BIRD_STEPS", "50")), learning_rate=2e-4,
    per_device_train_batch_size=1, gradient_accumulation_steps=8,
    remove_unused_columns=False, report_to="none", save_strategy="no", logging_steps=5,
    bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
    gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant":False})
model.config.use_cache = False
trainer = Trainer(model=model, args=args, train_dataset=train_qa, data_collator=collator)
trainer.train()
model.config.use_cache = True
after = evaluate(model, processor, evaluation)
print(after["tasks"])
model.save_pretrained(RUN / "adapter")
processor.save_pretrained(RUN / "adapter")
'''),
md('''## Эксперименты и выводы
1. Для каждого вопроса сравните с большинством ответов **в train**, а не test.
2. Перемешайте изображения: падает ли balanced accuracy? Учтите, что часть перемешанных
   фото может иметь тот же правильный ответ.
3. Сравните r=2 и r=8 при одинаковом числе шагов и данных.
4. Почему уменьшение teacher-forced loss не гарантирует правильную генерацию?
5. Каких вопросов модель не умеет решать? Не путайте новые формулировки с новыми задачами.
6. Бонус: добавьте вопросы о видимости частей из parts/part_locs.txt; не выводите
   видимость из отрицательной метки цвета.
'''),
code('''from birdlab.vlm import predict, MODEL_ID
from transformers import Qwen3_5ForConditionalGeneration
from peft import PeftModel
# Save a reference before releasing GPU memory; reload exactly the saved adapter.
reference = predict(model, processor, evaluation[0])
dtype = next(model.parameters()).dtype
del trainer, model
import gc
gc.collect()
if torch.cuda.is_available(): torch.cuda.empty_cache()
base = Qwen3_5ForConditionalGeneration.from_pretrained(MODEL_ID, dtype=dtype, attn_implementation="sdpa").to(DEVICE)
restored = PeftModel.from_pretrained(base, RUN / "adapter")
assert predict(restored, processor, evaluation[0]) == reference
print("Adapter reload OK:", reference)
''')]


def write(name, cells, solved):
    result = []
    for cell in cells:
        if isinstance(cell, tuple):
            result.extend([cell[0], cell[1 if solved else 2]])
        else:
            result.append(cell)
    for i, cell in enumerate(result):
        cell["id"] = f"cell-{i:03d}"
    notebook = dict(cells=result, metadata={"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                    "language_info": {"name": "python", "version": "3.12"}}, nbformat=4, nbformat_minor=5)
    path = ROOT / ("solutions" if solved else "notebooks") / name
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")


for name, cells in [("01_resnet.ipynb", resnet), ("02_vlm.ipynb", vlm)]:
    for solved in [False, True]:
        write(name, cells, solved)

write("00_kaggle_validation.ipynb", [md('''# bird-watcher: GPU validation
Включите Internet и GPU T4 x2. Этот ноутбук запускает эталон, упражнения находятся
в 01_resnet и 02_vlm. Запуск скачивает официальный CUB и веса Qwen.
'''), code('''import os, subprocess, sys
from pathlib import Path
assert Path("/kaggle/working").is_dir(), "This setup cell is for Kaggle only"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
repo = Path("/kaggle/working/bird-watcher")
if not repo.exists():
    subprocess.run(["git", "clone", "https://github.com/Mixanik-43/bird-watcher.git", str(repo)], check=True)
os.chdir(repo)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-e", ".[vlm]", "pytest>=8"], check=True)
# Kaggle currently bundles torchao 0.10; PEFT 0.20 rejects it even for plain LoRA.
# No quantization is used here. Only remove it in this disposable Kaggle environment.
subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "torchao"], check=True)
'''), code('''import torch
assert torch.cuda.is_available(), "Enable a Kaggle GPU"
print(torch.__version__, torch.cuda.get_device_name(), torch.cuda.get_device_properties(0).total_memory / 1e9)
subprocess.run([sys.executable, "-m", "pytest", "-q"], check=True)
if not Path("data/cub8/manifest.jsonl").exists():
    import shutil
    attached = list(Path("/kaggle/input").rglob("manifest.jsonl"))
    if len(attached) == 1:
        shutil.copytree(attached[0].parent, "data/cub8", dirs_exist_ok=True)
    else:
        subprocess.run([sys.executable, "scripts/prepare_data.py"], check=True)
'''), code('''subprocess.run([sys.executable, "scripts/train_resnet.py", "--epochs", "8"], check=True)
'''), code('''subprocess.run([sys.executable, "scripts/train_vlm.py", "--steps", "50", "--eval-size", "24"], check=True)
''')], False)
