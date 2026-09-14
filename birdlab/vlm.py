"""Explicit processor/collator, generation evaluation, and PEFT setup."""
from collections import Counter, defaultdict
import torch
from .data import open_rgb

MODEL_ID = "Qwen/Qwen3.5-0.8B"


def messages(example, answer=False):
    result = [{"role": "user", "content": [
        {"type": "image", "image": open_rgb(example["image"])},
        {"type": "text", "text": example["question"]}]}]
    if answer:
        result.append({"role": "assistant", "content": [{"type": "text", "text": example["answer"]}]})
    return result


def answer_labels(input_ids, attention_mask, prompt_lengths):
    labels = input_ids.clone()
    labels[attention_mask == 0] = -100
    for i, length in enumerate(prompt_lengths):
        labels[i, :length] = -100
    if (labels != -100).sum(dim=1).min().item() == 0:
        raise ValueError("No supervised answer tokens")
    return labels


class QACollator:
    def __init__(self, processor, label_function=answer_labels):
        self.processor = processor
        self.label_function = label_function
        self.processor.tokenizer.padding_side = "right"

    def __call__(self, examples):
        common = dict(tokenize=True, return_dict=True, return_tensors="pt", enable_thinking=False)
        full = self.processor.apply_chat_template([messages(x, True) for x in examples],
                                                  processor_kwargs={"padding": True}, add_generation_prompt=False, **common)
        lengths = []
        for i, example in enumerate(examples):
            prefix = self.processor.apply_chat_template(messages(example), add_generation_prompt=True, **common)
            ids = prefix["input_ids"][0]
            # Fail loudly if the model template doesn't make generation prompt a prefix.
            if not torch.equal(full["input_ids"][i, :len(ids)], ids):
                raise ValueError("Chat template prefix mismatch; inspect the token boundary")
            lengths.append(len(ids))
        full["labels"] = self.label_function(full["input_ids"], full["attention_mask"], lengths)
        return full


def load_model(adapters=True):
    from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration
    from peft import LoraConfig, get_peft_model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported() else torch.float32
    processor = AutoProcessor.from_pretrained(MODEL_ID, min_pixels=128*128, max_pixels=256*256)
    # SDPA/PyTorch path avoids requiring a separately compiled FlashAttention package.
    model = Qwen3_5ForConditionalGeneration.from_pretrained(MODEL_ID, dtype=dtype, attn_implementation="sdpa").to(device)
    if adapters:
        targets = [name for name, module in model.named_modules()
                   if isinstance(module, torch.nn.Linear) and "language_model" in name
                   and name.rsplit(".", 1)[-1] in {"q_proj", "v_proj"}]
        if not targets:
            raise ValueError("No language attention targets found; inspect model.named_modules()")
        model = get_peft_model(model, LoraConfig(r=8, lora_alpha=16, lora_dropout=.05,
                                               target_modules=targets, bias="none", task_type="CAUSAL_LM"))
        model.print_trainable_parameters()
    return model, processor


@torch.inference_mode()
def predict(model, processor, example):
    model.eval()
    batch = processor.apply_chat_template(messages(example), tokenize=True, add_generation_prompt=True,
        enable_thinking=False, return_dict=True, return_tensors="pt").to(model.device)
    output = model.generate(**batch, max_new_tokens=8, do_sample=False)
    return processor.decode(output[0, batch["input_ids"].shape[1]:], skip_special_tokens=True).strip().lower().rstrip(".! ")


def evaluate(model, processor, examples):
    records = [dict(image_id=x["image_id"], task=x["task"], truth=x["answer"],
                    prediction=predict(model, processor, x)) for x in examples]
    by_task = defaultdict(list)
    for record in records:
        by_task[record["task"]].append(record)
    metrics = {}
    for task, rows in by_task.items():
        recalls = [sum(r["prediction"] == answer for r in rows if r["truth"] == answer) /
                   sum(r["truth"] == answer for r in rows) for answer in ("yes", "no")
                   if any(r["truth"] == answer for r in rows)]
        metrics[task] = dict(n=len(rows), counts=dict(Counter(r["truth"] for r in rows)),
                            accuracy=sum(r["prediction"] == r["truth"] for r in rows)/len(rows),
                            balanced_accuracy=sum(recalls)/len(recalls),
                            invalid=sum(r["prediction"] not in {"yes", "no"} for r in rows)/len(rows))
    return {"tasks": metrics, "predictions": records,
            "macro_balanced_accuracy": sum(m["balanced_accuracy"] for m in metrics.values()) / len(metrics) if metrics else None}
