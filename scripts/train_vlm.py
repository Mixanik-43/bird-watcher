import argparse
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
import importlib.metadata
import json
from pathlib import Path
import random
import time
import torch
from transformers import Trainer, TrainingArguments, set_seed
from birdlab.data import make_qa
from birdlab.vlm import load_model, QACollator, evaluate


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/cub8")
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--eval-size", type=int, default=160)
    p.add_argument("--output", default="runs/vlm")
    p.add_argument("--targets", choices=["attention", "all-linear"], default="all-linear")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--balance", action="store_true")
    p.add_argument("--no-checkpointing", action="store_true")
    args = p.parse_args()
    set_seed(42)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    train = make_qa(args.data, "train", balance=args.balance)
    val = make_qa(args.data, "val", heldout_wording=True)
    random.Random(42).shuffle(val)
    val = val[:args.eval_size]
    if not train or not val:
        raise ValueError("No QA examples")
    if args.batch_size not in (1, 2, 4, 8):
        raise ValueError("Choose a batch size dividing effective batch 8")
    model, processor = load_model(target_mode=args.targets)
    collator = QACollator(processor)
    batch = collator(train[:2])
    loss = model(**{k: v.to(model.device) for k, v in batch.items()}).loss
    assert torch.isfinite(loss)
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for n, p in model.named_parameters() if "lora_B" in n)
    model.zero_grad(set_to_none=True)
    del batch, loss
    before = evaluate(model, processor, val)
    (output / "before.json").write_text(json.dumps(before, indent=2))
    bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    options = TrainingArguments(output_dir=str(output), max_steps=args.steps,
        per_device_train_batch_size=args.batch_size, gradient_accumulation_steps=8 // args.batch_size,
        learning_rate=2e-4, logging_steps=5, save_strategy="no", report_to="none",
        remove_unused_columns=False, bf16=bf16, fp16=False,
        gradient_checkpointing=not args.no_checkpointing, gradient_checkpointing_kwargs={"use_reentrant": False},
        dataloader_num_workers=0, seed=42)
    model.config.use_cache = False
    trainer = Trainer(model=model, args=options, train_dataset=train, data_collator=collator)
    start = time.perf_counter()
    trainer.train()
    seconds = time.perf_counter()-start
    model.save_pretrained(output / "adapter")
    processor.save_pretrained(output / "adapter")
    model.config.use_cache = True
    after = evaluate(model, processor, val)
    shuffled = [dict(x) for x in val]
    if len(shuffled) > 1:
        for i, x in enumerate(shuffled):
            x["image"] = val[(i+1) % len(val)]["image"]
    shuffled_score = evaluate(model, processor, shuffled)
    report = dict(steps=args.steps, targets=args.targets, microbatch=args.batch_size, balanced_train=args.balance,
        train_pairs=len(train), validation_pairs=len(val),
        seconds=seconds, versions={k: importlib.metadata.version(k) for k in ("torch", "transformers", "peft")},
        gpu=torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu",
        peak_memory_gb=torch.cuda.max_memory_allocated()/1e9 if torch.cuda.is_available() else None,
        before=before, after=after, shuffled=shuffled_score, history=trainer.state.log_history)
    (output / "metrics.json").write_text(json.dumps(report, indent=2))
    print({k: report[k] for k in ("steps", "train_pairs", "seconds", "gpu", "peak_memory_gb")}, flush=True)
    # A saved adapter is only useful if a fresh model reproduces its answer.
    from birdlab.vlm import predict, MODEL_ID
    from transformers import Qwen3_5ForConditionalGeneration
    from peft import PeftModel
    import gc
    reference = predict(model, processor, val[0])
    device, dtype = model.device, next(model.parameters()).dtype
    del trainer, model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    base = Qwen3_5ForConditionalGeneration.from_pretrained(MODEL_ID, dtype=dtype, attn_implementation="sdpa").to(device)
    restored = PeftModel.from_pretrained(base, output / "adapter")
    assert predict(restored, processor, val[0]) == reference
    report["adapter_reload_passed"] = True
    (output / "metrics.json").write_text(json.dumps(report, indent=2))
    print("Adapter reload passed", flush=True)


if __name__ == "__main__":
    main()
