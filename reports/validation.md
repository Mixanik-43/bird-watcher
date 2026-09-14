# Validation log

## Completed locally, 2026-09-14

- Official CUB archive downloaded; source metadata parsed successfully.
- Subset: 478 photos, 8 species. Train 190, validation 48, official test 240.
- No exact duplicate image files in this subset.
- Excluded two candidate questions after inspecting **train**: needle-shaped bill
  and yellow breast have no confident positive training examples. Four questions remain.
- Core tests: LoRA initialization, first-step gradients, merged weights, frozen
  ResNet BatchNorm state, selective unfreezing, EOS-safe answer loss mask passed.
- All five notebook files pass nbformat validation and Python cell compilation.
- ResNet solution notebook executed successfully from a fresh kernel on CPU
  (including its learner checks, training, checkpoint reload, and test evaluation).
- Actual Qwen3.5-0.8B processor tested on two CUB examples. Chat-template prefix
  assertion passes. Supervised labels decode to answer plus im_end, excluding prompt.
- Command-line ResNet prediction tested with the saved checkpoint on a held-out
  Gadwall image: Gadwall 0.9581, Mallard 0.0252, Bank Swallow 0.0075.
- ResNet reference trained for 8 epochs on CPU: 53.4 seconds for training/validation/test,
  excluding initial weights download. Validation accuracy 47/48 = 97.92%;
  test accuracy 214/240 = 89.17%. Small sample; not a claim about camera-trap images.
- CPU environment: PyTorch 2.14.0+cpu, torchvision 0.29.0+cpu,
  Transformers 5.17.0, PEFT 0.20.0, Python 3.12.

## Completed on Kaggle

- Tesla T4 allocated, 15.636 GB visible VRAM, PyTorch 2.10.0+cu128.
- Four tests passed on Kaggle (6.14 seconds).
- Official archive download returns HTTP 403 from Kaggle; a locally prepared
  ZIP was attached as a private input dataset. No competition was created.
- ResNet 8-epoch GPU run completed: 16.09 seconds, 0.402 GB peak allocated memory;
  validation accuracy 47/48, official test subset accuracy 214/240.
- Qwen attention-only LoRA: 50 steps, 422 balanced training pairs, 746.16 seconds,
  3.096 GB peak allocated GPU memory. Actual environment PEFT **0.19.1** (not local 0.20.0).
  On the 24-question pilot: before 15/24, after 15/24, shuffled images 12/24.
  All answers parse. This does NOT show improved quality after fine-tuning.
- Published VLM solution notebook passes in a **fresh Kaggle kernel**, using
  BIRD_STEPS=2 and BIRD_EVAL_SIZE=4: custom LoRA checks, processor and loss masking,
  real model backward, training, generation, saving and fresh adapter reload all pass.
  This smoke run tests notebook execution; the separate 50-step reference run tests training.

- Broader language-layer LoRA completed: 5,411,328 trainable parameters (0.6304%),
  rank 8, alpha 16, 50 steps, microbatch 4, effective batch 8, no gradient checkpointing.
  Training took 668.08 seconds; peak allocated GPU memory 5.370 GB. This is training
  time only: full autoregressive evaluation adds several minutes per pass on T4.
- All **157 validation questions** evaluated with held-out wording. Results:

| Variant | Correct | Accuracy | Macro balanced accuracy |
|---|---:|---:|---:|
| Original Qwen | 84/157 | 53.50% | 59.49% |
| Language-wide LoRA | 104/157 | 66.24% | 68.48% |
| Same adapter, shuffled images | 76/157 | 48.41% | 46.96% |
| Constant per-question train majority | 102/157 | 64.97% | 50.00% |

- All generated answers parsed. A newly loaded adapter reproduced its reference answer.
- The quality gain is useful for the exercise, but not uniform: black-wing accuracy
  slightly decreased. Most improvement was in conical bill recognition. Overall accuracy
  only narrowly exceeds the train-majority baseline, while balanced accuracy improves.
- These are one-seed validation results on 48 photos; questions sharing a photo are not
  independent. They do not establish significance or deployment performance. The VLM
  official test remains unused during configuration selection.
- The final notebooks use language-wide LoRA by default. BIRD_TARGETS=attention restores
  the small pilot configuration's target selection for comparison.
- The **final language-wide LoRA solution notebook** was fetched from published GitHub
  commit `36e7d09` and successfully executed in another fresh Kaggle kernel with
  BIRD_STEPS=2 and BIRD_EVAL_SIZE=4, including save/reload. The full 50-step recipe
  was validated by the reference script above; the final notebook's smoke uses the
  same target selection and microbatch, with fewer steps/questions.

Qwen3.5 mixes ordinary attention and linear-attention blocks. Selecting only q_proj/v_proj
does not adapt all language blocks. The narrow pilot trained 319,488 parameters; the broader
variant trained 5,411,328. This comparison also changed microbatch/checkpointing, so it is
not a controlled causal estimate of the target-module choice alone.

The VLM numbers above are validation diagnostics, not final held-out test performance.
