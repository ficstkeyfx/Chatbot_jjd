"""Single-GPU QLoRA; loss only on assistant tokens; held-out base/adapter comparison."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path


def encode(row, tokenizer, max_length):
    messages = row['messages']
    prompt = tokenizer.apply_chat_template(messages[:-1], tokenize=True, add_generation_prompt=True, return_dict=False)
    full = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False, return_dict=False)
    if full[:len(prompt)] != prompt:
        raise ValueError('Chat template prefix mismatch')
    if len(full) > max_length:
        raise ValueError(f"{row['scenario_id']} exceeds {max_length} tokens; increase --max-length")
    if len(full) <= len(prompt):
        raise ValueError('Missing assistant tokens')
    return {'input_ids': full, 'attention_mask': [1] * len(full), 'labels': [-100] * len(prompt) + full[len(prompt):]}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--max-length', type=int, default=1024)
    p.add_argument('--epochs', type=float, default=1)
    p.add_argument('--learning-rate', type=float, default=2e-5)
    p.add_argument('--allow-draft', action='store_true')
    a = p.parse_args()
    manifest = json.loads((a.data / 'manifest.json').read_text())
    if manifest.get('experimental') and not a.allow_draft:
        p.error('Draft dataset requires --allow-draft')
    for name, expected in manifest['files_sha256'].items():
        if hashlib.sha256((a.data / name).read_bytes()).hexdigest() != expected:
            raise ValueError('Dataset hash mismatch: ' + name)
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
    import torch
    from datasets import Dataset
    from huggingface_hub import HfApi
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Trainer, TrainingArguments, set_seed
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU required')
    set_seed(42)
    a.output.mkdir(parents=True, exist_ok=False)
    revision = HfApi().model_info(manifest['base_model']).sha
    tokenizer = AutoTokenizer.from_pretrained(manifest['base_model'], revision=revision)
    tokenizer.padding_side = 'right'
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    rows = {s: [json.loads(line) for line in (a.data / (s + '.jsonl')).read_text().splitlines() if line.strip()] for s in ('train', 'validation')}
    datasets = {s: Dataset.from_list([encode(r, tokenizer, a.max_length) for r in rs]) for s, rs in rows.items()}
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model = AutoModelForCausalLM.from_pretrained(manifest['base_model'], revision=revision, device_map={'': 0}, dtype=dtype, attn_implementation='sdpa', quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4', bnb_4bit_compute_dtype=dtype, bnb_4bit_use_double_quant=True))
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'], bias='none', task_type='CAUSAL_LM'))
    model.config.use_cache = False
    model.print_trainable_parameters()

    def collate(batch):
        size = max(len(r['input_ids']) for r in batch)
        return {key: torch.tensor([r[key] + [tokenizer.pad_token_id if key == 'input_ids' else (-100 if key == 'labels' else 0)] * (size - len(r[key])) for r in batch]) for key in ('input_ids', 'attention_mask', 'labels')}

    warmup_steps = math.ceil(math.ceil(len(rows['train']) / 8) * a.epochs * 0.1)
    args = TrainingArguments(output_dir=str(a.output / 'checkpoints'), num_train_epochs=a.epochs, per_device_train_batch_size=1, per_device_eval_batch_size=1, gradient_accumulation_steps=8, learning_rate=a.learning_rate, warmup_steps=warmup_steps, lr_scheduler_type='cosine', bf16=dtype == torch.bfloat16, fp16=dtype == torch.float16, gradient_checkpointing=True, optim='paged_adamw_8bit', eval_strategy='epoch', save_strategy='epoch', save_total_limit=2, load_best_model_at_end=True, metric_for_best_model='eval_loss', greater_is_better=False, logging_steps=1, report_to='none', seed=42, data_seed=42)
    trainer = Trainer(model=model, args=args, train_dataset=datasets['train'], eval_dataset=datasets['validation'], data_collator=collate, processing_class=tokenizer)

    def samples():
        model.eval()
        result = []
        for row in rows['validation']:
            ids = tokenizer.apply_chat_template(row['messages'][:-1], tokenize=True, add_generation_prompt=True, return_tensors='pt', return_dict=False).to(model.device)
            with torch.inference_mode():
                out = model.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=256, do_sample=False, use_cache=True, pad_token_id=tokenizer.pad_token_id)
            result.append({'scenario_id': row['scenario_id'], 'question': row['messages'][-2]['content'], 'reference': row['messages'][-1]['content'], 'response': tokenizer.decode(out[0, ids.shape[1]:], skip_special_tokens=True)})
        return result

    with model.disable_adapter():
        baseline = trainer.evaluate(metric_key_prefix='base')
        base_samples = samples()
    (a.output / 'base_predictions.json').write_text(json.dumps(base_samples, ensure_ascii=False, indent=2))
    train_result = trainer.train()
    metrics = trainer.evaluate()
    trainer.save_model(str(a.output / 'adapter'))
    tokenizer.save_pretrained(a.output / 'adapter')
    tuned_samples = samples()
    (a.output / 'tuned_predictions.json').write_text(json.dumps(tuned_samples, ensure_ascii=False, indent=2))
    report = {'base_model': manifest['base_model'], 'base_revision': revision, 'data_manifest': manifest, 'training_arguments': args.to_dict(), 'base_metrics': baseline, 'train_metrics': train_result.metrics, 'tuned_metrics': metrics, 'gpu': torch.cuda.get_device_name(), 'expert_approved': False, 'release_ready': False, 'note': 'Experimental draft; validation loss alone is not a deployment approval.'}
    (a.output / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
