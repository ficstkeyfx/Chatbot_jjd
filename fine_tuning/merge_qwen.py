"""Merge a trained adapter into its exact base revision for GGUF conversion."""
import argparse
import json
from pathlib import Path

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('run', type=Path)
    a = p.parse_args()
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    report = json.loads((a.run / 'report.json').read_text())
    if not report.get('base_revision'):
        raise ValueError('Missing exact base model revision')
    base = AutoModelForCausalLM.from_pretrained(report['base_model'], revision=report['base_revision'], torch_dtype=torch.float16, device_map='cpu', low_cpu_mem_usage=True)
    model = PeftModel.from_pretrained(base, a.run / 'adapter').merge_and_unload(safe_merge=True)
    model.save_pretrained(a.run / 'merged', safe_serialization=True)
    AutoTokenizer.from_pretrained(a.run / 'adapter').save_pretrained(a.run / 'merged')
