"""Prepare traceable experimental SFT data without changing expert review status."""
import argparse
import ast
import hashlib
import json
from pathlib import Path


def prepare(root, output, allow_draft=False):
    if not allow_draft:
        raise ValueError('Current source is draft data; experimental use requires --allow-draft.')
    paths = [root / 'fine_tuning/scenarios.jsonl', root / 'fine_tuning/initial_responses.jsonl']
    scenarios, responses = [[json.loads(line) for line in p.read_text(encoding='utf-8').splitlines() if line.strip()] for p in paths]
    answers = {r['scenario_id']: r['initial_chatbot_response'] for r in responses}
    if len(answers) != len(responses):
        raise ValueError('Duplicate response IDs')
    tree = ast.parse((root / 'core.py').read_text(encoding='utf-8-sig'))
    system = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'DEFAULT_SYSTEM_PROMPT' for t in n.targets)).strip()
    excluded = {'mf-005', 'mf-006', 'mf-007'}
    splits = {'train': [], 'validation': []}
    seen_ids, seen_questions = set(), set()
    for row in scenarios:
        sid = row['scenario_id']
        if sid in seen_ids:
            raise ValueError('Duplicate scenario ID: ' + sid)
        seen_ids.add(sid)
        if sid in excluded:
            continue
        question = row['draft_user_message'].strip()
        normalized = ' '.join(question.casefold().split())
        if normalized in seen_questions:
            raise ValueError('Duplicate question: ' + sid)
        seen_questions.add(normalized)
        answer = answers[sid].strip()
        if not question or not answer:
            raise ValueError('Empty conversation: ' + sid)
        splits[row['dataset_split']].append({'scenario_id': sid, 'review_status': 'draft', 'messages': [{'role': 'system', 'content': system}, {'role': 'user', 'content': question}, {'role': 'assistant', 'content': answer}]})
    output.mkdir(parents=True, exist_ok=True)
    for split, rows in splits.items():
        (output / (split + '.jsonl')).write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows), encoding='utf-8')
    manifest = {'base_model': 'Qwen/Qwen2.5-3B-Instruct', 'experimental': True, 'expert_approved': False, 'excluded_action_ids': sorted(excluded), 'sources_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}, 'split_ids': {s: [r['scenario_id'] for r in rows] for s, rows in splits.items()}}
    manifest['files_sha256'] = {s + '.jsonl': hashlib.sha256((output / (s + '.jsonl')).read_bytes()).hexdigest() for s in splits}
    (output / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    return {s: len(rows) for s, rows in splits.items()}


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--allow-draft', action='store_true')
    a = p.parse_args()
    print(prepare(Path(__file__).resolve().parents[1], a.output, a.allow_draft))
