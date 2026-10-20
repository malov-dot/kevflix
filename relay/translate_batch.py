"""Bounded CPU translation process; exit releases model and loader memory."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--model',required=True)
    parser.add_argument('--input',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--threads',type=int,default=2)
    parser.add_argument('--ct2-model',default='')
    args=parser.parse_args()
    if not 1<=args.threads<=8:raise ValueError('Invalid CPU translation thread limit')
    os.environ.update(CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
    import torch
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    from .language import LanguageTools
    from .media import atomic_json
    rows=json.loads(Path(args.input).read_text(encoding='utf-8'))
    if not isinstance(rows,list) or len(rows)>100 or any(len(str(r.get('text','')))>2000 for r in rows):
        raise ValueError('Translation batch exceeds a bounded movie window')
    tools=LanguageTools({'translation_model':args.model,'translation_ct2_model':args.ct2_model,'cpu_threads':args.threads},'cpu')
    results=[]
    for row in rows:
        english,basis=tools.english(row['text'],row['language'])
        results.append({'id':row['id'],'english':english,'translation':basis})
    atomic_json(args.output,results)


if __name__=='__main__':main()
