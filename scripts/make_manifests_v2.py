#!/usr/bin/env python3
"""Create paired, method-independent effective-update manifests."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
from rankstudy.v2.data import effective_manifest, read_jsonl

def main():
    p=argparse.ArgumentParser(); p.add_argument('--data-root',type=Path,default=Path('/var/tmp/dora-adaptive-rank/data_v2'))
    p.add_argument('--out',type=Path,default=Path('artifacts/manifests_v2')); p.add_argument('--updates',type=int,default=512)
    p.add_argument('--seeds',default='1001,1002,1003,2001,2002,2003,3001,3002,3003,3004,3005')
    a=p.parse_args(); train=read_jsonl(a.data_root/'train.jsonl'); a.out.mkdir(parents=True,exist_ok=True)
    index={}
    for seed_s in a.seeds.split(','):
        seed=int(seed_s); m=effective_manifest(train,seed,a.updates); text=json.dumps(m,sort_keys=True,separators=(',',':'))+'\n'
        (a.out/f'train_seed{seed}.json').write_text(text)
        index[str(seed)]={'path':str(a.out/f'train_seed{seed}.json'),'sha256':hashlib.sha256(text.encode()).hexdigest(),'updates':a.updates,'input_tokens':m['input_tokens'],'supervised_tokens':m['supervised_tokens']}
    (a.out/'index.json').write_text(json.dumps({'protocol':'adaptive-rank-v2','data_root':str(a.data_root),'rows':len(train),'manifests':index},indent=2)+'\n')
    print(json.dumps({'seeds':len(index),'rows':len(train),'out':str(a.out),'token_totals':{k:v['input_tokens'] for k,v in index.items()}},indent=2))
if __name__=='__main__': main()
