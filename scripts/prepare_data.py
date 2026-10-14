#!/usr/bin/env python3
"""Prepare grouped MetaMath pools and sealed GSM8K/MATH-500 evaluation manifests."""
from __future__ import annotations
import argparse,hashlib,json,re,random
from pathlib import Path
import pyarrow.parquet as pq
from transformers import AutoTokenizer


def norm(s):
    # Preserve numerals; only normalize text layout and harmless punctuation spacing.
    return re.sub(r'\s+',' ',str(s)).strip().casefold()

def key(s): return hashlib.sha256(norm(s).encode()).hexdigest()

def load_json(path): return json.loads(Path(path).read_text())
def load_parquet(path): return pq.read_table(path).to_pylist()

def dsu_groups(rows):
    parent=list(range(len(rows))); idx={}
    def find(x):
        while parent[x]!=x: parent[x]=parent[parent[x]]; x=parent[x]
        return x
    def union(a,b):
        a,b=find(a),find(b)
        if a!=b: parent[b]=a
    for i,row in enumerate(rows):
        fields=[row.get('original_question',''),row.get('query','')]
        for s in fields:
            if not s: continue
            k=key(s)
            if k in idx: union(i,idx[k])
            else: idx[k]=i
    groups={}
    for i in range(len(rows)): groups.setdefault(find(i),[]).append(i)
    return list(groups.values())

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--data-root',default='/var/tmp/dora-adaptive-rank/data'); ap.add_argument('--model',default='/var/tmp/dora-adaptive-rank/models/Qwen2.5-Math-1.5B'); ap.add_argument('--out',default='artifacts/data_manifest.json'); ap.add_argument('--seed',type=int,default=20261014); ap.add_argument('--train-groups',type=int,default=20000); ap.add_argument('--dev-groups',type=int,default=512); ap.add_argument('--allow-available',action='store_true',help='run with all eligible groups when the requested pool is unavailable; records a blocker')
    a=ap.parse_args(); root=Path(a.data_root); out=Path(a.out); out.parent.mkdir(parents=True,exist_ok=True)
    tok=AutoTokenizer.from_pretrained(a.model); eos=tok.eos_token or ''
    rows=load_json(root/'meta-math_MetaMathQA/MetaMathQA-395K.json')
    gsm_test=load_parquet(root/'openai_gsm8k/main/test-00000-of-00001.parquet')
    math_test=[json.loads(x) for x in (root/'HuggingFaceH4_MATH-500/test.jsonl').read_text().splitlines()]
    heldout_keys={key(x['question']) for x in gsm_test}|{key(x['problem']) for x in math_test}
    groups=dsu_groups(rows)
    records=[]; excluded={'overlap':0,'overlength':0,'empty':0}
    for gi,inds in enumerate(groups):
        # Deterministic representative: shortest query, then source index.
        inds=sorted(inds,key=lambda i:(len(str(rows[i].get('query',''))),i)); i=inds[0]; r=rows[i]
        q=str(r.get('original_question') or r.get('query') or '').strip(); resp=str(r.get('response') or '').strip()
        if not q or not resp: excluded['empty']+=1; continue
        if key(q) in heldout_keys or key(r.get('query','')) in heldout_keys: excluded['overlap']+=1; continue
        text=f'Question: {q}\nAnswer: {resp}{eos}'
        ids=tok(text,add_special_tokens=False)['input_ids']
        if len(ids)>2048: excluded['overlength']+=1; continue
        records.append({'group_id':f'g{gi:06d}','source_index':i,'source_type':r.get('type'),'question':q,'response':resp,'input_tokens':len(ids),'text_sha256':hashlib.sha256(text.encode()).hexdigest(),'duplicate_count':len(inds)})
    rng=random.Random(a.seed); rng.shuffle(records)
    dev_tune=records[:a.dev_groups]; dev_audit=records[a.dev_groups:2*a.dev_groups]; train=records[2*a.dev_groups:2*a.dev_groups+a.train_groups]
    blocker=None
    if len(train)<a.train_groups:
        blocker={'kind':'insufficient_unique_training_groups','requested':a.train_groups,'available':len(train),'reason':'MetaMathQA contains 13,928 connected original-question groups after the required duplicate grouping; 20,000 disjoint groups cannot be formed from this pinned revision.'}
        if not a.allow_available: raise RuntimeError(json.dumps(blocker))
    # Save raw rows separately so training can stream exact IDs and never infer splits.
    for name,data in [('dev_tune',dev_tune),('dev_audit',dev_audit),('train',train)]:
        p=out.parent/'data'/f'{name}.jsonl'; p.parent.mkdir(parents=True,exist_ok=True); p.write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in data))
    evals={'gsm8k_test': [{'id':i,'question':x['question'],'answer':x['answer']} for i,x in enumerate(gsm_test)], 'math500_test':math_test}
    (out.parent/'data'/'eval_sealed.json').write_text(json.dumps(evals,ensure_ascii=False,indent=2)+'\n')
    manifest={'schema':'adaptive-rank-data-v1','seed':a.seed,'model_tokenizer':a.model,'max_sequence_length':2048,'template':'Question: {question}\\nAnswer: {response}<EOS>','counts':{'all_rows':len(rows),'connected_groups':len(groups),'eligible_groups':len(records),'train':len(train),'dev_tune':len(dev_tune),'dev_audit':len(dev_audit),'gsm8k_test':len(gsm_test),'math500_test':len(math_test)},'excluded':excluded,'blocker':blocker,'overlap_keys_sha256':hashlib.sha256((''.join(sorted(heldout_keys))).encode()).hexdigest(),'split_group_sha256':{n:hashlib.sha256((''.join(x['group_id'] for x in d)).encode()).hexdigest() for n,d in [('train',train),('dev_tune',dev_tune),('dev_audit',dev_audit)]},'token_stats':{n:{'tokens':sum(x['input_tokens'] for x in d),'mean':sum(x['input_tokens'] for x in d)/len(d)} for n,d in [('train',train),('dev_tune',dev_tune),('dev_audit',dev_audit)]}}
    out.write_text(json.dumps(manifest,indent=2)+'\n'); print(json.dumps(manifest,indent=2))
if __name__=='__main__': main()
