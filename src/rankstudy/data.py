from __future__ import annotations
import json,random
from pathlib import Path
import torch
from torch.utils.data import Dataset

class CompletionDataset(Dataset):
    def __init__(self,path,tokenizer,max_length=2048):
        # U+2028/U+2029 can occur in mathematical prose; split only on the
        # record newline emitted by prepare_data.
        self.rows=[json.loads(x) for x in Path(path).read_text().split('\n') if x]; self.tok=tokenizer; self.max_length=max_length
    def __len__(self): return len(self.rows)
    def __getitem__(self,i):
        r=self.rows[i]; prompt=f"Question: {r['question']}\nAnswer: "; full=prompt+r['response']+(self.tok.eos_token or '')
        p=self.tok(prompt,add_special_tokens=False)['input_ids']; f=self.tok(full,add_special_tokens=False) ['input_ids'][:self.max_length]
        labels=[-100]*min(len(p),len(f))+f[len(p):]
        return {'input_ids':f,'labels':labels,'group_id':r['group_id']}


def collate(batch,pad):
    n=max(len(x['input_ids']) for x in batch); ids=[]; labels=[]; mask=[]
    for x in batch:
        k=n-len(x['input_ids']); ids.append(x['input_ids']+[pad]*k); labels.append(x['labels']+[-100]*k); mask.append([1]*len(x['input_ids'])+[0]*k)
    return {k:torch.tensor(v) for k,v in [('input_ids',ids),('labels',labels),('attention_mask',mask)]}


def stream_batches(ds,seed,batch_size=4):
    rng=random.Random(seed); order=list(range(len(ds))); rng.shuffle(order)
    for i in range(0,len(order),batch_size): yield collate([ds[j] for j in order[i:i+batch_size]],ds.tok.pad_token_id)
