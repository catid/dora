import json
from pathlib import Path
from rankstudy.data import CompletionDataset,collate

class TinyTok:
    eos_token='<eos>'; pad_token_id=0
    def __call__(self,text,add_special_tokens=False):
        return {'input_ids':[ord(c)%31+1 for c in text]}

def test_completion_masks_prompt_and_keeps_eos(tmp_path):
    p=tmp_path/'x.jsonl'; p.write_text(json.dumps({'question':'Q','response':'A\u2028B','group_id':'g'})+'\n')
    d=CompletionDataset(p,TinyTok(),100); row=d[0]
    assert row['input_ids'][-1] == ord('>')%31+1
    prompt_len=len(TinyTok()("Question: Q\nAnswer: ")['input_ids'])
    assert all(x==-100 for x in row['labels'][:prompt_len])
    assert all(x!=-100 for x in row['labels'][prompt_len:])

def test_collate_does_not_cross_attention_boundaries(tmp_path):
    p=tmp_path/'x.jsonl'; p.write_text('\n'.join(json.dumps({'question':q,'response':'A','group_id':q}) for q in ('Q1','Q2'))+'\n')
    d=CompletionDataset(p,TinyTok(),100); b=collate([d[0],d[1]],0)
    assert b['attention_mask'].shape == b['input_ids'].shape
    assert int(b['attention_mask'].sum()) == sum(int(d[i]['labels'].__len__()) for i in range(2))
