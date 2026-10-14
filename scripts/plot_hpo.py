#!/usr/bin/env python3
import argparse,json
from pathlib import Path
import matplotlib.pyplot as plt

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--input',required=True); ap.add_argument('--out',required=True); args=ap.parse_args()
 rows=json.loads(Path(args.input).read_text()); rows=sorted(rows,key=lambda x:x['best_dev_tune_nll'])
 fig,ax=plt.subplots(figsize=(11,8)); ax.barh([x['method'] for x in rows],[x['best_dev_tune_nll'] for x in rows],color='#4472c4'); ax.invert_yaxis(); ax.set_xlabel('Best 128-update dev_tune endpoint NLL (lower is better)'); ax.set_title('Accepted native HPO: one best trial per registered cell'); fig.tight_layout(); Path(args.out).parent.mkdir(parents=True,exist_ok=True); fig.savefig(args.out,dpi=160); print(args.out)
if __name__=='__main__': main()
