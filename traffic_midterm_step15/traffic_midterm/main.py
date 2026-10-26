#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path
from traffic.core import ROOT, analyze, read_json
from traffic.llm import Ollama, ModelError, DEFAULT_MODEL
from traffic.evaluate import run, review_metrics

def main():
    parser=argparse.ArgumentParser(description='免費本機車禍主張驗證研究原型')
    sub=parser.add_subparsers(dest='command',required=True)
    app=sub.add_parser('serve');app.add_argument('--port',type=int,default=8765)
    demo=sub.add_parser('demo');demo.add_argument('--input',type=Path);demo.add_argument('--output',type=Path)
    status=sub.add_parser('check-model');status.add_argument('--model',default=DEFAULT_MODEL);status.add_argument('--generate', action='store_true', help='Run a small local JSON generation test')
    ev=sub.add_parser('evaluate');ev.add_argument('--backend',choices=['offline','ollama'],default='offline');ev.add_argument('--model',default=DEFAULT_MODEL);ev.add_argument('--split',choices=['dev','test'],default='test');ev.add_argument('--gold',type=Path);ev.add_argument('--output',type=Path,default=ROOT/'results/local_run')
    ex=sub.add_parser('extract');ex.add_argument('--input',type=Path,required=True);ex.add_argument('--output',type=Path,required=True);ex.add_argument('--model',default=DEFAULT_MODEL)
    review=sub.add_parser('review-metrics');review.add_argument('input',type=Path)
    args=parser.parse_args()
    try:
        if args.command=='serve':
            from traffic.server import serve
            serve(args.port);return
        if args.command=='check-model':
            client = Ollama(args.model)
            result = client.smoke_test() if args.generate else client.status()
        elif args.command=='evaluate':result=run(args.output,args.backend,args.model,args.split,args.gold)
        elif args.command=='review-metrics':result=review_metrics(args.input)
        elif args.command=='extract':
            from traffic.extraction import extract_case
            payload=json.loads(args.input.read_text(encoding='utf8'))
            if not isinstance(payload,dict) or not isinstance(payload.get('description'),str):
                raise ValueError('擷取輸入必須是含description文字的JSON物件')
            client=Ollama(args.model);status=client.status()
            result=extract_case(payload['description'],client);result['model_status']=status
        else:
            payload=json.loads(args.input.read_text(encoding='utf8')) if args.input else read_json('demo.json')
            result=analyze(payload)
        text=json.dumps(result,ensure_ascii=False,indent=2)
        if args.command in ('demo','extract') and args.output:
            args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(text+'\n',encoding='utf8')
        else:print(text)
    except (OSError,ValueError,ModelError) as exc:
        print(str(exc),file=sys.stderr);sys.exit(1)
    if args.command=='evaluate' and result['metadata']['llm_experiment_status']=='INCOMPLETE':sys.exit(2)

if __name__=='__main__':main()
