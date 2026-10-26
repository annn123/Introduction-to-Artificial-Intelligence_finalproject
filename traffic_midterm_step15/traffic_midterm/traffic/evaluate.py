from __future__ import annotations
import collections
import csv
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from .core import LABELS, ROOT, Retriever, analyze, read_json
from .llm import Ollama, ModelError, classify, DEFAULT_MODEL, PROMPT_VERSION


def classification_metrics(gold, predicted):
    if len(gold)!=len(predicted): raise ValueError('Prediction count mismatch')
    if any(x not in LABELS for x in gold+predicted): raise ValueError('Unknown label')
    per={}
    for label in LABELS:
        tp=sum(g==p==label for g,p in zip(gold,predicted))
        fp=sum(g!=label and p==label for g,p in zip(gold,predicted))
        fn=sum(g==label and p!=label for g,p in zip(gold,predicted))
        precision=tp/(tp+fp) if tp+fp else 0.0
        recall=tp/(tp+fn) if tp+fn else 0.0
        per[label]={'support':sum(g==label for g in gold),'precision':precision,'recall':recall,'f1':2*precision*recall/(precision+recall) if precision+recall else 0.0}
    present=[v['f1'] for v in per.values() if v['support']]
    return {'n_claims':len(gold),'accuracy':sum(g==p for g,p in zip(gold,predicted))/len(gold) if gold else None,
            'macro_f1_present_gold_labels':sum(present)/len(present) if present else None,
            'macro_f1_all_five_labels':sum(v['f1'] for v in per.values())/len(LABELS),'per_label':per}


def retrieval_metrics(relevant, retrieved, k):
    relevant=set(relevant)
    if not relevant:return {'precision_at_k':None,'recall_at_k':None}
    hits=len(relevant.intersection(retrieved[:k]))
    return {'precision_at_k':hits/k,'recall_at_k':hits/len(relevant)}


def load_reference(path, cases, retriever):
    """Load supplied reference annotations without treating AI adjudication as human gold."""
    reference=json.loads(Path(path).read_text(encoding='utf8'))
    if not isinstance(reference,dict) or not isinstance(reference.get('cases'),list):
        raise ValueError('Reference annotations must contain a cases array')
    entries=reference['cases']
    if any(not isinstance(e,dict) or not isinstance(e.get('id'),str) for e in entries):
        raise ValueError('Invalid reference case')
    if len({e['id'] for e in entries})!=len(entries):
        raise ValueError('Duplicate reference case ID')
    expected=reference.get('cases_sha256')
    actual=hashlib.sha256((ROOT/'data/cases.json').read_bytes()).hexdigest()
    if expected and expected!=actual:
        raise ValueError('Reference annotations do not match the current cases.json')
    if reference.get('corpus_sha256') and reference['corpus_sha256']!=retriever.fingerprint:
        raise ValueError('Reference annotations do not match the current corpus.json')
    source_ids={d['id'] for d in retriever.documents}
    for case in cases:
        entry=next((e for e in entries if e['id']==case['id']),None)
        if not entry or not entry.get('reviewer_1') or not entry.get('reviewer_2') or not entry.get('adjudicator'):
            raise ValueError(f'{case["id"]}: reference annotation incomplete')
        labels=entry.get('labels')
        if not isinstance(labels,dict) or set(labels)!={c['id'] for c in case['claims']}:
            raise ValueError(f'{case["id"]}: reference claim IDs do not match')
        if any(label not in LABELS for label in labels.values()):
            raise ValueError('Invalid reference label')
        relevant=entry.get('relevant_document_ids')
        if relevant is not None and (not isinstance(relevant,list) or any(not isinstance(v,str) or v not in source_ids for v in relevant)):
            raise ValueError(f'{case["id"]}: invalid relevant source ID')
    return reference


def run(output, backend='offline', model=DEFAULT_MODEL, split='test', gold_path=None):
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    cases=[r for r in read_json('cases.json') if r['split']==split]
    retriever=Retriever()
    human=load_reference(gold_path,cases,retriever) if gold_path else None
    assisted=bool(human and human.get('annotation_type')=='TWO_INITIAL_REVIEWERS_AI_ASSISTED_USER_ADOPTED')
    annotation_type=('TWO_INITIAL_REVIEWERS_AI_ASSISTED_USER_ADOPTED' if assisted else 'HUMAN_REVIEWED') if human else 'AI_SILVER_SYNTHETIC_NOT_HUMAN_GOLD'
    metadata={'timestamp_utc':datetime.now(timezone.utc).isoformat(),'python':sys.version,'platform':platform.platform(),
              'backend':backend,'prompt_version':PROMPT_VERSION if backend=='ollama' else None,'model':model if backend=='ollama' else None,'corpus_sha256':retriever.fingerprint,
              'cases_sha256':hashlib.sha256((ROOT/'data/cases.json').read_bytes()).hexdigest(),
              'split':split,'annotation':annotation_type,
              'warning':('Small synthetic dataset. Initial labels from two reviewers; disagreements and source relevance finalized with AI assistance and adopted by the user. Not independent human-consensus gold. Some scenarios overlap development.' if assisted else 'Human-reviewed reference supplied; provenance must be assessed separately.' if human else 'Silver results are engineering smoke checks, not evidence of research effectiveness.'),
              'evaluation_version':'step12-20261024',
              'reference_sha256':hashlib.sha256(Path(gold_path).read_bytes()).hexdigest() if gold_path else None,
              'reference_provenance':human.get('provenance') if human else None,
              'verification_decision_policy':'rag_verify final labels are determined by structured evidence rules; not an isolated test of LLM reasoning',
              'retrieval_query_policy':'same description plus fixed claim texts as the RAG classification input',
              'retrieval_empty_relevance_policy':'queries with no relevant documents for a document kind are excluded from that kind mean',
              'llm_experiment_status':'NOT_RUN' if backend=='offline' else 'RUNNING',
              'extraction_evaluation':'NOT_RUN: fixed atomic claims supplied to every condition',
              'unmeasured':['hallucination_rate','unsupported_claim_rate','citation_entailment_correctness','extraction_F1','mediation_grounding_error_rate']}
    if backend=='ollama':
        metadata['ollama']=Ollama(model).status()
    modes=['offline_rules'] if backend=='offline' else ['llm_only','rag','rag_verify']
    all_runs=[];scores={}
    for mode in modes:
        gold=[];predicted=[];failures=0
        for case in cases:
            client = Ollama(model) if backend == 'ollama' else None
            try:
                result=analyze(case,retriever) if backend=='offline' else classify(case,mode,client,retriever)
                reference=(next(x for x in human['cases'] if x['id']==case['id'])['labels'] if human else dict(zip([c['id'] for c in case['claims']],case['silver_labels'])))
                gold.extend(reference[c['id']] for c in case['claims'])
                predicted.extend(c['label'] for c in result['claims'])
                row={'case_id':case['id'],'mode':mode,'status':'ok','result':result}
            except (ModelError,ValueError) as exc:
                failures+=1;row={'case_id':case['id'],'mode':mode,'status':'error','error':str(exc),'calls':client.calls if client else []}
            all_runs.append(row)
            # Incremental checkpoints survive slow models and interrupted runs.
            (out/'runs.json').write_text(json.dumps(all_runs,ensure_ascii=False,indent=2),encoding='utf8')
        scores[mode]={**classification_metrics(gold,predicted),'failed_cases':failures,'total_cases':len(cases),'complete':failures==0}
    retrieval={}
    for kind in ('law','case'):
        metrics=[]
        for case in cases:
            ids=[d['id'] for d in retriever.documents if d['kind']==kind]
            if human:
                entry=next(x for x in human['cases'] if x['id']==case['id'])
                if entry.get('relevant_document_ids') is None:continue
                rel=[x for x in entry['relevant_document_ids'] if x in ids]
            else:rel=[x for x in case['relevant_document_ids'] if x in ids]
            query=case['description']+' '+' '.join(c['text'] for c in case['claims'])
            hits=retriever.dual(query)[kind]
            metric=retrieval_metrics(rel,[h['id'] for h in hits],3)
            if metric['precision_at_k'] is not None:metrics.append(metric)
        retrieval[kind]={'n_queries':len(metrics),**{key:sum(m[key] for m in metrics)/len(metrics) if metrics else None for key in ('precision_at_k','recall_at_k')}}
    metadata['llm_experiment_status']='NOT_RUN' if backend=='offline' else ('COMPLETE' if all(s['complete'] for s in scores.values()) else 'INCOMPLETE')
    summary={'metadata':metadata,'classification':scores,'retrieval':retrieval}
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf8')
    with (out/'manual_output_review.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.writer(f);writer.writerow(['case_id','mode','claim_id','output_text','is_factual_0_or_1','unsupported_0_or_1','is_responsibility_conclusion_0_or_1','unsupported_responsibility_0_or_1','citation_entails_0_or_1','reviewer','notes'])
        for row in all_runs:
            for c in row.get('result',{}).get('claims',[]):
                writer.writerow([row['case_id'],row['mode'],c['id'],c['reason'],'','','','','','',''])
    return summary


def review_metrics(path):
    """Require explicit manual decisions; never infer hallucinations from label names."""
    groups=collections.defaultdict(list)
    with Path(path).open(encoding='utf-8-sig',newline='') as f:
        for r in csv.DictReader(f):
            if not r.get('reviewer'):raise ValueError('Manual review missing reviewer')
            for key in ('is_factual_0_or_1','unsupported_0_or_1','is_responsibility_conclusion_0_or_1','unsupported_responsibility_0_or_1','citation_entails_0_or_1'):
                if r.get(key) not in ('0','1','NA'):raise ValueError('Fill review values using 0/1/NA')
            if r['unsupported_0_or_1']=='1' and r['is_factual_0_or_1']!='1':raise ValueError('Unsupported factual statement must be factual')
            if r['unsupported_responsibility_0_or_1']=='1' and r['is_responsibility_conclusion_0_or_1']!='1':raise ValueError('Invalid responsibility review')
            groups[r['mode']].append(r)
    output={}
    for mode,rows in groups.items():
        facts=[r for r in rows if r['is_factual_0_or_1']=='1']
        resp=[r for r in rows if r['is_responsibility_conclusion_0_or_1']=='1']
        cites=[r for r in rows if r['citation_entails_0_or_1'] in ('0','1')]
        if any(r['unsupported_0_or_1'] not in ('0','1') for r in facts):raise ValueError('Missing factual decision')
        if any(r['unsupported_responsibility_0_or_1'] not in ('0','1') for r in resp):raise ValueError('Missing responsibility decision')
        output[mode]={'unit':'manually segmented review rows; split composite statements before scoring',
            'hallucination_rate':sum(r['unsupported_0_or_1']=='1' for r in facts)/len(facts) if facts else None,
            'unsupported_responsibility_conclusion_rate':sum(r['unsupported_responsibility_0_or_1']=='1' for r in resp)/len(resp) if resp else None,
            'citation_entailment_correctness':sum(r['citation_entails_0_or_1']=='1' for r in cites)/len(cites) if cites else None}
    return output
