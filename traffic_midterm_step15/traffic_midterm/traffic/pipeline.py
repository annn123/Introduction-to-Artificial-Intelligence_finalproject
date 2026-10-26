"""End-to-end demo path. Fixed-claim research evaluation remains separate."""
from __future__ import annotations
import copy
from .core import analyze, validate_payload
from .extraction import extract_case
from .llm import classify


def model_pipeline(payload, mode, client, retriever=None):
    if mode not in ('llm_only','rag','rag_verify'):raise ValueError('Invalid experiment mode')
    if not isinstance(payload,dict):raise ValueError('Input must be a JSON object')
    working=copy.deepcopy(payload)
    extraction=None
    if working.get('claims') is None:
        extraction=extract_case(working.get('description',''),client)
        working['claims']=extraction['claims']
    validate_payload(working)
    result=classify(working,mode,client,retriever)
    if extraction:
        result['case']=extraction['case']
        result['extraction']={k:v for k,v in extraction.items() if k not in ('calls','claims','case')}
        result['extraction']['generated_check_questions']=[c['id'] for c in extraction['claims'] if c.get('origin')=='generated_check_question']
    else:
        result['extraction']={'method':'supplied_claims','warning':'本次使用提供的主張，不是自動擷取評估。'}
    result['calls']=copy.deepcopy(client.calls)
    return result
