from __future__ import annotations

import collections
import hashlib
import json
import math
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LABELS = ('Supported', 'Partially Supported', 'Insufficient Evidence', 'Overgeneralized', 'Irrelevant')
PREDICATES = {
    'speeding': ('超速',), 'unlicensed': ('無照', '沒有駕照', '未領駕照'),
    'red_light': ('闖紅燈',), 'yield': ('未讓', '未禮讓', '未停讓'),
    'distance': ('未保持安全距離', '未保持車距', '追撞'),
    'lane_change': ('變換車道', '切換車道', '切車道'),
    'door': ('開車門', '開啟車門'), 'alcohol': ('酒駕', '酒後駕車'),
    'attention': ('未注意車前狀況',), 'cargo': ('掉落物', '輪胎掉落', '貨物掉落'),
}

def read_json(name):
    return json.loads((ROOT / 'data' / name).read_text(encoding='utf-8'))

def normalize(text):
    return re.sub(r'\s+', '', str(text)).replace('臺', '台')

def tokens(text):
    text = normalize(text).lower()
    # Character bigrams avoid a downloaded Chinese tokenizer; English words remain intact.
    chars = re.findall(r'[\u3400-\u9fff]', text)
    pairs = re.findall(r'[\u3400-\u9fff]{2,}', text)
    return chars + [s[i:i+2] for s in pairs for i in range(len(s)-1)] + re.findall(r'[a-z0-9_]+', text)

class Retriever:
    def __init__(self, documents=None):
        self.documents = read_json('corpus.json') if documents is None else documents
        ids = [d['id'] for d in self.documents]
        if len(ids) != len(set(ids)):
            raise ValueError('Duplicate document IDs')
        self.counts = [collections.Counter(tokens(d['title'] + ' ' + d['text'])) for d in self.documents]
        self.lengths = [sum(c.values()) for c in self.counts]
        self.avg = sum(self.lengths) / max(1, len(self.lengths))
        self.df = collections.Counter(t for c in self.counts for t in c)

    def search(self, query, kind=None, k=3, exclude=()):
        if not isinstance(k, int) or not 1 <= k <= 20:
            raise ValueError('k must be between 1 and 20')
        result = []
        n = len(self.documents)
        for d, counts, length in zip(self.documents, self.counts, self.lengths):
            if (kind and d['kind'] != kind) or d['id'] in exclude:
                continue
            score = 0.0
            for term in set(tokens(query)):
                frequency = counts[term]
                if frequency:
                    idf = math.log(1 + (n - self.df[term] + .5) / (self.df[term] + .5))
                    score += idf * frequency * 2.5 / (frequency + 1.5 * (.25 + .75 * length / (self.avg or 1)))
            if score > 0:
                result.append({**d, 'score': round(score, 6)})
        return sorted(result, key=lambda x: (-x['score'], x['id']))[:k]

    def dual(self, query, k=3, exclude=()):
        return {kind: self.search(query, kind, k, exclude) for kind in ('law', 'case')}

    @property
    def fingerprint(self):
        return hashlib.sha256(json.dumps(self.documents, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def infer_claim(text, index=1):
    if not isinstance(text, str) or not text.strip():
        raise ValueError('Claim text cannot be empty')
    # A/B take priority; 我/對方 are explicit aliases only, never guessed from a surname.
    match = re.search(r'(?<![A-Za-z])([AB])(?![A-Za-z])|對方|我', text)
    subject = match.group(0) if match else 'unknown'
    subject = {'我': 'A', '對方': 'B'}.get(subject, subject)
    actors = { {'我': 'A', '對方': 'B'}.get(m.group(), m.group()) for m in re.finditer(r'(?<![A-Za-z])([AB])(?![A-Za-z])|對方|我', text) }
    if len(actors) > 1:
        subject = 'unknown'
    kind, predicate = 'fact', 'unknown'
    if re.search(r'全責|全部責任|百分之百|100\s*[%％]|完全責任', text):
        kind, predicate = 'liability', 'liability:100'
    elif re.search(r'(?:\d+\s*[%％]|[一二三四五六七八九]成).*(?:責|肇責)|責任.*\d+\s*[%％]', text):
        kind, predicate = 'liability', 'liability:unspecified'
    else:
        for key, words in PREDICATES.items():
            if any(w in text for w in words):
                predicate = key
                break
        if re.search(r'造成|導致|肇事原因|因果關係', text):
            kind, predicate = 'causal', 'cause:' + predicate
        if re.search(r'態度|穿著|有錢|薪水|學歷', text) and predicate == 'unknown':
            kind = 'irrelevant'
    # Composite and uncertain statements cannot be verified as a single established fact.
    detected = [key for key, words in PREDICATES.items() if any(w in text for w in words)]
    if kind in ('fact', 'causal') and (len(detected) > 1 or re.search(r'未必|可能|不確定|據說|否認|並非沒有|不是沒有|沒有未', text)):
        predicate = 'unknown'
    # Unsupported/ambiguous negation is flagged rather than treated as a positive fact.
    negation = bool(re.search(r'沒有超速|未超速|並無超速|並非超速|不是超速|沒有闖紅燈|未闖紅燈|沒有酒駕|未酒駕|並非無照|不是無照|並非未讓|不是未讓|並非未保持|不是未保持|未變換車道|沒有變換車道|未開啟車門|沒有開車門', text))
    if kind == 'liability' and re.search(r'並非|不是|不應|並不|不負', text):
        negation = True
    return {'id': f'C{index}', 'text': text.strip(), 'subject': subject,
            'kind': kind, 'predicate': predicate, 'polarity': not negation,
            'extraction': 'heuristic_requires_review'}


def extract_claims(description):
    output = []
    for sentence in re.split(r'[。；;\n]+', description):
        sentence = sentence.strip(' ，,「」')
        if not sentence:
            continue
        parts = re.split(r'[，,]?\s*(?:所以|因此|故而)\s*', sentence, maxsplit=1)
        first = infer_claim(parts[0], len(output)+1)
        # Keep unknown statements, so omission cannot masquerade as successful extraction.
        output.append(first)
        if len(parts) == 2 and parts[1].strip():
            second = infer_claim(parts[1], len(output)+1)
            second['derived_from'] = [first['id']]
            # A missing subject stays unknown: inheritance could assign the wrong party.
            output.append(second)
    return output


def validate_payload(payload):
    if not isinstance(payload, dict):
        raise ValueError('Input must be a JSON object')
    description = payload.get('description', '')
    if not isinstance(description, str) or not description.strip() or len(description) > 12000:
        raise ValueError('事故描述必須為 1–12000 字')
    claims = payload.get('claims')
    if claims is None:
        claims = extract_claims(description)
    if not isinstance(claims, list) or not 1 <= len(claims) <= 60:
        raise ValueError('Claims must contain 1–60 items')
    cleaned, ids = [], set()
    for i, raw in enumerate(claims, 1):
        if isinstance(raw, str):
            claim = infer_claim(raw, i)
        elif isinstance(raw, dict):
            claim = {**infer_claim(raw.get('text', ''), i), **raw}
        else:
            raise ValueError('Invalid claim')
        if not isinstance(claim['id'], str) or not claim['id'] or claim['id'] in ids:
            raise ValueError('Duplicate or invalid claim ID')
        ids.add(claim['id'])
        if claim['subject'] not in ('A', 'B', 'unknown') or claim['kind'] not in ('fact', 'causal', 'liability', 'irrelevant'):
            raise ValueError('Invalid claim subject/kind')
        if type(claim['polarity']) is not bool or not isinstance(claim['predicate'], str):
            raise ValueError('Invalid claim polarity/predicate')
        cleaned.append(claim)
    for claim in cleaned:
        deps = claim.get('derived_from', [])
        if not isinstance(deps, list) or any(not isinstance(d, str) or d not in ids or d == claim['id'] for d in deps):
            raise ValueError('Invalid claim dependencies')
        checks=claim.get('required_checks',[])
        if not isinstance(checks,list) or any(not isinstance(d,str) or d not in ids or d==claim['id'] for d in checks):
            raise ValueError('Invalid required checks')
    evidence = payload.get('evidence', [])
    if not isinstance(evidence, list) or len(evidence) > 60:
        raise ValueError('Evidence must be an array of at most 60 items')
    evid_ids = set()
    for item in evidence:
        if not isinstance(item, dict) or not isinstance(item.get('id'), str) or not item['id'] or item['id'] in evid_ids:
            raise ValueError('Invalid/duplicate evidence ID')
        evid_ids.add(item['id'])
        if not isinstance(item.get('description', ''), str) or type(item.get('reviewed', False)) is not bool:
            raise ValueError('Invalid evidence description/review flag')
        assertions = item.get('assertions', [])
        if not isinstance(assertions, list):
            raise ValueError('Evidence assertions must be an array')
        for a in assertions:
            if not isinstance(a, dict) or a.get('subject') not in ('A', 'B') or not isinstance(a.get('predicate'), str) or type(a.get('polarity')) is not bool:
                raise ValueError('Invalid evidence assertion')
    return {'description': description, 'claims': cleaned, 'evidence': evidence}


def structure_case(description):
    participants = {'A': {'vehicle': None, 'actions': []}, 'B': {'vehicle': None, 'actions': []}}
    # Only actor-local clauses are considered; no broad cross-clause attribution.
    for clause in re.split(r'[，,。；;\n]', description):
        actors = list(re.finditer(r'(?<![A-Za-z])([AB])(?![A-Za-z])|對方|我', clause))
        for i, actor in enumerate(actors):
            who = {'我': 'A', '對方': 'B'}.get(actor.group(), actor.group())
            segment = clause[actor.end(): actors[i+1].start() if i+1 < len(actors) else len(clause)]
            for vehicle in ('機車', '小貨車', '貨車', '汽車', '自行車'):
                if vehicle in segment and participants[who]['vehicle'] is None:
                    participants[who]['vehicle'] = vehicle
                    break
            for action in ('直行', '左轉', '右轉', '變換車道', '開啟車門', '追撞'):
                if action in segment and action not in participants[who]['actions']:
                    participants[who]['actions'].append(action)
    return {'participants': participants, 'traffic_signal': None, 'right_of_way': None,
            'road_type': '交岔路口（依描述）' if '路口' in description else None,
            'evidence_mentions': [w for w in ('行車紀錄器', '監視器', '警方紀錄', '鑑定報告') if w in description],
            'court_findings': None, 'extraction_method': 'conservative_rules',
            'status': '僅擷取敘述，不代表事實成立；提及影片不代表已讀取影片。'}


def verify_claim(claim, evidence):
    supporting, contradicting = [], []
    if claim['subject'] != 'unknown' and claim['predicate'] != 'unknown':
        for e in evidence:
            if not e.get('reviewed', False):
                continue
            # Liability support requires a supplied case-specific court finding.
            if claim['kind'] == 'liability' and e.get('source_type') != 'court_finding':
                continue
            for a in e.get('assertions', []):
                if (a['subject'], a['predicate']) == (claim['subject'], claim['predicate']):
                    dest = supporting if a['polarity'] == claim['polarity'] else contradicting
                    if e['id'] not in dest:
                        dest.append(e['id'])
    if claim['kind'] == 'irrelevant':
        label, reason = 'Irrelevant', '此主張未說明與事故行為或因果爭點的關聯。'
    elif contradicting:
        label, reason = 'Insufficient Evidence', '提供的證據與主張相反或彼此衝突；五標籤原設計沒有 Refuted，另以 conflict 標記，不能稱為支持。'
    elif supporting:
        label, reason = 'Supported', '與使用者提供、標記已核對之證據轉錄相符；系統未獨立驗證原始檔案真偽。'
    elif claim['kind'] == 'liability' and claim['predicate'] == 'liability:100' and claim['polarity'] and (claim.get('derived_from') or re.search(r'所以|因此|只要|既然|就應|必須|一定', claim['text'])):
        label, reason = 'Overgeneralized', '目前推論由個別行為直接跳到全部責任，缺少本案因果與雙方行為的完整支持。'
    else:
        label, reason = 'Insufficient Evidence', '尚無可對應的本案證據；法規、相似裁判或證據名稱不能證實本案事實。'
    return {'id': claim['id'], 'label': label, 'reason': reason, 'evidence_ids': supporting,
            'contradicting_evidence_ids': contradicting, 'conflict': bool(contradicting),
            'needs_review': True}


def analyze(payload, retriever=None):
    data = validate_payload(payload)
    retriever = retriever or Retriever()
    out = []
    for c in data['claims']:
        hits = retriever.dual(c['text'])
        out.append({**c, **verify_claim(c, data['evidence']), 'retrieved': hits})
    return {'engine': 'offline_rules_NOT_LLM', 'case': structure_case(data['description']),
            'claims': out, 'evidence': data['evidence'], 'corpus_sha256': retriever.fingerprint,
            'mediator': mediation(out), 'limitations': ['規則式原型，不是 LLM 實驗結果。', '法規為查閱日快照，尚未按事故日自動選擇歷史版本。', '相似裁判僅供類比，不可移植其事實或責任比例。']}


def mediation(claims):
    details=[]
    for claim in claims:
        kind=claim.get('kind','unknown');label=claim['label']
        gate=claim.get('gate') or {}
        opposite=list(dict.fromkeys(claim.get('contradicting_evidence_ids',[])+gate.get('contradicting_evidence_ids',[])))
        conflict=bool(claim.get('conflict') or gate.get('conflict') or opposite)
        if conflict:
            refs='、'.join(opposite)
            message='已有與此主張相反的證據'+('（'+refs+'）' if refs else '')+'，或存在證據衝突，不能將此主張視為已獲支持；請核對證據內容與來源。'
        elif kind=='fact' and label=='Overgeneralized':
            message='此項是事實指控，卻被標為過度推論；需核對標籤與理由，不能把它改說成全責結論。'
        elif label=='Supported':
            message='輸出標示支持，請核對列出的本案證據及其轉錄，仍非系統獨立查證。'
        elif label=='Partially Supported':
            message='輸出標示部分支持，請指出已支持的部分及尚缺的證據。'
        elif label=='Insufficient Evidence':
            message='尚需補充或核對本案證據，不能把當事人指控當作已證實事實。'
        elif label=='Overgeneralized':
            message='此項推論的支持程度需要核對。'
            if kind=='liability':message='此項責任推論需要核對，不能只由個別行為直接確定全部責任。'
        else:message='請先釐清此項與事故爭點的關聯。'
        details.append({'id':claim['id'],'claim':claim['text'],'kind':kind,'label':label,'message':message})
    sentences=[d['id']+'「'+d['claim']+'」：'+d['message'] for d in details[:6]]
    if len(details)>6:sentences.append('其餘項目請逐項檢查。')
    return {'role':'neutral_mediator','engine':'template_NOT_LLM','details':details,
            'text':'我們先逐項確認證據，再討論責任。 '+ ' '.join(sentences)+' 目前不推估責任比例。'}
