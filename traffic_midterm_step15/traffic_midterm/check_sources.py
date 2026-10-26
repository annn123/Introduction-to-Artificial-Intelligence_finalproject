"""Optional source receipt refresh. Does not silently replace curated summaries."""
import concurrent.futures
import hashlib
import json
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from traffic.core import ROOT, read_json

def fetch(group):
    url,ids=group
    request=urllib.request.Request(url,headers={'User-Agent':'TrafficMidtermResearch/1.0'})
    record={'document_ids':ids,'url':url,'fetched_utc':datetime.now(timezone.utc).isoformat()}
    try:
        with urllib.request.urlopen(request,timeout=25) as response:
            data=response.read(4_000_000)
            text=data.decode('utf8',errors='replace')
            if response.status != 200 or len(data)<1000:raise ValueError('Unexpected/empty response')
            # Save receipts, not copyrighted third-party content. Source pages are government records.
            record.update(http_status=response.status,sha256=hashlib.sha256(data).hexdigest(),bytes=len(data),final_url=response.url,
                          page_contains_expected_heading=any(t in text for t in ('條文內容','裁判','判決','道路交通')),
                          note='HTTP availability receipt only; not an independent content or legal validity review.')
    except Exception as exc:record['error']=str(exc)
    return record

def main():
    grouped={}
    for d in read_json('corpus.json'):grouped.setdefault(d['url'],[]).append(d['id'])
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        records=list(pool.map(fetch,grouped.items()))
    path=ROOT/'results/source_receipts.json';path.write_text(json.dumps(records,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(f'{sum("error" not in r for r in records)}/{len(records)} URLs reachable; receipts: {path}')

if __name__=='__main__':main()
