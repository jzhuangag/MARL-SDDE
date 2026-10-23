"""Fresh read-only bibliography/metadata audit; never rewrites manuscript/BibTeX.

Only public bibliographic identifiers/titles are sent to scholarly resolvers.
Every result is preserved separately; unavailable indexes are not called passes.
"""
import argparse
import hashlib
import html
from html.parser import HTMLParser
import importlib.util
import json
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('existing_citation_records', HERE/'tools/verify_citations.py')
existing = importlib.util.module_from_spec(spec); spec.loader.exec_module(existing)


def dump(path, value):
    with Path(path).open('x', encoding='utf-8') as out:
        json.dump(value, out, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        out.write('\n')


def fetch(url):
    request = urllib.request.Request(url, headers={'User-Agent': 'Academic-bibliography-integrity-audit/20261016'})
    try:
        with urllib.request.urlopen(request, timeout=25) as response:
            body = response.read().decode('utf-8', errors='replace')
            return dict(status='AVAILABLE', http_status=response.status, url=response.geturl(),
                        response_sha256=hashlib.sha256(body.encode()).hexdigest(), body=body)
    except Exception as error:
        return dict(status='UNAVAILABLE', url=url, http_status=getattr(error, 'code', None), exception=repr(error))


class Meta(HTMLParser):
    def __init__(self):
        super().__init__(); self.values = {}

    def handle_starttag(self, tag, attrs):
        if tag == 'meta':
            attrs = dict(attrs)
            key = attrs.get('name', attrs.get('property', '')).lower()
            if key.startswith('citation_'):
                self.values.setdefault(key, []).append(html.unescape(attrs.get('content', '')))


def bib_entries(text):
    output = {}
    for match in re.finditer(r'@(article|inproceedings|book|misc)\s*\{\s*([^,]+),', text, re.I):
        pos, depth = match.end(), 1
        end = pos
        while depth and end < len(text):
            if text[end] == '{': depth += 1
            elif text[end] == '}': depth -= 1
            end += 1
        body = text[pos:end-1]; fields = {}; cursor = 0
        while True:
            field = re.search(r'(\w+)\s*=\s*', body[cursor:])
            if field is None: break
            key = field.group(1).lower(); start = cursor+field.end(); stop = start
            if body[start] == '{':
                depth = 1; stop += 1
                while depth:
                    depth += (body[stop] == '{')-(body[stop] == '}'); stop += 1
                value = body[start+1:stop-1]
            elif body[start] == '"':
                stop = body.index('"', start+1)+1; value = body[start+1:stop-1]
            else:
                comma = body.find(',', start); stop = len(body) if comma < 0 else comma
                value = body[start:stop].strip()
            fields[key] = value; cursor = stop
        name = match.group(2).strip()
        if name in output: raise ValueError('Duplicate BibTeX key '+name)
        output[name] = dict(kind=match.group(1).lower(), fields=fields)
    return output


def norm(value):
    value = re.sub(r'\\[\'`"^~=.]\s*\{?([A-Za-z])\}?', r'\1', value)
    value = value.replace(r'\c{s}', 's').replace(r'\c{c}', 'c')
    return existing.normalize(html.unescape(value))


def run(output, preserved_records=None):
    output.mkdir(exist_ok=False)
    (output/'records').mkdir()
    tex = (HERE/'main.tex').read_text(encoding='utf-8')
    appendix = (HERE/'appendices.tex').read_text(encoding='utf-8')
    bib = (HERE/'references.bib').read_text(encoding='utf-8')
    entries = bib_entries(bib)
    cited = {key.strip() for block in re.findall(r'\\cite\w*\{([^}]+)\}', tex+appendix) for key in block.split(',')}
    records = {r['key']: r for r in existing.RECORDS}
    assert set(records) == set(entries) == cited
    styles, results = [], []
    s2_unavailable = alex_unavailable = False
    for index, (key, item) in enumerate(entries.items()):
        preserved = preserved_records/f'{key}.json' if preserved_records else None
        if preserved and preserved.exists():
            row = json.loads(preserved.read_text(encoding='utf-8'))
            assert row['bib_fields'] == item['fields']
            dump(output/'records'/f'{key}.json', row); results.append(row)
            s2_unavailable = row['resolvers']['SemanticScholar']['status'] != 'AVAILABLE'
            alex_unavailable = row['resolvers']['OpenAlex']['status'] != 'AVAILABLE'
            print(json.dumps(dict(completed=index+1, key=key, status=row['status'],
                                  retained_same_day_record=True)), flush=True)
            continue
        expected, fields = records[key], item['fields']
        row = dict(key=key, bib_fields=fields, resolvers={}, definite_conflicts=[], unresolved=[], style_judgments=[])
        primary = (('https://api.crossref.org/works/'+urllib.parse.quote(fields['doi'], safe=''))
                   if fields.get('doi') else expected['url'])
        response = fetch(primary)
        metadata = {}
        if response['status'] == 'AVAILABLE':
            if fields.get('doi'):
                message = json.loads(response['body'])['message']
                metadata = dict(title=' '.join(message.get('title', [])),
                    year=message.get('issued', {}).get('date-parts', [[None]])[0][0],
                    authors=[' '.join(x.get(y, '') for y in ('given', 'family')).strip() for x in message.get('author', [])],
                    venue=' '.join(message.get('container-title', [])),
                    volume=message.get('volume'), number=message.get('issue'), pages=message.get('page'), doi=message.get('DOI'))
                resolver = 'Crossref'
            else:
                parser = Meta(); parser.feed(response['body']); data = parser.values
                get = lambda name: data.get(name, [None])[0]
                date = get('citation_publication_date') or get('citation_date')
                metadata = dict(title=get('citation_title'), authors=data.get('citation_author', []),
                    year=int(date[:4]) if date and date[:4].isdigit() else None,
                    venue=get('citation_conference_title') or get('citation_journal_title'),
                    volume=get('citation_volume'), number=get('citation_issue'),
                    pages=(get('citation_firstpage')+'--'+get('citation_lastpage'))
                        if get('citation_firstpage') and get('citation_lastpage') else None,
                    doi=get('citation_doi'))
                resolver = 'official_publisher_or_proceedings'
            if not metadata.get('title') or not metadata.get('authors') or metadata.get('year') is None:
                row['unresolved'].append('Primary structured title/authors/year incomplete; manual authoritative check required')
            else:
                row['definite_conflicts'] += existing.validate(expected, metadata)
                if len(metadata['authors']) != len(fields['author'].split(' and ')):
                    row['definite_conflicts'].append('author_count')
                if norm(fields['title']) != norm(metadata['title']):
                    row['definite_conflicts'].append('bib_title')
                if int(fields['year']) != metadata['year']:
                    row['definite_conflicts'].append('bib_year')
                for field in ('volume', 'number', 'pages', 'doi'):
                    if fields.get(field) and metadata.get(field):
                        if norm(fields[field]) != norm(str(metadata[field])):
                            row['definite_conflicts'].append('bib_'+field)
            row['resolvers']['authoritative'] = dict(resolver=resolver, metadata=metadata,
                                                     **{k: v for k, v in response.items() if k != 'body'})
        else:
            row['resolvers']['authoritative'] = response
            row['unresolved'].append('Primary resolver unavailable; not treated as verified')
        # Independent indexed lookup; stop a resolver batch on unavailability.
        if not s2_unavailable:
            query = ('DOI:'+fields['doi']) if fields.get('doi') else None
            url = ('https://api.semanticscholar.org/graph/v1/paper/'+urllib.parse.quote(query, safe=':/')+
                   '?fields=title,authors,year,externalIds,venue') if query else (
                   'https://api.semanticscholar.org/graph/v1/paper/search?limit=3&fields=title,authors,year,externalIds,venue&query='+
                   urllib.parse.quote(expected['title']))
            s2 = fetch(url)
            if s2['status'] == 'AVAILABLE':
                data = json.loads(s2['body']); row['resolvers']['SemanticScholar'] = {**{k: v for k, v in s2.items() if k != 'body'}, 'metadata': data}
            else:
                s2_unavailable = True; row['resolvers']['SemanticScholar'] = s2
            time.sleep(1)
        else:
            row['resolvers']['SemanticScholar'] = dict(status='UNAVAILABLE_BATCH_STOPPED', no_rate_limit_bypass=True)
        if not alex_unavailable:
            url = ('https://api.openalex.org/works/https://doi.org/'+fields['doi']) if fields.get('doi') else (
                'https://api.openalex.org/works?per-page=3&search='+urllib.parse.quote(expected['title']))
            alex = fetch(url)
            if alex['status'] == 'AVAILABLE':
                row['resolvers']['OpenAlex'] = {**{k: v for k, v in alex.items() if k != 'body'}, 'metadata': json.loads(alex['body'])}
            else:
                alex_unavailable = True; row['resolvers']['OpenAlex'] = alex
        else:
            row['resolvers']['OpenAlex'] = dict(status='UNAVAILABLE_BATCH_STOPPED', no_rate_limit_bypass=True)
        row['resolvers']['arXiv'] = dict(status='NOT_APPLICABLE_TO_PUBLISHED_RECORD' if not fields.get('eprint') else
                                               'PUBLISHED_RECORD_PRIMARY; ARXIV_PROVENANCE_NOT_YET_RECHECKED')
        # Project-specific IEEEtran rendering governs initialized full-name authors,
        # NeurIPS abbreviations, macro months and omitted PMLR location fields.
        if item['kind'] == 'inproceedings' and not fields.get('booktitle', '').startswith(('Proc.', 'Adv.')):
            styles.append(dict(key=key, definite='noncanonical proceedings venue'))
        if re.search(r'IEEE Transactions on', fields.get('journal', '')):
            styles.append(dict(key=key, definite='unabbreviated IEEE journal'))
        if any(value.strip() == '' for value in fields.values()):
            styles.append(dict(key=key, definite='empty BibTeX field'))
        if item['kind'] == 'inproceedings' and (not fields.get('month') or not fields.get('address')):
            row['style_judgments'].append('Local compact PMLR/NeurIPS convention omits month/location; not a definite error')
        row['status'] = 'PASS' if not row['definite_conflicts'] and not row['unresolved'] else 'NEEDS_RESOLUTION'
        dump(output/'records'/f'{key}.json', row); results.append(row)
        print(json.dumps(dict(completed=index+1, key=key, status=row['status'])), flush=True)
    dump(output/'citation_verification.json', dict(checked_on='2026-10-16', records=results,
        all_pass=all(r['status'] == 'PASS' for r in results),
        unresolved_material_conflicts=[r['key'] for r in results if r['status'] != 'PASS'],
        manuscript_sha256=hashlib.sha256((HERE/'main.tex').read_bytes()).hexdigest(),
        bibliography_sha256=hashlib.sha256((HERE/'references.bib').read_bytes()).hexdigest(),
        no_manuscript_or_bib_mutation=True, no_external_model_upload=True))
    dump(output/'bibliography_style.json', dict(entries=len(entries), cited=len(cited),
        canonical='Existing project IEEEtran.bst rendering and compact IEEE venues',
        definite_style_errors=styles, keys_preserved=True, bib_edited=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--preserved-records', type=Path)
    args = parser.parse_args(); run(args.output, args.preserved_records)
