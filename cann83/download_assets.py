"""Requests-only pinned model/data downloader. Resumes chunks and validates full files."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import threading
import time
from urllib.parse import urlsplit, urlunsplit, parse_qsl, urlencode

import requests

LOCAL = threading.local()


def checksum(path, kind):
    h = hashlib.new('sha256' if kind == 'sha256' else 'sha1')
    if kind == 'git-sha1':
        h.update(f'blob {path.stat().st_size}\0'.encode())
    with path.open('rb') as stream:
        for raw in iter(lambda: stream.read(8 * 1024**2), b''):
            h.update(raw)
    return h.hexdigest()


def get(url, **kwargs):
    if not hasattr(LOCAL, 'session'):
        LOCAL.session = requests.Session()
    for attempt in range(6):
        try:
            r = LOCAL.session.get(url, timeout=(10, 60), **kwargs)
            r.raise_for_status()
            return r
        except requests.RequestException:
            if attempt == 5:
                raise
            time.sleep(min(attempt + 1, 5))


def valid(p, row):
    return (p.is_file() and p.stat().st_size == row['bytes']
            and checksum(p, row['hash_kind']) == row['hash'])


def range_url(url, index):
    u = urlsplit(url)
    return urlunsplit(u._replace(query=urlencode(parse_qsl(u.query) + [('part', str(index))])))


def download(row, root, workers=16):
    relative = PurePosixPath(row['path'])
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError('Unsafe manifest path')
    dest = root.joinpath(*relative.parts)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if valid(dest, row):
        return 'existing'
    if dest.exists():
        raise RuntimeError(f'Preserving unverified existing file: {dest}')
    temp = dest.with_name(dest.name + '.assembling')
    if row['bytes'] < 16 * 1024**2:
        r = get(row['url'], headers={'Accept-Encoding': 'identity'})
        temp.write_bytes(r.content)
    else:
        chunk_size = 8 * 1024**2
        chunks = dest.with_name(dest.name + '.chunks')
        chunks.mkdir(exist_ok=True)
        meta = {k: row[k] for k in ('path', 'bytes', 'hash', 'hash_kind')}
        meta['chunk_size'] = chunk_size
        mp = chunks / 'download.json'
        if mp.exists() and json.loads(mp.read_text()) != meta:
            raise RuntimeError('Chunk metadata mismatch: ' + str(chunks))
        mp.write_text(json.dumps(meta, indent=2))
        jobs = [(i, start, min(start + chunk_size, row['bytes']) - 1)
                for i, start in enumerate(range(0, row['bytes'], chunk_size))]

        def chunk(job):
            index, start, end = job
            p = chunks / f'{index:06d}.chunk'
            if p.is_file() and p.stat().st_size == end - start + 1:
                return
            r = get(range_url(row['url'], index),
                    headers={'Range': f'bytes={start}-{end}', 'Accept-Encoding': 'identity'})
            expected = f'bytes {start}-{end}/{row["bytes"]}'
            if r.status_code != 206 or r.headers.get('Content-Range') != expected:
                raise RuntimeError(f'Server did not honor range: {r.status_code}, {r.headers.get("Content-Range")}')
            if len(r.content) != end - start + 1:
                raise RuntimeError('Truncated range')
            part = p.with_suffix('.part')
            part.write_bytes(r.content)
            part.replace(p)

        with ThreadPoolExecutor(max_workers=workers) as pool:
            for i, future in enumerate(as_completed([pool.submit(chunk, j) for j in jobs]), 1):
                future.result()
                if i % 32 == 0 or i == len(jobs):
                    print(f'{dest.name}: {i}/{len(jobs)} chunks', flush=True)
        with temp.open('wb') as out:
            for i, _, _ in jobs:
                with (chunks / f'{i:06d}.chunk').open('rb') as stream:
                    for block in iter(lambda: stream.read(8 * 1024**2), b''):
                        out.write(block)
            out.flush()
            os.fsync(out.fileno())
    if not valid(temp, row):
        raise RuntimeError('Full-file verification failed: ' + str(temp))
    temp.replace(dest)
    return 'downloaded'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=16)
    parser.add_argument('--only', choices=['model', 'data', 'all'], default='all')
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    if not 1 <= args.workers <= 32:
        parser.error('workers must be 1..32')
    manifest = json.loads((Path(__file__).parent / 'assets-manifest.json').read_text())
    rows = [r for r in manifest['files'] if args.only == 'all' or r['group'] == args.only]
    if args.verify_only:
        bad = [r['path'] for r in rows if not valid(args.output / r['path'], r)]
        if bad:
            raise SystemExit('Missing/invalid assets:\n' + '\n'.join(bad))
        print('VERIFIED', len(rows), 'files')
        return
    models = [r for r in rows if r['group'] == 'model']
    for i, row in enumerate(models, 1):
        print(f'Model {i}/{len(models)} {row["path"]}', flush=True)
        download(row, args.output, args.workers)
    data = [r for r in rows if r['group'] == 'data']
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(download, row, args.output, 1): row for row in data}
        for i, future in enumerate(as_completed(futures), 1):
            try:
                future.result()
            except Exception as e:
                raise RuntimeError(f'Dataset download failed: {futures[future]["path"]}: {e}') from e
            if i % 50 == 0 or i == len(data):
                print(f'Dataset: {i}/{len(data)} verified', flush=True)
    report = {'models': manifest['models'], 'datasets': manifest['datasets'],
              'verified_files': len(rows), 'only': args.only}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / f'verified-{args.only}.json').write_text(json.dumps(report, indent=2))
    print('COMPLETE:', args.output.resolve(), flush=True)


if __name__ == '__main__':
    main()
