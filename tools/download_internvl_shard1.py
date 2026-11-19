"""InternVL3-78B first shard: verified 16/32-worker resumable requests downloader.

Fixed chunk boundaries survive worker-count changes. Existing final files are
never replaced when their size/hash differs. No old .mt files are imported.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import re
import threading
import time
from urllib.parse import urlsplit

import requests

MIB = 1024 * 1024
DEFAULT_NAME = 'model-00001-of-00033.safetensors'
DEFAULT_URL = 'https://modelscope.cn/models/OpenGVLab/InternVL3-78B/resolve/master/' + DEFAULT_NAME
DEFAULT_SIZE = 4988569440
DEFAULT_SHA256 = 'c527e7dfe9c1d8c0b3da1a10a6940f7dcf3caa6872078c38c3bc90413f30f99a'
RETRY_STATUS = {429, 500, 502, 503, 504}


class DownloadError(RuntimeError):
    pass


class TransientHTTPError(requests.RequestException):
    def __init__(self, status):
        self.status = status
        super().__init__(f'HTTP {status}')


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * MIB), b''):
            digest.update(block)
    return digest.hexdigest()


def network_error(exc):
    # Exception strings may contain signed URLs or proxy passwords.
    if isinstance(exc, requests.exceptions.SSLError):
        return 'TLS/certificate error; check the company CA bundle'
    if isinstance(exc, requests.exceptions.ProxyError):
        return 'Proxy connection error'
    if isinstance(exc, requests.exceptions.Timeout):
        return 'Connection/read timeout'
    if isinstance(exc, TransientHTTPError):
        return f'HTTP {exc.status}'
    return f'Network interruption ({type(exc).__name__})'


def validate_range(response, first, last, size):
    if response.headers.get('Content-Encoding', 'identity').lower().strip() != 'identity':
        raise DownloadError('Compressed Range response cannot be appended')
    raw_range = response.headers.get('Content-Range', '').strip()
    length = response.headers.get('Content-Length')
    # Some servers ignore Range for small config files. A complete, unresumed
    # object with an exact length is equivalent; a partial/resumed request is not.
    if response.status_code == 200:
        if first != 0 or last != size - 1:
            raise DownloadError('Partial/resumed Range request returned HTTP 200; response was not appended')
        if length is None or not length.isdigit() or int(length) != size:
            raise DownloadError('HTTP 200 full-object response needs an exact Content-Length')
        if raw_range:
            match = re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)', raw_range)
            if not match or tuple(map(int, match.groups())) != (0, size - 1, size):
                raise DownloadError('HTTP 200 response has a conflicting Content-Range')
        return
    if response.status_code != 206:
        raise DownloadError(f'Range request returned HTTP {response.status_code}, expected 206; response was not appended')
    match = re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)', raw_range)
    if not match or tuple(map(int, match.groups())) != (first, last, size):
        actual = raw_range if match else '(missing or invalid)'
        raise DownloadError(f'Content-Range mismatch: expected bytes {first}-{last}/{size}; received {actual}. Stopped without appending or retrying this response.')
    if length is not None and (not length.isdigit() or int(length) != last - first + 1):
        actual = length if length.isdigit() else '(invalid)'
        raise DownloadError(f'Content-Length mismatch: expected {last-first+1}; received {actual}. Stopped without appending or retrying this response.')


def download(url, output, size, sha256, *, workers=16, chunk_size=64*MIB,
             verify=True, retries=4, retry_delay=2, progress_interval=3):
    parsed = urlsplit(url)
    if parsed.scheme not in {'http', 'https'} or not parsed.hostname or parsed.username or parsed.password:
        raise ValueError('Use an HTTP(S) URL without embedded credentials')
    sha256 = sha256.lower()
    if not re.fullmatch(r'[0-9a-f]{64}', sha256):
        raise ValueError('An official 64-character SHA256 is required')
    if type(size) is not int or size < 1 or type(chunk_size) is not int or chunk_size < 1:
        raise ValueError('size and chunk_size must be positive integers')
    if not 1 <= workers <= 32 or not 1 <= retries <= 20:
        raise ValueError('workers must be 1..32 and retries must be 1..20 (total attempts per chunk per run)')
    output = Path(output).expanduser().absolute()
    output.parent.mkdir(parents=True, exist_ok=True)
    parts = output.with_name(output.name + '.parts')
    lock = output.with_name(output.name + '.download.lock')
    try:
        lock_fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise DownloadError(f'Another download may be active: {lock.name}. If an earlier process crashed, confirm it has exited before removing only this lock file.') from exc
    os.write(lock_fd, str(os.getpid()).encode())
    os.close(lock_fd)
    stopped = threading.Event()
    printer = None
    try:
        if output.exists():
            if output.is_file() and output.stat().st_size == size and sha256_file(output) == sha256:
                print(f'Already verified: {output}', flush=True)
                return output
            raise DownloadError('Final output already exists with a different size/hash; preserve it and choose another output path')
        # The source fingerprint avoids saving signed query strings or proxy secrets.
        metadata = {'version': 1, 'source_sha256': hashlib.sha256(url.encode()).hexdigest(),
                    'source_host': parsed.hostname, 'size': size, 'sha256': sha256, 'chunk_size': chunk_size}
        if parts.exists() and not parts.is_dir():
            raise DownloadError('Parts path exists but is not a directory')
        parts.mkdir(exist_ok=True)
        meta_path = parts / 'metadata.json'
        if meta_path.exists():
            if json.loads(meta_path.read_text(encoding='utf-8')) != metadata:
                raise DownloadError('URL, expected size/hash or chunk size changed; preserve this parts directory and choose a new output path')
        else:
            if any(parts.iterdir()):
                raise DownloadError('Parts directory has files but no metadata; do not reuse unrelated download parts')
            with meta_path.open('x', encoding='utf-8') as stream:
                json.dump(metadata, stream, indent=2)
        ranges = [(i, start, min(start + chunk_size, size) - 1)
                  for i, start in enumerate(range(0, size, chunk_size))]
        paths = {i: parts / f'chunk-{i:06d}.part' for i, _, _ in ranges}
        for i, start, end in ranges:
            if paths[i].exists() and paths[i].stat().st_size > end-start+1:
                raise DownloadError(f'{paths[i].name}: local size {paths[i].stat().st_size} exceeds expected {end-start+1}. Stopped; preserve this directory and choose a new output path.')
        initial = sum(path.stat().st_size for path in paths.values() if path.exists())
        begun = time.monotonic()
        print(f'Output: {output}\nExpected bytes: {size}\nWorkers: {workers}; fixed chunks: {len(ranges)} x up to {chunk_size} bytes; max attempts/chunk: {retries}\nResume: {initial}/{size} bytes', flush=True)

        def progress():
            previous, previous_time = initial, begun
            while not stopped.wait(progress_interval):
                current = sum(path.stat().st_size for path in paths.values() if path.exists())
                now = time.monotonic()
                speed = (current-previous) / max(now-previous_time, 0.001) / MIB
                print(f'{current/MIB:.1f}/{size/MIB:.1f} MiB | {speed:.2f} MiB/s | workers={workers}', flush=True)
                previous, previous_time = current, now

        if progress_interval > 0:
            printer = threading.Thread(target=progress, daemon=True)
            printer.start()

        def fetch(item):
            index, first, last = item
            path = paths[index]
            expected = last-first+1
            with requests.Session() as session:
                session.max_redirects = 5
                # Default trust_env=True preserves configured corporate proxies/CA env.
                for attempt in range(retries):
                    if stopped.is_set():
                        raise DownloadError('Download interrupted; completed/partial chunks were retained')
                    have = path.stat().st_size if path.exists() else 0
                    if have > expected:
                        raise DownloadError(f'{path.name}: local size {have} exceeds expected {expected}; stopped without retry')
                    if have == expected:
                        return
                    start = first + have
                    response_start_have = have
                    try:
                        with session.get(url, headers={'Range': f'bytes={start}-{last}', 'Accept-Encoding': 'identity'},
                                         stream=True, timeout=(15, 30), verify=verify) as response:
                            if response.status_code in RETRY_STATUS:
                                raise TransientHTTPError(response.status_code)
                            validate_range(response, start, last, size)
                            with path.open('ab') as stream:
                                for block in response.iter_content(chunk_size=MIB):
                                    if stopped.is_set():
                                        raise DownloadError('Download interrupted; partial chunks were retained')
                                    if not block:
                                        continue
                                    if have + len(block) > expected:
                                        stream.truncate(response_start_have)
                                        stream.flush()
                                        os.fsync(stream.fileno())
                                        raise DownloadError(f'{path.name}: response body exceeded expected chunk size {expected}; observed at least {have+len(block)} bytes including existing prefix. This response was rolled back; stopped without retry.')
                                    stream.write(block)
                                    have += len(block)
                                stream.flush()
                                os.fsync(stream.fileno())
                            if have != expected:
                                raise requests.exceptions.ChunkedEncodingError('Response ended early')
                            return
                    except requests.RequestException as exc:
                        if isinstance(exc, requests.exceptions.SSLError):
                            raise DownloadError(f'{path.name}: {network_error(exc)}; use --ca-bundle, or explicit --insecure only to reproduce the current connection mode') from None
                        if attempt + 1 == retries:
                            raise DownloadError(f'{path.name}: {network_error(exc)}; stopped after {retries} attempts. Parts retained; rerun to resume.') from None
                        delay = min(retry_delay * 2**attempt, 8)
                        print(f'{path.name}: {network_error(exc)}; next attempt {attempt+2}/{retries} in {delay}s', flush=True)
                        if stopped.wait(delay):
                            raise DownloadError('Download interrupted; partial chunks were retained')
            raise DownloadError('Chunk did not complete')

        pool = ThreadPoolExecutor(max_workers=workers)
        futures = [pool.submit(fetch, item) for item in ranges]
        try:
            for future in as_completed(futures):
                future.result()
        except BaseException:
            stopped.set()
            for future in futures:
                future.cancel()
            raise
        finally:
            pool.shutdown(wait=True, cancel_futures=True)
        stopped.set()
        if printer:
            printer.join()
        assembled = output.with_name(output.name + '.assembling')
        digest = hashlib.sha256()
        with assembled.open('wb') as target:
            for i, first, last in ranges:
                if paths[i].stat().st_size != last-first+1:
                    raise DownloadError(f'Chunk length changed: {paths[i].name}')
                with paths[i].open('rb') as source:
                    for block in iter(lambda: source.read(8*MIB), b''):
                        target.write(block)
                        digest.update(block)
            target.flush()
            os.fsync(target.fileno())
        if digest.hexdigest() != sha256:
            raise DownloadError('SHA256 mismatch: assembled file and parts retained; no final file published. Do not import this file as model weights.')
        if output.exists():
            raise DownloadError('Output appeared during download; refusing to replace it')
        # Exclusive publication prevents overwriting a file even if another program
        # creates output after the existence check. Both files are on the same FS.
        try:
            os.link(assembled, output)
        except OSError as exc:
            raise DownloadError(f'Cannot publish final file exclusively ({type(exc).__name__}); verified .assembling retained') from None
        assembled.unlink()
        print(f'VERIFIED {output}\nbytes={size}\nsha256={sha256}\nChunks retained in {parts}', flush=True)
        return output
    finally:
        stopped.set()
        if printer:
            printer.join()
        lock.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default=DEFAULT_URL)
    parser.add_argument('--output', default=str(Path.home() / 'Downloads' / 'InternVL3-78B_verified' / DEFAULT_NAME))
    parser.add_argument('--size', type=int, default=DEFAULT_SIZE, help='Official file length in bytes')
    parser.add_argument('--sha256', default=DEFAULT_SHA256, help='SHA256 from the same official revision')
    parser.add_argument('--workers', type=int, default=16, help='1..32; 16 and 32 supported; may change safely after previous run exits')
    parser.add_argument('--retries', type=int, default=4, help='1..20 total attempts per chunk per invocation, including the first; protocol errors stop immediately')
    parser.add_argument('--chunk-mib', type=int, default=64, help='Fixed chunk size; do not change when resuming')
    tls = parser.add_mutually_exclusive_group()
    tls.add_argument('--ca-bundle', help='Company-approved CA PEM file')
    tls.add_argument('--insecure', action='store_true', help='Explicitly disable TLS verification for this invocation')
    args = parser.parse_args()
    if args.ca_bundle and not Path(args.ca_bundle).is_file():
        parser.error('CA bundle file does not exist')
    if args.insecure:
        print('TLS verification disabled by explicit --insecure; official SHA256 remains mandatory.', flush=True)
        requests.packages.urllib3.disable_warnings(requests.packages.urllib3.exceptions.InsecureRequestWarning)
    try:
        download(args.url, args.output, args.size, args.sha256, workers=args.workers,
                 chunk_size=args.chunk_mib*MIB, verify=args.ca_bundle or not args.insecure,
                 retries=args.retries)
        return 0
    except KeyboardInterrupt:
        print('Stopped; parts retained. Re-run the same command to resume.', flush=True)
        return 130
    except (DownloadError, ValueError, OSError) as exc:
        print(f'ERROR: {exc}', flush=True)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
