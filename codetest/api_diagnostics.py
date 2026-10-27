"""Read-only OpenAI model access checks and bounded, redacted HTTP diagnostics."""
import json
import os
import unicodedata
from urllib.parse import quote

import requests


MODELS_URL = 'https://api.openai.com/v1/models/'


def validate_api_key(key, variable):
    """Validate transport syntax only, without guessing a provider's key format."""
    if not isinstance(key, str) or not key:
        raise ValueError(f'Set {variable} in the environment')
    if not key.isascii() or any(c.isspace() or unicodedata.category(c).startswith('C') for c in key):
        raise ValueError(f'{variable} contains non-ASCII, whitespace, or control characters; paste the key again')
    return key


def redacted_text(value, key, limit):
    if not isinstance(value, str):
        return None
    # Redact before truncation so a key crossing the output limit cannot leak a prefix.
    if key:
        for secret in {key, json.dumps(key, ensure_ascii=True)[1:-1]}:
            value = value.replace(secret, '[REDACTED]')
    value = ''.join(c if c.isprintable() else ' ' for c in value)
    return value[:limit]


def response_diagnostics(response, key, *, include_detail=True):
    """Allowlist response metadata. Never inspect request headers or the request body."""
    headers = {k.lower(): v for k, v in response.headers.items() if isinstance(k, str)}
    body = response.content
    # The fallback also supports small response doubles used by the existing tests.
    body_bytes = len(body) if isinstance(body, bytes) else len(response.text.encode('utf-8'))
    result = {'status': response.status_code,
              'request_id': redacted_text(headers.get('x-request-id'), key, 256),
              'content_type': redacted_text(headers.get('content-type'), key, 256),
              'body_bytes': body_bytes,
              'server': redacted_text(headers.get('server'), key, 256),
              'retry_after': redacted_text(headers.get('retry-after'), key, 256)}
    if include_detail:
        result['detail'] = redacted_text(response.text, key, 2000)
    return result


def diagnose_openai(model):
    """GET model metadata once, with no completions, retry, redirect, or ledger access."""
    model = model.removeprefix('openai/')
    key = os.environ.get('OPENAI_API_KEY')
    result = {'provider': 'openai', 'model': redacted_text(model, key, 256),
              'method': 'GET', 'credential_present': bool(key), 'credential_format_valid': False,
              'ok': False, 'paid_requests': 0, 'ledger_touched': False,
              'limitation': 'Model access does not verify completion requests, schema acceptance, quota, or balance.'}
    try:
        validate_api_key(key, 'OPENAI_API_KEY')
    except ValueError as error:
        result.update(status='credential_malformed' if key else 'credential_missing', detail=str(error))
        return result
    result['credential_format_valid'] = True
    if not model or any(c.isspace() or unicodedata.category(c).startswith('C') for c in model):
        result.update(status='invalid_model', detail='The configured model ID is empty or contains whitespace or control characters.')
        return result
    try:
        response = requests.get(MODELS_URL + quote(model, safe=''),
                                headers={'Authorization': 'Bearer ' + key},
                                timeout=(10, 30), allow_redirects=False)
    except requests.RequestException as error:
        # Exception messages can contain a URL or headers, so only expose the exception class.
        result.update(status='timeout' if isinstance(error, requests.Timeout) else 'network_error',
                      error_type=type(error).__name__)
        return result
    result['http'] = response_diagnostics(response, key, include_detail=response.status_code != 200)
    status = response.status_code
    if status == 200:
        try:
            data = response.json()
        except ValueError:
            data = None
        if isinstance(data, dict) and data.get('object') == 'model' and data.get('id') == model:
            result.update(status='model_accessible', ok=True)
        else:
            result.update(status='unexpected_response', detail='The response was not metadata for the requested model.')
    elif status in (401, 403, 404, 429):
        result['status'] = {401: 'authentication_failed', 403: 'access_denied',
                            404: 'model_unavailable', 429: 'rate_limited'}[status]
    elif 300 <= status < 400:
        result['status'] = 'redirect_rejected'
    elif status >= 500:
        result['status'] = 'server_error'
    else:
        result['status'] = 'request_rejected'
    return result
