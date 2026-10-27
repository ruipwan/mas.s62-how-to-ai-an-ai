# Opt-in transport timeout recovery

The default client behavior is unchanged: `max_retries=0`, a 15-second connection timeout, and a 180-second read timeout. A failed or ambiguous request retains its reservation and stops the calling experiment.

An authorized launcher can opt in with:

```python
transport = {
    "connect_timeout": 15,
    "read_timeout": 600,
    "max_retries": 2,
    "retry_backoff": (5, 15),
    "retry_reserve_cap_usd": 1.0,
}
```

These are `OpenRouter` constructor options. They change transport handling, not the model, prompt, structured output schema, reasoning setting, response token allocation, round count, or experimental mechanism. Apply the same transport policy to compared conditions and record the policy and recovery boundary in their manifests.

A read timeout limits how long Requests waits for incoming data. It is not an absolute deadline for the whole request, and 600 seconds does not guarantee that a large non-streaming response will complete. Connection attempts and response reads can make total elapsed time longer.

## Which failures can retry

Only identified connection or read timeout exceptions from the HTTP send are eligible. This includes Requests' `ConnectionError` wrapper around an actual urllib3 timeout. A generic connection error, HTTP error, missing usage, invalid JSON, or a charged empty response does not trigger a transport retry. Exception text alone never establishes that an error is a timeout.

The maximum is two additional sends per logical request, three attempts in total. Backoff before attempts two and three is 5 and 15 seconds respectively. Stop files are checked during backoff. Every reservation also rechecks the live request lease, campaign budget, episode budget, and retry reservation allowance. No retry occurs inside an HTTP adapter or SDK.

After the final timeout, its full reservation is retained and the experiment raises `BudgetStop`. A supervisor must not bypass this limit by repeatedly creating new logical request IDs or new episodes. Other live experiments are not blocked by a transient `uncertain` state for an acknowledged timeout.

## Accounting

Every physical send has a fresh ledger call ID, fresh reservation, and, for OpenAI, fresh `X-Client-Request-Id`. An opted-in timeout is atomically moved from pending to `reserved_error`, with `actual` remaining `NULL`. Its entire reservation remains part of accounted spending. This does not assert a confirmed charge or a zero charge.

The retry allowance defaults to $1 per ledger run. It sums the original reservations of all attempts numbered greater than one, including attempts that later settle for a smaller actual charge. This conservative allowance survives client and ledger-object recreation. It is separate from, and cannot override, the ordinary episode and campaign caps. Those caps include retained unknown charges as well as confirmed charges.

A timeout does not prove that the server stopped processing. The retry might overlap with an earlier server-side generation and both may incur charges. Client request IDs provide traceability; they do not guarantee service-side deduplication. The local lease protects against duplicate local owners of a ledger run. The launcher must additionally protect its output directory and avoid launching the same recovery twice.

## Resuming an interrupted logical request

For durable retry accounting within the same ledger run, pass a stable, nonsecret identifier:

```python
client.generate(messages, max_tokens, phase, agent,
                request_id="episode-step-0042")
```

The ledger persists each attempt number and a hash of the request contents in `transport_attempts`. Reusing the identifier after a retained timeout continues its attempt count. Reusing it with different contents, after a confirmed success, or after the retry limit is reached cannot send another request. Identifiers must be at most 128 ASCII letters, digits, periods, underscores, colons, or hyphens. Do not put prompts or credentials in identifiers.

If no identifier is provided, each `generate` invocation creates a new logical request. A recovery launcher using that mode must reconstruct the unique interrupted operation and preserve its prior attempt count; it must not blindly call it again after an unknown outcome.

For an audited legacy timeout that predates transport metadata, `initial_attempts_used=1` counts the original send when constructing the recovery client. This seed applies only to the first `generate` invocation. That invocation starts at attempt two and has at most two new sends; later logical requests start at attempt one. This option does not settle or release the historical reservation. The launcher must retain historical unknown charges and carry their spending into the recovery's effective budget, especially if using a new ledger run.

Credentials stay in the client process. Transport events store call identifiers, attempt numbers, safe error types, waits, and accounting amounts; they do not copy request headers, exception messages, prompts, or API keys.
