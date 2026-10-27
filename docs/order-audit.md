# Corrected market metrics and offline ordering audit

This document defines version 2 of the market diagnostics. Earlier descriptions that interpret rising bid correlations as learned prices do not apply: bidders already receive candidate test results, and pooled bids are confounded by team size, wealth and round.

## Metric definitions

`top_bid_tie_averaged_top_gain_share` is the uniform probability that a candidate drawn from the highest-bid tie also has maximum standalone test gain, averaged over contested rounds. A tie between one best and one worse candidate contributes 0.5, not 1. The legacy `top_bid_was_top_gain_share` is an alias for this corrected definition. The tie rule matches the classic selector or the verified queue as configured.

Selection comparisons require every eligible candidate and its base to have finite scores in [0, 1], valid passed counts and equal positive test totals. Incomplete rounds are excluded with explicit counts. No usable evidence is reported as null.

`first_publication_top_gain_share` describes the first successfully accepted publication, possibly after rejected attempts. It does not describe the first attempted merge. Its hit count, denominator and missing-outcome counts are reported separately. `first_publication_raw_gain_regret_mean` compares the chosen candidate's standalone gain with the largest standalone gain in that round, not with an achievable final oracle.

`bid_gain_spearman_within_round_mean` is an equal-weight mean over complete comparable rounds with at least three candidates and nonconstant bids and gains. Existing pooled and early/late correlations remain descriptive. Completed-round merge rates exclude unfinished rounds.

## Offline replay

Run from an analysis checkout after all input runs have completed:

```bash
python -m codetest.order_audit \
  --run /path/to/completed-baseline \
  --run /path/to/completed-market \
  --out /path/to/new-audit-output \
  --random-seeds 7 17 27 --merge-slots 10
```

The command makes no model calls and never opens the campaign ledger. It requires a fresh output directory, complete input records, unchanged task inputs and the original locally available Docker image. It never builds or downloads an image. Fixture validation needs neither Docker nor an API credential.

Each round restarts at its observed public commit in an independent disposable Git clone. All orderings receive the same eligible candidates and use the same strict test-set gate: at least one additional passing test and no previously passing test lost. Before the package first scores, the original bootstrap exception applies. Every applied tree is freshly verified, including the first candidate. Historical bid attempt order is used when recoverable; otherwise seeded reconstruction is labelled explicitly. Undefined gain is not silently treated as zero. Random orders use three fixed seeds and report their descriptive mean and range, not confidence intervals across independent tasks.

These are one-round counterfactuals, not full replacement trajectories. Different accepted patches would change future agent prompts, work and wealth; those changes are not simulated. Current visible development tests do not become hidden evaluation tests by being rerun. The original merge-slots intervention changes both capacity and acceptance semantics, so it is not a pure throughput ablation.

The audit checks candidate and patch hashes, frozen verifier versions, and source files and Git references before and after execution. Repeated verification of an identical Git tree must agree or the audit fails without retrying. Corrected metrics, per-round and per-policy results, implementation hashes, verification work and status are written separately. Original metrics and event files are preserved. Failed or existing output directories are never overwritten or automatically resumed.
