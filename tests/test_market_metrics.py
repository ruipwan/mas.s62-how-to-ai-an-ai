import unittest
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'vendor')]
from codetest.report import market_metrics


def check(passed, total=100):
    return {'passed': passed, 'total': total, 'score': passed / total}


def snapshot(round_id, passed=10):
    return {'event': 'snapshot', 'round': round_id, 'feedback': check(passed)}


def candidate(team, bid, gain, round_id=0, base=10, eligible=True):
    return {'event': 'candidate', 'round': round_id, 'team': team, 'bid': bid,
            'eligible': eligible, 'check': check(base + gain)}


def publication(team, round_id=0, **extra):
    return {'event': 'publication', 'round': round_id, 'team': team, **extra}


def summary(events, merge_slots=10):
    return market_metrics({'config': {'merge_slots': merge_slots}}, events)


class MarketMetricDefinitionTests(unittest.TestCase):
    def test_equal_bids_use_uniform_expectation_not_any_best_candidate(self):
        events = [snapshot(-1), candidate('low', .1, 1), candidate('middle', .1, 2),
                  candidate('best', .1, 3), publication('low'), snapshot(0, 11)]
        result = summary(events)
        self.assertEqual(result['metrics_version'], 2)
        self.assertAlmostEqual(result['top_bid_tie_averaged_top_gain_share'], 1 / 3)
        self.assertEqual(result['top_bid_was_top_gain_share'], result['top_bid_tie_averaged_top_gain_share'])
        self.assertEqual(result['top_bid_tie_rounds'], 1)
        self.assertEqual(result['top_bid_tie_rounds_denominator'], 1)
        self.assertEqual(result['contested_rounds'], 1)
        self.assertEqual(result['first_publication_top_gain_share'], 0)
        self.assertEqual(result['first_publication_top_gain_hits'], 0)
        self.assertEqual(result['first_publication_comparable_contested_rounds'], 1)
        self.assertEqual(result['first_publication_raw_gain_regret_mean'], 2)
        self.assertEqual(result['first_publication_raw_gain_regret_total'], 2)
        self.assertEqual(result['first_publication_raw_gain_regret_rounds'], 1)
        self.assertIsNone(result['bid_gain_spearman_within_round_mean'])
        self.assertEqual(result['bid_gain_spearman_within_round_valid_rounds'], 0)

    def test_tie_group_matches_classic_isclose_and_queue_exact_ordering(self):
        events = [snapshot(-1), candidate('best-gain', .1, 3), candidate('highest-bid', .1 + 5e-11, 1)]
        classic, queue = summary(events, 1), summary(events, 10)
        self.assertEqual(classic['top_bid_tie_averaged_top_gain_share'], .5)
        self.assertEqual(classic['top_bid_tie_rounds'], 1)
        self.assertIn('isclose', classic['bid_tie_policy'])
        self.assertEqual(queue['top_bid_tie_averaged_top_gain_share'], 0)
        self.assertEqual(queue['top_bid_tie_rounds'], 0)
        self.assertIn('exact equality', queue['bid_tie_policy'])
        events[-1]['bid'] = .1 + 5e-9
        self.assertEqual(summary(events, 1)['top_bid_tie_rounds'], 0)

    def test_tie_expectation_is_round_weighted_not_candidate_weighted(self):
        events = [snapshot(-1), candidate('a', .1, 1), candidate('b', .1, 2), snapshot(0, 12),
                  candidate('c', .1, 1, 1, 12), candidate('d', .1, 2, 1, 12),
                  candidate('e', .1, 3, 1, 12), candidate('f', .1, 4, 1, 12)]
        result = summary(events)
        self.assertAlmostEqual(result['top_bid_tie_averaged_top_gain_share'], (.5 + .25) / 2)
        self.assertEqual(result['contested_rounds'], 2)

    def test_no_publication_is_missing_outcome_not_a_missed_best_candidate(self):
        events = [snapshot(-1), candidate('a', .2, 4), candidate('b', .1, 1),
                  {'event': 'merge_rejected', 'round': 0, 'team': 'a', 'reason': 'conflict'}]
        result = summary(events)
        self.assertEqual(result['contested_rounds_without_publication'], 1)
        self.assertEqual(result['contested_rounds_unmatched_first_publication'], 0)
        self.assertEqual(result['first_publication_comparable_contested_rounds'], 0)
        self.assertIsNone(result['first_publication_top_gain_share'])
        self.assertIsNone(result['first_publication_raw_gain_regret_mean'])
        self.assertEqual(result['first_publication_raw_gain_regret_rounds'], 0)

    def test_first_successful_publication_can_follow_rejection_and_have_lower_bid(self):
        events = [snapshot(-1), candidate('highest', .3, 5), candidate('lower', .1, 2),
                  {'event': 'merge_rejected', 'round': 0, 'team': 'highest', 'reason': 'breaks passing tests: t'},
                  publication('lower', before=check(10), after=check(50)), publication('highest')]
        result = summary(events)
        self.assertEqual(result['top_bid_tie_averaged_top_gain_share'], 1)
        self.assertEqual(result['first_publication_top_gain_share'], 0)
        self.assertEqual(result['first_publication_raw_gain_regret_mean'], 3)
        self.assertEqual(result['first_publication_comparable_contested_rounds'], 1)
        self.assertIn('possibly after rejected attempts', result['descriptive_caveat'])
        self.assertEqual(result['merges'], 2)

    def test_unmatched_first_publication_is_not_replaced_by_a_later_match(self):
        events = [snapshot(-1), candidate('a', .1, 1), candidate('b', .2, 2),
                  publication('unknown'), publication('b')]
        result = summary(events)
        self.assertEqual(result['contested_rounds_unmatched_first_publication'], 1)
        self.assertEqual(result['contested_rounds_without_publication'], 0)
        self.assertIsNone(result['first_publication_top_gain_share'])

    def test_within_round_correlations_are_equally_weighted_and_exclude_undefined_rounds(self):
        events = [snapshot(-1)]
        for round_id, gains in enumerate(([1, 2, 3], [5, 4, 3, 2, 1], [1, 2], [2, 2, 2])):
            if round_id:
                events.append(snapshot(round_id - 1))
            events.extend(candidate(str(i), (i + 1) / 10, gain, round_id) for i, gain in enumerate(gains))
        result = summary(events)
        self.assertAlmostEqual(result['bid_gain_spearman_within_round_mean'], 0)
        self.assertEqual(result['bid_gain_spearman_within_round_valid_rounds'], 2)
        self.assertIsNotNone(result['bid_gain_spearman'])
        self.assertIn('not evidence of price learning', result['descriptive_caveat'])

    def test_incomplete_candidate_checks_exclude_the_whole_selection_comparison(self):
        events = [snapshot(-1), candidate('a', .1, 1), candidate('b', .2, 2), candidate('missing', .9, 3)]
        events[-1]['check'] = {'score': None, 'passed': 0, 'total': 100}
        events.append(publication('b'))
        result = summary(events)
        self.assertEqual(result['eligible_candidates_scored'], 2)
        self.assertEqual(result['selection_comparable_rounds'], 0)
        self.assertEqual(result['selection_excluded_incomplete_rounds'], 1)
        self.assertEqual(result['contested_rounds'], 0)
        self.assertIsNone(result['first_publication_top_gain_share'])
        events[3]['eligible'] = False
        result = summary(events)
        self.assertEqual(result['selection_comparable_rounds'], 1)
        self.assertEqual(result['first_publication_top_gain_share'], 1)

    def test_invalid_start_or_candidate_score_excludes_selection_despite_valid_counts(self):
        for side in ('start', 'candidate'):
            for score in (None, float('nan'), float('inf'), -.01, 1.01, 'missing'):
                with self.subTest(side=side, score=score):
                    events = [snapshot(-1), candidate('a', .1, 1), candidate('b', .2, 2), publication('b')]
                    feedback = events[0]['feedback'] if side == 'start' else events[2]['check']
                    if score == 'missing':
                        del feedback['score']
                    else:
                        feedback['score'] = score
                    result = summary(events)
                    self.assertEqual(result['eligible_candidates_scored'], 0 if side == 'start' else 1)
                    self.assertEqual(result['selection_comparable_rounds'], 0)
                    self.assertEqual(result['selection_excluded_incomplete_rounds'], 1)
                    self.assertEqual(result['contested_rounds'], 0)
                    self.assertIsNone(result['top_bid_tie_averaged_top_gain_share'])
                    self.assertIsNone(result['first_publication_top_gain_share'])

    def test_valid_zero_score_start_is_comparable(self):
        events = [snapshot(-1, 0), candidate('a', .1, 1, base=0),
                  candidate('b', .2, 2, base=0), publication('b')]
        result = summary(events)
        self.assertEqual(result['eligible_candidates_scored'], 2)
        self.assertEqual(result['selection_comparable_rounds'], 1)
        self.assertEqual(result['top_bid_tie_averaged_top_gain_share'], 1)
        self.assertEqual(result['first_publication_top_gain_share'], 1)
        self.assertEqual(result['first_publication_raw_gain_regret_mean'], 0)

    def test_different_test_totals_and_missing_start_are_not_comparable(self):
        events = [snapshot(-1), candidate('a', .1, 1), candidate('b', .2, 2)]
        events[-1]['check'] = check(12, 101)
        self.assertEqual(summary(events)['selection_excluded_incomplete_rounds'], 1)
        self.assertEqual(summary(events[1:])['eligible_candidates_scored'], 0)

    def test_equal_gains_count_ties_but_not_contested_selection_rounds(self):
        events = [snapshot(-1), candidate('a', .1, 2), candidate('b', .1, 2), publication('a')]
        result = summary(events)
        self.assertEqual(result['top_bid_tie_rounds'], 1)
        self.assertEqual(result['top_bid_tie_rounds_denominator'], 1)
        self.assertEqual(result['contested_rounds'], 0)
        self.assertIsNone(result['top_bid_tie_averaged_top_gain_share'])
        self.assertIsNone(result['first_publication_top_gain_share'])

    def test_empty_and_swarm_samples_do_not_claim_selection_quality(self):
        for events in ([], [{'event': 'candidate', 'round': 0, 'task': 'module', 'eligible': True, 'bid': .1},
                           {'event': 'publication', 'round': 0, 'task': 'module'}]):
            with self.subTest(events=events):
                result = summary(events)
                self.assertEqual(result['eligible_candidates_scored'], 0)
                self.assertEqual(result['completed_rounds'], 0)
                self.assertEqual(result['first_publication_comparable_contested_rounds'], 0)
                self.assertIsNone(result['bid_gain_spearman'])
                self.assertIsNone(result['bid_gain_spearman_within_round_mean'])
                self.assertIsNone(result['top_bid_was_top_gain_share'])
                self.assertIsNone(result['merges_per_round'])

    def test_merge_rate_only_uses_completed_rounds_not_partial_next_round(self):
        events = [snapshot(-1), publication('a'), publication('b'), snapshot(0, 12),
                  publication('c', round_id=1)]
        result = summary(events)
        self.assertEqual(result['merges'], 3)
        self.assertEqual(result['completed_rounds'], 1)
        self.assertEqual(result['completed_round_merges'], 2)
        self.assertEqual(result['merges_per_round'], 2)
        partial = summary([snapshot(-1), publication('a')])
        self.assertEqual(partial['merges'], 1)
        self.assertEqual(partial['completed_round_merges'], 0)
        self.assertIsNone(partial['merges_per_round'])


if __name__ == '__main__':
    unittest.main()
