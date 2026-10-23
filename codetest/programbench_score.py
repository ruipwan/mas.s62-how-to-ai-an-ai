"""Use the pinned upstream scoring API, including active branches and ignored tests."""
from pathlib import Path


def summarize(root, instance_id):
    from programbench.eval.eval import EvaluationResult
    from programbench.eval.eval_batch import InstanceEvalSummary
    from programbench.utils.load_data import get_active_branches, get_ignored_tests, load_all_instances
    path = Path(root) / instance_id / f'{instance_id}.eval.json'
    result = EvaluationResult.model_validate_json(path.read_text())
    instances = {i['instance_id']: i for i in load_all_instances(include_tests=True)}
    if instance_id not in instances:
        raise ValueError('Instance missing from pinned upstream scoring data')
    instance = instances[instance_id]
    active = get_active_branches(instance)
    ignored_branches = set(result.test_branches) - set(active)
    result = result.for_branches(active).without_ignored(get_ignored_tests(instance))
    result.warnings = [w for w in result.warnings if not any(f'branch {b}' in w for b in ignored_branches)]
    summary = InstanceEvalSummary.from_eval_result(instance_id, result).model_dump()
    valid = summary['n_tests'] > 0 and not (summary['error_code'] or summary['test_branch_errors'] or summary['n_system_errors'])
    return {'status': 'native_scored' if valid else 'native_result_with_errors',
            'score': summary['score'] if valid else None, 'native_summary': summary,
            'passed': summary['n_resolved'], 'total': summary['n_tests'],
            'protocol': 'pinned_programbench_info_logic'}
