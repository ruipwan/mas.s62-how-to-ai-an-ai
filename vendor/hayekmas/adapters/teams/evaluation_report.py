"""Portable comparison dashboard; all model text is rendered as text, never HTML."""

import json
from pathlib import Path


TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>EoM · Teams and independent sampling</title><style>
:root{color-scheme:dark;font:16px/1.5 system-ui,sans-serif;background:#101724;color:#e9edf3}body{max-width:1200px;margin:auto;padding:32px}h1{font-size:28px;margin:0}h2{font-size:19px}p{color:#bac6d8}a{color:#90dfcf}select,button{font:inherit;color:inherit;background:#26344a;border:1px solid #53627a;border-radius:6px;padding:7px}table{width:100%;border-collapse:collapse}th,td{text-align:left;padding:12px;border-bottom:1px solid #344052;font-variant-numeric:tabular-nums}th{color:#afc0d6;font-weight:500}section{margin-top:24px;padding:20px;background:#192334;border:1px solid #344052;border-radius:10px}.scroll{overflow:auto}.bar{display:inline-block;height:9px;background:#85d9c5;border-radius:3px;margin-right:10px;min-width:1px}pre{white-space:pre-wrap;overflow-wrap:anywhere;font:14px/1.6 ui-monospace,monospace}details{margin:15px 0}summary{cursor:pointer}small{color:#a6b6ca}.controls{display:flex;gap:20px;flex-wrap:wrap}.badge{font-size:12px;letter-spacing:.5px;color:#90dfcf;margin:12px 0}.notice{border-left:3px solid #d9b67f;padding-left:14px}@media(max-width:640px){body{padding:16px}th,td{padding:8px}}
</style></head><body><h1>Teams and independent sampling</h1><div class="badge" id="status"></div>
<p>Same questions and model. Team submits one answer. Single agent is the first independent attempt. Pass@k estimates whether any of k independent attempts passes the grader.</p>
<p class="notice">Pass@k uses an oracle: it does not identify the correct answer without grading. Methods have different compute budgets. All pass@k values reuse one sample pool; their costs must not be added together.</p>
<section><h2>Evaluation results</h2><p id="progress"></p><div class="scroll"><table><thead><tr><th>Method</th><th>Pass rate</th><th>Calls incl. judge</th><th>Reference output tokens</th><th>USD incl. judge</th></tr></thead><tbody id="summary"></tbody></table></div><p id="total"></p><small>Pass@k compute columns estimate the cost of k random attempts from the recorded pool. The actual experiment also pays for all n attempts and any training. Native token counts and per-call USD appear in comparison.json and api_usage.jsonl.</small></section>
<section><h2>Inspect a matched task</h2><div class="controls"><label>Task <select id="task"></select></label><label>k <select id="k"></select></label></div><div id="detail"></div></section>
<section><h2>Interpretation</h2><p id="interpretation"></p><p>Test tasks start from independent copies of the same initial or trained team state. Test reflection is disabled; test outcomes never update the next task. Within-task negotiation and transfers remain active. Training is recorded separately.</p><p><a href="comparison.json">Full comparison data</a> · <a href="manifest.json">Configuration and provenance</a> · <a href="outcomes.jsonl">All recorded outcomes</a></p></section>
<script type="application/json" id="data">__DATA__</script><script>
'use strict';
const data=JSON.parse(document.getElementById('data').textContent),$=id=>document.getElementById(id);
const el=(tag,text)=>{const node=document.createElement(tag);if(text!==undefined)node.textContent=text;return node};
const pct=x=>(100*x).toFixed(1)+'%',num=x=>x===null||x===undefined?'unavailable':x.toFixed(2),usd=x=>x===null||x===undefined?'unavailable':'$'+x.toFixed(4);
$('status').textContent=`${data.backend==='demo'?'SCRIPTED DEMO · mechanism check only':data.backend} · ${data.environment} · ${data.dataset_split||'test'} split · ${data.status}`;
$('progress').textContent=`${data.completed_pairs}/${data.planned_pairs} task/replicate pairs completed · n=${data.samples_per_task} samples per task · pass threshold ${data.pass_threshold}`;
for(const row of data.summary){const tr=el('tr'),method=row.method==='single_agent'?'Single agent':row.method==='team'?'Team':row.method;tr.append(el('td',method));const cell=el('td'),bar=el('span');bar.className='bar';bar.style.width=(row.pass_rate*100)+'px';cell.append(bar,document.createTextNode(pct(row.pass_rate)));tr.append(cell,el('td',num(row.usage.calls)),el('td',num(row.usage.reference_output_tokens)),el('td',usd(row.usage.cost_usd)));$('summary').append(tr)}
if(!data.summary.length)$('summary').append(el('tr','Final aggregate results appear only after every planned task completes.'));
$('total').textContent=data.total_actual_usage?`Entire experiment: ${data.total_actual_usage.calls} calls · ${usd(data.total_actual_usage.cost_usd)} recorded cost (including training and the full sampling pool).`:'';
const rates=data.summary.filter(x=>x.method==='single_agent');$('interpretation').textContent=data.backend==='demo'?'These answers are scripted. Scores do not measure model ability or team emergence.':rates.some(r=>r.pass_rate>=.9)?'Single-agent accuracy is already high on this selection. A ceiling may hide collaboration gains; inspect the untouched test set and compute costs before drawing conclusions.':'Inspect task-level outcomes and grading quality before interpreting differences. A small task sample or one replicate cannot establish a reliable advantage.';
data.ks.forEach(k=>$('k').add(new Option(String(k),String(k))));
data.pairs.forEach((p,i)=>$('task').add(new Option(`${i+1}. ${p.subject||p.task_id} · seed ${p.seed}`,String(i))));
function render(){const root=$('detail');root.replaceChildren();const pair=data.pairs[Number($('task').value)];if(!pair){root.append(el('p','No complete matched tasks yet.'));return}const k=$('k').value;root.append(el('h3',pair.task_id),el('pre',pair.problem),el('p',`Team score ${num(pair.team.score)} · ${pair.team.passed?'passed':'failed'} | Single-agent score ${num(pair.samples[0].score)} | pass@${k}: ${pct(pair.pass_at_k[k])}`));const link=el('a','Open team formation and conversation replay');link.href=pair.team.replay;root.append(link);const team=el('details');team.append(el('summary','Team final answer'),el('pre',pair.team.answer||'No final answer'));root.append(team);for(const sample of pair.samples){const d=el('details');d.append(el('summary',`Independent attempt ${sample.sample+1} · score ${num(sample.score)} · ${sample.passed?'passed':'failed'}${sample.invalid?' · '+sample.invalid:''}`),el('pre',String(sample.answer??'No valid answer')));root.append(d)}}
$('task').onchange=render;$('k').onchange=render;render();
</script></body></html>"""


def write_report(data, out):
    payload = (
        json.dumps(data, ensure_ascii=False, allow_nan=False)
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
    )
    path = Path(out) / "comparison.html"
    temporary = path.with_suffix(".html.tmp")
    temporary.write_text(TEMPLATE.replace("__DATA__", payload), encoding="utf-8")
    temporary.replace(path)
