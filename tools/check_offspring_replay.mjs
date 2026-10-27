/** Browser assertions using an existing FREE scripted demo. Never imports the engine. */
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import assert from 'node:assert/strict';
import crypto from 'node:crypto';
import {createRequire} from 'node:module';
import {pathToFileURL,fileURLToPath} from 'node:url';
const args={};for(let i=2;i<process.argv.length;i+=2)args[process.argv[i]]=process.argv[i+1];
if(!args['--run']||!args['--playwright']||!args['--browser'])throw new Error('Required: --run SCRIPTED_RUN --playwright MODULE --browser BINARY');
const source=fs.realpathSync(args['--run']);
const hashes={};
const hash=p=>crypto.createHash('sha256').update(fs.readFileSync(p)).digest('hex');
for(const name of ['manifest.json','state.json','events.jsonl'])hashes[name]=hash(path.join(source,name));
const data={manifest:JSON.parse(fs.readFileSync(path.join(source,'manifest.json'))),state:JSON.parse(fs.readFileSync(path.join(source,'state.json'))),events:fs.readFileSync(path.join(source,'events.jsonl'),'utf8').trim().split(/\r?\n/).map(JSON.parse),metrics:{},patches:{}};
assert.equal(data.manifest.backend,'scripted_fixture','Only free fixtures may be used');
assert.equal(data.manifest.status,'completed');
assert.equal(data.manifest.usage?.calls,0);
const births=data.events.filter(e=>e.event==='agent_born'),retirements=data.events.filter(e=>e.event==='agent_retired');
assert.ok(births.length&&retirements.length,'Demo must exercise lifecycle events');
const viewer=args['--viewer']||path.resolve(path.dirname(fileURLToPath(import.meta.url)),'../codetest/viewer.html');
const template=fs.readFileSync(viewer,'utf8');
const temp=fs.mkdtempSync(path.join(os.tmpdir(),'offspring-viewer-check-'));
const req=createRequire(import.meta.url),{chromium}=req(args['--playwright']);
const browser=await chromium.launch({headless:true,executablePath:args['--browser'],args:['--disable-background-networking']});
const errors=[],checks=[];
async function load(page,fixture,name){
 const target=path.join(temp,name+'.html');
 fs.writeFileSync(target,template.replace('/*__DATA__*/null',()=>JSON.stringify(fixture).replaceAll('<','\\u003c')));
 await page.goto(pathToFileURL(target).href);
 await page.waitForFunction(()=>typeof renderOffspring==='function');
}
try{
 const page=await browser.newPage({viewport:{width:1440,height:1000},colorScheme:'light'});
 page.on('pageerror',e=>errors.push(String(e)));
 await page.route(/^https?:/,r=>r.abort());
 await load(page,data,'actual');
 const initial=await page.evaluate(()=>{move(0);return {active:current().snap.agents.length,historical:historicalIds(-1).length,population:$('population').textContent,options:[...$('agent-select').options].map(o=>o.value)}});
 assert.equal(initial.active,data.state.config.k);assert.equal(initial.historical,data.state.config.k);
 for(const e of retirements)assert.ok(initial.options.includes(e.agent),'Retired agent remains selectable');
 checks.push('Initial founders counted at round -1; retired IDs remain selectable');
 for(let i=0;i<births.length;i++){
  const e=births[i],seen=await page.evaluate(e=>{move(rounds.indexOf(e.round));selectAgent(e.agent);return {active:current().snap.agents.map(a=>a.id),historical:historicalIds(e.round).length,family:[...$('lineage-graph').querySelectorAll('[data-lineage-agent]')].map(a=>a.dataset.lineageAgent),summary:summary(e),relevant:relevant(e)}},e);
  assert.equal(seen.active.length,data.state.config.k);assert.ok(seen.active.includes(e.agent));assert.ok(!seen.active.includes(e.retired_agent));assert.equal(seen.historical,data.state.config.k+i+1);
  assert.ok(seen.family.includes(e.parent)&&seen.family.includes(e.agent));assert.ok(seen.relevant);
  const before=await page.evaluate(e=>{move(rounds.indexOf(e.round-1));selectAgent(e.agent);return {title:$('agent-title').textContent,wealth:$('agent-wealth').textContent}},e);
  assert.match(before.title,/Not yet born/);assert.equal(before.wealth,'—');
 }
 checks.push('Every birth preserves active k and adds a historical individual; lineage connects parent to child');
 const retired=retirements[0],latest=await page.evaluate(id=>{move(rounds.length-1);selectAgent(id);$('lineage-scope').value='all';renderOffspring(current().r,current().snap);return {title:$('agent-title').textContent,rows:[...$('membership').querySelectorAll('tbody tr')].map(r=>r.textContent),family:[...$('lineage-graph').querySelectorAll('[data-lineage-agent]')].map(a=>a.dataset.lineageAgent),parentFilter:relevant({parent:id}),childFilter:relevant({child:id}),victimFilter:relevant({retired_agent:id})}},retired.agent);
 assert.match(latest.title,/Retired/);assert.equal(latest.family.length,data.state.config.k+births.length);assert.ok(latest.parentFilter&&latest.childFilter&&latest.victimFilter);
 assert.ok(latest.rows.find(r=>r.startsWith(births[0].agent.replace('agent-','A')))?.includes('—'),'Absent before birth is not represented as Solo');
 checks.push('Retired agent lens, lifecycle filtering and absent membership cells are truthful');
 if(args['--screenshots']){
  const out=path.resolve(args['--screenshots']);
  const relative=path.relative(source,out);
  assert.ok(relative.startsWith('..'+path.sep)||path.isAbsolute(relative),'Screenshots must be outside the source run');
  fs.mkdirSync(out,{recursive:true});
  await page.locator('#offspring-panel').screenshot({path:path.join(out,'html-lineage.png')});
  await page.locator('#detail').screenshot({path:path.join(out,'retired-agent-lens.png')});
 }
 const changed=structuredClone(data),account=changed.state.retired_accounts.find(a=>a.id===retired.agent);
 account.wealth=123.456;
 await load(page,changed,'late-credit-fixture');
 const balances=await page.evaluate(e=>{move(rounds.indexOf(e.round));selectAgent(e.agent);const atRetirement=$('agent-wealth').textContent;move(rounds.length-1);selectAgent(e.agent);return {atRetirement,latest:$('agent-wealth').textContent}},retired);
 assert.equal(balances.atRetirement,Number(retired.wealth).toFixed(3));assert.equal(balances.latest,'123.456');
 checks.push('Later retired-account credit is not backfilled into earlier rounds');
 const partial=structuredClone(data),born=births[0];
 partial.manifest.status='interrupted';partial.events=partial.events.filter(e=>e.seq<=born.seq);partial.state.snapshots=partial.state.snapshots.filter(s=>s.round<born.round);partial.state.agents=partial.state.snapshots.at(-1).agents;
 await load(page,partial,'partial-before-membership');
 const roster=await page.evaluate(()=>current().snap.agents.map(a=>a.id));
 assert.equal(roster.length,data.state.config.k);assert.ok(roster.includes(born.agent));assert.ok(!roster.includes(born.retired_agent));
 checks.push('Birth before membership in an interrupted round updates active roster');
 const legacy=structuredClone(data);delete legacy.state.offspring;delete legacy.state.retired_accounts;legacy.events=legacy.events.filter(e=>!['agent_born','agent_retired','offspring_investment'].includes(e.event));
 await load(page,legacy,'legacy-fixture');
 const legacyView=await page.evaluate(()=>({hidden:$('offspring-panel').hidden,label:$('population').parentElement.querySelector('.label').textContent,population:$('population').textContent}));
 assert.equal(legacyView.hidden,true);assert.equal(legacyView.label,'Population');assert.equal(legacyView.population,String(data.state.config.k));
 checks.push('Legacy run without offspring keeps its existing population display and hides new panel');
 assert.deepEqual(errors,[]);
 for(const [name,before] of Object.entries(hashes))assert.equal(hash(path.join(source,name)),before,'Source file changed: '+name);
 console.log(JSON.stringify({passed:true,checks,source_unchanged:true,paid_calls:0},null,2));
}finally{
 await browser.close();
 const resolved=fs.realpathSync(temp),expectedParent=fs.realpathSync(os.tmpdir());
 assert.equal(path.dirname(resolved),expectedParent);assert.ok(path.basename(resolved).startsWith('offspring-viewer-check-'));
 fs.rmSync(resolved,{recursive:true,force:true});
}
