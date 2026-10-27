/** Local, offline video export from an existing experiment. Never imports Python/model code.
 * Windows example:
 * node export_replay_video.mjs --run \\wsl.localhost\Ubuntu\home\USER\repo\runs\RUN
 *   --out C:\OUTPUT\RUN --allow-partial --playwright C:\...\node_modules\playwright
 *   --browser C:\...\chrome.exe --ffmpeg C:\...\ffmpeg.exe
 * Output must be new and outside the source run. A manifest records source/render hashes.
 */
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import {createRequire} from 'node:module';
import {pathToFileURL} from 'node:url';
import {spawnSync} from 'node:child_process';

const args = {};
for (let i=2;i<process.argv.length;i++) {
  const key=process.argv[i];
  if (!key.startsWith('--')) throw new Error('Expected --option');
  if (key==='--allow-partial') args[key]=true;
  else args[key]=process.argv[++i];
}
if (!args['--run'] || !args['--out']) throw new Error('Required: --run SOURCE --out NEW_DIRECTORY');
const source=fs.realpathSync(args['--run']), out=path.resolve(args['--out']);
const inside=(a,b)=>{const r=path.relative(a,b);return r===''||(!r.startsWith('..'+path.sep)&&r!=='..'&&!path.isAbsolute(r));};
if (inside(source,out)||inside(out,source)||fs.existsSync(out)) throw new Error('Output must be new and separate from the source run.');
const hash=b=>crypto.createHash('sha256').update(b).digest('hex');
const sourceFiles=['manifest.json','state.json','events.jsonl'];
const captured=Object.fromEntries(sourceFiles.map(n=>[n,fs.readFileSync(path.join(source,n))]));
const manifest=JSON.parse(captured['manifest.json']), state=JSON.parse(captured['state.json']);
if (manifest.status==='running') throw new Error('Refusing a moving run: export after it stops or completes.');
if (manifest.status!=='completed'&&!args['--allow-partial']) throw new Error('Incomplete run requires --allow-partial.');
const events=captured['events.jsonl'].toString('utf8').split(/\r?\n/).filter(Boolean).map(s=>JSON.parse(s));
if (events.some((e,i)=>e.seq!==i)) throw new Error('Event sequence is incomplete.');
if (!state.snapshots?.length) throw new Error('No recorded state snapshots.');
const repo=path.dirname(path.dirname(source));
const viewerPath=args['--viewer']||path.join(repo,'codetest','viewer.html');
const template=fs.readFileSync(viewerPath,'utf8');
if (!template.includes('/*__DATA__*/null')||!template.includes('function drawGraph')) throw new Error('Unsupported viewer template.');
const metricPath=path.join(source,'metrics.json');
const metrics=fs.existsSync(metricPath)?JSON.parse(fs.readFileSync(metricPath,'utf8')):{};
const partial=manifest.status!=='completed';
const title=args['--title']||path.basename(source);
const data={manifest,state,events:events.filter(e=>!['decision','api_reserved'].includes(e.event)),metrics,patches:{}};
const encoded=JSON.stringify(data).replaceAll('<','\\u003c').replaceAll('\u2028','\\u2028').replaceAll('\u2029','\\u2029');

// Presentation-only adjustments. The inference renderer and run data stay untouched.
const css=`
:root{color-scheme:light;--bg:#eef1f5;--panel:#fff;--ink:#15283b;--muted:#607286;--line:#dce4ee;--soft:#edf2f7;--accent:#007d78;--accent-soft:#e3f3ee;--amber:#a55d00;--red:#b53742;--purple:#7452b0;--blue:#2672b0}
html,body{width:1920px;height:1080px;overflow:hidden!important}body{font-family:Segoe UI,Arial,sans-serif;font-size:17px}header{height:118px;padding:16px 28px;position:relative}h1{font-size:26px;margin:6px 0 0}.topline{gap:12px}.brand{font-size:19px}.version,.meta,.subtitle,#notice{display:none!important}.tag{font-size:15px}.mark{width:28px;height:28px}.shell{max-width:none;padding:16px 26px;display:grid;grid-template-columns:1.3fr 1fr;grid-template-rows:70px 42px 488px 264px;gap:14px;height:936px}.stats{grid-column:1/-1;grid-row:1;margin:0;gap:10px}.value{font-size:29px}.stat .label,.hint{font-size:15px}.toolbar{grid-column:1/-1;grid-row:2;margin:0}.toolbar button,.toolbar label,.toolbar input{display:none}.round{font-size:23px;min-width:220px}#video-scene{font-size:19px;color:var(--muted)}.workspace{display:contents}.workspace>section{grid-column:1;grid-row:3}.panel-head{padding:11px 16px}h2{font-size:19px}.eyebrow{font-size:11px}.legend{font-size:13px}.graph-wrap{height:382px;min-height:0;padding:4px 10px}#graph{height:374px!important;width:100%;max-height:374px}.graph-footer{padding:8px 14px;font-size:13px;display:block}.graph-footer>span:last-child{display:block;font-size:12px;margin-top:3px}.side{grid-column:2;grid-row:3;display:block}.side>#detail,.events-controls{display:none}.side>section:last-child{height:488px}.event-list{height:431px;padding:5px 9px;overflow:hidden}.event{padding:10px 12px;font-size:17px;margin-bottom:4px}.event.chosen{background:#e1f2ed;box-shadow:inset 4px 0 #007d78}.event-meta{font-size:13px}.event-summary{font-size:15px;line-height:1.4}.evolution{grid-column:1;grid-row:4;margin:0}.evolution .scroll-table{max-height:186px;overflow:hidden;padding:4px 10px}.evolution-table td{min-width:42px}.evolution-cell{height:40px}.evolution-cell .size-number{font-size:14px}.evolution-cell .size-delta{font-size:11px;top:18px}.evolution-caption{display:none}.trajectory{grid-column:2;grid-row:4;margin:0}.trajectory-grid{display:block;padding:7px 12px}.trajectory-grid>div>p,.trajectory-table{display:none}.trajectory svg{height:200px;max-height:200px}.bottom,.usage,footer,#benchmark-result,[id^=handoff]{display:none!important}.tiny{font-size:13px}#video-footer{position:fixed;bottom:0;left:0;width:1920px;height:30px;background:#15283b;color:#e8eff6;font-size:14px;padding:4px 28px;display:flex;justify-content:space-between}#video-banner{position:absolute;right:28px;top:65px;font-weight:650;font-size:18px;color:${partial?'#a54712':'#007d78'}}
.group-title{font-size:13px}.node-label{font-size:16px}.node-wealth{font-size:13px}.repo-text{font-size:14px}.graph-footer>span{display:block}.graph-footer>span:last-child,.graph-footer #round-outcome{display:none}.trajectory svg{height:174px;max-height:174px}.evolution table{font-size:14px}
`;
const presentation={title,partial,status:manifest.status,completed:state.completed_rounds??state.snapshots.filter(s=>s.round>=0).length,source:path.basename(source)};
const adapter=`
<script>
const VIDEO=${JSON.stringify(presentation).replaceAll('<','\\u003c')};
document.querySelector('h1').textContent=VIDEO.title;
$('backend').textContent=M.backend==='scripted_fixture'?'SCRIPTED FIXTURE':(S.config.provider||'openrouter').toUpperCase();
$('spend').parentElement.querySelector('.label').textContent='Usage-based cost estimate';
document.querySelector('.toolbar').insertAdjacentHTML('beforeend','<span id="video-scene"></span>');
document.querySelector('header').insertAdjacentHTML('beforeend','<div id="video-banner"></div>');
$('video-banner').textContent=VIDEO.partial?'PARTIAL / 未完成 · '+VIDEO.completed+' complete round(s)':'COMPLETED / 已完成 · '+VIDEO.completed+' rounds';
document.body.insertAdjacentHTML('beforeend','<div id="video-footer"><span>Visible development tests · not a hidden generalization score</span><span>Virtual wealth ≠ API dollars · recorded round summaries</span></div>');
const eventKinds=new Set(['invitation','join','membership','candidate','ballot','publication','transfer','burn','merge_rejected','merge_skipped','no_publication','regression_blocked','rent','stopped','answer_checkpoint']);
const baseSummary=summary;
summary=function(e){
 if(e.event==='candidate')return teamName(e.team)+' · bid '+num(e.bid)+' · '+(e.check?.score==null?'unscored':e.check.passed+'/'+e.check.total+' visible tests')+' · '+(e.eligible?'eligible':'ineligible');
 if(e.event==='publication')return teamName(e.team)+' published · bid '+num(e.bid)+' · '+(e.after?.score==null?'unscored':e.after.passed+'/'+e.after.total+' visible tests');
 if(e.event==='membership')return 'Recorded membership for '+e.agents.length+' agents';
 if(e.event==='invitation')return short(e.from)+' invited '+short(e.to);
 if(e.event==='ballot')return short(e.agent)+' · '+(e.yes?'yes':'no')+' · pledge '+num(e.pledge);
 if(e.event==='stopped')return 'Experiment stopped before the planned run completed';
 return baseSummary(e);
};
const originalEvents=renderEvents;
renderEvents=function(rows){let visible=rows.filter(e=>eventKinds.has(e.event));if(selectedEvent)visible=visible.filter(e=>e.seq<=selectedEvent.seq);originalEvents(visible.slice(-5));};
window.exportReplayFrame=function(round,seq,focus,label){
 selected=focus||'';selectedEvent=null;index=Math.max(0,rounds.indexOf(round));render();
 selectedEvent=seq==null?null:E.find(e=>e.seq===seq);renderEvents(current().events);
 $('video-scene').textContent=label;
 const lastComplete=Math.max(...snaps.map(s=>s.round));
 $('round-label').textContent=round<0?'Initial state':('Round '+(round+1)+(round>lastComplete?' · partial':' · completed'));
 const charges=E.filter(e=>e.round<=round&&e.event==='api_charged');
 const ledger=Number.isFinite(M.usage?.accounted_usd)?' · run incl. unresolved $'+M.usage.accounted_usd.toFixed(6):'';
 $('calls').textContent=charges.length+' calls'+ledger;
 $('network-note').textContent=focus?'Agent focus: '+short(focus)+' · recorded round interactions':'Coalition summary · recorded round payments';
 document.querySelector('.trajectory h2').textContent='Full recorded score trajectory';
 if(!snaps.some(s=>s.feedback?.score!=null)&&!E.some(e=>e.event==='candidate'&&e.check?.score!=null))$('trajectory').innerHTML='<rect x="0" y="0" width="650" height="205" rx="8" fill="var(--soft)"/><text x="325" y="90" text-anchor="middle" font-size="20" fill="var(--ink)">No valid numeric test scores recorded</text><text x="325" y="123" text-anchor="middle" font-size="15" fill="var(--muted)">Missing scores are not zero scores.</text>';
 return {round,seq,focus,score:$('score').textContent,spend:$('spend').textContent};
};
</script>`;
let html=template.replace('/*__DATA__*/null',()=>encoded).replace('<!--__HANDOFF_VIEW__-->','');
html=html.replace('Math.floor(w/240)','Math.floor(w/175)').replace('</style>',()=>css+'</style>').replace('</body>',()=>adapter+'</body>');
html=html.replace('${privateCount} private messages · ${publicCount} public posts','${privateCount} msgs · ${publicCount} posts');
for(const n of sourceFiles)if(hash(fs.readFileSync(path.join(source,n)))!==hash(captured[n]))throw new Error('Source changed during snapshot.');
fs.mkdirSync(out,{recursive:true});
const htmlPath=path.join(out,'presentation.html');fs.writeFileSync(htmlPath,html);
const frameDir=path.join(out,'frames');fs.mkdirSync(frameDir);
const scenes=[{round:-1,seq:null,focus:'',label:'Initial population · original scaffold',seconds:3}];
const recordedRounds=[...new Set(events.map(e=>e.round).filter(r=>r>=0))].sort((a,b)=>a-b);
for(const round of recordedRounds){
 const rows=events.filter(e=>e.round===round);
 const membership=rows.find(e=>e.event==='membership');
 scenes.push({round,seq:membership?.seq??null,focus:'',label:'Team formation and round summary',seconds:3});
 const candidates=rows.filter(e=>e.event==='candidate');
 if(candidates.length){const c=candidates.reduce((a,b)=>(b.bid??0)>(a.bid??0)?b:a);scenes.push({round,seq:c.seq,focus:c.members?.[0]||'',label:'Highest recorded candidate bid · '+candidates.length+' candidate(s)',seconds:3});}
 const publications=rows.filter(e=>e.event==='publication');
 if(publications.length){const e=publications.at(-1);scenes.push({round,seq:e.seq,focus:'',label:publications.length+' accepted contribution(s) in this round',seconds:4});}
 else {const stopped=rows.find(e=>e.event==='stopped');scenes.push({round,seq:stopped?.seq??rows.at(-1)?.seq??null,focus:'',label:stopped?'Partial round · interrupted before completion':'No accepted contribution in this round',seconds:4});}
}
scenes.push({round:recordedRounds.at(-1)??-1,seq:null,focus:'',label:partial?'PARTIAL preview · no completed-run comparison claim':'Completed replay · compare quality, spend and elapsed time',seconds:4});
const req=createRequire(import.meta.url);
const {chromium}=req(args['--playwright']||'playwright');
const launch={headless:true,args:['--disable-gpu','--disable-background-networking']};
if(args['--browser'])launch.executablePath=args['--browser'];
const browser=await chromium.launch(launch);
const errors=[];
try{
 const page=await browser.newPage({viewport:{width:1920,height:1080},deviceScaleFactor:1,colorScheme:'light',reducedMotion:'reduce'});
 page.on('pageerror',e=>errors.push(String(e)));
 await page.route(/^https?:/,r=>r.abort());
 await page.goto(pathToFileURL(htmlPath).href,{waitUntil:'load'});
 if(errors.length)throw new Error('Replay page errors: '+errors.join('\n'));
 await page.waitForFunction(()=>typeof window.exportReplayFrame==='function');
 await page.evaluate(()=>document.fonts.ready);
 for(let i=0;i<scenes.length;i++){
  const s=scenes[i];s.rendered=await page.evaluate(s=>window.exportReplayFrame(s.round,s.seq,s.focus,s.label),s);
  await page.evaluate(()=>new Promise(resolve=>requestAnimationFrame(()=>requestAnimationFrame(resolve))));
  s.frame=String(i).padStart(4,'0')+'.png';
  await page.screenshot({path:path.join(frameDir,s.frame),animations:'disabled'});
 }
 if(errors.length)throw new Error('Replay page errors: '+errors.join('\n'));
 await page.close();
}finally{await browser.close();}
const concat=scenes.map(s=>'file frames/'+s.frame+'\nduration '+s.seconds).join('\n')+'\nfile frames/'+scenes.at(-1).frame+'\n';
fs.writeFileSync(path.join(out,'frames.txt'),concat);
const videoPath=path.join(out,'replay.mp4');
const duration=scenes.reduce((a,s)=>a+s.seconds,0);
const ff=spawnSync(args['--ffmpeg']||'ffmpeg',['-hide_banner','-loglevel','warning','-n','-f','concat','-safe','0','-i','frames.txt','-vf','fps=30','-t',String(duration),'-c:v','libx264','-preset','medium','-crf','20','-pix_fmt','yuv420p','-movflags','+faststart','replay.mp4'],{cwd:out,encoding:'utf8',maxBuffer:1024*1024});
if(ff.error||ff.status!==0)throw new Error('ffmpeg failed: '+(ff.error||ff.stderr));
for(const n of sourceFiles)if(hash(fs.readFileSync(path.join(source,n)))!==hash(captured[n]))throw new Error('Source changed while rendering; output is a snapshot, not a current complete run.');
const exportManifest={version:1,source,partial,source_status:manifest.status,title,completed_rounds:presentation.completed,
 source_sha256:Object.fromEntries(sourceFiles.map(n=>[n,hash(captured[n])])),viewer_sha256:hash(template),
 exporter_sha256:hash(fs.readFileSync(new URL(import.meta.url))),presentation_sha256:hash(html),video_sha256:hash(fs.readFileSync(videoPath)),
 width:1920,height:1080,fps:30,duration_seconds:duration,
 note:'Offline presentation of real recorded round summaries. Event highlights are excerpts, not a continuous event-state reconstruction. Costs from charged events are usage-based estimates, not confirmed bills; the run accounting total includes unresolved reserves. Partial runs must not be compared as completed experiments. No model calls or source writes.',scenes};
fs.writeFileSync(path.join(out,'export-manifest.json'),JSON.stringify(exportManifest,null,2)+'\n');
console.log(JSON.stringify({video:videoPath,presentation:htmlPath,partial,duration_seconds:exportManifest.duration_seconds,scenes:scenes.length},null,2));
