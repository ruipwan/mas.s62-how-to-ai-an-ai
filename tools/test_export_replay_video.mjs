import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {spawnSync} from 'node:child_process';

const exporter=fileURLToPath(new URL('./export_replay_video.mjs',import.meta.url));
function fixture(t,status='interrupted',events=[{seq:0,event:'started',round:-1}]){
 const base=fs.mkdtempSync(path.join(os.tmpdir(),'replay-video-test-'));
 t.after(()=>fs.rmSync(base,{recursive:true,force:true}));
 const run=path.join(base,'run'),out=path.join(base,'output');fs.mkdirSync(run);
 fs.writeFileSync(path.join(run,'manifest.json'),JSON.stringify({status}));
 fs.writeFileSync(path.join(run,'state.json'),JSON.stringify({snapshots:[{round:-1}]}));
 fs.writeFileSync(path.join(run,'events.jsonl'),events.map(e=>JSON.stringify(e)).join('\n')+'\n');
 const before=Object.fromEntries(fs.readdirSync(run).map(n=>[n,fs.readFileSync(path.join(run,n),'utf8')]));
 return {run,out,checkUnchanged(){assert.deepEqual(Object.fromEntries(fs.readdirSync(run).map(n=>[n,fs.readFileSync(path.join(run,n),'utf8')])),before);}};
}
function invoke(f,...flags){return spawnSync(process.execPath,[exporter,'--run',f.run,'--out',f.out,...flags],{encoding:'utf8'});}
test('refuses running source even with allow-partial before renderer or output',t=>{
 const f=fixture(t,'running'),r=invoke(f,'--allow-partial');assert.notEqual(r.status,0);assert.match(r.stderr,/Refusing a moving run/);assert.equal(fs.existsSync(f.out),false);f.checkUnchanged();
});
test('requires explicit partial export, without changing source',t=>{
 const f=fixture(t),r=invoke(f);assert.notEqual(r.status,0);assert.match(r.stderr,/requires --allow-partial/);assert.equal(fs.existsSync(f.out),false);f.checkUnchanged();
});
test('refuses source-contained output before any write',t=>{
 const f=fixture(t,'completed');f.out=path.join(f.run,'video');const r=invoke(f);assert.notEqual(r.status,0);assert.match(r.stderr,/new and separate/);assert.equal(fs.existsSync(f.out),false);f.checkUnchanged();
});
test('refuses discontinuous log instead of silently presenting incomplete history',t=>{
 const f=fixture(t,'completed',[{seq:1,event:'started',round:-1}]),r=invoke(f);assert.notEqual(r.status,0);assert.match(r.stderr,/sequence is incomplete/);assert.equal(fs.existsSync(f.out),false);f.checkUnchanged();
});
