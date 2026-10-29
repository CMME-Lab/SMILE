/* SMILE interactive webpage. All narratives are scripted, not live model inference.
   Clinical case excerpts: WACV supplement, Tables S4/S5 (pages 5/6).
   Quantitative metrics: rebuttal, supplement, and previously shared main-paper
   values; confirm all main-paper values against the final camera-ready version. */
(()=>{
'use strict';
const $=(sel,root=document)=>root.querySelector(sel);
const $$=(sel,root=document)=>[...root.querySelectorAll(sel)];
const reduce=window.matchMedia('(prefers-reduced-motion: reduce)').matches;
const cases=[
 {image:'case_01.webp',title:'Echogenicity and echotexture recovered',description:'The judge recovers diffuse parenchymal findings despite an initially normal radiology description.',type:'Evidence recovery',outcome:'recovery',reference:'The liver has a plump shape, a smooth capsule, dense and enhanced parenchymal echoes, and slightly attenuated backfield echoes.',a:'The liver has a normal size and shape. The capsule is not clearly visualized. The liver parenchyma demonstrates normal echogenicity with a relatively homogeneous echotexture. Posterior acoustic attenuation is not evident.',b:'Lobular activity, mild; portal–periportal activity, none; steatosis, none; fibrosis, none.',c:'Echogenicity increased? Yes.\nEchotexture coarse? Yes.\nPosterior attenuation increased? No.',final:'The hepatic parenchyma demonstrates increased echogenicity with a coarse and heterogeneous appearance. Liver size and capsular contour cannot be confidently characterized from the available views. Posterior acoustic attenuation is not clearly demonstrated.',lesson:'Partial recovery: increased echogenicity and heterogeneous echotexture are represented in the final report. The reference attenuation finding is not fully recovered.'},
 {image:'case_02.webp',title:'Contour and attenuation recovered',description:'The consolidated report recovers liver contour, heterogeneous echoes, and posterior attenuation.',type:'Evidence recovery',outcome:'recovery',reference:'The liver has a plump shape, a smooth capsule, dense and uneven parenchymal echoes, attenuated rear echoes, and enhanced liver–kidney contrast.',a:'The liver has a normal size and shape, with the capsule not clearly visualized. The parenchymal echoes are relatively homogeneous, and posterior attenuation is unremarkable. The portal venous system shows no obvious abnormality.',b:'Steatohepatitis, mild; lobular activity, none; portal–periportal activity, none; fibrosis, none; steatosis, none.',c:'Echogenicity increased? Yes.\nEchotexture coarse? Yes.\nPosterior attenuation increased? No.',final:'A plump shape of the liver is present, with a smooth capsule. The liver parenchyma demonstrates dense and uneven echoes, with attenuated rear echoes. The liver and kidney echoes are enhanced, and the portal venous system is not clearly displayed.',lesson:'Evidence recovery: the consolidated report recovers the plump contour, uneven parenchymal echoes, and attenuation described in the reference.'},
 {image:'case_03.webp',title:'Prior cholecystectomy missed',description:'Absent surgical history leads the judge to retain an incorrect gallbladder statement.',type:'Failure case',outcome:'failure',reference:'The reference report describes the liver and states that the patient has undergone cholecystectomy.',a:'The liver has a normal size and shape, with a smooth capsule. The liver parenchyma demonstrates normal echogenicity and echotexture. The gallbladder is normal in size and echogenicity, with no obvious abnormal echo in the cavity.',b:'Lobular activity, none; portal–periportal activity, none; steatosis, mild; fibrosis, none.',c:'Echogenicity increased? Yes.\nEchotexture coarse? No.\nPosterior attenuation increased? No.',final:'The liver parenchyma shows no definite focal abnormality. The gallbladder size and echogenicity are normal, and no obvious abnormal echo is identified in the cavity. The visualized biliary structures show no definite dilation.',lesson:'Failure: the reference reports a prior cholecystectomy, but this detail is absent from the specialist evidence. The judge incorrectly describes the gallbladder as normal.'},
 {image:'case_04.webp',title:'Right-lobe cyst omitted',description:'A focal lesion absent from specialist outputs remains missing after consensus.',type:'Failure case',outcome:'failure',reference:'A cystic structure can be seen in the right lobe of the liver, with a clear border and enhanced rear echo.',a:'The liver has a normal size and shape, with a smooth capsule. The liver parenchyma demonstrates relatively homogeneous echotexture. There is no evidence of a focal lesion in the visualized liver parenchyma.',b:'Lobular activity, none; portal–periportal activity, mild; steatosis, none; fibrosis, none.',c:'Echogenicity increased? Yes.\nEchotexture coarse? No.\nPosterior attenuation increased? No.',final:'The liver parenchyma appears relatively homogeneous, with no focal lesion identified in the available views. No additional focal abnormality is evident in the visualized hepatic parenchyma.',lesson:'Failure: the right-lobe cyst is missed. The VQA questions cover diffuse parenchymal attributes rather than focal lesions, and consensus cannot recover evidence that is not provided.'}
];
const agents=['a','b','c'];
let activeCase=0;
const opened=new Set();
const agentTokens={a:0,b:0,c:0};
let judgeToken=0;
const wait=ms=>new Promise(r=>setTimeout(r,ms));

// Cartoon chirps so each agent has its own voice. The context is created on the
// first interaction, so no audio is opened before the visitor clicks.
const voices={a:[340,620],b:[430,780],c:[520,940]};
let actx=null,soundOn=true;
try{soundOn=localStorage.getItem('smile-sound')!=='off';}catch(e){}
function audioContext(){
 if(!actx){const Ctor=window.AudioContext||window.webkitAudioContext;if(!Ctor)return null;actx=new Ctor();}
 return actx;
}
function chirp(f0,f1,delay,dur,type,vol,cutoff){
 const ctx=audioContext();if(!ctx)return;
 const osc=ctx.createOscillator(),gain=ctx.createGain(),filter=ctx.createBiquadFilter();
 const t=ctx.currentTime+delay;
 osc.type=type;
 osc.frequency.setValueAtTime(f0,t);
 osc.frequency.exponentialRampToValueAtTime(f1,t+dur*.55);
 osc.frequency.exponentialRampToValueAtTime(f0*1.12,t+dur);
 filter.type='lowpass';filter.frequency.setValueAtTime(cutoff,t);
 gain.gain.setValueAtTime(.0001,t);
 gain.gain.exponentialRampToValueAtTime(vol,t+.02);
 gain.gain.exponentialRampToValueAtTime(.0001,t+dur);
 osc.connect(filter);filter.connect(gain);gain.connect(ctx.destination);
 osc.start(t);osc.stop(t+dur+.03);
}
function play(kind){
 if(!soundOn)return;
 const ctx=audioContext();if(!ctx)return;
 if(ctx.state==='suspended')ctx.resume();
 if(kind==='close'){chirp(420,250,0,.14,'triangle',.05,1800);return;}
 if(kind==='judge'){
  [0,.1,.2].forEach((d,i)=>chirp(300+i*120,560+i*210,d,.21,'sawtooth',.045,2400));
  chirp(175,175,.3,.55,'sine',.05,900);
  return;
 }
 const [f0,f1]=voices[kind];
 chirp(f0,f1,0,.19,'sawtooth',.06,2200);
 chirp(f0*2,f1*2,.04,.12,'triangle',.02,3200);
}
const soundToggle=$('#sound-toggle');
function renderSound(){
 soundToggle.setAttribute('aria-pressed',String(soundOn));
 soundToggle.setAttribute('aria-label',soundOn?'Mute agent sounds':'Unmute agent sounds');
 $('#sound-icon').textContent=soundOn?'🔊':'🔇';
}
soundToggle.addEventListener('click',()=>{
 soundOn=!soundOn;renderSound();
 try{localStorage.setItem('smile-sound',soundOn?'on':'off');}catch(e){}
 if(soundOn)play('a');
});
renderSound();

async function typeInto(node,str,isValid,{duration=1400}={}){
 node.textContent='';node.classList.add('typing');
 if(reduce){node.textContent=str;node.classList.remove('typing');return;}
 const chunk=Math.max(2,Math.round(str.length/60));
 const ms=Math.max(16,duration/(str.length/chunk));
 for(let n=chunk;n<str.length+chunk;n+=chunk){
  if(!isValid()){node.classList.remove('typing');return;}
  node.textContent=str.slice(0,n);await wait(ms);
 }
 node.textContent=str;node.classList.remove('typing');
}
// Lexical provenance. Content words of the consolidated report are matched
// against each agent response and the reference, which is the same kind of
// unigram overlap the ROUGE columns report. Nothing here is model output.
const STOP=new Set(['the','a','an','and','or','of','in','on','at','to','for','with','without','from','by','as','is','are','was','were','be','been','being','has','have','had','no','not','its','it','this','that','there','their','they','which','can','cannot','but','any','all','also','more','than','then','out','into','over','under','does','do','obvious','available','further']);
function norm(word){
 let w=word.toLowerCase().replace(/[^a-z–-]/g,'');
 if(w.length>5&&w.endsWith('ies'))return w.slice(0,-3)+'y';
 if(w.length>4&&w.endsWith('s')&&!w.endsWith('ss'))w=w.slice(0,-1);
 if(w.length>5&&w.endsWith('ion'))w=w.slice(0,-3);
 if(w.length>6&&w.endsWith('ed'))w=w.slice(0,-2);
 if(w.length>7&&w.endsWith('ing'))w=w.slice(0,-3);
 return w;
}
function isContent(w){return w.length>2&&!STOP.has(w);}
function bagOf(str){
 const bag=new Set();
 for(const word of str.split(/\s+/)){const w=norm(word);if(isContent(w))bag.add(w);}
 return bag;
}
// Stemming alone misses pairs like parenchyma/parenchymal, so long stems also
// match on a shared prefix.
function bagHas(bag,w){
 if(bag.has(w))return true;
 if(w.length<6)return false;
 for(const b of bag){if(b.length>=6&&(b.startsWith(w)||w.startsWith(b)))return true;}
 return false;
}
const traceSources=[['a','Agent A'],['c','Agent C'],['ref','Reference report']];
// Agent B returns severity grades rather than prose, so word overlap cannot
// score it. Its grades are listed as they are instead of being forced onto a bar.
function gradesOf(str){
 return str.replace(/\.\s*$/,'').split(';').map(s=>s.trim()).filter(Boolean).map(s=>{
  const i=s.lastIndexOf(',');
  return i<0?{attr:s,grade:''}:{attr:s.slice(0,i).trim(),grade:s.slice(i+1).trim()};
 });
}
function renderGrades(d){
 const host=$('#grade-chips');host.replaceChildren();
 for(const g of gradesOf(d.b)){
  const chip=document.createElement('span');chip.className='grade-chip';
  const attr=document.createElement('span');attr.className='grade-attr';attr.textContent=g.attr;
  chip.appendChild(attr);
  if(g.grade){
   chip.dataset.grade=g.grade;
   const val=document.createElement('span');val.className='grade-value';val.textContent=g.grade;
   chip.appendChild(val);
  }
  host.appendChild(chip);
 }
}
function renderTrace(d){
 const bags={a:bagOf(d.a),c:bagOf(d.c),ref:bagOf(d.reference)};
 const host=$('#final-text');host.replaceChildren();
 const counts={a:0,c:0,ref:0};let total=0;
 for(const chunk of d.final.split(/(\s+)/)){
  if(!chunk)continue;
  if(/^\s+$/.test(chunk)){host.append(chunk);continue;}
  const w=norm(chunk);
  if(!isContent(w)){host.append(chunk);continue;}
  total++;
  const src=traceSources.map(([k])=>k).filter(k=>bagHas(bags[k],w));
  src.forEach(k=>{counts[k]++;});
  const span=document.createElement('span');
  span.className='w';span.dataset.src=src.join(' ');span.textContent=chunk;
  host.append(span);
 }
 const rows=$('#trace-rows');rows.replaceChildren();
 for(const [key,name] of traceSources){
  const pct=total?Math.round(counts[key]/total*100):0;
  const row=document.createElement('button');
  row.type='button';row.className='trace-row';row.dataset.key=key;
  const label=document.createElement('span');label.className='trace-name';label.textContent=name;
  const track=document.createElement('span');track.className='trace-track';
  const fill=document.createElement('span');fill.className='trace-fill';fill.dataset.width=pct;track.appendChild(fill);
  const val=document.createElement('strong');val.className='trace-pct';val.textContent=pct+'%';
  row.append(label,track,val);
  row.addEventListener('mouseenter',()=>setFocus(key));
  row.addEventListener('focus',()=>setFocus(key));
  row.addEventListener('mouseleave',()=>setFocus(null));
  row.addEventListener('blur',()=>setFocus(null));
  rows.appendChild(row);
 }
 animateWidths(rows);
 renderGrades(d);
 $('#trace-panel').hidden=false;
}
function setFocus(key){
 const panel=$('#judgement');
 if(key)panel.dataset.focus=key;else delete panel.dataset.focus;
 $$('.trace-row').forEach(r=>r.classList.toggle('is-focus',r.dataset.key===key));
 $('#trace-grades').classList.toggle('is-focus',key==='b');
}
function setAgentOpen(agent,open){
 const pod=$(`.agent-pod[data-character="${agent}"]`);
 const bubble=$(`.bubble[data-bubble="${agent}"]`);
 const link=$(`.stage-link[data-link="${agent}"]`);
 pod.classList.toggle('is-open',open);
 link.classList.toggle('is-live',open);
 pod.setAttribute('aria-expanded',String(open));
 bubble.hidden=!open;
 if(!open)pod.classList.remove('talks');
}
function pop(el){el.classList.remove('pops');void el.offsetWidth;el.classList.add('pops');}
function renderState(){
 const count=opened.size;
 const judge=$('#judge-button');
 judge.disabled=count<3;
 if(count===3)judge.classList.remove('has-spoken');
 $('#stage-hint').textContent=count<3?`${count} of 3 agents heard`:'The judge is awake — tap to consolidate';
}
async function toggleAgent(agent){
 const pod=$(`.agent-pod[data-character="${agent}"]`);
 const bubble=$(`.bubble[data-bubble="${agent}"]`);
 if(opened.has(agent)){
  opened.delete(agent);agentTokens[agent]++;setAgentOpen(agent,false);bubble.classList.remove('thinking');play('close');renderState();return;
 }
 opened.add(agent);const token=++agentTokens[agent];
 setAgentOpen(agent,true);pop(pod);play(agent);renderState();
 const text=$(`[data-report="${agent}"]`);
 bubble.classList.add('thinking');pod.classList.add('ponders');
 await wait(reduce?0:760);
 bubble.classList.remove('thinking');pod.classList.remove('ponders');
 if(token!==agentTokens[agent])return;
 pod.classList.add('talks');
 await typeInto(text,cases[activeCase][agent],()=>token===agentTokens[agent],{duration:agent==='a'?1650:1200});
 if(token===agentTokens[agent])pod.classList.remove('talks');
}
function resetCase(){
 for(const a of agents){agentTokens[a]++;setAgentOpen(a,false);$(`[data-report="${a}"]`).textContent='';}
 opened.clear();judgeToken++;renderState();
 $('#judge-button').classList.remove('has-spoken','talks');
 $('#judgement-content').hidden=true;$('#judgement-content').classList.remove('appeared');
 $('#judge-placeholder').hidden=false;
 $('#final-text').textContent='';$('#trace-panel').hidden=true;setFocus(null);
 $$('.bubble').forEach(b=>b.classList.remove('thinking'));
 $$('.agent-pod').forEach(p=>p.classList.remove('ponders','talks'));
 $('#stage').classList.remove('is-flowing');
 $('#reference-box').hidden=true;
 $('#reference-toggle').setAttribute('aria-expanded','false');
 $('#reference-toggle').innerHTML='Reference finding <span>＋</span>';
}
function showCase(i){
 activeCase=(i+cases.length)%cases.length;
 const d=cases[activeCase];resetCase();
 $('#case-counter').textContent=`${String(activeCase+1).padStart(2,'0')} / 04`;
 $('#case-number').textContent=String(activeCase+1).padStart(2,'0');
 $('#case-heading').textContent=d.title;$('#case-description').textContent=d.description;
 $('#case-type').textContent=d.type;$('#case-type').classList.toggle('is-failure',d.outcome==='failure');
 $('#reference-report').textContent=d.reference;
 const img=$('#ultrasound');img.src='assets/ultrasound/'+d.image;img.alt=`Deidentified liver ultrasound case ${activeCase+1}`;
 $$('#case-dots button').forEach((b,k)=>{b.classList.toggle('is-current',k===activeCase);b.setAttribute('aria-current',k===activeCase?'true':'false');});
}
const dots=$('#case-dots');cases.forEach((d,i)=>{const b=document.createElement('button');b.setAttribute('type','button');b.setAttribute('aria-label',`Go to case ${i+1}: ${d.title}`);b.addEventListener('click',()=>showCase(i));dots.appendChild(b);});
$('#case-prev').addEventListener('click',()=>{stopPlay();showCase(activeCase-1);});
$('#case-next').addEventListener('click',()=>{stopPlay();showCase(activeCase+1);});
$$('[data-character]').forEach(b=>b.addEventListener('click',()=>{stopPlay();toggleAgent(b.dataset.character);}));
for(const agent of agents){
 const targets=[$(`.agent-pod[data-character="${agent}"]`),$(`.bubble[data-bubble="${agent}"]`)];
 if(agent==='b')targets.push($('#trace-grades'));
 for(const el of targets){
  el.addEventListener('mouseenter',()=>{if(!$('#trace-panel').hidden)setFocus(agent);});
  el.addEventListener('mouseleave',()=>{if(!$('#trace-panel').hidden)setFocus(null);});
 }
}
// Pupils drift toward the pointer so the council feels like it is watching.
if(!reduce){
 const eyeGroups=$$('.pod-avatar .eyes, .judge-face .eyes');
 let pointer=null,queued=false;
 const trackEyes=()=>{
  queued=false;
  if(!pointer)return;
  for(const group of eyeGroups){
   const box=group.getBoundingClientRect();
   if(!box.width)continue;
   const dx=pointer.x-(box.left+box.width/2),dy=pointer.y-(box.top+box.height/2);
   const dist=Math.hypot(dx,dy)||1,reach=Math.min(2.6,dist/26);
   group.style.transform=`translate(${(dx/dist*reach).toFixed(2)}px,${(dy/dist*reach).toFixed(2)}px)`;
  }
 };
 addEventListener('pointermove',e=>{
  pointer={x:e.clientX,y:e.clientY};
  if(!queued){queued=true;requestAnimationFrame(trackEyes);}
 },{passive:true});
}
$('#reference-toggle').addEventListener('click',()=>{
 const box=$('#reference-box');const open=box.hidden;box.hidden=!open;
 $('#reference-toggle').setAttribute('aria-expanded',String(open));
 $('#reference-toggle').innerHTML=open?'Hide reference finding <span>−</span>':'Reference finding <span>＋</span>';
});
async function askJudge(){
 if(opened.size<3)return;
 const token=++judgeToken;const d=cases[activeCase];const content=$('#judgement-content');
 const judge=$('#judge-button'),stage=$('#stage');
 play('judge');pop(judge);
 judge.classList.add('has-spoken');
 stage.classList.remove('is-flowing');void stage.offsetWidth;stage.classList.add('is-flowing');
 $('#stage-hint').textContent='The judge has consolidated the evidence';
 $('#judge-placeholder').hidden=true;content.hidden=false;content.classList.remove('appeared');
 void content.offsetWidth;content.classList.add('appeared');
 $('#final-text').textContent='';$('#trace-panel').hidden=true;
 const lesson=$('.case-lesson');lesson.classList.toggle('is-limitation',d.outcome==='failure');
 $('#lesson-icon').textContent=d.outcome==='failure'?'!':'✓';$('#lesson').textContent=d.lesson;
 $('#report-stamp').textContent=d.outcome==='failure'?'Important limitation':'Selected example';
 await wait(reduce?0:900);if(token!==judgeToken)return;
 stage.classList.remove('is-flowing');
 judge.classList.add('talks');
 await typeInto($('#final-text'),d.final,()=>judgeToken===token,{duration:2300});
 if(token!==judgeToken)return;
 judge.classList.remove('talks');
 renderTrace(d);
}
$('#judge-button').addEventListener('click',askJudge);

// Autoplay walks the whole case so a visitor sees the pipeline without clicking.
let playToken=0;
function setPlaying(on){
 $('#play-button').classList.toggle('is-playing',on);
 $('#play-icon').textContent=on?'■':'▶';
 $('#play-label').textContent=on?'Stop':'Play case';
}
function stopPlay(){playToken++;setPlaying(false);}
async function playCase(){
 const token=++playToken;setPlaying(true);
 showCase(activeCase);
 await wait(reduce?0:420);
 for(const a of agents){
  if(token!==playToken)return;
  await toggleAgent(a);
  if(token!==playToken)return;
  await wait(reduce?0:420);
 }
 if(token!==playToken)return;
 await askJudge();
 if(token===playToken)setPlaying(false);
}
$('#play-button').addEventListener('click',()=>{
 if($('#play-button').classList.contains('is-playing'))stopPlay();else playCase();
});
showCase(0);

// Reported model outcomes. Values not available for a specific comparison are
// intentionally displayed as N/R, never imputed.
const modelNames=['Qwen-2B','MedMO-4B','MedGemma-4B','MedAgents','SMILE'];
const reports={
 d1:{r1:[.10,.69,.44,.60,.72],rl:[.09,.59,.37,null,.61],meteor:[.02,.47,.24,null,.56],ce:[.01,.69,.53,.73,.82],vqa:[.77,.76,.62,.41,.82]},
 d2:{rl:[.19,.38,.42,.07,.50],meteor:[.08,.29,.30,.03,.32],vqa:[.44,.57,.61,.39,.65]}
};
const metrics={r1:'ROUGE-1',rl:'ROUGE-L',meteor:'METEOR',ce:'Clinical entity F1',vqa:'VQA F1'};
let dataset='d1';
function selectDataset(ds){dataset=ds;
 $$('[data-dataset]').forEach(b=>{let on=b.dataset.dataset===ds;b.classList.toggle('is-active',on);b.setAttribute('aria-pressed',String(on));});
 const sel=$('#result-metric');
 [...sel.options].forEach(o=>{o.disabled=reports[ds][o.value]===undefined;o.hidden=o.disabled;});
 if(!reports[ds][sel.value])sel.value='rl';renderResult();
}
function animateWidths(group){requestAnimationFrame(()=>requestAnimationFrame(()=>group.querySelectorAll('[data-width]').forEach(el=>el.style.width=el.dataset.width+'%')));}
function renderResult(){
 const key=$('#result-metric').value,data=reports[dataset][key],shell=$('#result-bars');
 $('#result-chart-title').textContent=metrics[key];
 shell.replaceChildren();
 modelNames.forEach((name,i)=>{
  const val=data[i],row=document.createElement('div');row.className=`bar-item ${name==='SMILE'?'smile':''} ${name==='MedAgents'?'medagents':''} ${val===null?'missing':''}`;
  const label=document.createElement('span');label.className='bar-name';label.textContent=name;
  const track=document.createElement('div');track.className='bar-track';const fill=document.createElement('div');fill.className='bar-fill';fill.dataset.width=val===null?0:(val*100);track.appendChild(fill);
  const value=document.createElement('strong');value.className='bar-value';value.textContent=val===null?'N/R':val.toFixed(2);
  row.append(label,track,value);shell.appendChild(row);
 });
 animateWidths(shell);
}
$$('[data-dataset]').forEach(b=>b.addEventListener('click',()=>selectDataset(b.dataset.dataset)));
$('#result-metric').addEventListener('change',renderResult);selectDataset('d1');
// Selected Dataset-1 ablations from the previously shared manuscript results.
// Capacity series are also corroborated in the attached supplement Figure S3.
const ablation={
 specialists:{title:'Individual agents versus combinations',rows:[['Agent A',.44,.53],['Agent B',.25,.03],['Agent C',.18,.26],['A + B',.70,.70],['A + C',.42,.73],['B + C',.29,.62],['SMILE',.72,.82]]},
 judge:{title:'Inputs to the judge',rows:[['Image only',.17,.20],['Agents only',.48,.74],['Image + agents (SMILE)',.72,.82]]},
 fusion:{title:'Output aggregation strategies',rows:[['Simple ensembling',.41,.53],['Cross-attention',.23,.42],['SMILE',.72,.82]]},
 council:{title:'Specialization and backbone choice',rows:[['General agents',.30,.72],['Homogeneous council',.68,.77],['Qwen-2B judge',.57,.77],['SMILE',.72,.82]]},
 capacity:{title:'Single-model adaptation capacity',rows:[['MedGemma · 38.5M',.44,.53],['MedGemma · 77.0M',.59,.72],['MedGemma · 115.5M',.70,.78],['SMILE · 55.63M',.72,.82]]}
};
let abGroup='specialists',abMetric='r1';
function renderAblation(){
 const d=ablation[abGroup],shell=$('#ablation-bars');
 $('#ablation-title').textContent=d.title;
 shell.replaceChildren();
 d.rows.forEach(([name,r1,ce])=>{
  const val=abMetric==='r1'?r1:ce;const div=document.createElement('div');div.className=`ab-bar ${name.includes('SMILE')?'smile':''}`;
  const label=document.createElement('span');label.className='ab-name';label.textContent=name;
  const track=document.createElement('div');track.className='ab-track';const fill=document.createElement('div');fill.className='ab-fill';fill.dataset.width=val*100;track.appendChild(fill);
  const value=document.createElement('strong');value.className='ab-value';value.textContent=val.toFixed(2);
  div.append(label,track,value);shell.appendChild(div);
 });animateWidths(shell);
}
$$('[data-ablation]').forEach(b=>b.addEventListener('click',()=>{
 abGroup=b.dataset.ablation;
 $$('[data-ablation]').forEach(x=>{let on=x===b;x.classList.toggle('active',on);x.setAttribute('aria-pressed',String(on));});renderAblation();
}));
$$('[data-ab-metric]').forEach(b=>b.addEventListener('click',()=>{
 abMetric=b.dataset.abMetric;
 $$('[data-ab-metric]').forEach(x=>{let on=x===b;x.classList.toggle('active',on);x.setAttribute('aria-pressed',String(on));});renderAblation();
}));renderAblation();
})();

const methodDialog=document.querySelector('#method-dialog');
if(methodDialog){document.querySelector('#method-expand')?.addEventListener('click',()=>methodDialog.showModal());document.querySelector('#method-close')?.addEventListener('click',()=>methodDialog.close());methodDialog.addEventListener('click',e=>{if(e.target===methodDialog)methodDialog.close()});}
