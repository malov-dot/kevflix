'use strict';
const $=id=>document.getElementById(id);
let token=sessionStorage.getItem('movie-dub-token')||'',timer=null,eventSeq=0,selectedReview=null,blobUrl=null,refreshing=false;
$('token').value=token;
function notice(text,error=false){$('notice').textContent=text;$('notice').classList.toggle('error',error);}
async function api(path,data){
 const response=await fetch(path,{method:data===undefined?'GET':'POST',headers:{Authorization:'Bearer '+token,...(data===undefined?{}:{'Content-Type':'application/json'})},body:data===undefined?undefined:JSON.stringify(data)});
 const value=await response.json();if(!response.ok)throw Error(value.error||'Request failed');return value;
}
function element(tag,text,cls){const e=document.createElement(tag);if(text!==undefined)e.textContent=text;if(cls)e.className=cls;return e;}
function action(text,fn){const b=element('button',text);b.onclick=async()=>{b.disabled=true;try{await fn();await refresh();}catch(e){notice(e.message,true);}finally{b.disabled=false;}};return b;}
function time(seconds){return new Date(Math.max(0,seconds)*1000).toISOString().slice(11,19);}
async function connect(){
 token=$('token').value.trim();sessionStorage.setItem('movie-dub-token',token);await refresh();
 try{const sections=await api('/api/plex/sections');$('sections').replaceChildren(...sections.map(s=>{const o=element('option',s.title);o.value=s.key;return o;}));}catch(e){notice(e.message,true);}
 clearInterval(timer);timer=setInterval(()=>refresh().catch(e=>notice(e.message,true)),2000);
}
$('connect').onsubmit=e=>{e.preventDefault();connect().catch(e=>notice(e.message,true));};
$('logout').onclick=()=>{clearInterval(timer);token='';sessionStorage.removeItem('movie-dub-token');$('token').value='';notice('Disconnected.');};
$('pair-tv').onclick=async()=>{try{
 const result=await api('/api/tv/pair-code',{});
 $('pair-code').textContent='TV code: '+result.code+' · valid for 5 minutes · '+result.worker_url;
 const devices=await api('/api/tv/devices');
 $('tv-devices').replaceChildren(...devices.filter(d=>!d.revoked).map(d=>action('Revoke '+d.name,async()=>{await api('/api/tv/revoke',{id:d.id});notice('TV access revoked.');})));
}catch(e){notice(e.message,true);}};
async function refresh(){
 if(!token||refreshing)return;
 refreshing=true;try{
 const status=await api('/api/status');$('version').textContent=status.version+' · awaiting validation';
 const session=status.sessions[0]?.payload;
 $('session').textContent=session?session.title+' · '+session.state+' · '+time(session.position):'No playback session on the selected TV.';
 $('worker').textContent='Worker: '+status.config.device+' · English speech: '+status.config.speech_engine+' · '+(status.watch_enabled?'Watch context enabled':'Watch context off');
 $('plex-error').textContent=status.plex_error||'';
 $('jobs').replaceChildren(...status.jobs.map(job=>{
  const p=job.payload,c=element('article',undefined,'card');c.append(element('h3',p.title),element('span',job.state,'pill'));
  c.append(element('p',(p.chunks_ready||0)+' / '+p.chunks_total+' segments ready'+(p.publication_state?' · publication '+p.publication_state:'')+(p.publication_error?'\n'+p.publication_error:'')));
  const buttons=element('div',undefined,'actions');
  buttons.append(action('Details / review',()=>showJob(job.id)));
  if(['cancelled','error'].includes(job.state))buttons.append(action('Resume',()=>api('/api/jobs/'+job.id+'/resume',{})));
  else if(!['complete','needs_review'].includes(job.state)||p.publication_state==='working')buttons.append(action('Cancel',()=>api('/api/jobs/'+job.id+'/cancel',{})));
  if(job.state==='complete'&&p.publication_state!=='working')buttons.append(action('Publish to Plex',()=>api('/api/jobs/'+job.id+'/publish',{})));
  if(p.published)buttons.append(action('Play English version on TV',()=>api('/api/jobs/'+job.id+'/play',{position:0})));
  c.append(buttons);return c;
 }));
 const events=await api('/api/events?after='+eventSeq);
 for(const event of events){eventSeq=event.seq;$('events').textContent+=new Date(event.at*1000).toLocaleTimeString()+' · '+event.kind+' · '+JSON.stringify(event.payload)+'\n';}
 $('events').textContent=$('events').textContent.slice(-18000);
 }finally{refreshing=false;}
}
$('search').onsubmit=async e=>{e.preventDefault();try{await library($('sections').value,$('query').value);}catch(error){notice(error.message,true);}};
async function library(section,query){
 const items=await api('/api/plex/library?section='+encodeURIComponent(section)+'&query='+encodeURIComponent(query));
 renderLibrary(items);
}
function renderLibrary(items){
 $('library').replaceChildren(...items.map(item=>{
  const card=element('article',undefined,'card');card.append(element('h3',item.title+(item.year?' ('+item.year+')':'')));
  const buttons=element('div',undefined,'actions');
  if(!['movie','episode'].includes(item.type)){
   buttons.append(action('Open',async()=>renderLibrary(await api('/api/plex/children?key='+item.ratingKey))));
  }else{
   buttons.append(action('Prepare English version',async()=>{await api('/api/plex/prepare',{key:item.ratingKey});notice('English preparation queued.');}));
  }
  card.append(buttons);return card;
 }));
}
$('local').onsubmit=async e=>{e.preventDefault();try{await api('/api/jobs',{source:$('source').value,subtitles:$('subtitles').value||null,mode:'prepared'});notice('Preparation queued.');await refresh();}catch(error){notice(error.message,true);}};
document.querySelectorAll('[data-control]').forEach(b=>b.onclick=()=>api('/api/plex/control',{action:b.dataset.control}).then(refresh).catch(e=>notice(e.message,true)));
$('seek').onsubmit=e=>{e.preventDefault();api('/api/plex/control',{action:'seekTo',position:Number($('seek-position').value)}).then(refresh).catch(e=>notice(e.message,true));};
$('watch').onclick=()=>api('/api/watch/start',{}).then(refresh).catch(e=>notice(e.message,true));
$('watch-stop').onclick=()=>api('/api/watch/stop',{}).then(refresh).catch(e=>notice(e.message,true));
async function showJob(id,index){
 const job=await api('/api/jobs/'+id);const chunk=(index===undefined?null:job.chunks.find(x=>x.idx===index))||job.chunks.find(x=>x.state==='needs_review')||job.chunks.find(x=>x.state==='error')||job.chunks.at(-1);
 if(!chunk){notice('No segments have been processed yet.');return;}
 selectedReview={job,chunk};$('review').hidden=false;
 $('review-segment').replaceChildren(...job.chunks.map(c=>{const o=element('option',time(c.payload.start)+' · '+c.state);o.value=c.idx;return o;}));$('review-segment').value=chunk.idx;
 $('review-segment').onchange=()=>showJob(id,Number($('review-segment').value)).catch(e=>notice(e.message,true));
 $('review-description').textContent=job.payload.title+' · '+time(chunk.payload.start)+'–'+time(chunk.payload.end)+' · '+chunk.state;
 $('issues').replaceChildren(...(chunk.payload.issues||[]).map(x=>element('p',x.kind+': '+(x.reason||x.heard||x.cue||''),'warning')));
 if(chunk.payload.error)$('issues').append(element('p',chunk.payload.error,'warning'));
 $('preview').replaceChildren(...['original','bed','audio'].filter(k=>chunk.payload[k]).map(kind=>action('Listen: '+kind,async()=>{
  const response=await fetch('/api/jobs/'+id+'/chunks/'+chunk.idx+'/artifact?kind='+kind,{headers:{Authorization:'Bearer '+token}});
  if(!response.ok)throw Error((await response.json()).error);
  if(blobUrl)URL.revokeObjectURL(blobUrl);blobUrl=URL.createObjectURL(await response.blob());$('audio').src=blobUrl;
 })));
 $('cues').replaceChildren(...(chunk.payload.cues||[]).map(cue=>{
  const row=element('div',undefined,'cue');row.dataset.id=cue.id;
  row.append(element('p',time(cue.start)));
  const original=element('textarea');original.dataset.field='source_text';original.value=cue.text||'';original.setAttribute('aria-label','Original dialogue');
  const english=element('textarea');english.dataset.field='english_text';english.value=cue.english||'';
  const language=element('input');language.dataset.field='source_language';language.value=cue.source_language||'';language.placeholder='Source language code';
  const speaker=element('select');speaker.dataset.field='speaker';const empty=element('option','Choose the speaker');empty.value='';speaker.append(empty);
  for(const [key,value] of Object.entries(job.speakers)){const option=element('option',(value.label||key)+' · '+(value.voice.name||''));option.value=key;speaker.append(option);}
  speaker.value=cue.speaker||'';row.append(original,english,language,speaker);
  for(const input of row.querySelectorAll('[data-field]'))input.dataset.original=input.value;
  return row;
 }));
 $('approve').disabled=chunk.state!=='needs_review'||!chunk.payload.audio;
 $('save-corrections').disabled=chunk.state!=='needs_review';
 $('review').scrollIntoView({behavior:'smooth'});
}
$('save-corrections').onclick=async()=>{try{
 const corrections={};for(const row of $('cues').children){const fields={};for(const input of row.querySelectorAll('[data-field]'))if(input.value.trim()&&input.value!==input.dataset.original)fields[input.dataset.field]=input.value.trim();if(Object.keys(fields).length)corrections[row.dataset.id]=fields;}
 if(!Object.keys(corrections).length)throw Error('Change the uncertain text, language or speaker before saving.');
 await api('/api/jobs/'+selectedReview.job.id+'/review',{chunk:selectedReview.chunk.idx,corrections});notice('Corrections saved; affected segments will be regenerated.');$('review').hidden=true;await refresh();
}catch(e){notice(e.message,true);}};
$('approve').onclick=async()=>{try{
 await api('/api/jobs/'+selectedReview.job.id+'/review',{chunk:selectedReview.chunk.idx,approve_hash:selectedReview.chunk.payload.sha256});notice('Approval saved for this exact audio.');$('review').hidden=true;await refresh();
}catch(e){notice(e.message,true);}};
if(token)connect().catch(e=>notice(e.message,true));
