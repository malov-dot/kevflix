(function () {
  'use strict';
  var VERSION='0.3.1', $=function(id){return document.getElementById(id);};
  var token=localStorage.getItem('sem-tv-token')||'', device=localStorage.getItem('sem-tv-device')||'';
  var folder=null,parent=null,offset=0,rows=[],session=null,loadedGeneration=0,wantedPlay=false;
  var operation=0,polling=false,loading=false,pollTimer=null,hideTimer=null,lastHeartbeat=0,reportedError='';
  var video=$('video'),desiredPosition=0,mediaStart=0,loadAbort=null,loadSerial=0;
  var sourceMode='folders',posterUrls=[],posterEpoch=0,captionKey='',setupOnly=false;
  $('worker-address').textContent='Worker: '+location.origin;
  function message(text){$('notice').textContent=text||'';}
  function playerMessage(text){$('player-message').textContent=text||'';}
  function clock(seconds){
    seconds=Math.floor(Math.max(0,Number(seconds)||0));
    var h=Math.floor(seconds/3600),m=Math.floor(seconds/60)%60,s=seconds%60;
    return(h?h+':':'')+(h&&m<10?'0':'')+m+':'+(s<10?'0':'')+s;
  }
  async function api(path,body,publicRequest){
    var controller=new AbortController(),timeout=setTimeout(function(){controller.abort();},45000);
    try{
      var response=await fetch('/tv-api/'+path,{method:body===undefined?'GET':'POST',
        headers:Object.assign({},publicRequest?{}:{Authorization:'Bearer '+token},body===undefined?{}:{'Content-Type':'application/json'}),
        body:body===undefined?undefined:JSON.stringify(body),signal:controller.signal,cache:'no-store'});
      var value=await response.json();
      if(!response.ok)throw Error(value.error||'The worker could not complete this request.');
      return value;
    }finally{clearTimeout(timeout);}
  }
  function pairPanel(show){
    $('pair-panel').hidden=!show;$('library-panel').hidden=show;
    if(show)$('pair-code').focus();
  }
  $('settings-button').onclick=function(){pairPanel($('pair-panel').hidden);};
  $('pair-form').onsubmit=async function(event){
    event.preventDefault();
    try{
      var result=await api('pair',{code:$('pair-code').value.trim(),name:'Samsung QN90B'},true);
      token=result.token;device=result.device_id;
      localStorage.setItem('sem-tv-token',token);localStorage.setItem('sem-tv-device',device);
      $('pair-code').value='';pairPanel(false);message('TV paired.');await browse(null);
    }catch(error){message(error.message);}
  };
  $('forget').onclick=async function(){
    await leave();localStorage.removeItem('sem-tv-token');localStorage.removeItem('sem-tv-device');
    token='';device='';pairPanel(true);message('Local pairing removed. You can also revoke this TV on the PC dashboard.');
  };
  function button(item){
    var b=document.createElement('button');b.className='movie'+(item.folder?' folder':'');
    var title=document.createElement('strong');title.textContent=item.title;
    var hint=document.createElement('span');hint.textContent=item.folder?'Open folder':'Play movie';
    b.appendChild(title);b.appendChild(hint);
    if(item.plex&&item.poster){var image=document.createElement('img');image.alt='';image.dataset.key=item.id;b.insertBefore(image,title);b.classList.add('has-poster');}
    if(item.resume)hint.textContent='Resume at '+clock(item.resume);
    b.onclick=function(){(item.folder?browse(item.id):openMovie(item)).catch(function(e){message(e.message);});};
    return b;
  }
  async function browse(id,append){
    message('Loading library…');
    var result;
    if(sourceMode==='plex'){
      if(!$('plex-section').value)await loadPlexSections();
      result=await api('plex/library?'+new URLSearchParams({section:$('plex-section').value,offset:append?offset:0,query:$('search').value.trim()}));
    }else result=await api('library?'+new URLSearchParams({folder:id||'',offset:append?offset:0,query:$('search').value.trim()}));
    folder=id;parent=result.parent;rows=append?rows.concat(result.items):result.items;offset=rows.length;
    $('library').replaceChildren.apply($('library'),rows.map(button));
    $('more').hidden=!result.more;$('empty').hidden=rows.length!==0;
    $('back-folder').disabled=!folder||sourceMode==='plex';message('');
    void loadPosters();
    var first=$('library').querySelector('button');if(first)first.focus();
  }
  async function loadPlexSections(){
    var items=await api('plex/sections');
    $('plex-section').replaceChildren.apply($('plex-section'),items.map(function(item){var option=document.createElement('option');option.value=item.id;option.textContent=item.title;return option;}));
    if(!items.length)throw Error('No Plex movie library is available. Pair Plex on the PC, or use Movie folders.');
  }
  async function loadPosters(){
    var epoch=++posterEpoch,images=Array.prototype.slice.call($('library').querySelectorAll('img[data-key]'));
    posterUrls.forEach(function(url){URL.revokeObjectURL(url);});posterUrls=[];
    async function next(){
      while(images.length&&epoch===posterEpoch){
        var image=images.shift();
        try{
          var response=await fetch('/tv-api/plex/poster?key='+encodeURIComponent(image.dataset.key),{headers:{Authorization:'Bearer '+token}});
          if(!response.ok)continue;
          var blob=await response.blob();if(epoch!==posterEpoch)return;
          var url=URL.createObjectURL(blob);posterUrls.push(url);image.src=url;
        }catch(ignore){}
      }
    }
    await Promise.all([next(),next()]);
  }
  $('library-source').onchange=async function(){
    sourceMode=$('library-source').value;$('plex-section').hidden=sourceMode!=='plex';
    try{if(sourceMode==='plex')await loadPlexSections();await browse(null);}catch(error){message(error.message);}
  };
  $('plex-section').onchange=function(){browse(null).catch(function(e){message(e.message);});};
  $('back-folder').onclick=function(){browse(parent).catch(function(e){message(e.message);});};
  $('reload').onclick=function(){browse(folder).catch(function(e){message(e.message);});};
  $('more').onclick=function(){browse(folder,true).catch(function(e){message(e.message);});};
  $('search-form').onsubmit=function(event){event.preventDefault();browse(folder).catch(function(e){message(e.message);});};
  function position(){
    if(!session)return 0;
    if(loadedGeneration!==session.generation)return desiredPosition;
    return Math.min(session.duration,session.base+Math.max(0,(Number(video.currentTime)||0)-mediaStart));
  }
  function showControls(){
    $('player-controls').classList.remove('controls-hidden');clearTimeout(hideTimer);
    if(session&&wantedPlay&&!video.paused&&!reportedError)
      hideTimer=setTimeout(function(){$('player-controls').classList.add('controls-hidden');},7000);
  }
  function updateStatus(value){
    if(!session||session.id!==value.id)return;
    var changed=session.generation!==value.generation;
    session=value;
    $('movie-title').textContent=session.title;
    $('profile').textContent=session.profile+' · '+(session.dub?'English dubbing':'Original audio');
    $('dub').textContent='English dubbing: '+(session.dub?'on':'off');
    $('watch').textContent='Watch with Sem: '+(session.watch?'on':'off');
    $('play-pause').textContent=wantedPlay?'Pause':'Play';
    $('time').textContent=clock(position())+' / '+clock(session.duration);
    $('progress-fill').style.width=(100*position()/Math.max(1,session.duration))+'%';
    if(value.error){reportedError=value.error;playerMessage(value.error);video.pause();showControls();}
    else if(value.resource_wait)playerMessage(value.resource_wait);
    if(changed){loadedGeneration=0;desiredPosition=value.position;video.pause();if(loadAbort)loadAbort.abort();loading=false;loadSerial++;}
    if(value.ready&&loadedGeneration!==value.generation&&!loading&&!value.error)void loadStream();
    if(!value.ready){$('buffering').hidden=false;$('buffer-title').textContent=session.dub?'Preparing English audio':'Preparing compatible video';$('buffer-detail').textContent='Playback waits for real audio and video. You can turn dubbing off to return to the original.';}
    updateCaptions(false);
  }
  async function openMovie(item){
    if(setupOnly){message('Setup-only mode: TV pairing is available; playback and dubbing remain disabled.');return;}
    await leave();
    var number=++operation;
    message('Opening movie…');
    var value=await api('open',item.plex?{plex_key:item.id,resume:true}:{node:item.id});
    if(number!==operation){await api('control',{id:value.id,action:'stop'});return;}
    session=value;desiredPosition=value.position;wantedPlay=true;loadedGeneration=0;reportedError='';
    $('player').hidden=false;$('header').hidden=true;$('library-panel').hidden=true;message('');
    $('audio-track').replaceChildren.apply($('audio-track'),value.audio_tracks.map(function(track){
      var option=document.createElement('option');option.value=track.index;
      option.textContent=track.language+' · '+track.title;return option;
    }));
    $('audio-track').value=value.audio_index;
    $('subtitle-track').replaceChildren.apply($('subtitle-track'),[
      {index:'off',title:'Off'},{index:'auto',title:'External subtitle file'},{index:'dub',title:'English dubbing text'}
    ].concat(value.subtitle_tracks).map(function(track){var o=document.createElement('option');o.value=track.index;o.textContent=(track.language?track.language+' · ':'')+track.title;return o;}));
    $('quality').value=value.quality;$('upscale').value=value.upscale?'on':'off';$('hdr-mode').value=value.hdr_mode;
    $('subtitle-track').value=value.subtitles;$('watch-mode').value=value.watch_mode;$('progress-sync').value=value.sync_progress?'on':'off';
    updateStatus(value);showControls();$('play-pause').focus();
    clearInterval(pollTimer);pollTimer=setInterval(poll,2000);
  }
  function waitFor(event,number,timeout,signal){
    return new Promise(function(resolve,reject){
      var timer;
      function cleanup(){clearTimeout(timer);video.removeEventListener(event,done);video.removeEventListener('error',failed);if(signal)signal.removeEventListener('abort',cancelled);}
      function done(){cleanup();number===operation?resolve():reject(Error('Playback changed.'));}
      function failed(){cleanup();reject(Error('The TV could not load this stream. Try a lower output limit or SDR conversion in Playback settings.'));}
      function cancelled(){cleanup();reject(Error('Playback changed.'));}
      timer=setTimeout(function(){cleanup();reject(Error('The TV is still waiting for playback data.'));},timeout);
      video.addEventListener(event,done,{once:true});video.addEventListener('error',failed,{once:true});
      if(signal)signal.addEventListener('abort',cancelled,{once:true});
    });
  }
  async function loadStream(){
    if(!session)return;loading=true;
    var number=operation,generation=session.generation,id=session.id,target=desiredPosition-session.base,serial=++loadSerial;
    loadAbort=new AbortController();var signal=loadAbort.signal;
    try{
      video.pause();$('buffering').hidden=false;
      var ready=waitFor('loadedmetadata',number,30000,signal);
      video.src=session.url;video.load();await ready;
      if(!session||id!==session.id||generation!==session.generation)return;
      loadedGeneration=generation;
      mediaStart=(session.dub||session.compatibility)&&video.seekable.length?video.seekable.start(0):0;
      updateCaptions(true);
      if(Math.abs(video.currentTime-(mediaStart+target))>.05){
        var seeked=waitFor('seeked',number,20000,signal);
        video.currentTime=Math.max(0,mediaStart+target);await seeked;
      }
      if(number!==operation||!session||generation!==session.generation)return;
      $('buffering').hidden=true;reportedError='';playerMessage('');
      if(wantedPlay)await video.play();
      showControls();
    }catch(error){
      if(number===operation&&serial===loadSerial){reportedError=error.message;playerMessage(error.message);showControls();}
    }finally{if(serial===loadSerial)loading=false;}
  }
  async function poll(){
    if(!session||polling)return;polling=true;
    var id=session.id,number=operation;
    try{
      var state=video.ended?'ended':loading||loadedGeneration!==session.generation?'buffering':video.paused?'paused':'playing';
      var value=await api('heartbeat',{id:id,position:position(),state:state,generation:session.generation});
      if(number===operation&&session&&session.id===id)updateStatus(value);
    }catch(error){if(number===operation)playerMessage(error.message);}
    finally{polling=false;}
  }
  async function change(action,fields){
    if(!session)return;
    var id=session.id,where=position(),number=++operation;
    if(loadAbort)loadAbort.abort();loading=false;loadSerial++;
    video.pause();loadedGeneration=0;desiredPosition=fields&&fields.position!==undefined?fields.position:where;
    reportedError='';playerMessage('Applying playback change…');showControls();
    try{
      var value=await api('control',Object.assign({id:id,action:action,position:desiredPosition},fields||{}));
      if(number===operation&&session&&session.id===id){updateStatus(value);void poll();}
    }catch(error){if(number===operation){playerMessage(error.message);reportedError=error.message;}}
  }
  async function playPause(){
    if(!session)return;
    wantedPlay=!wantedPlay;
    if(wantedPlay&&loadedGeneration===session.generation){try{await video.play();}catch(e){playerMessage(e.message);}}
    else video.pause();
    $('play-pause').textContent=wantedPlay?'Pause':'Play';showControls();void poll();
  }
  async function leave(){
    operation++;clearInterval(pollTimer);clearTimeout(hideTimer);
    if(loadAbort)loadAbort.abort();loading=false;loadSerial++;
    var old=session;session=null;wantedPlay=false;loadedGeneration=0;
    video.pause();video.removeAttribute('src');video.load();
    $('player').hidden=true;$('header').hidden=false;$('library-panel').hidden=!token;
    $('playback-panel').hidden=true;captionKey='';
    if(old){try{await api('control',{id:old.id,action:'stop'});}catch(ignore){}}
  }
  $('leave-player').onclick=function(){leave().then(function(){var b=$('library').querySelector('button');if(b)b.focus();});};
  $('play-pause').onclick=playPause;
  $('rewind').onclick=function(){if(session)void change('seek',{position:Math.max(0,position()-30)});};
  $('forward').onclick=function(){if(session)void change('seek',{position:Math.min(session.duration-.1,position()+30)});};
  $('dub').onclick=function(){if(session)void change('dub',{enabled:!session.dub});};
  $('playback-settings').onclick=function(){$('playback-panel').hidden=false;$('quality').focus();clearTimeout(hideTimer);};
  $('close-playback-settings').onclick=function(){$('playback-panel').hidden=true;showControls();$('playback-settings').focus();};
  $('apply-quality').onclick=function(){
    void change('quality',{quality:Number($('quality').value),upscale:$('upscale').value==='on',hdr_mode:$('hdr-mode').value});
  };
  async function setting(action,fields){
    if(!session)return;
    try{updateStatus(await api('control',Object.assign({id:session.id,action:action},fields)));}
    catch(error){playerMessage(error.message);}
  }
  $('subtitle-track').onchange=function(){void setting('subtitles',{subtitles:$('subtitle-track').value});};
  $('watch-mode').onchange=function(){void setting('watch_mode',{watch_mode:$('watch-mode').value});};
  $('progress-sync').onchange=function(){void setting('progress_sync',{enabled:$('progress-sync').value==='on'});};
  function updateCaptions(force){
    if(!session||loadedGeneration!==session.generation)return;
    var key=session.generation+':'+session.subtitles+':'+(session.subtitles==='dub'?session.ready_through:'');
    if(!force&&key===captionKey)return;captionKey=key;
    Array.prototype.forEach.call(video.querySelectorAll('track'),function(track){track.remove();});
    if(!session.caption_url)return;
    var track=document.createElement('track');track.kind='subtitles';track.label='Movie subtitles';track.srclang='en';track.default=true;
    track.onload=function(){
      if(mediaStart&&track.track&&track.track.cues)Array.prototype.forEach.call(track.track.cues,function(cue){cue.startTime+=mediaStart;cue.endTime+=mediaStart;});
      track.track.mode='showing';
    };
    track.src=session.caption_url;video.appendChild(track);
  }
  $('audio-track').onchange=function(){void change('audio',{audio_index:Number($('audio-track').value)});};
  $('watch').onclick=async function(){
    if(!session)return;
    try{updateStatus(await api('control',{id:session.id,action:'watch',enabled:!session.watch}));playerMessage('Ask Sem to watch with you in an existing voice call.');}
    catch(error){playerMessage(error.message);}
  };
  video.addEventListener('waiting',function(){if(session){$('buffering').hidden=false;$('buffer-title').textContent='Buffering';$('buffer-detail').textContent='Waiting for the next complete segment…';}});
  video.addEventListener('playing',function(){$('buffering').hidden=true;showControls();});
  video.addEventListener('timeupdate',function(){if(session){$('time').textContent=clock(position())+' / '+clock(session.duration);$('progress-fill').style.width=100*position()/session.duration+'%';}});
  video.addEventListener('ended',function(){wantedPlay=false;$('play-pause').textContent='Play';showControls();void poll();});
  video.addEventListener('error',function(){if(session){playerMessage('Playback could not continue. Lower the output limit, select SDR conversion, or turn dubbing off.');showControls();}});
  document.addEventListener('visibilitychange',function(){if(document.hidden&&session){wantedPlay=false;video.pause();void poll();}});
  window.addEventListener('pagehide',function(){
    if(session)fetch('/tv-api/control',{method:'POST',headers:{Authorization:'Bearer '+token,'Content-Type':'application/json'},body:JSON.stringify({id:session.id,action:'stop'}),keepalive:true}).catch(function(){});
  });
  function focusMove(direction){
    var scope=session?($('playback-panel').hidden?$('player-controls'):$('playback-panel')):($('pair-panel').hidden?$('library-panel'):$('pair-panel'));
    var targets=Array.prototype.filter.call(scope.querySelectorAll('button,input,select'),function(e){return !e.disabled&&e.getClientRects().length;});
    var active=document.activeElement,rect=active.getBoundingClientRect(),cx=rect.left+rect.width/2,cy=rect.top+rect.height/2;
    var best=null,score=Infinity;
    targets.forEach(function(e){
      if(e===active)return;
      var r=e.getBoundingClientRect(),dx=r.left+r.width/2-cx,dy=r.top+r.height/2-cy;
      if((direction===37&&dx>=-5)||(direction===39&&dx<=5)||(direction===38&&dy>=-5)||(direction===40&&dy<=5))return;
      var value=(direction===37||direction===39)?Math.abs(dx)+3*Math.abs(dy):Math.abs(dy)+3*Math.abs(dx);
      if(value<score){best=e;score=value;}
    });
    (best||targets[0])?.focus();
  }
  document.addEventListener('keydown',function(event){
    var key=event.keyCode;
    if(key===10009||key===27){event.preventDefault();if(session&&!$('playback-panel').hidden)$('close-playback-settings').click();else if(session)void leave();else if(!$('pair-panel').hidden)pairPanel(false);else if(folder)void browse(parent);return;}
    if(session){
      showControls();
      if([10252,415,19,32].indexOf(key)>=0){event.preventDefault();void playPause();return;}
      if(key===413){event.preventDefault();void leave();return;}
      if(key===412){event.preventDefault();$('rewind').click();return;}
      if(key===417){event.preventDefault();$('forward').click();return;}
      if(key===403){event.preventDefault();$('dub').click();return;}
    }
    if(key>=37&&key<=40&&document.activeElement.tagName!=='INPUT'&&document.activeElement.tagName!=='SELECT'){event.preventDefault();focusMove(key);}
  });
  try{['MediaPlay','MediaPause','MediaPlayPause','MediaStop','MediaRewind','MediaFastForward','Info','ColorF0Red'].forEach(function(key){tizen.tvinputdevice.registerKey(key);});}catch(ignore){}
  $('version').textContent=VERSION+' · development';
  api('version',undefined,true).then(function(value){
    setupOnly=value.setup_only===true;
    $('version').textContent='KevFLIX '+VERSION+' · worker '+value.version+(setupOnly?' · setup only':' · '+value.player_release.slice(0,8));
    if(setupOnly)message('Setup-only mode: pairing is available. Playback and dubbing are disabled.');
  }).catch(function(){});
  if(token)api('settings').then(function(value){
    if(value.plex_url){$('plex-admin').href=value.plex_url.replace(/\/$/,'')+'/web/';$('plex-admin').hidden=false;}
  }).catch(function(){});
  if(token){pairPanel(false);browse(null).catch(function(e){message(e.message);pairPanel(true);});}
  else pairPanel(true);
}());
