(function () {
  'use strict';
  var VERSION='0.3.2', $=function(id){return document.getElementById(id);};
  var token=localStorage.getItem('sem-tv-token')||'', device=localStorage.getItem('sem-tv-device')||'';
  var folder=null,parent=null,offset=0,rows=[],session=null,loadedGeneration=0,wantedPlay=false;
  var operation=0,polling=false,loading=false,pollTimer=null,hideTimer=null,lastHeartbeat=0,reportedError='';
  var video=$('video'),desiredPosition=0,mediaStart=0,loadAbort=null,loadSerial=0;
  var sourceMode='folders',posterUrls=[],posterEpoch=0,captionKey='',setupOnly=false;
  var page='home',selectedItem=null,selectedCard=null,choiceReturn=null,noticeTimer=null,editingSearch=false;
  var defaults={quality:2160,upscale:false,hdr_mode:'auto',dub:false,watch:false,commentary:'balanced'},resumeMap={};
  try{Object.assign(defaults,JSON.parse(localStorage.getItem('kevflix-defaults')||'{}'));resumeMap=JSON.parse(localStorage.getItem('kevflix-resume')||'{}');}catch(ignore){}
  $('worker-address').textContent='Worker: '+location.origin;
  function message(text){
    $('notice').textContent=text||'';$('notice').hidden=!text;clearTimeout(noticeTimer);
    if(text)noticeTimer=setTimeout(function(){$('notice').hidden=true;},8000);
    if(!$('pair-panel').hidden)$('pair-message').textContent=text||'';
  }
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
    $('main').hidden=show;$('sidebar').hidden=show;
    if(show)$('pin-first').focus();
    else if(token){navigate(page,false);$('nav-'+page).focus();}
  }
  $('settings-button').onclick=function(){pairPanel($('pair-panel').hidden);};
  var pairing=false;
  async function submitPair(event){
    if(event)event.preventDefault();
    if(pairing)return;
    var code=$('pair-code').value.trim();
    if(!/^[0-9]{6}$/.test(code)){message('Enter all six digits using the keypad, then choose Pair TV.');$('pin-first').focus();return;}
    pairing=true;$('pair-submit').disabled=true;message('Pairing this TV…');
    try{
      var result=await api('pair',{code:code,name:'KevFLIX TV'},true);
      token=result.token;device=result.device_id;
      localStorage.setItem('sem-tv-token',token);localStorage.setItem('sem-tv-device',device);
      $('pair-code').value='';pairPanel(false);message('TV paired.');await browse(null);
    }catch(error){message(error.message);$('pin-first').focus();}
    finally{pairing=false;$('pair-submit').disabled=false;}
  }
  $('pair-form').onsubmit=submitPair;$('pair-submit').onclick=submitPair;
  function pinDigit(digit){
    if($('pair-code').value.length>=6)return;
    $('pair-code').value+=digit;
    if($('pair-code').value.length===6)$('pair-submit').focus();
  }
  Array.prototype.forEach.call(document.querySelectorAll('[data-digit]'),function(button){button.onclick=function(){pinDigit(button.dataset.digit);};});
  $('pin-delete').onclick=function(){$('pair-code').value=$('pair-code').value.slice(0,-1);};
  $('pin-clear').onclick=function(){$('pair-code').value='';$('pin-first').focus();};
  $('forget').onclick=async function(){
    await leave();localStorage.removeItem('sem-tv-token');localStorage.removeItem('sem-tv-device');
    token='';device='';pairPanel(true);message('Local pairing removed. You can also revoke this TV on the PC dashboard.');
  };
  function button(item){
    var b=document.createElement('button');b.className='movie'+(item.folder?' folder':'');
    var poster=document.createElement('div');poster.className='poster';
    var title=document.createElement('strong');title.textContent=item.title;
    var hint=document.createElement('span');hint.className='subtitle';hint.textContent=item.folder?'Folder':item.year||'Movie';
    if(item.plex&&item.poster){var image=document.createElement('img');image.alt='';image.dataset.key=item.id;poster.appendChild(image);}
    else{var symbol=document.createElement('span');symbol.className='poster-symbol';symbol.textContent=item.folder?'▤':'▷';poster.appendChild(symbol);}
    b.appendChild(poster);b.appendChild(title);b.appendChild(hint);
    if(item.resume)hint.textContent='Resume at '+clock(item.resume);
    b.onclick=function(){if(item.folder)browse(item.id).catch(function(e){message(e.message);});else showDetails(item,b);};
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
    if(!id&&sourceMode==='folders'&&rows.length===1&&rows[0].folder)return browse(rows[0].id);
    rows.forEach(function(item){if(!item.plex&&resumeMap[item.id])item.resume=resumeMap[item.id].position;});
    $('library').replaceChildren.apply($('library'),rows.map(button));
    $('more').hidden=!result.more;$('empty').hidden=rows.length!==0;
    $('back-folder').disabled=!folder||sourceMode==='plex';$('back-folder').hidden=!folder||sourceMode==='plex';message('');
    var continuing=page==='home'?rows.filter(function(item){return !item.folder&&item.resume>10;}).slice(0,5):[];
    $('continue-section').hidden=!continuing.length;$('continue-row').replaceChildren.apply($('continue-row'),continuing.map(button));
    $('library-heading').textContent=page==='search'?'Search results':sourceMode==='plex'?'Your Plex movies':'Your movie folders';
    void loadPosters();
    var first=$('library').querySelector('button');if(first)first.focus();
  }
  async function loadPlexSections(){
    var items=await api('plex/sections');
    $('plex-section').replaceChildren.apply($('plex-section'),items.map(function(item){var option=document.createElement('option');option.value=item.id;option.textContent=item.title;return option;}));
    if(!items.length)throw Error('No Plex movie library is available. Pair Plex on the PC, or use Movie folders.');
  }
  async function loadPosters(){
    var epoch=++posterEpoch,images=Array.prototype.slice.call($('main').querySelectorAll('img[data-key]'));
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
  function finishSearch(){ $('search').readOnly=true;$('search').blur();$('find-movies').focus(); }
  $('search-form').onsubmit=function(event){event.preventDefault();finishSearch();browse(folder).catch(function(e){message(e.message);});};
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
  async function openMovie(item,restart){
    if(setupOnly){message('Setup-only mode: TV pairing is available; playback and dubbing remain disabled.');return;}
    await leave();
    var number=++operation;
    message('Opening movie…');
    var value=await api('open',item.plex?{plex_key:item.id,resume:!restart}:{node:item.id});
    if(!item.plex&&!restart&&item.resume>10)value=await api('control',{id:value.id,action:'seek',position:item.resume});
    if(defaults.quality!==value.quality||defaults.upscale!==value.upscale||defaults.hdr_mode!==value.hdr_mode)
      value=await api('control',{id:value.id,action:'quality',quality:defaults.quality,upscale:defaults.upscale,hdr_mode:defaults.hdr_mode,position:value.position});
    if(defaults.dub)value=await api('control',{id:value.id,action:'dub',enabled:true,position:value.position});
    if(defaults.watch)value=await api('control',{id:value.id,action:'watch',enabled:true});
    value=await api('control',{id:value.id,action:'watch_mode',watch_mode:defaults.commentary});
    if(number!==operation){await api('control',{id:value.id,action:'stop'});return;}
    session=value;desiredPosition=value.position;wantedPlay=true;loadedGeneration=0;reportedError='';
    $('details').hidden=true;$('player').hidden=false;$('main').hidden=true;$('sidebar').hidden=true;message('');
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
    if(session&&selectedItem){
      var saved=position();resumeMap[selectedItem.id]={position:saved>session.duration-30?0:saved};
      try{localStorage.setItem('kevflix-resume',JSON.stringify(resumeMap));}catch(ignore){}
      selectedItem.resume=resumeMap[selectedItem.id].position;
    }
    operation++;clearInterval(pollTimer);clearTimeout(hideTimer);
    if(loadAbort)loadAbort.abort();loading=false;loadSerial++;
    var old=session;session=null;wantedPlay=false;loadedGeneration=0;
    video.pause();video.removeAttribute('src');video.load();
    $('player').hidden=true;$('header').hidden=false;$('main').hidden=false;$('sidebar').hidden=false;$('library-panel').hidden=!token||page==='settings';
    $('playback-panel').hidden=true;captionKey='';
    if(old){try{await api('control',{id:old.id,action:'stop'});}catch(ignore){}}
    if(old&&selectedItem)showDetails(selectedItem,selectedCard);
  }
  $('leave-player').onclick=function(){void leave();};
  $('play-pause').onclick=playPause;
  $('rewind').onclick=function(){if(session)void change('seek',{position:Math.max(0,position()-10)});};
  $('forward').onclick=function(){if(session)void change('seek',{position:Math.min(session.duration-.1,position()+10)});};
  $('dub').onclick=function(){if(session)void change('dub',{enabled:!session.dub});};
  $('playback-settings').onclick=function(){$('playback-panel').hidden=false;renderPlaybackSettings();clearTimeout(hideTimer);};
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
    var scope=!$('choice-panel').hidden?$('choice-panel'):!$('pair-panel').hidden?$('pair-panel'):session?($('playback-panel').hidden?$('player-controls'):$('playback-panel')):!$('details').hidden?$('details'):document;
    var targets=Array.prototype.filter.call(scope.querySelectorAll('button,input,select,a[href]'),function(e){return !e.disabled&&e.tabIndex>=0&&e.getClientRects().length;});
    var active=document.activeElement,rect=active.getBoundingClientRect(),cx=rect.left+rect.width/2,cy=rect.top+rect.height/2;
    if(targets.indexOf(active)<0){if(targets[0])targets[0].focus();return;}
    var best=null,score=Infinity;
    targets.forEach(function(e){
      if(e===active)return;
      var r=e.getBoundingClientRect(),dx=r.left+r.width/2-cx,dy=r.top+r.height/2-cy;
      if((direction===37&&dx>=-5)||(direction===39&&dx<=5)||(direction===38&&dy>=-5)||(direction===40&&dy<=5))return;
      var value=(direction===37||direction===39)?Math.abs(dx)+3*Math.abs(dy):Math.abs(dy)+3*Math.abs(dx);
      if(value<score){best=e;score=value;}
    });
    if(best){best.focus();best.scrollIntoView({block:'nearest',inline:'nearest'});}
  }
  window.addEventListener('keydown',function(event){
    var key=event.keyCode||event.which||({ArrowLeft:37,ArrowUp:38,ArrowRight:39,ArrowDown:40,Enter:13,Backspace:8,Escape:27,Return:10009}[event.key]);
    var active=document.activeElement;
    if(event.isComposing)return;
    if(session&&$('choice-panel').hidden&&$('playback-panel').hidden&&$('player-controls').classList.contains('controls-hidden')&&[13,37,38,39,40].indexOf(key)>=0){
      event.preventDefault();showControls();$('play-pause').focus();return;
    }
    if(active===$('search')&&!active.readOnly){
      if(key===13||key===65376||key===65385||key===10009){event.preventDefault();finishSearch();}
      return;
    }
    if(!$('pair-panel').hidden&&!session){
      var digit=key>=48&&key<=57?key-48:key>=96&&key<=105?key-96:null;
      if(digit!==null){event.preventDefault();pinDigit(String(digit));return;}
      if(key===8){event.preventDefault();$('pin-delete').click();return;}
    }
    if(key===13){
      event.preventDefault();
      if(active===$('search')){active.readOnly=false;active.blur();active.focus();active.click();return;}
      if(active&&active.tagName==='SELECT'){
        active.selectedIndex=(active.selectedIndex+1)%active.options.length;active.dispatchEvent(new Event('change',{bubbles:true}));return;
      }
      if(active&&/^(BUTTON|A)$/.test(active.tagName)&&!active.disabled){active.click();return;}
      focusMove(40);return;
    }
    if(key===10009||key===27){
      event.preventDefault();
      if(!$('choice-panel').hidden)closeChoice();
      else if(session&&!$('playback-panel').hidden)$('close-playback-settings').click();
      else if(session&&!$('player-controls').classList.contains('controls-hidden')){$('player-controls').classList.add('controls-hidden');}
      else if(session)void leave();
      else if(!$('details').hidden)closeDetails();
      else if(!$('pair-panel').hidden)$('pair-back').click();
      else if(folder&&sourceMode==='folders'&&parent)void browse(parent);
      else $('nav-'+page).focus();
      return;
    }
    if(session){
      showControls();
      if([10252,415,19,32].indexOf(key)>=0){event.preventDefault();void playPause();return;}
      if(key===413){event.preventDefault();void leave();return;}
      if(key===412){event.preventDefault();$('rewind').click();return;}
      if(key===417){event.preventDefault();$('forward').click();return;}
      if(key===403){event.preventDefault();$('dub').click();return;}
    }
    if(key>=37&&key<=40){
      event.preventDefault();
      if(session&&active===$('timeline')&&(key===37||key===39)){void change('seek',{position:Math.max(0,Math.min(session.duration-.1,position()+(key===39?10:-10)))});return;}
      if(active&&active.tagName==='SELECT'&&(key===37||key===39)){
        active.selectedIndex=(active.selectedIndex+(key===39?1:-1)+active.options.length)%active.options.length;
        active.dispatchEvent(new Event('change',{bubbles:true}));return;
      }
      focusMove(key);
    }
  },true);
  try{['MediaPlay','MediaPause','MediaPlayPause','MediaStop','MediaRewind','MediaFastForward','Info','ColorF0Red'].forEach(function(key){tizen.tvinputdevice.registerKey(key);});}catch(ignore){}
  $('version').textContent=VERSION+' · development';
  api('version',undefined,true).then(function(value){
    setupOnly=value.setup_only===true;
    $('version').textContent='KevFLIX '+VERSION+' · worker '+value.version+(setupOnly?' · setup only':' · '+value.player_release.slice(0,8));
    $('worker-mode').textContent=setupOnly?'Setup only · playback disabled':'PC processing · local stream';
    if(setupOnly)message('Setup-only mode: pairing is available. Playback and dubbing are disabled.');
  }).catch(function(){});
  function navigate(next,load){
    page=next;$('heading').textContent={home:'Home',movies:'Movies',search:'Search',settings:'Settings'}[next];
    $('settings-page').hidden=next!=='settings';$('library-panel').hidden=next==='settings';$('search-bar').hidden=next!=='search';
    Array.prototype.forEach.call(document.querySelectorAll('[data-page]'),function(b){b.classList.toggle('active',b.dataset.page===next);});
    if(next==='settings'){renderDefaults();if(load!==false)$('settings-source').focus();return;}
    if(next!=='search')$('search').value='';
    if(next==='search'){if(load!==false)$('search').focus();return;}
    if(load!==false)browse(folder).catch(function(e){message(e.message);});
  }
  Array.prototype.forEach.call(document.querySelectorAll('[data-page]'),function(b){b.onclick=function(){navigate(b.dataset.page);};});
  function saveDefaults(){try{localStorage.setItem('kevflix-defaults',JSON.stringify(defaults));}catch(ignore){}renderDefaults();}
  function renderDefaults(){
    $('default-quality-value').textContent=defaults.quality===2160?'4K':defaults.quality+'p';
    $('default-upscale-value').textContent=defaults.upscale?'Lanczos':'Off';
    $('default-hdr-value').textContent=defaults.hdr_mode==='auto'?'Automatic':'Convert to SDR';
    $('default-dub-value').textContent=defaults.dub?'On':'Off';$('default-watch-value').textContent=defaults.watch?'On':'Off';
    $('default-commentary-value').textContent={balanced:'Balanced',quiet:'Quiet','ask-only':'Only when asked'}[defaults.commentary];
  }
  function closeChoice(){
    $('choice-panel').hidden=true;
    if(choiceReturn&&document.contains(choiceReturn))choiceReturn.focus();
  }
  function choose(title,options,current,onSelect){
    choiceReturn=document.activeElement;$('choice-title').textContent=title;$('choice-panel').hidden=false;
    $('choice-list').replaceChildren.apply($('choice-list'),options.map(function(option){
      var b=document.createElement('button');b.className='choice-option'+(String(option.value)===String(current)?' selected':'');
      b.textContent=(String(option.value)===String(current)?'✓  ':'')+option.label;
      b.onclick=async function(){try{await onSelect(option.value);var index=choiceReturn&&choiceReturn.dataset.settingIndex;closeChoice();if(session&&!$('playback-panel').hidden)renderPlaybackSettings(Number(index)||0);}catch(e){message(e.message);}};
      return b;
    }));
    ($('choice-list').querySelector('.selected')||$('choice-list').querySelector('button')||$('choice-back')).focus();
  }
  $('choice-back').onclick=closeChoice;
  var qualities=[{value:2160,label:'4K · 2160p'},{value:1080,label:'1080p'},{value:720,label:'720p'},{value:480,label:'480p'}];
  var modes=[{value:'balanced',label:'Balanced reactions'},{value:'quiet',label:'Quiet reactions'},{value:'ask-only',label:'Only when I ask Sem'}];
  $('default-quality').onclick=function(){choose('Output limit',qualities,defaults.quality,function(v){defaults.quality=Number(v);saveDefaults();});};
  $('default-upscale').onclick=function(){choose('Upscaling',[{value:'off',label:'Off · preserve source size'},{value:'on',label:'Lanczos upscaling on the PC'}],defaults.upscale?'on':'off',function(v){defaults.upscale=v==='on';saveDefaults();});};
  $('default-hdr').onclick=function(){choose('HDR',[{value:'auto',label:'Preserve HDR when supported'},{value:'sdr',label:'Convert to SDR on the PC'}],defaults.hdr_mode,function(v){defaults.hdr_mode=v;saveDefaults();});};
  $('default-dub').onclick=function(){defaults.dub=!defaults.dub;saveDefaults();};
  $('default-watch').onclick=function(){defaults.watch=!defaults.watch;saveDefaults();};
  $('default-commentary').onclick=function(){choose('Sem commentary',modes,defaults.commentary,function(v){defaults.commentary=v;saveDefaults();});};
  $('connection-button').onclick=function(){pairPanel(true);};
  $('pair-back').onclick=function(){if(token)pairPanel(false);else{try{tizen.application.getCurrentApplication().exit();}catch(ignore){history.back();}}};
  async function chooseLibrary(){
    var options=[{value:'folders',label:'Movie folders'}];
    try{
      if(!$('plex-section').options.length)await loadPlexSections();
      Array.prototype.forEach.call($('plex-section').options,function(o){options.push({value:'plex:'+o.value,label:'Plex · '+o.textContent});});
    }catch(ignore){}
    choose('Movie library',options,sourceMode==='plex'?'plex:'+$('plex-section').value:'folders',async function(value){
      if(value==='folders')sourceMode='folders';
      else{sourceMode='plex';$('plex-section').value=value.slice(5);}
      $('library-source').value=sourceMode;folder=null;parent=null;
      var label=sourceMode==='folders'?'Movie folders':$('plex-section').selectedOptions[0].textContent;
      $('source-button').textContent=label;$('source-label').textContent=label;
      await browse(null);
    });
  }
  $('source-button').onclick=function(){void chooseLibrary();};$('settings-source').onclick=function(){void chooseLibrary();};
  function showDetails(item,card){
    selectedItem=item;selectedCard=card;$('details').hidden=false;$('main').hidden=true;$('sidebar').hidden=true;
    $('detail-title').textContent=item.title;$('detail-meta').textContent=item.year?String(item.year):'Local movie';
    $('detail-summary').textContent=item.summary||'';
    $('detail-play').textContent=item.resume>10?'Resume at '+clock(item.resume):'Play';
    $('detail-restart').hidden=!(item.resume>10);
    $('detail-dub').textContent='English dubbing: '+(defaults.dub?'on':'off');$('detail-watch').textContent='Watch with Sem: '+(defaults.watch?'on':'off');
    $('detail-poster').replaceChildren();
    var existing=card&&card.querySelector('img');
    if(existing&&existing.src){var img=document.createElement('img');img.src=existing.src;img.alt='';$('detail-poster').appendChild(img);}
    else $('detail-poster').textContent='▷';
    $('detail-play').focus();
  }
  function closeDetails(){
    $('details').hidden=true;$('main').hidden=false;$('sidebar').hidden=false;
    if(selectedCard&&document.contains(selectedCard))selectedCard.focus();else $('nav-'+page).focus();
  }
  $('details-back').onclick=closeDetails;
  $('detail-play').onclick=function(){if(selectedItem)openMovie(selectedItem,false).catch(function(e){message(e.message);});};
  $('detail-restart').onclick=function(){if(selectedItem)openMovie(selectedItem,true).catch(function(e){message(e.message);});};
  $('detail-dub').onclick=function(){defaults.dub=!defaults.dub;saveDefaults();$('detail-dub').textContent='English dubbing: '+(defaults.dub?'on':'off');};
  $('detail-watch').onclick=function(){defaults.watch=!defaults.watch;saveDefaults();$('detail-watch').textContent='Watch with Sem: '+(defaults.watch?'on':'off');};
  function renderPlaybackSettings(index){
    if(!session)return;
    var rows=[
      ['Output limit',session.quality===2160?'4K':session.quality+'p',function(){choose('Output limit',qualities,session.quality,function(v){return change('quality',{quality:Number(v),upscale:session.upscale,hdr_mode:session.hdr_mode});});}],
      ['Upscaling',session.upscale?'Lanczos':'Off',function(){choose('Upscaling',[{value:'off',label:'Off'},{value:'on',label:'Lanczos on the PC'}],session.upscale?'on':'off',function(v){return change('quality',{quality:session.quality,upscale:v==='on',hdr_mode:session.hdr_mode});});}],
      ['HDR',session.hdr_mode==='auto'?'Automatic':'SDR',function(){choose('HDR',[{value:'auto',label:'Preserve HDR signal'},{value:'sdr',label:'Convert to SDR'}],session.hdr_mode,function(v){return change('quality',{quality:session.quality,upscale:session.upscale,hdr_mode:v});});}],
      ['Audio track','Select',function(){choose('Audio track',session.audio_tracks.map(function(t){return{value:t.index,label:t.language+' · '+t.title};}),session.audio_index,function(v){return change('audio',{audio_index:Number(v)});});}],
      ['Subtitles',session.subtitles==='off'?'Off':'On',function(){choose('Subtitles',[{value:'off',label:'Off'},{value:'auto',label:'External subtitle file'}].concat(session.dub?[{value:'dub',label:'English dubbing text'}]:[]).concat(session.subtitle_tracks.map(function(t){return{value:String(t.index),label:t.language+' · '+t.title};})),session.subtitles,function(v){return setting('subtitles',{subtitles:v});});}],
      ['Sem commentary',session.watch_mode,function(){choose('Sem commentary',modes,session.watch_mode,function(v){return setting('watch_mode',{watch_mode:v});});}],
      ['Save progress to Plex',session.sync_progress?'On':'Off',function(){return setting('progress_sync',{enabled:!session.sync_progress});}]
    ];
    $('playback-options').replaceChildren.apply($('playback-options'),rows.map(function(row,number){
      var b=document.createElement('button');b.className='setting-row';var label=document.createElement('span'),value=document.createElement('span');
      b.dataset.settingIndex=number;label.textContent=row[0];value.textContent=row[1];b.appendChild(label);b.appendChild(value);
      b.onclick=async function(){await row[2]();if($('choice-panel').hidden)renderPlaybackSettings(number);};return b;
    }));
    var options=$('playback-options').querySelectorAll('button');options[Math.min(Number(index)||0,options.length-1)].focus();
  }
  $('timeline').onclick=function(){showControls();};
  renderDefaults();
  if(token){
    pairPanel(false);
    api('settings').then(async function(value){
      if(value.plex_url){$('plex-admin').hidden=false;$('plex-admin').onclick=function(){window.open(value.plex_url.replace(/\/$/,'')+'/web/','_blank');};}
      if(value.plex_configured){try{await loadPlexSections();sourceMode='plex';$('source-button').textContent='Plex movies';$('source-label').textContent='Plex movies';}catch(ignore){}}
      await browse(null);
    }).catch(function(e){message(e.message);pairPanel(true);});
  }else pairPanel(true);
}());
