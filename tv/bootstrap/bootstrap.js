(function () {
  'use strict';
  var input=document.getElementById('address'),message=document.getElementById('message');
  var button=document.getElementById('open-player'),edit=document.getElementById('edit-address'),busy=false,currentRequest=null,attempt=0,editing=false;
  var form=document.getElementById('connect'),instructions=document.getElementById('instructions'),change=document.getElementById('change-address');
  try{input.value=localStorage.getItem('kevflix-worker-address')||localStorage.getItem('sem-worker-address')||'';}catch(ignore){}
  function editAddress(text){
    attempt++;if(currentRequest){currentRequest.abort();currentRequest=null;}
    busy=false;button.disabled=false;form.hidden=false;instructions.hidden=false;change.hidden=true;
    editing=false;input.readOnly=true;input.blur();message.textContent=text||'';
    (input.value?button:edit).focus();
  }
  function startEditing(){
    editing=true;input.readOnly=false;input.focus();input.click();
  }
  function finishEditing(){
    editing=false;input.readOnly=true;input.blur();button.focus();
  }
  function connect(event){
    if(event)event.preventDefault();
    if(busy)return;
    var origin;
    try{
      var address=input.value.trim();
      if(!/^https?:\/\//i.test(address))address='http://'+address;
      var url=new URL(address);
      if(!/^https?:$/.test(url.protocol)||url.username||url.password)throw Error('Enter the PC worker HTTP or HTTPS address.');
      var h=url.hostname;
      var parts=h.split('.'),ipv4=parts.length===4&&parts.every(function(n){return /^\d{1,3}$/.test(n)&&Number(n)<=255;});
      var privateIP=ipv4&&(Number(parts[0])===10||Number(parts[0])===127||Number(parts[0])===192&&Number(parts[1])===168||Number(parts[0])===172&&Number(parts[1])>=16&&Number(parts[1])<=31);
      if(!(privateIP||h==='localhost'||/\.local$/.test(h)))
        throw Error('Use your PC’s private LAN address and worker port.');
      if(!url.port)url.port='8787';
      origin=url.origin;
      input.value=origin;
    }catch(error){editAddress(error.message);return;}
    finishEditing();busy=true;button.disabled=true;form.hidden=true;instructions.hidden=true;
    message.textContent='Opening KevFLIX…';change.hidden=false;change.focus();
    var number=++attempt,xhr=new XMLHttpRequest();currentRequest=xhr;
    xhr.open('GET',origin+'/tv-api/version',true);xhr.timeout=8000;
    function fail(text){if(number!==attempt)return;currentRequest=null;editAddress(text);}
    xhr.onload=function(){
      var result;
      if(number!==attempt)return;
      try{result=JSON.parse(xhr.responseText);}catch(ignore){}
      if(xhr.status!==200||!result||result.product!=='KevFLIX'){fail('That address did not return a KevFLIX worker. Check the address and port.');return;}
      try{localStorage.setItem('kevflix-worker-address',origin);localStorage.setItem('sem-worker-address',origin);}catch(ignore){}
      form.hidden=true;instructions.hidden=true;change.hidden=true;message.textContent='Opening your home screen…';
      var launcher=location.href.split('#')[0];
      window.location.replace(origin+'/tv/#launcher='+encodeURIComponent(launcher));
    };
    xhr.onerror=function(){fail('Cannot reach the PC worker. Start it, check the address, and allow its port on your private network.');};
    xhr.ontimeout=function(){fail('The PC did not respond. Check that the worker is running and the TV is on the same local network.');};
    try{xhr.send();}catch(error){fail(error.message);}
  }
  document.getElementById('connect').onsubmit=connect;
  button.onclick=connect;
  edit.onclick=startEditing;
  change.onclick=function(){editAddress();};
  input.addEventListener('blur',function(){if(editing)setTimeout(finishEditing,0);});
  function consume(event){event.preventDefault();event.stopPropagation();}
  window.addEventListener('keydown',function(event){
    var key=event.keyCode||event.which||({ArrowLeft:37,ArrowUp:38,ArrowRight:39,ArrowDown:40,Enter:13,Escape:27,Return:10009}[event.key]);
    if(event.isComposing)return;
    if(editing){
      // Only an explicit Edit opens the native keyboard; let it own its arrow keys.
      if([13,65376,65385,10009,27].indexOf(key)>=0){consume(event);finishEditing();}
      return;
    }
    if([37,38,39,40].indexOf(key)>=0){
      consume(event);
      if(busy){change.focus();return;}
      var buttons=[button,edit].filter(function(b){return !b.disabled&&b.getClientRects().length;});
      var index=buttons.indexOf(document.activeElement),step=(key===37||key===38)?-1:1;
      if(buttons.length)buttons[index<0?0:(index+step+buttons.length)%buttons.length].focus();
      return;
    }
    if(key===13){
      consume(event);if(event.repeat)return;
      var focused=document.activeElement;
      if(focused===button||focused===edit||focused===change){if(!focused.disabled)focused.click();}
      else (busy?change:input.value?button:edit).focus();
      return;
    }
    if(key===10009||key===27){
      consume(event);if(busy){editAddress();return;}
      try{tizen.application.getCurrentApplication().exit();}catch(ignore){history.back();}
    }
  },true);
  if(input.value&&location.hash!=='#configure')connect();else editAddress();
}());
