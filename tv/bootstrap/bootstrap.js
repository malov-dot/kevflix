(function () {
  'use strict';
  var input=document.getElementById('address'),message=document.getElementById('message');
  var button=document.getElementById('open-player'),busy=false,currentRequest=null,attempt=0;
  var form=document.getElementById('connect'),instructions=document.getElementById('instructions'),change=document.getElementById('change-address');
  try{input.value=localStorage.getItem('kevflix-worker-address')||localStorage.getItem('sem-worker-address')||'';}catch(ignore){}
  function editAddress(text){
    attempt++;if(currentRequest){currentRequest.abort();currentRequest=null;}
    busy=false;button.disabled=false;form.hidden=false;instructions.hidden=false;change.hidden=true;
    message.textContent=text||'';input.focus();
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
      if(!(/^(10\.|192\.168\.|127\.|172\.(1[6-9]|2[0-9]|3[01])\.)/.test(h)||h==='localhost'||/\.local$/.test(h)))
        throw Error('Use your PC’s private LAN address and worker port.');
      if(!url.port)url.port='8787';
      origin=url.origin;
      input.value=origin;
    }catch(error){message.textContent=error.message;return;}
    busy=true;button.disabled=true;message.textContent='Opening KevFLIX…';change.hidden=false;
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
  change.onclick=function(){editAddress();};
  window.addEventListener('keydown',function(event){
    if(event.keyCode===13&&document.activeElement===change){event.preventDefault();editAddress();return;}
    if(event.keyCode===13&&document.activeElement===button){connect(event);return;}
    if(busy&&(event.keyCode===10009||event.keyCode===27)){event.preventDefault();editAddress();return;}
    if(busy&&[37,38,39,40].indexOf(event.keyCode)>=0){event.preventDefault();change.focus();return;}
    if(event.keyCode===40){event.preventDefault();button.focus();}
    if(event.keyCode===38){event.preventDefault();input.focus();}
    if(event.keyCode===10009){event.preventDefault();try{tizen.application.getCurrentApplication().exit();}catch(ignore){}}
  });
  if(input.value&&location.hash!=='#configure')connect();else editAddress();
}());
