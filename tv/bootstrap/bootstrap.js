(function () {
  'use strict';
  var input=document.getElementById('address'),message=document.getElementById('message');
  var button=document.getElementById('open-player'),busy=false;
  try{input.value=localStorage.getItem('sem-worker-address')||'';}catch(ignore){}
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
    busy=true;button.disabled=true;message.textContent='Connecting to your PC…';
    var xhr=new XMLHttpRequest();
    xhr.open('GET',origin+'/tv-api/version',true);xhr.timeout=8000;
    function fail(text){busy=false;button.disabled=false;message.textContent=text;button.focus();}
    xhr.onload=function(){
      var result;
      try{result=JSON.parse(xhr.responseText);}catch(ignore){}
      if(xhr.status!==200||!result||!result.version){fail('That address did not return a KevFLIX worker. Check the address and port.');return;}
      try{localStorage.setItem('sem-worker-address',origin);}catch(ignore){}
      message.textContent=result.setup_only?'Connected. Opening TV pairing; playback is disabled.':'Connected. Opening KevFLIX…';
      window.location.assign(origin+'/tv/');
    };
    xhr.onerror=function(){fail('Cannot reach the PC worker. Start it, check the address, and allow its port on your private network.');};
    xhr.ontimeout=function(){fail('The PC did not respond. Check that the worker is running and the TV is on the same local network.');};
    try{xhr.send();}catch(error){fail(error.message);}
  }
  document.getElementById('connect').onsubmit=connect;
  button.onclick=connect;
  document.addEventListener('keydown',function(event){
    if(event.keyCode===13&&document.activeElement===button){connect(event);return;}
    if(event.keyCode===40){event.preventDefault();button.focus();}
    if(event.keyCode===38){event.preventDefault();input.focus();}
    if(event.keyCode===10009){event.preventDefault();try{tizen.application.getCurrentApplication().exit();}catch(ignore){}}
  });
  input.focus();
}());
