(function () {
  'use strict';
  var input=document.getElementById('address'), message=document.getElementById('message');
  input.value=localStorage.getItem('sem-worker-address')||'';
  document.getElementById('connect').onsubmit=function(event){
    event.preventDefault();
    try {
      var url=new URL(input.value);
      if(!/^https?:$/.test(url.protocol)||url.username||url.password)throw Error('Enter the PC worker HTTP or HTTPS address.');
      var h=url.hostname;
      if(!(/^(10\.|192\.168\.|127\.|172\.(1[6-9]|2[0-9]|3[01])\.)/.test(h)||h==='localhost'||/\.local$/.test(h)))
        throw Error('Use your PC’s private LAN address, for example http://192.168.1.100:8787.');
      localStorage.setItem('sem-worker-address',url.origin);
      window.location.assign(url.origin+'/tv/');
    } catch(error) {message.textContent=error.message;}
  };
  document.addEventListener('keydown',function(event){
    if(event.keyCode===40){event.preventDefault();document.querySelector('button').focus();}
    if(event.keyCode===38){event.preventDefault();input.focus();}
    if(event.keyCode===10009){event.preventDefault();try{tizen.application.getCurrentApplication().exit();}catch(ignore){}}
  });
  input.focus();
}());
