/* KevFLIX YouTube view (stage 1). The real youtube.com/tv page runs in a background browser on the PC; this file shows its
   screen (multipart MJPEG in an <img>) fullscreen and sends the TV remote keys back to the PC. A LONG Back press (>= 700 ms)
   always returns to the KevFLIX home; a short Back goes to YouTube. Loaded BEFORE player.js so its key listener runs first. */
(function () {
  'use strict';
  var LONG_BACK_MS = 700, DOUBLE_BACK_MS = 900, HINT_MS = 6000, REPEAT_DELAY_MS = 380, REPEAT_MS = 90, FIRST_FRAME_MS = 25000, STATUS_MS = 1000, PLAY_MAX_AGE_S = 20;
  // Samsung remote key codes -> names the PC understands (relay/youtube_host.py KEYS).
  var NAMES = {37: 'Left', 38: 'Up', 39: 'Right', 40: 'Down', 13: 'Enter', 415: 'Play', 19: 'Pause',
    10252: 'PlayPause', 417: 'FastForward', 412: 'Rewind'};
  var BACK = {10009: 1, 27: 1, 8: 1}, ARROWS = {37: 1, 38: 1, 39: 1, 40: 1};
  var $ = function (id) { return document.getElementById(id); };
  var active = false, state = 'idle', ticket = '', serial = 0, returnFocus = null, firstFrameTimer = 0, statusTimer = 0;
  var held = {}, repeats = {}, downNow = {}, backHeld = false, backTimer = 0, swallowBack = false, lastBackAt = 0, hintTimer = 0;
  var queue = [], sending = false, failures = 0, enabled = false, pageName = 'home';
  // Hand-off (stage 3): when the picked video's id reaches us (key reply or the 1 s status poll) the picture stream stops and KevFLIX's own
  // player (player.js, window.KevFlixPlayer.openYoutube) downloads and plays it; its exit (Back / end of video / any error) calls resume().
  var seenSeq = 0, handoff = false, notice = '';
  // Picture transport. Pull (/tv-api/youtube/frame): ONE picture per request and the next is asked for only after the last arrived, so a slow
  // Wi-Fi link can never queue seconds of pictures (the old push stream, one endless multipart <img>, did, which felt like a laggy menu).
  // The TV also reports how long each picture took (download + decode) so the PC can send smaller pictures when the link is slow.
  // The push stream stays as the fallback when the PC or the TV lacks support.
  var pullOk = false, pullAbort = null, pullUrl = '', pullFailures = 0, lastCounter = 0, lastMs = 0;

  function token() { try { return localStorage.getItem('sem-tv-token') || ''; } catch (ignore) { return ''; } }
  function mapKey(code) { return BACK[code] ? 'Back' : (NAMES[code] || null); }

  function call(path, body, ms) {
    var controller = typeof AbortController === 'function' ? new AbortController() : null;
    var timer = setTimeout(function () { if (controller) controller.abort(); }, ms || 15000);
    return fetch('/tv-api/' + path, {
      method: body === undefined ? 'GET' : 'POST', cache: 'no-store', signal: controller ? controller.signal : undefined,
      headers: Object.assign({ Authorization: 'Bearer ' + token() }, body === undefined ? {} : { 'Content-Type': 'application/json' }),
      body: body === undefined ? undefined : JSON.stringify(body)
    }).then(function (response) {
      return response.json().then(function (value) {
        if (!response.ok) throw new Error(value && value.error || 'The PC could not complete this request.');
        return value;
      });
    }).then(function (value) { clearTimeout(timer); return value; }, function (error) { clearTimeout(timer); throw error; });
  }

  function setState(next, text) {
    state = next;
    $('youtube-status').textContent = text || '';
    $('youtube-status').hidden = !text;
  }

  // ---- keys to the PC (serialised so a key-up can never overtake its key-down) --------------------------------------
  function send(name, action, isRepeat) {
    if (isRepeat && queue.length > 2) return;                     // a slow link drops repeats instead of queueing them
    queue.push({ key: name, action: action });
    pump();
  }
  function pump() {
    if (sending || !queue.length) return;
    sending = true;
    var item = queue.shift();
    call('youtube/key', item, 8000).then(function (value) { failures = 0; if (value && value.play) considerPlay(value.play); }, function () {
      failures += 1;
      if (failures >= 3 && active && state === 'live') setState('error', 'Lost the connection to YouTube on the PC. Press OK to retry, or Back to return.');
    }).then(function () { sending = false; pump(); });
  }
  function releaseAll(notify) {
    Object.keys(repeats).forEach(function (code) { clearTimeout(repeats[code]); clearInterval(repeats[code]); });
    repeats = {};
    Object.keys(held).forEach(function (code) { if (notify && held[code] !== 'stale') send(held[code], 'up'); });
    held = {};
    clearTimeout(backTimer); backTimer = 0; backHeld = false;
  }

  // ---- the view -----------------------------------------------------------------------------------------------------
  function connect() {
    var mine = ++serial;
    ticket = '';
    stopPicture();
    setState('connecting', 'Starting YouTube on your PC…');
    call('youtube/open', {}, 70000).then(function (result) {
      if (mine !== serial || !active) return;
      ticket = result.screen_ticket || '';
      pullOk = result.frame_pull === true && typeof fetch === 'function' && typeof URL === 'function' && typeof URL.createObjectURL === 'function';
      seenSeq = Math.max(seenSeq, result.play_seq || 0);         // a selection made before this view opened is not a new pick
      showStream(mine);
    }).catch(function (error) {
      if (mine !== serial || !active) return;
      setState('error', ((error && error.message) || 'YouTube could not start') + '. Press OK to retry, or Back to return.');
    });
  }
  function firstPicture(mine) {
    if (mine !== serial) return;
    clearTimeout(firstFrameTimer);
    if (state !== 'live') {
      setState('live', notice || 'Press Back twice, or hold Back, to return to KevFLIX');   // how to leave: a single Back belongs to YouTube
      notice = '';
      clearTimeout(hintTimer);
      hintTimer = setTimeout(function () { if (active && state === 'live') setState('live', ''); }, HINT_MS);
    }
  }
  function stopPicture() {
    if (pullAbort) { try { pullAbort.abort(); } catch (ignore) {} pullAbort = null; }
    if (pullUrl) { try { URL.revokeObjectURL(pullUrl); } catch (ignore) {} pullUrl = ''; }
    lastCounter = 0; lastMs = 0; pullFailures = 0;
    var image = $('youtube-screen');
    image.onload = image.onerror = null;
    image.removeAttribute('src');                                  // also closes the push stream; the PC stops it when no client is left
  }
  function pullPicture(mine) {
    if (mine !== serial || !active) return;
    var controller = typeof AbortController === 'function' ? new AbortController() : null, headersAt = 0;
    pullAbort = controller;
    var path = '/tv-api/youtube/frame?ticket=' + encodeURIComponent(ticket) + '&after=' + lastCounter + (lastMs > 0 ? '&ms=' + Math.round(lastMs) : '');
    lastMs = 0;
    fetch(path, { cache: 'no-store', signal: controller ? controller.signal : undefined }).then(function (response) {
      headersAt = Date.now();
      var counter = Number(response.headers && response.headers.get('X-Frame'));
      if (counter >= 0) lastCounter = counter;
      if (response.status === 204) return null;                    // the page did not change: ask again
      if (!response.ok) throw new Error('picture ' + response.status);
      return response.blob();
    }).then(function (blob) {
      if (mine !== serial || !active) return;
      pullFailures = 0;
      if (blob && blob.size) showPulled(mine, blob, headersAt);
      pullPicture(mine);                                           // the next picture is asked for as soon as this one has arrived
    }).catch(function () {
      if (mine !== serial || !active) return;
      pullFailures += 1;
      if (pullFailures >= 4) { clearTimeout(firstFrameTimer); setState('error', 'The YouTube picture stopped. Press OK to retry, or Back to return.'); return; }
      setTimeout(function () { pullPicture(mine); }, 250 * pullFailures);
    });
  }
  function showPulled(mine, blob, headersAt) {
    var image = $('youtube-screen'), url = URL.createObjectURL(blob), previous = pullUrl;
    pullUrl = url;
    image.onload = function () {
      if (mine !== serial) return;
      lastMs = Date.now() - headersAt;                             // download + decode of this picture, reported with the next request
      firstPicture(mine);
    };
    image.onerror = function () { lastMs = 0; };
    image.src = url;
    if (previous) setTimeout(function () { try { URL.revokeObjectURL(previous); } catch (ignore) {} }, 1000);
  }
  function showStream(mine) {
    var image = $('youtube-screen');
    if (pullOk) {
      lastCounter = 0; lastMs = 0; pullFailures = 0;
      pullPicture(mine);
      clearTimeout(firstFrameTimer);
      firstFrameTimer = setTimeout(function () {
        if (mine === serial && state !== 'live') setState('error', 'YouTube did not show a picture in time. Press OK to retry, or Back to return.');
      }, FIRST_FRAME_MS);
      return;
    }
    image.onload = function () { firstPicture(mine); };
    image.onerror = function () {
      if (mine !== serial || !active) return;
      clearTimeout(firstFrameTimer);
      setState('error', 'The YouTube picture stopped. Press OK to retry, or Back to return.');
    };
    image.src = '/tv-api/youtube/screen?ticket=' + encodeURIComponent(ticket) + '&n=' + Date.now();
    clearTimeout(firstFrameTimer);
    firstFrameTimer = setTimeout(function () {
      if (mine === serial && state !== 'live') setState('error', 'YouTube did not show a picture in time. Press OK to retry, or Back to return.');
    }, FIRST_FRAME_MS);
  }
  function watchHost() {
    clearInterval(statusTimer);
    statusTimer = setInterval(function () {
      if (!active || state !== 'live') return;
      call('youtube/status', undefined, 6000).then(function (value) {
        if (active && state === 'live' && value.state !== 'running') setState('error', 'YouTube stopped on the PC. Press OK to start it again, or Back to return.');
        else if (value.play) considerPlay(value.play);
      }).catch(function () {});
    }, STATUS_MS);
  }
  // ---- hand-off to the KevFLIX player ---------------------------------------------------------------------------------
  function considerPlay(play) {
    if (!active || state !== 'live' || handoff || !play || !play.video_id || !(play.seq > seenSeq)) return;
    seenSeq = play.seq;
    if (play.age > PLAY_MAX_AGE_S) return;                         // an old selection (for example after a reload) is not a request
    startHandoff(play.video_id);
  }
  function startHandoff(videoId) {
    var player = window.KevFlixPlayer;
    handoff = true;
    releaseAll(false); queue = [];
    Object.keys(downNow).forEach(function (code) { held[code] = 'stale'; });   // the OK that picked the video must not leak into the player
    active = false; serial += 1; state = 'handoff';
    clearTimeout(hintTimer); clearTimeout(firstFrameTimer); clearInterval(statusTimer);
    stopPicture();                                                 // the picture stream stops while KevFLIX plays the video
    $('youtube-view').hidden = true;
    if (!player || typeof player.openYoutube !== 'function') { resume('The KevFLIX player is not ready.'); return; }
    var opened;
    try { opened = player.openYoutube(videoId); } catch (error) { opened = Promise.reject(error); }
    Promise.resolve(opened).catch(function (error) {
      resume(error && error.message === 'Cancelled' ? '' : ((error && error.message) || 'This YouTube video could not be played.'));   // never trap the viewer: back to the YouTube view with the reason
    });
  }
  // Called by player.js when the hand-off ends (Back, end of the video) and by the error path above. Safe to call twice.
  function resume(reason) {
    if (!handoff || active) return;
    handoff = false; notice = reason ? String(reason) : '';
    Object.keys(downNow).forEach(function (code) { held[code] = 'stale'; });
    active = true; failures = 0; queue = []; sending = false;
    $('youtube-view').hidden = false;
    setState('connecting', 'Returning to YouTube…');
    var mine = ++serial;
    // the PC unblocks the page's media and sends Back to its watch screen FIRST; only then is the picture stream reopened
    call('youtube/resume', {}, 10000).catch(function () {}).then(function () { if (mine === serial && active) { connect(); watchHost(); } });
  }
  function open() {
    if (active || handoff) return;
    active = true; failures = 0; queue = []; sending = false; notice = '';
    returnFocus = document.activeElement;
    releaseAll(false);
    Object.keys(downNow).forEach(function (code) { held[code] = 'stale'; });   // the key that opened the view is not a YouTube key
    $('youtube-view').hidden = false;
    connect();
    watchHost();
  }
  function exit() {
    if (!active) return;
    var wasBack = backHeld;
    releaseAll(true);
    active = false; serial += 1; state = 'idle'; lastBackAt = 0;
    clearTimeout(hintTimer); clearTimeout(firstFrameTimer); clearInterval(statusTimer);
    stopPicture();                                                 // closes the stream; the PC stops it when no client is left
    $('youtube-view').hidden = true;
    swallowBack = wasBack;                                         // the rest of a held Back press must not reach KevFLIX
    var target = returnFocus && document.contains(returnFocus) && !returnFocus.hidden ? returnFocus : $('youtube-tile');
    if (target && target.focus) target.focus();
  }

  // ---- keys ---------------------------------------------------------------------------------------------------------
  function onKeyDown(event) {
    var code = event.keyCode || event.which;
    if (!active) {
      if (!event.repeat) downNow[code] = true;
      if (swallowBack && BACK[code]) { event.preventDefault(); event.stopImmediatePropagation(); }
      return;
    }
    event.preventDefault(); event.stopImmediatePropagation();
    if (event.repeat || held[code]) return;                        // our own repeat timers drive held arrows
    if (BACK[code]) {
      if (backHeld) return;
      if (state !== 'live') { backHeld = true; exit(); return; }   // nothing to forward to: any Back leaves at once
      backHeld = true;
      backTimer = setTimeout(function () { backTimer = 0; exit(); }, LONG_BACK_MS);
      return;
    }
    var name = NAMES[code];
    if (!name) return;
    if (state === 'error') { if (code === 13) { held[code] = 'stale'; connect(); } return; }
    if (state !== 'live') return;
    held[code] = name;
    send(name, 'down');
    if (ARROWS[code]) {
      repeats[code] = setTimeout(function () {
        repeats[code] = setInterval(function () { send(name, 'down', true); }, REPEAT_MS);
      }, REPEAT_DELAY_MS);
    }
  }
  function onKeyUp(event) {
    var code = event.keyCode || event.which;
    delete downNow[code];
    if (!active) {
      if (swallowBack && BACK[code]) { swallowBack = false; event.preventDefault(); event.stopImmediatePropagation(); }
      return;
    }
    event.preventDefault(); event.stopImmediatePropagation();
    if (BACK[code]) {
      if (!backHeld) return;
      backHeld = false;
      if (backTimer) {                                             // released before 700 ms: a normal Back for YouTube
        clearTimeout(backTimer); backTimer = 0;
        if (state === 'live') {
          var now = Date.now();
          if (lastBackAt && now - lastBackAt < DOUBLE_BACK_MS) { lastBackAt = 0; exit(); return; }   // two quick Backs leave YouTube
          lastBackAt = now;
          send('Back', 'press');
        }
      }
      return;
    }
    if (!held[code]) return;
    clearTimeout(repeats[code]); clearInterval(repeats[code]); delete repeats[code];
    var name = held[code];
    delete held[code];
    if (name !== 'stale') send(name, 'up');
  }
  window.addEventListener('keydown', onKeyDown, true);
  window.addEventListener('keyup', onKeyUp, true);
  window.addEventListener('blur', function () { if (active) releaseAll(true); });

  // ---- home tile ----------------------------------------------------------------------------------------------------
  function applyTile() {
    $('youtube-section').hidden = !(enabled && pageName === 'home');
    if ($('nav-youtube')) $('nav-youtube').hidden = !enabled;     // the side menu entry, beside the libraries
  }
  function refresh() {
    if (!token()) { enabled = false; applyTile(); return Promise.resolve(); }
    return call('youtube/status', undefined, 8000).then(function (value) { enabled = !!value.enabled; applyTile(); },
      function () { enabled = false; applyTile(); });
  }
  function setPage(next) { pageName = next; applyTile(); if (next === 'home') refresh(); }
  $('youtube-tile').onclick = open;
  if ($('nav-youtube')) $('nav-youtube').onclick = open;
  try { ['MediaPlay', 'MediaPause', 'MediaPlayPause', 'MediaRewind', 'MediaFastForward'].forEach(function (name) { tizen.tvinputdevice.registerKey(name); }); } catch (ignore) {}
  window.KevFlixYoutube = {
    active: function () { return active; }, state: function () { return state; }, open: open, exit: exit, resume: resume,
    handingOff: function () { return handoff; },
    page: setPage, refresh: refresh, mapKey: mapKey, longBackMs: LONG_BACK_MS
  };
  refresh();
})();
