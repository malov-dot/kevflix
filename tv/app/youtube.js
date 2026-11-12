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
  // Timing report: the TV's console cannot be read, so every 10 s while the view is live ONE line goes to the PC's log (diagnostic stage
  // 'youtube-view'): pictures per second, KB, wait for the PC, download and decode time per picture, how late this page's own timers run
  // (lag: a busy TV page delays everything, including the remote keys) and the time from a remote key press to the next picture on screen.
  var stats = null, statsTimer = 0, lagTimer = 0, lagLast = 0, keyAt = 0;
  function resetStats() { stats = { n: 0, bytes: 0, wait: 0, xfer: 0, xferMax: 0, dec: 0, decMax: 0, lag: 0, lagMax: 0, lagN: 0, keys: [], since: Date.now() }; }
  function startStats() {
    stopStats(); resetStats(); lagLast = Date.now();
    lagTimer = setInterval(function () {
      var now = Date.now(), late = Math.max(0, now - lagLast - 100);
      lagLast = now;
      if (stats) { stats.lag += late; stats.lagN += 1; if (late > stats.lagMax) stats.lagMax = late; }
    }, 100);
    statsTimer = setInterval(reportStats, 10000);
  }
  function stopStats() { clearInterval(lagTimer); clearInterval(statsTimer); lagTimer = statsTimer = 0; stats = null; keyAt = 0; }
  function reportStats() {
    if (!stats || !active || state !== 'live') { if (stats) resetStats(); return; }
    var s = stats, seconds = Math.max(1, (Date.now() - s.since) / 1000), n = Math.max(1, s.n);
    var keys = s.keys.slice().sort(function (a, b) { return a - b; });
    resetStats();
    if (!s.n && !keys.length) return;
    call('diagnostic', { stage: 'youtube-view', youtube: {
      pictures: s.n, fps: Math.round(10 * s.n / seconds) / 10, kb: Math.round(s.bytes / n / 1024), waitMs: Math.round(s.wait / n),
      xferMs: Math.round(s.xfer / n), xferMax: s.xferMax, decodeMs: Math.round(s.dec / n), decodeMax: s.decMax,
      lagAvg: Math.round(s.lag / Math.max(1, s.lagN)), lagMax: s.lagMax,
      keys: keys.length, keyMedian: keys.length ? keys[keys.length >> 1] : 0, keyMax: keys.length ? keys[keys.length - 1] : 0, pull: pullOk ? 1 : 0,
      path: path, resets: resets, errs: decodeErrors, queueMax: queueMax } }, 5000).catch(function () {});
    queueMax = 0;
    noteWindow(keys.length ? keys[keys.length >> 1] : 0, keys.length);
  }
  // Once per page load: which browser features this TV has for a future video-based picture (the PC reads it from its log).
  var capsReported = false;
  function reportCapabilities() {
    if (capsReported) return;
    capsReported = true;
    var caps = { videoDecoder: typeof VideoDecoder === 'function' ? 1 : 0, imageBitmap: typeof createImageBitmap === 'function' ? 1 : 0,
      webSocket: typeof WebSocket === 'function' ? 1 : 0, offscreen: typeof OffscreenCanvas === 'function' ? 1 : 0,
      mse: typeof MediaSource === 'function' ? 1 : 0, streams: typeof ReadableStream === 'function' ? 1 : 0, h264hw: -2 };
    var send = function () { call('diagnostic', { stage: 'youtube-caps', youtube: caps }, 5000).catch(function () {}); };
    try {
      if (caps.videoDecoder && typeof VideoDecoder.isConfigSupported === 'function') {
        VideoDecoder.isConfigSupported({ codec: 'avc1.640028', codedWidth: 1920, codedHeight: 1080, hardwareAcceleration: 'prefer-hardware' })
          .then(function (result) { caps.h264hw = result && result.supported ? 1 : 0; }, function () { caps.h264hw = -1; }).then(send);
        return;
      }
    } catch (ignore) {}
    send();
  }
  function notePicture(bytes, requestAt, headersAt, blobAt, shownAt) {
    if (!stats) return;
    var xfer = blobAt - headersAt, dec = shownAt - blobAt;
    stats.n += 1; stats.bytes += bytes; stats.wait += headersAt - requestAt;
    stats.xfer += xfer; if (xfer > stats.xferMax) stats.xferMax = xfer;
    stats.dec += dec; if (dec > stats.decMax) stats.decMax = dec;
    if (keyAt) { stats.keys.push(shownAt - keyAt); keyAt = 0; }
  }

  // ---- F20: the picture as low-latency video ------------------------------------------------------------------------------
  // The PC encodes the page as H.264 and pushes it over ONE WebSocket (/tv-api/youtube/ws) that also carries the remote keys, so a key and
  // the picture it causes share a connection that is already open. Three ways to show the page, tried in this order, chosen by the TV:
  //   'webcodecs'  the TV's own decoder through VideoDecoder, drawn on a canvas: a frame is on screen the moment it is decoded;
  //   'mse'        the same stream as fragmented MP4 in a <video> through Media Source, kept at its newest frame;
  //   'jpeg'       the pictures (pull, or the push stream): what the TV did before, always available.
  // A way that shows no picture in time, loses its decoder twice or stays slow (key to picture above KEY_SLOW_MS in two reports running) is
  // left at once, for this visit; after two such visits (or one slow one) this TV remembers it (localStorage) and starts with the next way.
  // VIDEO_VERSION: raise it when this code changes, the memory is then forgotten and every way is tried again.
  var VIDEO_VERSION = 1, PATHS = ['webcodecs', 'mse', 'jpeg'], MEMORY_KEY = 'kevflix-youtube-path';
  var VIDEO_FIRST_MS = 4000, VIDEO_SILENT_MS = 15000, HEARTBEAT_MS = 5000, KEY_SLOW_MS = 400, DECODE_QUEUE_MAX = 6, APPEND_QUEUE_MAX = 60;
  var path = 'jpeg', videoOffered = false, videoForced = false, sessionFailed = {}, retried = {}, sock = null, sockOpen = false, epoch = -1, needKey = true;
  var decoder = null, mediaSource = null, sourceBuffer = null, appendQueue = [], pending = [], videoFps = 30, shownOnSocket = 0;
  var videoFirstTimer = 0, heartbeatTimer = 0, frameTimer = 0, frameWatch = 0, lastMessageAt = 0, resets = 0, decodeErrors = 0, slowWindows = 0, queueMax = 0;

  function loadMemory() {
    var memory = null;
    try { memory = JSON.parse(localStorage.getItem(MEMORY_KEY) || 'null'); } catch (ignore) {}
    return memory && memory.v === VIDEO_VERSION && memory.bad && memory.ms ? memory : { v: VIDEO_VERSION, bad: {}, ms: {} };
  }
  function saveMemory(memory) { try { localStorage.setItem(MEMORY_KEY, JSON.stringify(memory)); } catch (ignore) {} }
  function capable(name) {
    if (name === 'jpeg') return true;
    if (!videoOffered || typeof WebSocket !== 'function') return false;
    try {
      if (name === 'webcodecs') return typeof VideoDecoder === 'function' && typeof EncodedVideoChunk === 'function';
      return typeof MediaSource === 'function' && MediaSource.isTypeSupported('video/mp4; codecs="avc1.42c028"');
    } catch (ignore) { return false; }
  }
  function choosePath() {
    var memory = loadMemory();
    for (var i = 0; i < PATHS.length; i++) {
      var name = PATHS[i], mark = memory.bad[name];
      if (capable(name) && !sessionFailed[name] && (videoForced || !(mark && mark.n >= 2))) return name;
    }
    return 'jpeg';
  }
  function startPath(mine, why) {
    if (videoOffered) {                                          // one line in the PC's log: which way this TV uses now, and why it left the last one
      call('diagnostic', { stage: 'youtube-path', youtube: { path: path, why: String(why || '').slice(0, 80),
        webcodecs: capable('webcodecs') ? 1 : 0, mse: capable('mse') ? 1 : 0, v: VIDEO_VERSION } }, 5000).catch(function () {});
    }
    if (path === 'jpeg') { showStream(mine); return; }
    openSocket(mine);
  }
  // Leave the way in use and start the next one. strike 1 = count it against this way on this TV; 2 = remember it at once (measured slow).
  function failPath(why, strike) {
    if (path === 'jpeg' || !active) return;
    var memory = loadMemory(), mark = memory.bad[path] || { n: 0 };
    sessionFailed[path] = why;
    if (strike) { memory.bad[path] = { n: strike === 2 ? 2 : mark.n + 1, why: String(why).slice(0, 60) }; saveMemory(memory); }
    stopVideo();
    path = choosePath();
    startPath(serial, why);
  }
  function wsSend(value) { try { sock.send(JSON.stringify(value)); return true; } catch (ignore) { return false; } }
  function openSocket(mine) {
    var socket, place = window.location;
    epoch = -1; needKey = true; pending = []; shownOnSocket = 0; lastMessageAt = Date.now();
    try {
      socket = new WebSocket((place.protocol === 'https:' ? 'wss://' : 'ws://') + place.host + '/tv-api/youtube/ws?ticket=' +
        encodeURIComponent(ticket) + '&format=' + (path === 'webcodecs' ? 'avc' : 'fmp4'));
      socket.binaryType = 'arraybuffer';
    } catch (error) { failPath('socket'); return; }
    sock = socket;
    socket.onopen = function () { if (sock === socket) { sockOpen = true; pump(); } };
    socket.onmessage = function (event) {
      if (sock !== socket || mine !== serial) return;
      lastMessageAt = Date.now();
      try { if (typeof event.data === 'string') onText(mine, event.data); else onBinary(mine, event.data); }
      catch (error) { decoderFailed(mine, (error && error.name) || 'error'); }
    };
    socket.onclose = socket.onerror = function () {
      if (sock !== socket) return;
      sock = null; sockOpen = false;
      if (mine !== serial || !active) return;
      if (state === 'live' && !retried[path]) { retried[path] = 1; closeDecoder(); closeMedia(); openSocket(mine); return; }   // once per visit: the PC may have restarted
      failPath('socket');
    };
    clearTimeout(videoFirstTimer);
    videoFirstTimer = setTimeout(function () { if (mine === serial && sock === socket && !shownOnSocket) failPath('no picture', 1); }, VIDEO_FIRST_MS);
    clearInterval(heartbeatTimer);
    heartbeatTimer = setInterval(function () {
      if (sock !== socket) return;
      if (Date.now() - lastMessageAt > VIDEO_SILENT_MS) { failPath('silent'); return; }
      if (sockOpen) wsSend({ t: 'hb', n: Date.now() % 1000000 });
    }, HEARTBEAT_MS);
  }
  function onText(mine, text) {
    var message = JSON.parse(text);
    if (message.t === 'config') { epoch = message.epoch; configure(mine, message); }
    else if (message.t === 'play') considerPlay(message.play);
    else if (message.t === 'error' && !message.key) failPath('pc: ' + message.error);
  }
  function bytesOf(text) {
    var raw = atob(text), out = new Uint8Array(raw.length);
    for (var i = 0; i < raw.length; i++) out[i] = raw.charCodeAt(i);
    return out;
  }
  // A new stream begins (the first one, or after a reset): set up the decoder for it.
  function configure(mine, config) {
    needKey = true; pending = []; videoFps = config.fps || 30;
    closeDecoder(); closeMedia();
    $('youtube-screen').hidden = true;
    if (path === 'webcodecs') {
      var canvas = $('youtube-canvas'), context;
      canvas.width = config.width; canvas.height = config.height;
      context = canvas.getContext('2d');
      decoder = new VideoDecoder({
        output: function (frame) {
          try { context.drawImage(frame, 0, 0, canvas.width, canvas.height); } finally { frame.close(); }
          pictureShown(mine, pending.shift());
        },
        error: function (error) { decoderFailed(mine, (error && error.name) || 'decoder'); }
      });
      decoder.configure({ codec: config.codec, description: bytesOf(config.description), codedWidth: config.width, codedHeight: config.height,
        hardwareAcceleration: 'prefer-hardware', optimizeForLatency: true });
      canvas.hidden = false;
      return;
    }
    var video = $('youtube-video'), source = new MediaSource();
    mediaSource = source; appendQueue = [];
    video.muted = true; video.hidden = false;
    video.onerror = function () { decoderFailed(mine, 'video element'); };
    source.addEventListener('sourceopen', function () {
      if (mediaSource !== source) return;
      try {
        sourceBuffer = source.addSourceBuffer('video/mp4; codecs="' + config.codec + '"');
        sourceBuffer.addEventListener('updateend', function () { if (mediaSource === source) drain(mine); });
        sourceBuffer.addEventListener('error', function () { decoderFailed(mine, 'source buffer'); });
        drain(mine);
      } catch (error) { decoderFailed(mine, (error && error.name) || 'media source'); }
    });
    video.src = URL.createObjectURL(source);
    watchFrames(mine, video);
  }
  // One binary message: 8 bytes (kind 1 = MP4 start, 2 = frame; flags bit 0 = key frame; stream epoch; frame number), then the data.
  function onBinary(mine, buffer) {
    if (buffer.byteLength < 8 || epoch < 0) return;
    var view = new DataView(buffer), kind = view.getUint8(0), key = view.getUint8(1) & 1, number = view.getUint32(4);
    if (view.getUint16(2) !== (epoch & 0xFFFF)) return;          // left over from a stream that was reset
    var data = new Uint8Array(buffer, 8), now = Date.now();
    if (kind === 2) {
      if (needKey && !key) return;
      needKey = false;
    }
    if (path === 'webcodecs') {
      if (kind !== 2 || !decoder || decoder.state !== 'configured') return;
      if (decoder.decodeQueueSize > queueMax) queueMax = decoder.decodeQueueSize;
      if (decoder.decodeQueueSize > DECODE_QUEUE_MAX) { requestReset('decode queue'); return; }   // the TV decodes slower than frames arrive: start again at the newest picture
      pending.push({ n: number, at: now, bytes: data.byteLength });
      decoder.decode(new EncodedVideoChunk({ type: key ? 'key' : 'delta', timestamp: Math.round(number * 1000000 / videoFps), data: data }));
      return;
    }
    if (kind === 2) pending.push({ n: number, at: now, bytes: data.byteLength });
    appendQueue.push(data);
    if (appendQueue.length > queueMax) queueMax = appendQueue.length;
    if (appendQueue.length > APPEND_QUEUE_MAX) { requestReset('append queue'); return; }
    drain(mine);
  }
  // Media Source: append everything that has arrived in one go, keep the playhead at the newest frame, forget what was shown.
  function drain(mine) {
    var video = $('youtube-video');
    if (mine !== serial || !sourceBuffer || sourceBuffer.updating || !mediaSource || mediaSource.readyState !== 'open') return;
    try {
      if (appendQueue.length) {
        var total = 0, offset = 0, i;
        for (i = 0; i < appendQueue.length; i++) total += appendQueue[i].byteLength;
        var joined = new Uint8Array(total);
        for (i = 0; i < appendQueue.length; i++) { joined.set(appendQueue[i], offset); offset += appendQueue[i].byteLength; }
        appendQueue = [];
        sourceBuffer.appendBuffer(joined);
        return;
      }
      var ranges = video.buffered;
      if (!ranges.length) return;
      var start = ranges.start(0), end = ranges.end(ranges.length - 1);
      if (video.paused) { var playing = video.play(); if (playing && playing.catch) playing.catch(function () {}); }
      if (end - video.currentTime > 0.25 || video.currentTime < start) video.currentTime = Math.max(start, end - 0.04);
      if (end - start > 12) sourceBuffer.remove(start, end - 6);
    } catch (error) { decoderFailed(mine, (error && error.name) || 'append'); }
  }
  // Media Source: which frame is on screen (for the first-picture check and the key-to-picture time).
  function watchFrames(mine, video) {
    var watch = ++frameWatch;
    function upTo(seconds) {
      var last = null;
      while (pending.length && pending[0].n / videoFps <= seconds + 0.001) last = pending.shift();
      if (last) pictureShown(mine, last);
    }
    clearInterval(frameTimer);
    if (typeof video.requestVideoFrameCallback === 'function') {
      var next = function (now, shown) { if (watch !== frameWatch || mine !== serial) return; upTo(shown.mediaTime); video.requestVideoFrameCallback(next); };
      video.requestVideoFrameCallback(next);
    } else {
      frameTimer = setInterval(function () { if (watch === frameWatch && mine === serial) upTo(video.currentTime); }, 16);
    }
  }
  function pictureShown(mine, item) {
    if (mine !== serial || !item) return;
    shownOnSocket += 1;
    if (shownOnSocket === 1) {                                   // this way works: an earlier single strike against it is forgotten
      clearTimeout(videoFirstTimer);
      var memory = loadMemory(), mark = memory.bad[path];
      if (mark && mark.n < 2) { delete memory.bad[path]; saveMemory(memory); }
    }
    firstPicture(mine);
    notePicture(item.bytes, item.at, item.at, item.at, Date.now());
  }
  // The decoder is behind or broken: ask the PC for a new stream (it begins with a key frame of the page as it is now).
  function requestReset(why) {
    resets += 1;
    if (resets > 3) { failPath('resets: ' + why, 1); return; }
    epoch = -2; needKey = true; pending = []; appendQueue = [];   // everything still on its way belongs to the old stream
    if (!sockOpen || !wsSend({ t: 'reset', why: String(why).slice(0, 40) })) failPath('socket');
  }
  function decoderFailed(mine, what) {
    if (mine !== serial || path === 'jpeg' || !active) return;
    decodeErrors += 1;
    closeDecoder(); closeMedia();
    if (decodeErrors >= 2) failPath('decoder: ' + what, 1); else requestReset(what);
  }
  function closeDecoder() {
    var old = decoder;
    decoder = null;
    if (old) { try { if (old.state !== 'closed') old.close(); } catch (ignore) {} }
  }
  function closeMedia() {
    var video = $('youtube-video');
    frameWatch += 1; clearInterval(frameTimer);
    if (!mediaSource) return;
    mediaSource = null; sourceBuffer = null; appendQueue = [];
    video.onerror = null;
    try { video.pause(); } catch (ignore) {}
    if (video.src) { try { URL.revokeObjectURL(video.src); } catch (ignore) {} }
    video.removeAttribute('src');
    try { video.load(); } catch (ignore) {}
  }
  function stopVideo() {
    var socket = sock;
    clearTimeout(videoFirstTimer); clearInterval(heartbeatTimer);
    sock = null; sockOpen = false; epoch = -1; pending = [];
    if (socket) { try { socket.close(); } catch (ignore) {} }
    closeDecoder(); closeMedia();
    $('youtube-canvas').hidden = true; $('youtube-video').hidden = true; $('youtube-screen').hidden = false;
  }
  // Every 10 s report (reportStats): the key-to-picture median of this way. Slow twice running: leave it and remember.
  function noteWindow(median, count) {
    if (count < 3) return;
    var memory = loadMemory();
    memory.ms[path] = median;
    if (path !== 'jpeg') {
      slowWindows = median > KEY_SLOW_MS ? slowWindows + 1 : 0;
      saveMemory(memory);
      if (slowWindows >= 2) { slowWindows = 0; failPath('slow ' + median + ' ms', 2); }
      return;
    }
    PATHS.forEach(function (name) {                              // a way left for being slow that was still faster than these pictures is tried again next visit
      var mark = memory.bad[name], was = mark && /^slow (\d+)/.exec(mark.why || '');
      if (was && Number(was[1]) < median) delete memory.bad[name];
    });
    saveMemory(memory);
  }

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
    if (!isRepeat && action !== 'up') keyAt = Date.now();         // timing report: key press -> next picture on screen
    queue.push({ key: name, action: action });
    pump();
  }
  function pump() {
    // F20: while the video socket is open the keys travel on it, in order and without a request each; the PC answers a picked video with 'play'
    while (sockOpen && sock && !sending && queue.length) {
      if (!wsSend(Object.assign({ t: 'key' }, queue[0]))) break;
      queue.shift(); failures = 0;
    }
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
      videoOffered = result.video === true; videoForced = result.video_forced === true;
      path = choosePath();                                       // F20: video through the TV's decoder when it can, else the pictures
      startPath(mine, '');
    }).catch(function (error) {
      if (mine !== serial || !active) return;
      setState('error', ((error && error.message) || 'YouTube could not start') + '. Press OK to retry, or Back to return.');
    });
  }
  function firstPicture(mine) {
    if (mine !== serial) return;
    clearTimeout(firstFrameTimer);
    if (state !== 'live') {
      startStats();
      setState('live', notice || 'Press Back twice, or hold Back, to return to KevFLIX');   // how to leave: a single Back belongs to YouTube
      notice = '';
      clearTimeout(hintTimer);
      hintTimer = setTimeout(function () { if (active && state === 'live') setState('live', ''); }, HINT_MS);
    }
  }
  function stopPicture() {
    stopVideo();
    if (pullAbort) { try { pullAbort.abort(); } catch (ignore) {} pullAbort = null; }
    if (pullUrl) { try { URL.revokeObjectURL(pullUrl); } catch (ignore) {} pullUrl = ''; }
    lastCounter = 0; lastMs = 0; pullFailures = 0;
    stopStats();
    var image = $('youtube-screen');
    image.onload = image.onerror = null;
    image.removeAttribute('src');                                  // also closes the push stream; the PC stops it when no client is left
  }
  function pullPicture(mine) {
    if (mine !== serial || !active) return;
    var controller = typeof AbortController === 'function' ? new AbortController() : null, headersAt = 0, requestAt = Date.now();
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
      if (blob && blob.size) showPulled(mine, blob, requestAt, headersAt, Date.now());
      pullPicture(mine);                                           // the next picture is asked for as soon as this one has arrived
    }).catch(function () {
      if (mine !== serial || !active) return;
      pullFailures += 1;
      if (pullFailures >= 4) { clearTimeout(firstFrameTimer); setState('error', 'The YouTube picture stopped. Press OK to retry, or Back to return.'); return; }
      setTimeout(function () { pullPicture(mine); }, 250 * pullFailures);
    });
  }
  function showPulled(mine, blob, requestAt, headersAt, blobAt) {
    var image = $('youtube-screen'), url = URL.createObjectURL(blob), previous = pullUrl;
    pullUrl = url;
    image.onload = function () {
      if (mine !== serial) return;
      var shownAt = Date.now();
      lastMs = shownAt - headersAt;                                // download + decode of this picture, reported with the next request
      firstPicture(mine);
      notePicture(blob.size, requestAt, headersAt, blobAt, shownAt);
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
    pausePosters(true);
    setState('connecting', 'Returning to YouTube…');
    var mine = ++serial;
    // the PC unblocks the page's media and sends Back to its watch screen FIRST; only then is the picture stream reopened
    call('youtube/resume', {}, 10000).catch(function () {}).then(function () { if (mine === serial && active) { connect(); watchHost(); } });
  }
  function open() {
    if (active || handoff) return;
    active = true; failures = 0; queue = []; sending = false; notice = '';
    sessionFailed = {}; retried = {}; resets = 0; decodeErrors = 0; slowWindows = 0;
    returnFocus = document.activeElement;
    releaseAll(false);
    Object.keys(downNow).forEach(function (code) { held[code] = 'stale'; });   // the key that opened the view is not a YouTube key
    $('youtube-view').hidden = false;
    pausePosters(true);
    reportCapabilities();
    connect();
    watchHost();
  }
  // the library's poster loading steps aside while YouTube is on screen (see player.js pausePosters)
  function pausePosters(pause) {
    var player = window.KevFlixPlayer;
    try { if (player && typeof (pause ? player.pausePosters : player.resumePosters) === 'function') (pause ? player.pausePosters : player.resumePosters)(); } catch (ignore) {}
  }
  function exit() {
    if (!active) return;
    var wasBack = backHeld;
    releaseAll(true);
    active = false; serial += 1; state = 'idle'; lastBackAt = 0;
    clearTimeout(hintTimer); clearTimeout(firstFrameTimer); clearInterval(statusTimer);
    stopPicture();                                                 // closes the stream; the PC stops it when no client is left
    $('youtube-view').hidden = true;
    pausePosters(false);
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
