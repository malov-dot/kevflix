/* KevFLIX on-screen playback controls ("skins").
   A Plex library video gets a control set that looks and navigates like Plex's own TV player, a YouTube video one that looks and navigates like
   YouTube's own TV player. KevFLIX's extra functions (quality, upscaling, HDR, English dubbing, sound, audio track, subtitles, Watch with Sem) live
   INSIDE those control sets. Every other session (a library folder movie) keeps the original controls of index.html untouched.
   Loaded BEFORE player.js (after youtube.js) so its capture-phase key listener runs first and can stop a key before player.js's old handler sees it.
   player.js hands over its API with window.KevFlixControls.attach(window.KevFlixPlayerUI); every action calls the player's own functions.
   All state is local to the TV: a key press changes the picture in the same frame, the PC is only asked to do the real work (seek, quality, dubbing).
   No inline styles (CSP): the elements get their look from controls.css; the few dynamic values go through element.style (CSSOM). */
(function () {
  'use strict';
  var SVGNS = 'http://www.w3.org/2000/svg';
  var HIDE_MS = 5000, NOTICE_MS = 5000, UPNEXT_SECONDS = 5, REQUEST_MS = 8000, MENU_ROWS = 8;
  var SEEK = { plex: { back: 10, forward: 30 }, youtube: { back: 10, forward: 10 } };
  var SPEEDS = [0.5, 0.75, 1, 1.25, 1.5, 2];
  var LEVELS = [1, 0.75, 0.5, 0.25, 0];
  var QUALITIES = [2160, 1080, 720, 480];
  var WATCH_MODES = [{ value: 'balanced', label: 'Balanced reactions' }, { value: 'quiet', label: 'Quiet reactions' }, { value: 'ask-only', label: 'Only when I ask Sem' }];
  // Samsung remote key codes (and the desktop equivalents) -> actions.
  var KEY_CODES = { 37: 'left', 38: 'up', 39: 'right', 40: 'down', 13: 'ok', 10009: 'back', 27: 'back', 8: 'back',
    415: 'play', 19: 'pause', 10252: 'playpause', 32: 'playpause', 417: 'ff', 412: 'rw' };
  var KEY_NAMES = { ArrowLeft: 37, ArrowUp: 38, ArrowRight: 39, ArrowDown: 40, Enter: 13, Backspace: 8, Escape: 27 };

  // ================================================================================================================================
  // Pure functions (no DOM, no player): key -> action, number/time formatting, mode choice, focus movement, button and menu models.
  // ================================================================================================================================
  function keyCodeOf(event) { return event.keyCode || event.which || KEY_NAMES[event.key] || 0; }
  function keyAction(code) { return KEY_CODES[code] || null; }
  function num(value) { value = Number(value); return isFinite(value) ? value : null; }
  function clock(seconds) {
    seconds = Math.floor(Math.max(0, Number(seconds) || 0));
    var h = Math.floor(seconds / 3600), m = Math.floor(seconds / 60) % 60, s = seconds % 60;
    return (h ? h + ':' + (m < 10 ? '0' : '') : '') + m + ':' + (s < 10 ? '0' : '') + s;
  }
  function clockTime(date) {
    var h = date.getHours(), m = date.getMinutes();
    return ((h % 12) || 12) + ':' + (m < 10 ? '0' : '') + m + ' ' + (h < 12 ? 'AM' : 'PM');
  }
  function endsAt(date, remainingSeconds) {
    return 'Ends at ' + clockTime(new Date(date.getTime() + Math.max(0, Number(remainingSeconds) || 0) * 1000));
  }
  // 'plex' when the status has a plex object (or the TV opened a Plex item), 'youtube' when it has a youtube object, else null (old controls stay).
  function modeOf(status, item) {
    if (!status || typeof status !== 'object') { return null; }
    if (status.youtube && typeof status.youtube === 'object') { return 'youtube'; }
    if (status.plex && typeof status.plex === 'object') { return 'plex'; }
    if (item && item.plex && !item.youtube) { return 'plex'; }
    return null;
  }
  function neighbour(value) { return value && typeof value === 'object' && value.key ? { key: String(value.key), title: String(value.title || '') } : null; }
  function plexInfo(status, item) {
    var p = status && status.plex && typeof status.plex === 'object' ? status.plex : {};
    var type = p.type === 'episode' || (!p.type && p.show) ? 'episode' : 'movie';
    return {
      key: String(p.key || (item && item.id) || ''), type: type,
      title: String(p.title || (status && status.title) || (item && item.title) || ''), show: type === 'episode' ? String(p.show || '') : '',
      season: num(p.season), episode: num(p.episode), year: num(p.year) || num(item && item.year),
      prev: type === 'episode' ? neighbour(p.prev) : null, next: type === 'episode' ? neighbour(p.next) : null
    };
  }
  function plexHeader(info) {
    var code = info.season !== null && info.episode !== null ? 'S' + info.season + ' · E' + info.episode : info.episode !== null ? 'E' + info.episode : '';
    var parts = [];
    if (info.type === 'episode' && code) { parts.push(code); }
    if (info.year) { parts.push(String(info.year)); }
    return { show: info.show, title: info.title, line: parts.join(' · ') };
  }
  function focusable(button) { return !button.hidden && button.enabled; }
  // Focus along a row of buttons: skips hidden/disabled ones and stops at the ends. An unusable current id falls back to Play/Pause.
  function moveFocus(list, id, delta) {
    var order = list.filter(focusable).map(function (button) { return button.id; });
    var at = order.indexOf(id);
    if (at < 0) { return order.indexOf('playpause') >= 0 ? 'playpause' : (order[0] || id); }
    return order[Math.max(0, Math.min(order.length - 1, at + delta))];
  }
  // Scrubbing the progress bar: 10 s per press, faster while the key is held.
  function scrubStep(repeats) { return repeats > 20 ? 60 : repeats > 6 ? 30 : 10; }
  function scrubTarget(from, direction, repeats, duration) {
    var limit = Math.max(0, (Number(duration) || 0) - 0.1);
    return Math.max(0, Math.min(limit, (Number(from) || 0) + direction * scrubStep(repeats)));
  }
  function stepRow(rows, index, delta) {
    var i = index + delta;
    while (i >= 0 && i < rows.length) { if (rows[i].enabled) { return i; } i += delta; }
    return index;
  }
  function firstIndex(rows) {
    var i;
    for (i = 0; i < rows.length; i++) { if (rows[i].selected && rows[i].enabled) { return i; } }
    for (i = 0; i < rows.length; i++) { if (rows[i].enabled) { return i; } }
    return 0;
  }
  function windowStart(total, index, size, start) {
    if (total <= size) { return 0; }
    start = start || 0;
    if (index < start) { start = index; }
    if (index >= start + size) { start = index - size + 1; }
    return Math.max(0, Math.min(start, total - size));
  }
  function qualityText(q) { q = Number(q); return q === 2160 ? '4K' : q ? q + 'p' : ''; }
  function levelText(level) { return level === 0 ? 'Off' : Math.round(level * 100) + '%'; }
  function speedText(rate) { return rate === 1 ? 'Normal' : rate + '×'; }
  function nearestLevel(state) {
    if (!state || state.muted) { return 0; }
    var best = LEVELS[0], i;
    for (i = 0; i < LEVELS.length; i++) {
      if (LEVELS[i] > 0 && Math.abs(LEVELS[i] - state.volume) < Math.abs(best - state.volume)) { best = LEVELS[i]; }
    }
    return best;
  }
  function watchText(status) { return !status.watch ? 'Off' : (WATCH_MODES.filter(function (m) { return m.value === status.watch_mode; })[0] || WATCH_MODES[0]).label; }

  // Buttons of the control row. hidden = not drawn at all; enabled false = drawn dimmed and skipped by focus.
  function plexButtons(c) {
    var episode = c.plex.type === 'episode';
    return [
      { id: 'prev', icon: 'prev', label: 'Previous episode', hidden: !episode, enabled: !!c.plex.prev },
      { id: 'rew', icon: 'rew10', label: 'Back 10 seconds', hidden: false, enabled: true },
      { id: 'playpause', icon: c.playing ? 'pause' : 'play', label: c.playing ? 'Pause' : 'Play', hidden: false, enabled: true, primary: true },
      { id: 'fwd', icon: 'fwd30', label: 'Forward 30 seconds', hidden: false, enabled: true },
      { id: 'next', icon: 'next', label: 'Next episode', hidden: !episode, enabled: !!c.plex.next },
      { id: 'audio', icon: 'audio', label: 'Audio', hidden: false, enabled: c.audioTracks.length > 1, group: 'right' },
      { id: 'subs', icon: 'subtitles', label: 'Subtitles', hidden: false, enabled: true, group: 'right' },
      { id: 'settings', icon: 'gear', label: 'Settings', hidden: false, enabled: true, group: 'right' }
    ];
  }
  function youtubeButtons(c) {
    return [
      { id: 'playpause', icon: c.playing ? 'pause' : 'play', label: c.playing ? 'Pause' : 'Play', hidden: false, enabled: true, primary: true },
      { id: 'prev', icon: 'prev', label: 'Previous video', hidden: c.live, enabled: c.history > 0 },
      { id: 'next', icon: 'next', label: 'Next video', hidden: c.live, enabled: !!c.upNext },
      { id: 'like', icon: c.rating === 'like' ? 'like-on' : 'like', label: 'Like', hidden: !c.ratingAvailable, enabled: true, pressed: c.rating === 'like' },
      { id: 'dislike', icon: c.rating === 'dislike' ? 'dislike-on' : 'dislike', label: 'Dislike', hidden: !c.ratingAvailable, enabled: true, pressed: c.rating === 'dislike' },
      { id: 'captions', icon: 'subtitles', label: 'Captions', hidden: c.captions.length === 0, enabled: true },
      { id: 'settings', icon: 'gear', label: 'Settings', hidden: false, enabled: true }
    ];
  }
  function buttonsFor(mode, c) { return mode === 'plex' ? plexButtons(c) : youtubeButtons(c); }

  // Menus. A row = {id, label, value (right-hand text), selected (check mark), enabled, note, act}; act says what OK does:
  //   {kind:'open', pick}  opens a pick list   {kind:'set', what, value}  applies a choice   {kind:'toggle', what}  flips On/Off
  function openRow(pick, label, value) { return { id: pick, label: label, value: value, selected: false, enabled: true, note: '', act: { kind: 'open', pick: pick } }; }
  function rootRows(mode, c) {
    var s = c.status, rows = [];
    if (mode === 'youtube' && !c.live) { rows.push(openRow('speed', 'Playback speed', speedText(c.speed))); }
    rows.push(openRow('quality', 'Quality', qualityText(s.quality)));
    rows.push(openRow('upscale', 'Upscaling', s.upscale ? 'On' : 'Off'));
    rows.push(openRow('hdr', 'HDR', s.hdr_mode === 'sdr' ? 'Convert to SDR' : 'Automatic'));
    if (!c.live) {
      var unsupported = !s.dub && s.dub_unsupported ? String(s.dub_unsupported) : '';
      rows.push({ id: 'dub', label: 'English dubbing', value: s.dub ? 'On' : 'Off', selected: false, enabled: !unsupported, note: unsupported, act: { kind: 'toggle', what: 'dub' } });
    }
    if (c.hasSound) { rows.push(openRow('sound', 'Sound', levelText(c.level))); }
    if (mode === 'plex') { rows.push(openRow('surround', 'Surround', s.surround === 'stereo' ? 'Stereo' : '5.1 when available')); }
    rows.push(openRow('watch', 'Watch with Sem', watchText(s)));
    if (mode === 'plex') {
      var row = { id: 'progress', label: 'Save progress to Plex', value: s.sync_progress ? 'On' : 'Off', selected: false, enabled: true, note: '', act: { kind: 'toggle', what: 'progress_sync' } };
      row.keepOpen = true;
      rows.push(row);
    }
    return rows;
  }
  function pickList(kind, c) {
    var s = c.status || {}, rows = [], title = '';
    function add(value, label, selected, enabled) {
      rows.push({ id: kind + ':' + value, label: label, value: '', selected: !!selected, enabled: enabled !== false, note: '', act: { kind: 'set', what: kind, value: value } });
    }
    function none(text) { rows.push({ id: kind + ':none', label: text, value: '', selected: false, enabled: false, note: '', act: null }); }
    if (kind === 'quality') {
      title = 'Quality';
      QUALITIES.forEach(function (q) { add(q, q === 2160 ? '4K · 2160p' : q + 'p', Number(s.quality) === q); });
    } else if (kind === 'upscale') {
      title = 'Upscaling';
      add('off', 'Off', !s.upscale); add('on', 'On · Lanczos on the PC', !!s.upscale);
    } else if (kind === 'hdr') {
      title = 'HDR';
      add('auto', 'Automatic', s.hdr_mode !== 'sdr'); add('sdr', 'Convert to SDR', s.hdr_mode === 'sdr');
    } else if (kind === 'sound') {
      title = 'Sound';
      LEVELS.forEach(function (l) { add(l, l === 0 ? 'Off' : levelText(l), c.level === l); });
    } else if (kind === 'surround') {
      title = 'Surround';
      add('auto', '5.1 surround when the film has it', s.surround !== 'stereo'); add('stereo', 'Stereo', s.surround === 'stereo');
    } else if (kind === 'watch') {
      title = 'Watch with Sem';
      add('off', 'Off', !s.watch);
      WATCH_MODES.forEach(function (m) { add(m.value, m.label, !!s.watch && (s.watch_mode || 'balanced') === m.value); });
    } else if (kind === 'speed') {
      title = 'Playback speed';
      SPEEDS.forEach(function (r) { add(r, speedText(r), c.speed === r); });
    } else if (kind === 'audio') {
      title = 'Audio';
      c.audioTracks.forEach(function (t) { add(t.index, [t.language, t.title].filter(Boolean).join(' · ') || 'Track ' + t.index, Number(s.audio_index) === Number(t.index)); });
      if (!rows.length) { none('No other audio tracks'); }
    } else if (kind === 'subs') {
      title = 'Subtitles';
      add('off', 'Off', String(s.subtitles) === 'off'); add('auto', 'External subtitle file', String(s.subtitles) === 'auto');
      if (s.dub) { add('dub', 'English dubbing text', String(s.subtitles) === 'dub'); }
      (Array.isArray(s.subtitle_tracks) ? s.subtitle_tracks : []).forEach(function (t) {
        add(String(t.index), [t.language, t.title].filter(Boolean).join(' · ') || 'Track ' + t.index, String(s.subtitles) === String(t.index));
      });
    } else if (kind === 'captions') {
      title = 'Captions';
      add('off', 'Off', String(s.subtitles) === 'off');
      c.captions.forEach(function (t) { add(String(t.index), t.title, String(s.subtitles) === String(t.index)); });
    }
    return { title: title, rows: rows };
  }
  function levelBuild(level, mode, c) {
    return level.pick === 'root' ? { title: 'Settings', rows: rootRows(mode, c) } : pickList(level.pick, c);
  }

  // ================================================================================================================================
  // State
  // ================================================================================================================================
  var S = { mode: null, sessionId: '', status: null, visible: false, peek: false, zone: 'buttons', focus: 'playpause', scrub: null, scrubRepeats: 0,
    menu: null, notice: '', msg: '', upnext: null, history: [], switching: false, speed: 1 };
  var Y = { infoFor: '', infoState: 'idle', info: null, rating: null, signedIn: null, busy: false, rateMissing: false };
  var ui = null, video = null, player = null, root = null, R = null, bound = false;
  var hideTimer = 0, noticeTimer = 0, upTimer = 0;

  function token() { try { return localStorage.getItem('sem-tv-token') || ''; } catch (ignore) { return ''; } }
  function isLive() { return !!(S.status && (S.status.live || (S.status.youtube && S.status.youtube.live))); }
  function videoId(status) { return status && status.youtube && status.youtube.id ? String(status.youtube.id) : ''; }
  function audioApi() { return window.KevFlixAudio && typeof window.KevFlixAudio.get === 'function' && typeof window.KevFlixAudio.set === 'function' ? window.KevFlixAudio : null; }
  function nextOf(status) {
    var y = status && status.youtube, up = y && y.up_next;
    if (!(up && up.id) && Y.info && Y.infoFor === videoId(status)) { up = Y.info.up_next; }
    return up && up.id ? { id: String(up.id), title: String(up.title || '') } : null;
  }
  function channelOf(status) {
    var y = (status && status.youtube) || {};
    return String(y.channel || (Y.info && Y.infoFor === videoId(status) && Y.info.channel) || y.uploader || '');
  }
  // Caption tracks of a YouTube video: the status entries whose index starts with 'yt:' (else the info endpoint's list).
  function captionTracks(status) {
    var list = (Array.isArray(status.subtitle_tracks) ? status.subtitle_tracks : []).filter(function (t) { return String(t.index).indexOf('yt:') === 0; })
      .map(function (t) { return { index: String(t.index), title: String(t.title || t.language || t.index) }; });
    if (!list.length && Y.info && Y.infoFor === videoId(status) && Array.isArray(Y.info.captions)) {
      list = Y.info.captions.filter(function (t) { return t && t.code; }).map(function (t) { return { index: 'yt:' + t.code, title: String(t.name || t.code) + (t.auto ? ' (auto-generated)' : '') }; });
    }
    return list;
  }
  function context() {
    var status = S.status || {}, item = ui && ui.item ? ui.item() : null;
    var sound = audioApi();
    return {
      mode: S.mode, status: status, playing: ui ? !!ui.isPlaying() : false, position: ui ? Number(ui.position()) || 0 : 0,
      duration: Number(status.duration) || 0, live: isLive(),
      plex: S.mode === 'plex' ? plexInfo(status, item) : null,
      hasSound: !!sound, level: sound ? nearestLevel(sound.get()) : 1, speed: S.speed,
      audioTracks: Array.isArray(status.audio_tracks) ? status.audio_tracks : [],
      history: S.history.length, upNext: S.mode === 'youtube' ? nextOf(status) : null,
      captions: S.mode === 'youtube' ? captionTracks(status) : [],
      rating: Y.rating, ratingAvailable: S.mode === 'youtube' && Y.infoState === 'ok' && !Y.rateMissing
    };
  }

  // ================================================================================================================================
  // Network (YouTube extras). Every endpoint may be missing or fail: a failure hides or disables that button and never touches playback.
  // ================================================================================================================================
  function request(method, path, body) {
    var controller = typeof AbortController === 'function' ? new AbortController() : null;
    var timer = setTimeout(function () { if (controller) { controller.abort(); } }, REQUEST_MS);
    var headers = { Authorization: 'Bearer ' + token() };
    if (body !== undefined) { headers['Content-Type'] = 'application/json'; }
    return fetch('/tv-api/' + path, { method: method, cache: 'no-store', headers: headers, signal: controller ? controller.signal : undefined,
      body: body === undefined ? undefined : JSON.stringify(body) }).then(function (response) {
      return Promise.resolve(response.json()).catch(function () { return null; }).then(function (value) {
        if (!response.ok) {
          var error = new Error((value && value.error) || 'The PC could not complete this request.');
          error.status = response.status;
          throw error;
        }
        return value;
      });
    }).then(function (value) { clearTimeout(timer); return value; }, function (error) { clearTimeout(timer); throw error; });
  }
  function ratingOf(value) { return value === 'like' || value === 'dislike' ? value : null; }
  function loadInfo(id) {
    Y.infoFor = id; Y.info = null; Y.infoState = 'loading'; Y.rating = null; Y.signedIn = null; Y.rateMissing = false; Y.busy = false;
    request('GET', 'youtube/info?video=' + encodeURIComponent(id)).then(function (value) {
      if (Y.infoFor !== id) { return; }
      Y.info = value && typeof value === 'object' ? value : {};
      Y.infoState = 'ok'; Y.rating = ratingOf(Y.info.rating); Y.signedIn = typeof Y.info.signed_in === 'boolean' ? Y.info.signed_in : null;
      render();
    }, function () {
      if (Y.infoFor !== id) { return; }
      Y.info = null; Y.infoState = 'missing';              // 404 or any failure: no rating buttons, no PC-side channel/up-next/captions; playback is untouched
      render();
    });
  }
  function rate(which) {
    if (S.mode !== 'youtube' || Y.busy) { return; }
    var id = videoId(S.status), target = Y.rating === which ? 'none' : which;
    if (Y.signedIn === false) { notice('Sign in to YouTube on your PC to rate videos.'); return; }
    Y.busy = true;
    request('POST', 'youtube/rate', { video: id, rating: target }).then(function (reply) {
      Y.busy = false;
      if (videoId(S.status) !== id) { return; }
      if (reply && reply.signed_in === false) { Y.signedIn = false; notice('Sign in to YouTube on your PC to rate videos.'); render(); return; }
      Y.rating = reply && Object.prototype.hasOwnProperty.call(reply, 'rating') ? ratingOf(reply.rating) : (target === 'none' ? null : target);
      render();
    }, function (error) {
      Y.busy = false;
      if (error && error.status === 404) { Y.rateMissing = true; notice('Rating is not available yet.'); }
      else if (error && (error.status === 401 || error.status === 403 || /sign.?in|signed/i.test(error.message || ''))) { Y.signedIn = false; notice('Sign in to YouTube on your PC to rate videos.'); }
      else { notice('Could not rate this video' + (error && error.message ? ': ' + error.message : '.')); }
      render();
    });
  }

  // ================================================================================================================================
  // Actions
  // ================================================================================================================================
  function seekBy(delta) { if (ui && !isLive()) { ui.seekBy(delta); } }
  function setSpeed(rate) {
    S.speed = rate;
    applySpeed();
  }
  function applySpeed() {
    if (!video) { return; }
    try { video.defaultPlaybackRate = S.speed; video.playbackRate = S.speed; } catch (ignore) {}
  }
  function resetSpeed() {
    if (!video) { return; }
    try { if (video.playbackRate !== 1 || video.defaultPlaybackRate !== 1) { video.defaultPlaybackRate = 1; video.playbackRate = 1; } } catch (ignore) {}
  }
  function returnToYoutube(reason) {
    var y = window.KevFlixYoutube;
    try { if (y && typeof y.resume === 'function') { y.resume(reason || ''); } } catch (ignore) {}
  }
  // Another YouTube video inside the player (Next, Previous, autoplay): KevFLIX's own openYoutube keeps the player screen up. Any failure returns to
  // the YouTube view with the reason, exactly like the first hand-off does.
  function switchVideo(id) {
    var opener = window.KevFlixPlayer;
    S.switching = true; stopUpNext(); hideBar();
    var opened;
    try {
      if (!opener || typeof opener.openYoutube !== 'function') { throw new Error('The KevFLIX player is not ready.'); }
      opened = opener.openYoutube(id);
    } catch (error) { opened = Promise.reject(error); }
    Promise.resolve(opened).catch(function (error) {
      S.switching = false; S.history = [];
      var reason = error && error.message === 'Cancelled' ? '' : ((error && error.message) || 'This YouTube video could not be played.');
      Promise.resolve(ui && ui.leave ? ui.leave() : null).catch(function () {}).then(function () { returnToYoutube(reason); });
    });
  }
  function goNext() {
    var n = nextOf(S.status);
    if (!n || isLive()) { return; }
    S.history.push(videoId(S.status));
    switchVideo(n.id);
  }
  function goPrevious() {
    if (!S.history.length || isLive()) { return; }
    switchVideo(S.history.pop());
  }
  function openEpisode(target) {
    var opener = window.KevFlixPlayer;
    if (!target || !opener || typeof opener.openPlex !== 'function') { return; }
    hideBar();
    var opened;
    try { opened = opener.openPlex(target.key, target.title); } catch (error) { opened = Promise.reject(error); }
    Promise.resolve(opened).catch(function (error) { notice((error && error.message) || 'This episode could not be opened.'); });
  }
  function startUpNext(n) {
    stopUpNext(); clearTimeout(hideTimer); hideTimer = 0;
    S.visible = false; S.peek = false; S.menu = null; S.scrub = null;
    S.upnext = { id: n.id, title: n.title, left: UPNEXT_SECONDS };
    upTimer = setInterval(function () {
      if (!S.upnext) { stopUpNext(); return; }
      S.upnext.left -= 1;
      if (S.upnext.left <= 0) { playUpNext(); } else { render(); }
    }, 1000);
    render();
  }
  function stopUpNext() { if (upTimer) { clearInterval(upTimer); upTimer = 0; } S.upnext = null; }
  function playUpNext() { stopUpNext(); goNext(); }
  function setWatch(value, s) {
    if (value === 'off') { if (s.watch) { ui.change('watch', { enabled: false }); } return; }
    var first = s.watch ? Promise.resolve() : Promise.resolve(ui.change('watch', { enabled: true })).then(function () {
      ui.message('Ask Sem to watch with you in an existing voice call.');
    });
    first.then(function () { if (value !== (S.status && S.status.watch_mode)) { return ui.change('watch_mode', { watch_mode: value }); } return null; }).catch(function () {});
  }
  // Applies one menu choice with the player's own functions (the same fields the old playback settings build).
  function applyAct(act) {
    var s = S.status || {}, v = act.value;
    if (act.kind === 'toggle') {
      if (act.what === 'dub') { ui.dub(); }                         // the old Dubbing button: refuses live streams and still-downloading videos with a message
      else if (act.what === 'progress_sync') { ui.change('progress_sync', { enabled: !s.sync_progress }); }
      return;
    }
    switch (act.what) {
      case 'quality': if (Number(v) !== Number(s.quality)) { ui.change('quality', { quality: Number(v), upscale: !!s.upscale, hdr_mode: s.hdr_mode }); } break;
      case 'upscale': if ((v === 'on') !== !!s.upscale) { ui.change('quality', { quality: s.quality, upscale: v === 'on', hdr_mode: s.hdr_mode }); } break;
      case 'hdr': if (v !== s.hdr_mode) { ui.change('quality', { quality: s.quality, upscale: !!s.upscale, hdr_mode: v }); } break;
      case 'surround': if (v !== (s.surround || 'auto')) { ui.change('quality', { quality: s.quality, upscale: !!s.upscale, hdr_mode: s.hdr_mode, surround: v }); } break;
      case 'sound': if (audioApi()) { audioApi().set(Number(v)); } break;
      case 'speed': setSpeed(Number(v)); break;
      case 'watch': setWatch(v, s); break;
      case 'audio': if (Number(v) !== Number(s.audio_index)) { ui.change('audio', { audio_index: Number(v) }); } break;
      case 'subs': case 'captions': if (String(v) !== String(s.subtitles)) { ui.change('subtitles', { subtitles: String(v) }); } break;
      default: break;
    }
  }
  function activate(id) {
    var c = context();
    switch (id) {
      case 'playpause': ui.playPause(); break;
      case 'rew': seekBy(-SEEK.plex.back); break;
      case 'fwd': seekBy(SEEK.plex.forward); break;
      case 'prev': if (S.mode === 'plex') { openEpisode(c.plex.prev); } else { goPrevious(); } return;
      case 'next': if (S.mode === 'plex') { openEpisode(c.plex.next); } else { goNext(); } return;
      case 'like': rate('like'); break;
      case 'dislike': rate('dislike'); break;
      case 'audio': openMenu('audio'); return;
      case 'subs': openMenu('subs'); return;
      case 'captions': openMenu('captions'); return;
      case 'settings': openMenu('root'); return;
      default: break;
    }
    arm(); render();
  }

  // ---- menus -------------------------------------------------------------------------------------------------------------------
  function topLevel() { return S.menu && S.menu.levels[S.menu.levels.length - 1]; }
  function newLevel(pick) {
    var level = { pick: pick, index: 0, start: 0 };
    level.index = firstIndex(levelBuild(level, S.mode, context()).rows);
    return level;
  }
  function openMenu(pick) {
    S.menu = { origin: pick, levels: [newLevel(pick)] };
    clearTimeout(hideTimer); hideTimer = 0;
    render();
  }
  function closeMenu() { S.menu = null; arm(); render(); }
  function menuBack() {
    if (!S.menu) { return; }
    if (S.menu.levels.length > 1) { S.menu.levels.pop(); render(); } else { closeMenu(); }
  }
  function activateRow(row) {
    if (!row || !row.enabled || !row.act) { return; }
    if (row.act.kind === 'open') { S.menu.levels.push(newLevel(row.act.pick)); render(); return; }
    applyAct(row.act);
    if (row.keepOpen) { render(); } else { closeMenu(); }
  }
  function handleMenu(action) {
    var level = topLevel(), rows = levelBuild(level, S.mode, context()).rows;
    if (action === 'up') { level.index = stepRow(rows, level.index, -1); }
    else if (action === 'down') { level.index = stepRow(rows, level.index, 1); }
    else if (action === 'ok') { activateRow(rows[level.index]); return; }
    else if (action === 'left' || action === 'back') { menuBack(); return; }
    render();
  }

  // ---- showing, hiding, notices -----------------------------------------------------------------------------------------------
  function arm() {
    clearTimeout(hideTimer); hideTimer = 0;
    if (S.visible && !S.menu && !S.upnext && !(S.status && S.status.error)) { hideTimer = setTimeout(function () { hideTimer = 0; if (!S.menu) { hideBar(); } }, HIDE_MS); }
  }
  function showBar(peek) { S.visible = true; S.peek = !!peek; arm(); render(); }
  function reveal() { S.focus = 'playpause'; S.zone = 'buttons'; S.scrub = null; showBar(false); }
  function hideBar() {
    S.visible = false; S.peek = false; S.scrub = null; S.zone = 'buttons';
    clearTimeout(hideTimer); hideTimer = 0;
    render();
  }
  function notice(text) {
    S.notice = String(text || '');
    clearTimeout(noticeTimer); noticeTimer = 0;
    if (S.notice) {
      noticeTimer = setTimeout(function () { noticeTimer = 0; S.notice = ''; render(); }, NOTICE_MS);
      if (!S.visible) { showBar(true); return; }
    }
    render();
  }

  // ---- keys ----------------------------------------------------------------------------------------------------------------------
  function mediaKey(action, repeat) {
    var back = SEEK[S.mode].back, forward = SEEK[S.mode].forward;
    if (action === 'play') { if (!ui.isPlaying()) { ui.playPause(); } }
    else if (action === 'pause') { if (ui.isPlaying()) { ui.playPause(); } }
    else if (action === 'playpause') { if (!repeat) { ui.playPause(); } }
    else if (action === 'ff') { seekBy(forward); }
    else if (action === 'rw') { seekBy(-back); }
    if (S.visible && !S.peek) { arm(); render(); } else { showBar(true); }
  }
  // Keys while the bar is hidden (or only peeking after a seek): Left/Right keep seeking, Up/Down/OK bring up the full controls.
  function handleHidden(action, repeat) {
    var seek = SEEK[S.mode];
    switch (action) {
      case 'left': if (!isLive()) { seekBy(-seek.back); showBar(true); } break;
      case 'right': if (!isLive()) { seekBy(seek.forward); showBar(true); } break;
      case 'up': case 'down': reveal(); break;
      case 'ok': if (!repeat) { ui.playPause(); reveal(); } break;
      case 'back': if (S.visible) { hideBar(); } else if (!repeat) { ui.leave(); } break;
      default: break;
    }
  }
  function goBar() { if (isLive()) { return; } S.zone = 'bar'; S.scrub = null; S.scrubRepeats = 0; }
  function layoutBarAbove() { return S.mode === 'plex'; }
  function handleBar(action, repeat) {
    var c;
    switch (action) {
      case 'left': case 'right':
        if (isLive()) { break; }
        c = context();
        if (S.scrub === null) { S.scrub = c.position; S.scrubRepeats = 0; }
        S.scrubRepeats = repeat ? S.scrubRepeats + 1 : 0;
        S.scrub = scrubTarget(S.scrub, action === 'left' ? -1 : 1, S.scrubRepeats, c.duration);
        break;
      case 'ok':
        if (repeat) { break; }
        if (S.scrub !== null) { var target = S.scrub; S.scrub = null; ui.seekTo(target); } else { ui.playPause(); }
        break;
      case 'up': if (!layoutBarAbove()) { S.zone = 'buttons'; S.scrub = null; } break;
      case 'down': if (layoutBarAbove()) { S.zone = 'buttons'; S.scrub = null; } break;
      default: break;
    }
  }
  function handleVisible(action, repeat) {
    arm();
    if (action === 'back') {
      if (repeat) { return; }
      if (S.scrub !== null) { S.scrub = null; render(); } else { hideBar(); }
      return;
    }
    if (S.zone === 'bar') { handleBar(action, repeat); render(); return; }
    switch (action) {
      case 'left': S.focus = moveFocus(buttonsFor(S.mode, context()), S.focus, -1); break;
      case 'right': S.focus = moveFocus(buttonsFor(S.mode, context()), S.focus, 1); break;
      case 'up': if (layoutBarAbove()) { goBar(); } break;
      case 'down': if (!layoutBarAbove()) { goBar(); } break;
      case 'ok': if (!repeat) { activate(S.focus); return; } break;
      default: break;
    }
    render();
  }
  function handleUpNext(action, repeat) {
    if (repeat) { return; }
    if (action === 'ok') { playUpNext(); }
    else if (action === 'back') { stopUpNext(); render(); ui.leave(); }
  }
  function handle(action, repeat) {
    if (S.upnext) { handleUpNext(action, repeat); return; }
    if (action === 'play' || action === 'pause' || action === 'playpause' || action === 'ff' || action === 'rw') { mediaKey(action, repeat); return; }
    if (S.menu) { handleMenu(action); return; }
    if (!S.visible || S.peek) { handleHidden(action, repeat); return; }
    handleVisible(action, repeat);
  }
  function onKeyDown(event) {
    if (!ui || !S.mode) { return; }                               // not a Plex / YouTube session: the old controls own every key
    var y = window.KevFlixYoutube;
    if (y && typeof y.active === 'function' && y.active()) { return; }
    if (!(ui.session && ui.session())) { return; }
    var action = keyAction(keyCodeOf(event));
    if (!action) { return; }
    event.preventDefault(); event.stopImmediatePropagation();     // player.js's own handler never also acts on a key the skin handled
    handle(action, !!event.repeat);
  }

  // ================================================================================================================================
  // Player events
  // ================================================================================================================================
  function fixFocus() {
    var list = buttonsFor(S.mode, context());
    var current = list.filter(function (b) { return b.id === S.focus; })[0];
    if (!current || !focusable(current)) { S.focus = 'playpause'; }
  }
  function newSession(mode, status) {
    S.sessionId = status.id; S.mode = mode; S.switching = false;
    S.visible = false; S.peek = false; S.menu = null; S.scrub = null; S.zone = 'buttons'; S.focus = 'playpause';
    S.notice = ''; S.msg = ''; stopUpNext();
    clearTimeout(hideTimer); hideTimer = 0;
    S.status = status;
    if (!mode) { setActive(false); return; }
    ensureSkin(mode); setActive(true);
    if (mode === 'youtube') { applySpeed(); } else { resetSpeed(); }
    reveal();
  }
  function deactivate() {
    clearTimeout(hideTimer); clearTimeout(noticeTimer); hideTimer = noticeTimer = 0; stopUpNext();
    S.visible = false; S.peek = false; S.menu = null; S.scrub = null; S.zone = 'buttons'; S.focus = 'playpause'; S.notice = ''; S.msg = '';
    if (S.mode === 'youtube') { resetSpeed(); }
    if (!S.switching) {                                           // the viewing is over (Back, error): forget the previous-video stack
      S.history = []; Y.infoFor = ''; Y.info = null; Y.infoState = 'idle'; Y.rating = null; Y.signedIn = null; Y.busy = false; Y.rateMissing = false;
    }
    S.mode = null; S.sessionId = ''; S.status = null;
    setActive(false);
  }
  function onStatus(status) {
    if (!status) { deactivate(); return; }
    var mode = modeOf(status, ui && ui.item ? ui.item() : null);
    if (mode !== S.mode || status.id !== S.sessionId) {
      var video0 = mode === 'youtube' ? videoId(status) : '';
      if (video0 && video0 !== Y.infoFor) { loadInfo(video0); }
      newSession(mode, status);
      render();
      return;
    }
    S.status = status;
    if (S.mode === 'youtube') { var id = videoId(status); if (id && id !== Y.infoFor) { loadInfo(id); } }
    if (S.mode) { fixFocus(); render(); }
  }
  function onMessage(text) {
    S.msg = String(text || '');
    if (!S.mode) { return; }
    if (S.msg && !S.visible && !S.upnext) { showBar(true); return; }
    render();
  }
  function onEnded(event) {
    if (!ui || !S.mode) { return; }
    if (S.mode === 'plex') { if (!S.menu) { showBar(false); } return; }                   // player.js does its end-of-video handling after this
    var n = !isLive() && nextOf(S.status);
    if (!n) { return; }                                           // no up-next: player.js leaves to the YouTube view as before
    event.stopImmediatePropagation();                             // autoplay: the video that is next plays by itself unless the viewer presses Back
    startUpNext(n);
  }
  function onTime() { if (S.mode && S.visible) { renderProgress(); } }
  function onVideoState() { if (S.mode && S.visible) { render(); } }
  function onLoaded() { if (S.mode === 'youtube') { applySpeed(); } }
  function bindVideo() {
    if (bound) { return; }
    video = (ui && ui.video) || document.getElementById('video');
    if (!video) { return; }
    bound = true;
    video.addEventListener('ended', onEnded);
    video.addEventListener('timeupdate', onTime);
    video.addEventListener('play', onVideoState);
    video.addEventListener('pause', onVideoState);
    video.addEventListener('loadedmetadata', onLoaded);
  }
  function attach(api) {
    if (ui === api) { return; }
    ui = api;
    bindVideo();
    if (ui.video) { video = ui.video; }
    ui.onStatus(onStatus);
    ui.onMessage(onMessage);
    var current = ui.session && ui.session();
    if (current) { onStatus(current); }
  }

  // ================================================================================================================================
  // Drawing
  // ================================================================================================================================
  function el(tag, className, text) {
    var node = document.createElement(tag);
    if (className) { node.className = className; }
    if (text !== undefined && text !== null) { node.textContent = text; }
    return node;
  }
  function clear(node) { while (node.firstChild) { node.removeChild(node.firstChild); } }
  function setText(node, text) { text = String(text); if (node._text !== text) { node._text = text; node.textContent = text; } }
  function setClass(node, name) { if (node._cls !== name) { node._cls = name; node.className = name; } }
  function setHidden(node, hidden) { hidden = !!hidden; if (node.hidden !== hidden) { node.hidden = hidden; } }
  function setStyle(node, prop, value) {
    if (!node._st) { node._st = {}; }
    if (node._st[prop] !== value) { node._st[prop] = value; node.style[prop] = value; }
  }
  function setAria(node, label) { if (node._aria !== label) { node._aria = label; node.setAttribute('aria-label', label); } }

  // Icons: 48x48 shapes drawn with createElementNS (no markup is parsed).
  function gearPath(cx, cy, teeth, outer, inner, hole) {
    var parts = [], step = 360 / teeth, i;
    function point(deg, r) { var rad = deg * Math.PI / 180; return (cx + r * Math.cos(rad)).toFixed(2) + ' ' + (cy + r * Math.sin(rad)).toFixed(2); }
    for (i = 0; i < teeth; i++) {
      var a = i * step;
      parts.push(point(a - step * 0.30, inner), point(a - step * 0.16, outer), point(a + step * 0.16, outer), point(a + step * 0.30, inner));
    }
    return 'M' + parts.join('L') + 'Z' + 'M' + (cx + hole) + ' ' + cy + 'a' + hole + ' ' + hole + ' 0 1 0 ' + (-2 * hole) + ' 0a' + hole + ' ' + hole + ' 0 1 0 ' + (2 * hole) + ' 0Z';
  }
  var THUMB = 'M20 21l5.5-12.5c3.4 0 5.4 2.4 4.8 5.6L29 20h10c3 0 5 2.8 4.2 5.6l-3 11.6C39.6 39 37.8 40.5 35.6 40.5H20z';
  var CUFF = 'M6 21h9v19.5H6z';
  var FILL = { fill: 'currentColor' }, ROUND = { 'stroke-linecap': 'round', 'stroke-linejoin': 'round' };
  var ICONS = {
    back: [['path', { d: 'M30 10 16 24l14 14', fill: 'none', stroke: 'currentColor', 'stroke-width': 5, 'stroke-linecap': 'round', 'stroke-linejoin': 'round' }]],
    play: [['path', { d: 'M16 8v32l26-16z', fill: 'currentColor' }]],
    pause: [['path', { d: 'M13 8h8v32h-8zM27 8h8v32h-8z', fill: 'currentColor' }]],
    prev: [['path', { d: 'M10 9h5v30h-5zM41 9v30L19 24z', fill: 'currentColor' }]],
    next: [['path', { d: 'M33 9h5v30h-5zM7 9v30l22-15z', fill: 'currentColor' }]],
    rew10: [['path', { d: 'M24 9a15 15 0 1 1-13.2 8', fill: 'none', stroke: 'currentColor', 'stroke-width': 3.6, 'stroke-linecap': 'round' }], ['path', { d: 'M24 3 15 9l9 6z', fill: 'currentColor' }],
      ['text', { x: 24, y: 31, 'text-anchor': 'middle', 'font-size': 14, 'font-weight': 700, fill: 'currentColor', 'font-family': 'Arial,Helvetica,sans-serif' }, '10']],
    fwd30: [['path', { d: 'M24 9a15 15 0 1 0 13.2 8', fill: 'none', stroke: 'currentColor', 'stroke-width': 3.6, 'stroke-linecap': 'round' }], ['path', { d: 'M24 3l9 6-9 6z', fill: 'currentColor' }],
      ['text', { x: 24, y: 31, 'text-anchor': 'middle', 'font-size': 14, 'font-weight': 700, fill: 'currentColor', 'font-family': 'Arial,Helvetica,sans-serif' }, '30']],
    audio: [['path', { d: 'M6 18h8l10-9v30l-10-9H6z', fill: 'currentColor' }],
      ['path', { d: 'M31 17a10 10 0 0 1 0 14M36 11a18 18 0 0 1 0 26', fill: 'none', stroke: 'currentColor', 'stroke-width': 3.6, 'stroke-linecap': 'round' }]],
    subtitles: [['rect', { x: 5, y: 9, width: 38, height: 30, rx: 5, fill: 'none', stroke: 'currentColor', 'stroke-width': 3.6 }],
      ['path', { d: 'M12 22h9M25 22h11M12 30h5M21 30h15', fill: 'none', stroke: 'currentColor', 'stroke-width': 3.6, 'stroke-linecap': 'round' }]],
    gear: [['path', { d: gearPath(24, 24, 8, 21, 15.5, 6.5), fill: 'currentColor', 'fill-rule': 'evenodd' }]],
    like: [['path', { d: THUMB + CUFF, fill: 'none', stroke: 'currentColor', 'stroke-width': 3, 'stroke-linejoin': 'round' }]],
    'like-on': [['path', { d: THUMB + CUFF, fill: 'currentColor', stroke: 'currentColor', 'stroke-width': 3, 'stroke-linejoin': 'round' }]],
    dislike: [['path', { d: THUMB + CUFF, fill: 'none', stroke: 'currentColor', 'stroke-width': 3, 'stroke-linejoin': 'round', transform: 'rotate(180 24 24)' }]],
    'dislike-on': [['path', { d: THUMB + CUFF, fill: 'currentColor', stroke: 'currentColor', 'stroke-width': 3, 'stroke-linejoin': 'round', transform: 'rotate(180 24 24)' }]],
    check: [['path', { d: 'M9 25l10 10 20-22', fill: 'none', stroke: 'currentColor', 'stroke-width': 5, 'stroke-linecap': 'round', 'stroke-linejoin': 'round' }]],
    chevron: [['path', { d: 'M17 9l15 15-15 15', fill: 'none', stroke: 'currentColor', 'stroke-width': 4.5, 'stroke-linecap': 'round', 'stroke-linejoin': 'round' }]]
  };
  var VIEWBOX = { rew10: '3 2 42 42', fwd30: '3 2 42 42' };         // the circular arrows are drawn smaller than the other icons: crop them tighter
  function iconNode(name) {
    var svg = document.createElementNS(SVGNS, 'svg');
    svg.setAttribute('viewBox', VIEWBOX[name] || '0 0 48 48');
    svg.setAttribute('aria-hidden', 'true');
    (ICONS[name] || []).forEach(function (shape) {
      var node = document.createElementNS(SVGNS, shape[0]), key;
      for (key in shape[1]) { if (Object.prototype.hasOwnProperty.call(shape[1], key)) { node.setAttribute(key, String(shape[1][key])); } }
      if (shape[2] !== undefined) { node.textContent = shape[2]; }
      svg.appendChild(node);
    });
    return svg;
  }

  function buildBar(r) {
    r.bar = el('div', 'sk-bar');
    r.track = el('div', 'sk-track');
    r.buf = el('div', 'sk-buf');
    r.played = el('div', 'sk-played');
    r.track.appendChild(r.buf); r.track.appendChild(r.played);
    r.dot = el('div', 'sk-dot');
    r.bubble = el('div', 'sk-bubble'); r.bubble.hidden = true;
    r.bar.appendChild(r.track); r.bar.appendChild(r.dot); r.bar.appendChild(r.bubble);
    r.bar.onclick = function (event) {                            // pointer remotes: a click on the bar seeks there
      if (!ui || isLive() || !r.bar.getBoundingClientRect) { return; }
      var box = r.bar.getBoundingClientRect(), width = box.width || 1;
      ui.seekTo(Math.max(0, Math.min(1, (event.clientX - box.left) / width)) * (Number(S.status && S.status.duration) || 0));
      S.zone = 'bar'; S.scrub = null; showBar(false);
    };
    return r.bar;
  }
  function buildButton(r, id, big) {
    var b = el('div', 'sk-btn' + (big ? ' big' : ''));
    b.setAttribute('role', 'button');
    r.buttons[id] = { el: b, icon: '' };
    b.onclick = function () {
      if (!ui || !S.visible) { return; }
      S.focus = id; S.zone = 'buttons';
      activate(id);
    };
    return b;
  }
  function buildMenu(r) {
    r.menu = el('div', 'sk-menu'); r.menu.hidden = true;
    r.menuTitle = el('div', 'sk-menu-title');
    r.moreUp = el('div', 'sk-more up'); r.moreUp.appendChild(iconNode('chevron'));
    r.menuList = el('div', 'sk-list');
    r.moreDown = el('div', 'sk-more down'); r.moreDown.appendChild(iconNode('chevron'));
    r.menu.appendChild(r.menuTitle); r.menu.appendChild(r.moreUp); r.menu.appendChild(r.menuList); r.menu.appendChild(r.moreDown);
    r.menuSig = '';
    return r.menu;
  }
  function buildPlex(r) {
    var top = el('div', 'sk-ui sk-top');
    var back = el('div', 'sk-back'); back.appendChild(iconNode('back'));
    back.onclick = function () { if (ui && S.visible) { ui.leave(); } };
    var title = el('div', 'sk-title');
    r.show = el('div', 'sk-show'); r.name = el('div', 'sk-name'); r.line = el('div', 'sk-line');
    title.appendChild(r.show); title.appendChild(r.name); title.appendChild(r.line);
    top.appendChild(back); top.appendChild(title);
    var bottom = el('div', 'sk-ui sk-bottom');
    r.times = el('div', 'sk-times');
    r.elapsed = el('span', 'sk-elapsed'); r.ends = el('span', 'sk-ends');
    r.times.appendChild(r.elapsed); r.times.appendChild(r.ends);
    var row = el('div', 'sk-row'), center = el('div', 'sk-center'), right = el('div', 'sk-right');
    ['prev', 'rew', 'playpause', 'fwd', 'next'].forEach(function (id) { center.appendChild(buildButton(r, id, id === 'playpause')); });
    ['audio', 'subs', 'settings'].forEach(function (id) { right.appendChild(buildButton(r, id, false)); });
    row.appendChild(center); row.appendChild(right);
    bottom.appendChild(buildBar(r)); bottom.appendChild(r.times); bottom.appendChild(row);
    r.notice = el('div', 'sk-notice'); r.notice.hidden = true;
    root.appendChild(top); root.appendChild(bottom); root.appendChild(r.notice); root.appendChild(buildMenu(r));
  }
  function buildYoutube(r) {
    var bottom = el('div', 'sk-ui sk-bottom');
    var info = el('div', 'sk-info');
    r.name = el('div', 'sk-name'); r.channel = el('div', 'sk-channel');
    info.appendChild(r.name); info.appendChild(r.channel);
    var row = el('div', 'sk-row');
    ['playpause', 'prev', 'next', 'like', 'dislike', 'captions', 'settings'].forEach(function (id) { row.appendChild(buildButton(r, id, false)); });
    r.times = el('div', 'sk-times');
    r.elapsed = el('span', 'sk-elapsed'); r.live = el('span', 'sk-live', 'LIVE'); r.live.hidden = true;
    r.times.appendChild(r.elapsed); r.times.appendChild(r.live);
    bottom.appendChild(info); bottom.appendChild(row); bottom.appendChild(buildBar(r)); bottom.appendChild(r.times);
    r.notice = el('div', 'sk-notice'); r.notice.hidden = true;
    r.upnext = el('div', 'sk-upnext'); r.upnext.hidden = true;
    r.upLabel = el('div', 'sk-up-label'); r.upTitle = el('div', 'sk-up-title'); r.upHint = el('div', 'sk-up-hint');
    r.upnext.appendChild(r.upLabel); r.upnext.appendChild(r.upTitle); r.upnext.appendChild(r.upHint);
    root.appendChild(bottom); root.appendChild(r.notice); root.appendChild(buildMenu(r)); root.appendChild(r.upnext);
  }
  function ensureSkin(mode) {
    if (!root) { return; }
    if (R && R.kind === mode) { return; }
    clear(root);
    R = { kind: mode, buttons: {} };
    root.className = 'sk-root sk-' + mode;
    root._cls = root.className;
    if (mode === 'plex') { buildPlex(R); } else { buildYoutube(R); }
  }
  function setActive(on) {
    if (player) {
      player.classList.toggle('skin-on', !!on);
      player.classList.toggle('skin-plex', !!on && S.mode === 'plex');
      player.classList.toggle('skin-youtube', !!on && S.mode === 'youtube');
    }
    if (root) { root.hidden = !on; }
  }

  function renderProgress() {
    if (!R || !S.mode || !S.status) { return; }
    var live = isLive(), duration = Number(S.status.duration) || 0, position = ui ? Number(ui.position()) || 0 : 0;
    var shown = S.scrub !== null ? S.scrub : position;
    var played = live ? 1 : duration > 0 ? Math.max(0, Math.min(1, shown / duration)) : 0;
    var ready = live ? 1 : duration > 0 && typeof S.status.ready_through === 'number' ? Math.max(0, Math.min(1, S.status.ready_through / duration)) : 0;
    setStyle(R.played, 'transform', 'scaleX(' + played.toFixed(4) + ')');
    setStyle(R.buf, 'transform', 'scaleX(' + ready.toFixed(4) + ')');
    setStyle(R.dot, 'left', (played * 100).toFixed(2) + '%');
    setHidden(R.bubble, S.scrub === null);
    if (S.scrub !== null) { setStyle(R.bubble, 'left', (played * 100).toFixed(2) + '%'); setText(R.bubble, clock(S.scrub)); }
    if (S.mode === 'plex') {
      setText(R.elapsed, clock(shown));
      setText(R.ends, duration > 0 ? endsAt(new Date(Date.now()), duration - shown) : '');
    } else {
      setHidden(R.live, !live); setHidden(R.elapsed, live);
      setText(R.elapsed, clock(shown) + ' / ' + clock(duration));
    }
  }
  function renderMenu(c) {
    var open = !!S.menu;
    setHidden(R.menu, !open);
    if (!open) { R.menuSig = ''; return; }
    var level = topLevel(), built = levelBuild(level, S.mode, c), rows = built.rows;
    level.start = windowStart(rows.length, level.index, MENU_ROWS, level.start);
    var shown = rows.slice(level.start, level.start + MENU_ROWS), root0 = level.pick === 'root';
    var sig = JSON.stringify([built.title, level.index, level.start, rows.length, shown.map(function (r) { return [r.label, r.value, r.selected, r.enabled, r.note]; })]);
    if (sig === R.menuSig) { return; }
    R.menuSig = sig;
    setText(R.menuTitle, built.title);
    setClass(R.menuList, 'sk-list ' + (root0 ? 'root' : 'pick'));
    setHidden(R.moreUp, level.start === 0); setHidden(R.moreDown, level.start + MENU_ROWS >= rows.length);
    clear(R.menuList);
    shown.forEach(function (row, offset) {
      var index = level.start + offset;
      var item = el('div', 'sk-item' + (index === level.index ? ' focus' : '') + (row.enabled ? '' : ' off') + (row.selected ? ' sel' : ''));
      var mark = el('span', 'sk-mark');
      if (row.selected) { mark.appendChild(iconNode('check')); }
      if (!root0) { item.appendChild(mark); }
      item.appendChild(el('span', 'sk-label', row.label));
      if (root0) {
        item.appendChild(el('span', 'sk-value', row.value));
        var chev = el('span', 'sk-chev');                         // toggles get an empty slot so every value lines up
        if (row.act && row.act.kind === 'open') { chev.appendChild(iconNode('chevron')); }
        item.appendChild(chev);
      }
      if (row.note) { item.appendChild(el('div', 'sk-note', row.note)); }
      item.onclick = function () { if (S.menu && topLevel() === level) { level.index = index; activateRow(rows[index]); } };
      R.menuList.appendChild(item);
    });
  }
  function render() {
    if (!R || !S.mode || !root) { return; }
    fixFocus();                                                   // a focused button that has just disappeared or been dimmed hands the focus to Play/Pause
    var c = context(), plex = S.mode === 'plex';
    setClass(root, 'sk-root sk-' + S.mode + (S.visible ? ' sk-on' : '') + (S.visible && (S.zone === 'bar' || S.peek) ? ' sk-bar-focus' : '') + (S.menu ? ' sk-menu-open' : '') + (c.live ? ' sk-is-live' : ''));
    if (plex) {
      var head = plexHeader(c.plex);
      setText(R.show, head.show); setHidden(R.show, !head.show);
      setText(R.name, head.title);
      setText(R.line, head.line); setHidden(R.line, !head.line);
    } else {
      setText(R.name, c.status.title || '');
      var channel = channelOf(c.status);
      setText(R.channel, channel); setHidden(R.channel, !channel);
    }
    var list = buttonsFor(S.mode, c);
    list.forEach(function (b) {
      var ref = R.buttons[b.id];
      if (!ref) { return; }
      setHidden(ref.el, b.hidden);
      var focused = S.visible && !S.peek && S.zone === 'buttons' && S.focus === b.id;
      setClass(ref.el, 'sk-btn' + (b.primary && plex ? ' big' : '') + (focused ? ' focus' : '') + (b.enabled ? '' : ' off') + (b.pressed ? ' on' : '') + (S.menu && S.focus === b.id && !focused ? ' open' : ''));
      setAria(ref.el, b.label);
      if (ref.icon !== b.icon) { ref.icon = b.icon; clear(ref.el); ref.el.appendChild(iconNode(b.icon)); }
    });
    var text = S.notice || S.msg;
    setText(R.notice, text); setHidden(R.notice, !text);
    renderProgress();
    renderMenu(c);
    if (!plex) {
      setHidden(R.upnext, !S.upnext);
      if (S.upnext) {
        setText(R.upLabel, 'Up next');
        setText(R.upTitle, S.upnext.title || 'Next video');
        setText(R.upHint, 'Playing in ' + S.upnext.left + ' · OK to play now · Back to cancel');
      }
    }
  }

  // What a test (or a person with a debugger) can look at without touching the DOM.
  function snapshot() {
    var c = S.mode ? context() : null, level = S.menu ? topLevel() : null, built = level ? levelBuild(level, S.mode, c) : null;
    return {
      mode: S.mode, visible: S.visible, peek: S.peek, zone: S.zone, focus: S.focus, scrub: S.scrub, notice: S.notice || S.msg,
      buttons: c ? buttonsFor(S.mode, c).map(function (b) { return { id: b.id, icon: b.icon, enabled: b.enabled, hidden: !!b.hidden, pressed: !!b.pressed }; }) : [],
      menu: S.menu ? { depth: S.menu.levels.length, pick: level.pick, title: built.title, index: level.index,
        rows: built.rows.map(function (r) { return { id: r.id, label: r.label, value: r.value, selected: r.selected, enabled: r.enabled, note: r.note }; }) } : null,
      upnext: S.upnext ? { id: S.upnext.id, title: S.upnext.title, left: S.upnext.left } : null,
      header: c && S.mode === 'plex' ? plexHeader(c.plex) : c ? { title: c.status.title || '', channel: channelOf(c.status) } : null,
      elapsed: R && R.elapsed ? R.elapsed.textContent : '', ends: R && R.ends ? R.ends.textContent : '', bubble: R && R.bubble && !R.bubble.hidden ? R.bubble.textContent : '',
      live: c ? c.live : false, history: S.history.slice(), speed: S.speed, rating: Y.rating, infoState: Y.infoState, switching: S.switching
    };
  }

  // ---- start-up --------------------------------------------------------------------------------------------------------------------
  player = document.getElementById('player');
  root = document.getElementById('skin');
  if (!root && player) { root = document.createElement('div'); root.id = 'skin'; root.hidden = true; player.appendChild(root); }
  bindVideo();
  window.addEventListener('keydown', onKeyDown, true);
  window.KevFlixControls = {
    attach: attach, snapshot: snapshot, mode: function () { return S.mode; },
    pure: { keyAction: keyAction, clock: clock, clockTime: clockTime, endsAt: endsAt, modeOf: modeOf, plexInfo: plexInfo, plexHeader: plexHeader, moveFocus: moveFocus,
      scrubStep: scrubStep, scrubTarget: scrubTarget, stepRow: stepRow, windowStart: windowStart, buttonsFor: buttonsFor, rootRows: rootRows, pickList: pickList,
      nearestLevel: nearestLevel, gearPath: gearPath }
  };
  if (window.KevFlixPlayerUI) { attach(window.KevFlixPlayerUI); }
}());
