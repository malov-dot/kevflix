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
  var SEEK_COMMIT_MS = 600;            // seek presses are collected; ONE seek is sent this long after the last press (0 = every press seeks at once, as before)
  var PLEX_UPNEXT_SECONDS = 10;        // the next Plex episode starts by itself this many seconds after an episode ends, Back cancels (0 = off: the Next button is focused instead)
  var SEEK = { plex: { back: 10, forward: 30 }, youtube: { back: 10, forward: 10 } };
  var SPEEDS = [0.5, 0.75, 1, 1.25, 1.5, 2];
  var LEVELS = [1, 0.75, 0.5, 0.25, 0];
  var QUALITIES = [2160, 1080, 720, 480];
  var WATCH_MODES = [{ value: 'balanced', label: 'Balanced reactions' }, { value: 'quiet', label: 'Quiet reactions' }, { value: 'ask-only', label: 'Only when I ask Sem' }];
  // Samsung remote key codes (and the desktop equivalents) -> actions.
  var KEY_CODES = { 37: 'left', 38: 'up', 39: 'right', 40: 'down', 13: 'ok', 10009: 'back', 27: 'back', 8: 'back',
    415: 'play', 19: 'pause', 10252: 'playpause', 32: 'playpause', 417: 'ff', 412: 'rw', 10232: 'prevkey', 10233: 'nextkey' };
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
    menu: null, notice: '', msg: '', msgSeen: '', upnext: null, history: [], switching: false, switchSerial: 0, speed: 1, rateRetry: false,
    pending: null, ended: false };
  var Y = { infoFor: '', infoState: 'idle', info: null, rating: null, signedIn: null, busy: false, rateMissing: false };
  var ui = null, video = null, player = null, root = null, R = null, bound = false;
  var hideTimer = 0, noticeTimer = 0, upTimer = 0, seekTimer = 0, msgTimer = 0;

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
  // Self-check. Nobody can read the TV's console, so the skin measures itself and tells the PC: POST /tv-api/diagnostic
  // {stage: 'controls', session, controls: {...}} CHECK_REPORT_MS into a session, again when it ends, and at once if a skin handler throws. The PC
  // writes one 'tv_diagnostic' log line and keeps at most 24 flat fields of up to 160 characters (relay/tv.py diagnostic_record), so every value
  // here is a number, a flag or a short text. Two things are acted on right here: a TV that draws the skin slowly gets the light look (class
  // sk-lite: no blurred shadows, no transitions), and a skin that throws hands the session back to the old controls.
  // ================================================================================================================================
  var CHECK_REPORT_MS = 20000, PAINT_SAMPLES = 200, FRAME_RUN = 90, FRAME_RUNS = 6, CODE_KINDS = 12;   // 12 codes as 'code x times' stay under the PC's 160 characters
  var LITE_FRAME_MS = 40, LITE_PAINT_MS = 100, LITE_DAYS = 30, LITE_KEY = 'kevflix-skin-lite';
  var D = null, reportTimer = 0, lite = false, brokenSession = '';
  function freshCheck() {
    return { codes: {}, repeat: false, presses: 0, paint: [], frames: [], frameWorst: null, runs: 0, probing: false,
      seekPresses: 0, seekCommits: 0, seekDropped: 0, rateAsked: null, rateGot: null, ended: 0, upShown: 0, upPlayed: 0, upCancelled: 0,
      errors: 0, errorFirst: '' };
  }
  function nextFrame(fn) {
    if (typeof window.requestAnimationFrame !== 'function') { return false; }
    window.requestAnimationFrame(fn);
    return true;
  }
  function median(list) {
    if (!list.length) { return null; }
    var sorted = list.slice().sort(function (a, b) { return a - b; });
    return sorted[Math.floor(sorted.length / 2)];
  }
  function worst(list) { return list.length ? Math.max.apply(null, list) : null; }
  function loadLite() {
    try { var since = Number(localStorage.getItem(LITE_KEY)); lite = since > 0 && Date.now() - since < LITE_DAYS * 86400000; } catch (ignore) { lite = false; }
  }
  function setLite() {
    lite = true;
    try { localStorage.setItem(LITE_KEY, String(Date.now())); } catch (ignore) {}
    render();
  }
  // Slow by the TV's own numbers: the median frame of two bar showings above LITE_FRAME_MS, or the median key-to-paint of eight presses above LITE_PAINT_MS.
  function judge() {
    if (lite || !D) { return; }
    if ((D.frames.length >= 2 && median(D.frames) > LITE_FRAME_MS) || (D.paint.length >= 8 && median(D.paint) > LITE_PAINT_MS)) { setLite(); }
  }
  // Every key the skin sees in its session: the code (which keys this remote really sends), whether the TV marks repeats, and for a handled key the
  // time from the press to the frame AFTER the one that painted the change (two animation frames; at most PAINT_SAMPLES samples).
  function noteKey(code, repeat, handled) {
    var d = D;
    if (!d) { return; }
    if (repeat) { d.repeat = true; }
    var name = String(code);
    if (d.codes[name] || Object.keys(d.codes).length < CODE_KINDS) { d.codes[name] = (d.codes[name] || 0) + 1; }
    if (!handled) { return; }
    d.presses += 1;
    if (d.paint.length >= PAINT_SAMPLES) { return; }
    var pressed = Date.now();
    nextFrame(guard(function () { nextFrame(guard(function () { d.paint.push(Date.now() - pressed); if (d === D) { judge(); } })); }));
  }
  // FRAME_RUN frames after the bar or a menu comes up, at most FRAME_RUNS times a session: how long the TV takes per frame while the skin is drawn.
  function probeFrames() {
    var d = D;
    if (!d || d.probing || d.runs >= FRAME_RUNS) { return; }
    var left = FRAME_RUN, last = 0, deltas = [];
    var step = guard(function () {
      var now = Date.now();
      if (last) { deltas.push(now - last); }
      last = now; left -= 1;
      if (left > 0 && d === D && (S.visible || S.menu) && nextFrame(step)) { return; }
      d.probing = false;
      if (deltas.length >= 10) {
        d.runs += 1; d.frames.push(median(deltas)); d.frameWorst = Math.max(d.frameWorst || 0, worst(deltas));
        if (d === D) { judge(); }
      }
    });
    d.probing = nextFrame(step);
  }
  function checkReport(when) {
    var d = D, keys = window.KevFlixKeys || {};
    if (!d || !S.sessionId) { return; }
    request('POST', 'diagnostic', { stage: 'controls', session: S.sessionId, controls: {
      when: when, mode: S.mode, keys_ok: (keys.ok || []).join(','), keys_failed: (keys.failed || []).join(','), keys_error: String(keys.error || ''),
      codes: Object.keys(d.codes).map(function (code) { return code + 'x' + d.codes[code]; }).join(','), repeat_seen: d.repeat, presses: d.presses,
      paint_n: d.paint.length, paint_median_ms: median(d.paint), paint_worst_ms: worst(d.paint),
      frame_n: d.frames.length, frame_median_ms: median(d.frames), frame_worst_ms: d.frameWorst, lite: lite,
      seek_presses: d.seekPresses, seek_commits: d.seekCommits, seek_dropped: d.seekDropped, rate_asked: d.rateAsked, rate_got: d.rateGot,
      ended: d.ended, upnext: d.upShown + '/' + d.upPlayed + '/' + d.upCancelled, errors: d.errors, error_first: d.errorFirst } }).catch(function () {});
  }
  function startCheck() {
    D = freshCheck();
    clearTimeout(reportTimer);
    reportTimer = setTimeout(guard(function () { reportTimer = 0; checkReport('start'); }), CHECK_REPORT_MS);
  }
  function endCheck(when) {
    clearTimeout(reportTimer); reportTimer = 0;
    if (D && S.mode) { checkReport(when); }
    D = null;
  }
  // A skin handler threw: say so, then the old controls (still in the page) take over for the rest of this session.
  function skinFailed(error) {
    var text = String((error && error.message) || error).slice(0, 150);
    if (D) { D.errors += 1; if (!D.errorFirst) { D.errorFirst = text; } }
    if (!S.mode) { return; }
    brokenSession = S.sessionId;
    try { endCheck('error'); } catch (ignore) { D = null; }
    try { deactivate(); } catch (ignore) {
      S.mode = null; S.sessionId = ''; S.status = null;
      if (player) { player.classList.remove('skin-on'); player.classList.remove('skin-plex'); player.classList.remove('skin-youtube'); }
      if (root) { root.hidden = true; }
    }
  }
  function guard(fn) {
    return function () {
      try { return fn.apply(this, arguments); } catch (error) { skinFailed(error); return undefined; }
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
    request('GET', 'youtube/info?video=' + encodeURIComponent(id)).then(guard(function (value) {
      if (Y.infoFor !== id) { return; }
      Y.info = value && typeof value === 'object' ? value : {};
      Y.infoState = 'ok'; Y.rating = ratingOf(Y.info.rating); Y.signedIn = typeof Y.info.signed_in === 'boolean' ? Y.info.signed_in : null;
      render();
    }), guard(function () {
      if (Y.infoFor !== id) { return; }
      Y.info = null; Y.infoState = 'missing';              // 404 or any failure: no rating buttons, no PC-side channel/up-next/captions; playback is untouched
      render();
    }));
  }
  function rate(which) {
    if (S.mode !== 'youtube' || Y.busy) { return; }
    var id = videoId(S.status), target = Y.rating === which ? 'none' : which;
    if (Y.signedIn === false) { notice('Sign in to YouTube on your PC to rate videos.'); return; }
    Y.busy = true;
    request('POST', 'youtube/rate', { video: id, rating: target }).then(guard(function (reply) {
      Y.busy = false;
      if (videoId(S.status) !== id) { return; }
      if (reply && reply.signed_in === false) { Y.signedIn = false; notice('Sign in to YouTube on your PC to rate videos.'); render(); return; }
      Y.rating = reply && Object.prototype.hasOwnProperty.call(reply, 'rating') ? ratingOf(reply.rating) : (target === 'none' ? null : target);
      render();
    }), guard(function (error) {
      Y.busy = false;
      if (error && error.status === 404) { Y.rateMissing = true; notice('Rating is not available yet.'); }
      else if (error && (error.status === 401 || error.status === 403 || /sign.?in|signed/i.test(error.message || ''))) { Y.signedIn = false; notice('Sign in to YouTube on your PC to rate videos.'); }
      else { notice('Could not rate this video' + (error && error.message ? ': ' + error.message : '.')); }
      render();
    }));
  }

  // ================================================================================================================================
  // Actions
  // ================================================================================================================================
  // Seek presses are collected: each one moves a pending target that the bar shows at once, and ONE seek goes to the player SEEK_COMMIT_MS after the
  // last press. (Every press used to seek by itself; beyond the prepared part each of them restarted the stream on the PC.)
  function seekBy(delta) {
    if (!ui || isLive()) { return; }
    var duration = Number(S.status && S.status.duration) || 0, limit = duration > 0 ? Math.max(0, duration - 0.1) : Infinity;
    var from = S.pending !== null ? S.pending : Number(ui.position()) || 0;
    S.pending = Math.max(0, Math.min(limit, from + delta));
    if (D) { D.seekPresses += 1; }
    clearTimeout(seekTimer); seekTimer = 0;
    if (SEEK_COMMIT_MS > 0) { seekTimer = setTimeout(guard(commitSeek), SEEK_COMMIT_MS); } else { commitSeek(); }
  }
  function commitSeek() {
    clearTimeout(seekTimer); seekTimer = 0;
    if (S.pending === null) { return; }
    var target = S.pending;
    S.pending = null;
    if (ui && S.mode && !isLive() && Math.abs(target - (Number(ui.position()) || 0)) >= 1) { if (D) { D.seekCommits += 1; } seekTo(target); }
    render();
  }
  function dropSeek() { if (D && S.pending !== null) { D.seekDropped += 1; } clearTimeout(seekTimer); seekTimer = 0; S.pending = null; }
  function seekTo(target) { S.ended = false; ui.seekTo(target); }           // a seek after the end: Play then continues from there, not from the beginning
  // While the PC is still preparing the stream (status.ready false, or not startable yet) a press that only wanted to look must not switch the automatic
  // start off: OK with the bar hidden then just brings up the controls.
  function preparing() { var s = S.status; return !!s && !s.error && !!ui.isPlaying() && (s.ready === false || s.startable === false); }
  function togglePlay() {
    if (S.ended && !ui.isPlaying()) { ui.seekTo(0); }             // after the end: play again from the beginning, not from where the loaded part happens to start
    S.ended = false;
    ui.playPause();
  }
  function setSpeed(rate) {
    S.speed = rate; S.rateRetry = false;
    applySpeed();
  }
  function applySpeed() {
    if (!video) { return; }
    try { video.defaultPlaybackRate = S.speed; video.playbackRate = S.speed; } catch (ignore) {}
    if (D && S.mode === 'youtube' && S.speed !== 1) { D.rateAsked = S.speed; D.rateGot = Number(video.playbackRate); }
    readSpeed();
  }
  // What the TV really plays at. A decoder that refuses a rate throws, leaves playbackRate where it was or snaps it back ('ratechange'): the menu then
  // shows the real speed and the refused one is not remembered for the next video.
  function readSpeed() {
    var real = video ? Number(video.playbackRate) : NaN;
    if (S.mode !== 'youtube' || !isFinite(real) || real <= 0 || real === S.speed) { return; }
    var wanted = S.speed;
    S.speed = SPEEDS.indexOf(real) >= 0 ? real : 1;
    notice('This TV cannot play at ' + speedText(wanted) + '.');
  }
  function onRate() {
    if (S.mode !== 'youtube' || !video) { return; }
    if (Number(video.playbackRate) === S.speed) { S.rateRetry = false; return; }
    if (!S.rateRetry) { S.rateRetry = true; try { video.playbackRate = S.speed; } catch (ignore) {} return; }   // once more: a new load may only have reset it
    S.rateRetry = false;
    if (D) { D.rateGot = Number(video.playbackRate); }
    readSpeed(); render();
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
  function forgetViewing() { S.history = []; Y.infoFor = ''; Y.info = null; Y.infoState = 'idle'; Y.rating = null; Y.signedIn = null; Y.busy = false; Y.rateMissing = false; }
  function switchVideo(id) {
    var opener = window.KevFlixPlayer, mine = ++S.switchSerial;
    S.switching = true; stopUpNext(); hideBar();
    var opened;
    try {
      if (!opener || typeof opener.openYoutube !== 'function') { throw new Error('The KevFLIX player is not ready.'); }
      opened = opener.openYoutube(id);
    } catch (error) { opened = Promise.reject(error); }
    Promise.resolve(opened).then(function () {
      if (mine !== S.switchSerial || (ui && ui.session && ui.session())) { return; }
      S.switching = false; forgetViewing();                       // the open ended without a video (cancelled, or a newer open won): this viewing is over
    }, function (error) {
      S.switching = false; S.history = [];
      var reason = error && error.message === 'Cancelled' ? '' : ((error && error.message) || 'This YouTube video could not be played.');
      Promise.resolve(ui && ui.leave ? ui.leave() : null).catch(function () {}).then(function () { returnToYoutube(reason); });
    });
  }
  function goNext() {
    var n = nextOf(S.status);
    if (!n || isLive() || S.switching) { return; }
    S.history.push(videoId(S.status));
    switchVideo(n.id);
  }
  function goPrevious() {
    if (!S.history.length || isLive() || S.switching) { return; }
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
  // n = {id, title} (the next YouTube video) or {key, title, plex: true} (the next Plex episode; only used when PLEX_UPNEXT_SECONDS > 0).
  function startUpNext(n, seconds) {
    stopUpNext(); clearTimeout(hideTimer); hideTimer = 0; dropSeek();
    S.visible = false; S.peek = false; S.menu = null; S.scrub = null;
    S.upnext = { id: n.plex ? String(n.key) : n.id, title: n.title, left: seconds || UPNEXT_SECONDS, plex: !!n.plex };
    if (D) { D.upShown += 1; }
    upTimer = setInterval(guard(function () {
      if (!S.upnext) { stopUpNext(); return; }
      S.upnext.left -= 1;
      if (S.upnext.left <= 0) { playUpNext(); } else { render(); }
    }), 1000);
    render();
  }
  function stopUpNext() { if (upTimer) { clearInterval(upTimer); upTimer = 0; } S.upnext = null; }
  function playUpNext() {
    var up = S.upnext;
    stopUpNext();
    if (D && up) { D.upPlayed += 1; }
    if (up && up.plex) { openEpisode({ key: up.id, title: up.title }); } else { goNext(); }
  }
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
      case 'playpause': togglePlay(); break;
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
  // A list can get shorter while it is open (dubbing switched off removes its subtitle row, a caption list arrives): the focus never stays past the end.
  function clampLevel(level, rows) {
    if (level.index >= rows.length) { level.index = Math.max(0, rows.length - 1); }
    if (rows.length && !rows[level.index].enabled) { level.index = firstIndex(rows); }
  }
  function newLevel(pick) {
    var level = { pick: pick, index: 0, start: 0 };
    level.index = firstIndex(levelBuild(level, S.mode, context()).rows);
    return level;
  }
  function openMenu(pick) {
    S.menu = { origin: pick, levels: [newLevel(pick)] };
    clearTimeout(hideTimer); hideTimer = 0;
    render(); probeFrames();
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
    clampLevel(level, rows);
    if (action === 'up') { level.index = stepRow(rows, level.index, -1); }
    else if (action === 'down') { level.index = stepRow(rows, level.index, 1); }
    else if (action === 'ok') { activateRow(rows[level.index]); return; }
    else if (action === 'left' || action === 'back') { menuBack(); return; }
    render();
  }

  // ---- showing, hiding, notices -----------------------------------------------------------------------------------------------
  function arm() {
    clearTimeout(hideTimer); hideTimer = 0;
    if (S.visible && !S.menu && !S.upnext && !(S.status && S.status.error)) { hideTimer = setTimeout(guard(function () { hideTimer = 0; if (!S.menu) { hideBar(); } }), HIDE_MS); }
  }
  function showBar(peek) { S.visible = true; S.peek = !!peek; arm(); render(); probeFrames(); }
  function reveal() { S.focus = 'playpause'; S.zone = 'buttons'; S.scrub = null; showBar(false); }
  function hideBar() {
    S.visible = false; S.peek = false; S.scrub = null; S.zone = 'buttons'; dropSeek();
    clearTimeout(hideTimer); hideTimer = 0;
    render();
  }
  function notice(text) {
    S.notice = String(text || '');
    clearTimeout(noticeTimer); noticeTimer = 0;
    if (S.notice) {
      noticeTimer = setTimeout(guard(function () { noticeTimer = 0; S.notice = ''; render(); }), NOTICE_MS);
      if (!S.visible) { showBar(true); return; }
    }
    render();
  }

  // ---- keys ----------------------------------------------------------------------------------------------------------------------
  function mediaKey(action, repeat) {
    var back = SEEK[S.mode].back, forward = SEEK[S.mode].forward;
    if (action === 'play') { if (!ui.isPlaying()) { togglePlay(); } }
    else if (action === 'pause') { if (ui.isPlaying()) { ui.playPause(); } }
    else if (action === 'playpause') { if (!repeat) { togglePlay(); } }
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
      case 'ok':
        if (repeat) { break; }
        if (S.pending !== null) { commitSeek(); showBar(true); break; }       // OK sends the collected seek at once
        if (!preparing()) { togglePlay(); }
        reveal();
        break;
      case 'back':
        if (S.pending !== null) { dropSeek(); render(); }                     // Back drops a seek that was not sent yet
        else if (S.visible) { hideBar(); } else if (!repeat) { ui.leave(); }
        break;
      default: break;
    }
  }
  // The remote's own previous / next keys (MediaTrackPrevious 10232, MediaTrackNext 10233): the same as OK on the Previous / Next button. With nothing to go
  // to (a film, the last episode, no earlier video) the bar just shows.
  function trackKey(id, repeat) {
    if (repeat) { return; }
    var target = buttonsFor(S.mode, context()).filter(function (b) { return b.id === id; })[0];
    if (!target || target.hidden || !target.enabled) { showBar(true); return; }
    activate(id);
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
        if (S.scrub !== null) { var target = S.scrub; S.scrub = null; seekTo(target); } else { togglePlay(); }
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
      if (S.scrub !== null || S.pending !== null) { S.scrub = null; dropSeek(); render(); } else { hideBar(); }
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
    else if (action === 'back') {
      var episode = !!(S.upnext && S.upnext.plex);
      if (D) { D.upCancelled += 1; }
      stopUpNext();
      if (episode) { S.focus = 'next'; S.zone = 'buttons'; showBar(false); }       // an episode: only the countdown is cancelled, Next stays one OK away
      else { render(); ui.leave(); }
    }
  }
  function handle(action, repeat) {
    if (S.upnext) { handleUpNext(action, repeat); return; }
    if (action === 'prevkey' || action === 'nextkey') { trackKey(action === 'nextkey' ? 'next' : 'prev', repeat); return; }
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
    var code = keyCodeOf(event), action = keyAction(code);
    noteKey(code, !!event.repeat, !!action);
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
    endCheck('leave');                                            // a session that changes without a leave in between still reports
    S.sessionId = status.id; S.mode = mode; S.switching = false;
    S.visible = false; S.peek = false; S.menu = null; S.scrub = null; S.zone = 'buttons'; S.focus = 'playpause';
    S.notice = ''; S.msg = ''; S.msgSeen = ''; S.ended = false; stopUpNext(); dropSeek();
    clearTimeout(hideTimer); hideTimer = 0; clearTimeout(msgTimer); msgTimer = 0;
    S.status = status;
    if (!mode) { setActive(false); return; }
    ensureSkin(mode); setActive(true);
    startCheck();
    if (mode === 'youtube') { applySpeed(); } else { resetSpeed(); }
    reveal();
  }
  function deactivate() {
    endCheck('leave');
    clearTimeout(hideTimer); clearTimeout(noticeTimer); clearTimeout(msgTimer); hideTimer = noticeTimer = msgTimer = 0; stopUpNext(); dropSeek();
    S.visible = false; S.peek = false; S.menu = null; S.scrub = null; S.zone = 'buttons'; S.focus = 'playpause'; S.notice = ''; S.msg = ''; S.msgSeen = '';
    S.ended = false;
    if (S.mode === 'youtube') { resetSpeed(); }
    if (!S.switching) {                                           // the viewing is over (Back, error): forget the previous-video stack
      forgetViewing();
    }
    S.mode = null; S.sessionId = ''; S.status = null;
    setActive(false);
  }
  function onStatus(status) {
    if (!status) { brokenSession = ''; deactivate(); return; }
    if (status.id === brokenSession) { return; }                  // the skin failed in this session: the old controls keep it
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
  // player.js repeats an unchanged message with every status (every 2 s while an error or a wait lasts): only a NEW text is shown, it brings the bar up
  // once, and it goes away after NOTICE_MS unless the status still carries the error / wait it describes.
  function expireMsg() {
    msgTimer = 0;
    if (S.status && (S.status.error || S.status.resource_wait)) { msgTimer = setTimeout(guard(expireMsg), NOTICE_MS); return; }
    if (S.msg) { S.msg = ''; render(); }
  }
  function onMessage(text) {
    text = String(text || '');
    if (text === S.msgSeen) { return; }
    S.msgSeen = text; S.msg = text;
    clearTimeout(msgTimer); msgTimer = 0;
    if (!S.mode) { return; }
    if (S.msg) {
      msgTimer = setTimeout(guard(expireMsg), NOTICE_MS);
      if (!S.visible && !S.upnext) { showBar(true); return; }
    }
    render();
  }
  function onEnded(event) {
    if (!ui || !S.mode) { return; }
    if (D) { D.ended += 1; }
    if (S.mode === 'plex') {                                      // player.js does its end-of-video handling after this (heartbeat 'ended', watched state)
      S.ended = true; dropSeek();
      var following = context().plex.next;
      if (following && PLEX_UPNEXT_SECONDS > 0 && !S.menu) { startUpNext({ key: following.key, title: following.title, plex: true }, PLEX_UPNEXT_SECONDS); return; }
      if (!S.menu) { if (following) { S.focus = 'next'; S.zone = 'buttons'; } showBar(false); }
      return;
    }
    var n = !isLive() && nextOf(S.status);
    if (!n) { return; }                                           // no up-next: player.js leaves to the YouTube view as before
    event.stopImmediatePropagation();                             // autoplay: the video that is next plays by itself unless the viewer presses Back
    startUpNext(n);
  }
  function onTime() { if (S.mode && S.visible) { renderProgress(); } }
  function onVideoState(event) { if (event && event.type === 'play') { S.ended = false; } if (S.mode && S.visible) { render(); } }
  function onLoaded() { if (S.mode === 'youtube') { applySpeed(); } }
  function bindVideo() {
    if (bound) { return; }
    video = (ui && ui.video) || document.getElementById('video');
    if (!video) { return; }
    bound = true;
    video.addEventListener('ended', guard(onEnded));
    video.addEventListener('timeupdate', guard(onTime));
    video.addEventListener('play', guard(onVideoState));
    video.addEventListener('pause', guard(onVideoState));
    video.addEventListener('loadedmetadata', guard(onLoaded));
    video.addEventListener('ratechange', guard(onRate));
  }
  function attach(api) {
    if (ui === api) { return; }
    ui = api;
    bindVideo();
    if (ui.video) { video = ui.video; }
    ui.onStatus(guard(onStatus));
    ui.onMessage(guard(onMessage));
    var current = ui.session && ui.session();
    if (current) { guard(onStatus)(current); }
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
    r.pos = el('div', 'sk-pos');                                  // carries the dot and the time bubble along the bar with a transform (no layout at every time update)
    r.pos.appendChild(r.dot); r.pos.appendChild(r.bubble);
    r.bar.appendChild(r.track); r.bar.appendChild(r.pos);
    r.bar.onclick = guard(function (event) {                      // pointer remotes: a click on the bar seeks there
      if (!ui || isLive() || !r.bar.getBoundingClientRect) { return; }
      var box = r.bar.getBoundingClientRect(), width = box.width || 1;
      seekTo(Math.max(0, Math.min(1, (event.clientX - box.left) / width)) * (Number(S.status && S.status.duration) || 0));
      S.zone = 'bar'; S.scrub = null; showBar(false);
    });
    return r.bar;
  }
  function buildButton(r, id, big) {
    var b = el('div', 'sk-btn' + (big ? ' big' : ''));
    b.setAttribute('role', 'button');
    r.buttons[id] = { el: b, icon: '' };
    b.onclick = guard(function () {
      if (!ui || !S.visible) { return; }
      S.focus = id; S.zone = 'buttons';
      activate(id);
    });
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
  function buildUpNext(r) {
    r.upnext = el('div', 'sk-upnext'); r.upnext.hidden = true;
    r.upLabel = el('div', 'sk-up-label'); r.upTitle = el('div', 'sk-up-title'); r.upHint = el('div', 'sk-up-hint');
    r.upnext.appendChild(r.upLabel); r.upnext.appendChild(r.upTitle); r.upnext.appendChild(r.upHint);
    return r.upnext;
  }
  function buildPlex(r) {
    var top = el('div', 'sk-ui sk-top');
    var back = el('div', 'sk-back'); back.appendChild(iconNode('back'));
    back.onclick = guard(function () { if (ui && S.visible) { ui.leave(); } });
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
    root.appendChild(top); root.appendChild(bottom); root.appendChild(r.notice); root.appendChild(buildMenu(r)); root.appendChild(buildUpNext(r));
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
    root.appendChild(bottom); root.appendChild(r.notice); root.appendChild(buildMenu(r)); root.appendChild(buildUpNext(r));
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
    var preview = S.scrub !== null ? S.scrub : S.pending;         // a scrub on the bar, or seek presses that are still being collected
    var shown = preview !== null ? preview : position;
    var played = live ? 1 : duration > 0 ? Math.max(0, Math.min(1, shown / duration)) : 0;
    var ready = live ? 1 : duration > 0 && typeof S.status.ready_through === 'number' ? Math.max(0, Math.min(1, S.status.ready_through / duration)) : 0;
    setStyle(R.played, 'transform', 'scaleX(' + played.toFixed(4) + ')');
    setStyle(R.buf, 'transform', 'scaleX(' + ready.toFixed(4) + ')');
    setStyle(R.pos, 'transform', 'translateX(' + (played * 100).toFixed(2) + '%)');
    setHidden(R.bubble, preview === null);
    if (preview !== null) { setText(R.bubble, clock(preview)); }
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
    clampLevel(level, rows);
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
      item.onclick = guard(function () { if (S.menu && topLevel() === level) { level.index = index; activateRow(rows[index]); } });
      R.menuList.appendChild(item);
    });
  }
  function render() {
    if (!R || !S.mode || !root) { return; }
    fixFocus();                                                   // a focused button that has just disappeared or been dimmed hands the focus to Play/Pause
    var c = context(), plex = S.mode === 'plex';
    setClass(root, 'sk-root sk-' + S.mode + (S.visible ? ' sk-on' : '') + (S.visible && (S.zone === 'bar' || S.peek) ? ' sk-bar-focus' : '') + (S.menu ? ' sk-menu-open' : '') + (c.live ? ' sk-is-live' : '') + (lite ? ' sk-lite' : ''));
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
    setHidden(R.upnext, !S.upnext);
    if (S.upnext) {
      setText(R.upLabel, S.upnext.plex ? 'Next episode' : 'Up next');
      setText(R.upTitle, S.upnext.title || (S.upnext.plex ? 'Next episode' : 'Next video'));
      setText(R.upHint, 'Playing in ' + S.upnext.left + ' · OK to play now · Back to cancel');
    }
  }

  // What a test (or a person with a debugger) can look at without touching the DOM.
  function snapshot() {
    var c = S.mode ? context() : null, level = S.menu ? topLevel() : null, built = level ? levelBuild(level, S.mode, c) : null;
    return {
      mode: S.mode, visible: S.visible, peek: S.peek, zone: S.zone, focus: S.focus, scrub: S.scrub, pending: S.pending, notice: S.notice || S.msg,
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
  loadLite();
  bindVideo();
  window.addEventListener('keydown', guard(onKeyDown), true);
  window.KevFlixControls = {
    attach: attach, snapshot: snapshot, mode: function () { return S.mode; },
    pure: { keyAction: keyAction, clock: clock, clockTime: clockTime, endsAt: endsAt, modeOf: modeOf, plexInfo: plexInfo, plexHeader: plexHeader, moveFocus: moveFocus,
      scrubStep: scrubStep, scrubTarget: scrubTarget, stepRow: stepRow, windowStart: windowStart, buttonsFor: buttonsFor, rootRows: rootRows, pickList: pickList,
      nearestLevel: nearestLevel, gearPath: gearPath }
  };
  if (window.KevFlixPlayerUI) { attach(window.KevFlixPlayerUI); }
}());
