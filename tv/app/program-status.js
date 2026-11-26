// KevFLIX program player: works out one line that says what the English dub is doing when it is not simply playing:
//   * this part plays with the film's own sound because its dub was not ready in time
//     (session status original_audio_now; player.js keeps the fields in window.KevFlixDub while a film is open);
//   * the dub is waiting for the PC's graphics card (GET /tv-api/producer: state 'preempted' or waiting_gpu true,
//     or session status dub_blocked);
//   * the dubbing producer failed or was stopped.
// The line is NEVER drawn on the TV (owner, 2026-11-25). Each new line is reported once to the PC's log
// (POST /tv-api/diagnostic, stage 'dub-notice'). Nothing is asked or reported outside a playback (window.KevFlixDub is null
// on the library screens).
(function () {
  'use strict';
  var POLL_MS = 3000, SHOW_MS = 1000, producer = null, shown = '', reported = '';
  function token() { try { return localStorage.getItem('sem-tv-token') || ''; } catch (ignore) { return ''; } }
  // Pure. dub: window.KevFlixDub ({session, original, blocked, fallback}) or null; value: the last /tv-api/producer reply or null.
  function text(dub, value) {
    if (!dub) { return ''; }
    if (dub.original === true) { return 'Original sound for now: the English dub of this part was not ready in time.'; }
    var state = value && value.state;
    if (state === 'failed') {
      return 'English dubbing stopped: the PC reported a fault' + (value.reason ? ' (' + String(value.reason).slice(0, 80) + ')' : '') + '.';
    }
    if ((dub.blocked && dub.blocked.code === 'gpu_hot') || (value && value.reason === 'gpu_hot')) {   // F28 heat guard
      return 'The graphics card is hot. Dubbing continues when it has cooled.';
    }
    if (dub.blocked || state === 'preempted' || (value && value.waiting_gpu === true)) {
      return 'English dubbing is waiting for the PC\'s graphics card. ' +
        (dub.fallback === true ? 'The film continues with its original sound meanwhile.' : 'Parts that are already dubbed keep playing.');
    }
    if (state === 'stopped') { return 'English dubbing is stopped on the PC.'; }
    return '';
  }
  // Owner, 2026-11-25: nothing is drawn over the film ("some stupid caption above the video that I NEVER ASKED FOR AND
  // DISTRACTS FROM THE VIDEO"). The line is only reported to the PC's log, once per new line, so the PC still knows
  // what the dub was doing while the film played.
  function report(line) {
    var bearer = token(), dub = window.KevFlixDub;
    if (!bearer || !dub || line === reported) { return; }
    reported = line;
    var body = { stage: 'dub-notice', message: line.slice(0, 160), session: String(dub.session || ''),
      probe: { drawn: false, original: dub.original === true, blocked: dub.blocked ? String(dub.blocked.code || 'yes') : '' } };
    fetch('/tv-api/diagnostic', { method: 'POST', cache: 'no-store', headers: { Authorization: 'Bearer ' + bearer, 'Content-Type': 'application/json' },
      body: JSON.stringify(body) }).catch(function () {});
  }
  function show(line) {
    if (line === shown) { return; }
    shown = line;
    if (!line) { reported = ''; return; }
    report(line);
  }
  function refresh() { show(text(window.KevFlixDub || null, producer)); }
  function poll() {
    var bearer = token();
    if (!bearer || !window.KevFlixDub) { producer = null; setTimeout(poll, POLL_MS); return; }   // nothing is asked while no film is open
    fetch('/tv-api/producer', { headers: { Authorization: 'Bearer ' + bearer }, cache: 'no-store' })
      .then(function (response) { return response.ok ? response.json() : null; })
      .then(function (value) { producer = value && value.state ? value : null; })
      .catch(function () { /* unreachable worker: the player shows its own connection errors */ })
      .then(function () { refresh(); setTimeout(poll, POLL_MS); });
  }
  window.KevFlixProgramStatus = { text: text };
  setInterval(refresh, SHOW_MS);
  poll();
}());
