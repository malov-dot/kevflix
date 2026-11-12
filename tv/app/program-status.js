// KevFLIX program player: one readable line at the top of the PLAYER view that says what the English dub is doing
// when it is not simply playing:
//   * this part plays with the film's own sound because its dub was not ready in time
//     (session status original_audio_now; player.js keeps the fields in window.KevFlixDub while a film is open);
//   * the dub is waiting for the PC's graphics card (GET /tv-api/producer: state 'preempted' or waiting_gpu true,
//     or session status dub_blocked);
//   * the dubbing producer failed or was stopped.
// The line is informational. It never hides the buffering card: when the picture really waits, the card says so and
// this line says why. Nothing is shown outside a playback (window.KevFlixDub is null on the library screens).
// Nobody has to look at the TV to know the line was readable: every new line is reported once to the PC's log
// (POST /tv-api/diagnostic, stage 'dub-notice') with the size it was drawn at; a line wider than the screen allows is
// drawn smaller first.
(function () {
  'use strict';
  var POLL_MS = 3000, SHOW_MS = 1000, FONT_PX = 28, SMALL_PX = 24, producer = null, notice = null, shown = '', reported = '';
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
  function font(px) { return '600 ' + px + 'px/1.3 Arial,Helvetica,sans-serif'; }
  function report(line, px) {
    var bearer = token(), dub = window.KevFlixDub;
    if (!bearer || !dub || line === reported) { return; }
    reported = line;
    var box = notice.getBoundingClientRect ? notice.getBoundingClientRect() : { width: 0, height: 0, top: 0, left: 0 };
    var body = { stage: 'dub-notice', message: line.slice(0, 160), session: String(dub.session || ''),
      probe: { width: Math.round(box.width), height: Math.round(box.height), top: Math.round(box.top), left: Math.round(box.left),
        font_px: px, clipped: notice.scrollWidth > notice.clientWidth, screen_w: window.innerWidth || 0, screen_h: window.innerHeight || 0,
        original: dub.original === true, blocked: dub.blocked ? String(dub.blocked.code || 'yes') : '' } };
    fetch('/tv-api/diagnostic', { method: 'POST', cache: 'no-store', headers: { Authorization: 'Bearer ' + bearer, 'Content-Type': 'application/json' },
      body: JSON.stringify(body) }).catch(function () {});
  }
  function show(line) {
    if (line === shown) { return; }
    shown = line;
    if (!line) { reported = ''; if (notice) { notice.hidden = true; } return; }
    if (!notice) {
      notice = document.createElement('div');
      notice.id = 'producer-notice';
      notice.setAttribute('role', 'status');
      var s = notice.style;                       // CSSOM only: the page's CSP has no inline styles
      s.position = 'fixed'; s.top = '0'; s.left = '50%'; s.transform = 'translateX(-50%)'; s.zIndex = '2147483647';
      s.padding = '6px 28px'; s.background = 'rgba(24,24,24,.94)'; s.color = '#fff';
      s.borderRadius = '0 0 12px 12px'; s.maxWidth = '90vw';
      s.whiteSpace = 'nowrap'; s.overflow = 'hidden'; s.textOverflow = 'ellipsis'; s.pointerEvents = 'none';
      document.body.appendChild(notice);
    }
    var px = FONT_PX;
    notice.style.font = font(px);
    notice.textContent = line;
    notice.hidden = false;
    if (notice.scrollWidth > notice.clientWidth) { px = SMALL_PX; notice.style.font = font(px); }   // too wide for 90 % of the screen: one size down
    report(line, px);
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
