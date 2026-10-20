// KevFLIX program player (W10): shows the dubbing producer's state in the player view.
// GET /tv-api/producer (the paired device's bearer token) returns the app's movie pipeline state, {state, reason}:
//   running   -> no notice
//   preempted -> "Dubbing paused: GPU reclaimed (<reason>)"  (it resumes by itself when the GPU lease returns)
//   failed    -> "Dubbing stopped: producer failed (<reason>)"
//   stopped   -> "Dubbing stopped"
//   finished  -> "Dubbing complete"
// While the state is not running, the release's buffering spinner is kept hidden, so the view never shows processing
// progress that is not happening. Already published segments stay playable; playback itself is not touched.
(function () {
  'use strict';
  var POLL_MS = 3000, stopped = false, notice = null;
  var buffering = document.getElementById('buffering');
  function token() { return localStorage.getItem('sem-tv-token') || ''; }
  function message(value) {
    var why = value.reason ? ' (' + value.reason + ')' : '';
    if (value.state === 'preempted') { return 'Dubbing paused: GPU reclaimed' + why; }
    if (value.state === 'failed') { return 'Dubbing stopped: producer failed' + (why || ' (no reason given)'); }
    if (value.state === 'finished') { return 'Dubbing complete'; }
    return 'Dubbing stopped';
  }
  function show(text) {
    if (!notice) {
      notice = document.createElement('div');
      notice.id = 'producer-notice';
      notice.setAttribute('role', 'alert');
      var s = notice.style;
      s.position = 'fixed'; s.top = '0'; s.left = '50%'; s.transform = 'translateX(-50%)'; s.zIndex = '2147483647';
      s.padding = '6px 16px'; s.background = '#7a1f00'; s.color = '#fff'; s.font = '600 15px/1.4 sans-serif';
      s.borderRadius = '0 0 6px 6px'; s.maxWidth = '90vw'; s.pointerEvents = 'none';
      document.body.appendChild(notice);
    }
    notice.textContent = text;
    notice.hidden = false;
  }
  function hide() { if (notice) { notice.hidden = true; } }
  if (buffering && typeof MutationObserver !== 'undefined') {
    new MutationObserver(function () {
      if (stopped && !buffering.hidden) { buffering.hidden = true; }
    }).observe(buffering, { attributes: true, attributeFilter: ['hidden'] });
  }
  function poll() {
    var bearer = token();
    if (!bearer) { setTimeout(poll, POLL_MS); return; }
    fetch('/tv-api/producer', { headers: { Authorization: 'Bearer ' + bearer }, cache: 'no-store' })
      .then(function (response) { return response.ok ? response.json() : null; })
      .then(function (value) {
        if (value && value.state && value.state !== 'running') {
          stopped = true;
          show(message(value) + (value.state === 'finished' ? '.' : '. Published parts stay playable.'));
          if (buffering) { buffering.hidden = true; }
        } else {
          stopped = false;
          hide();
        }
      })
      .catch(function () { /* unreachable worker: the player shows its own connection errors */ })
      .then(function () { setTimeout(poll, POLL_MS); });
  }
  poll();
}());
