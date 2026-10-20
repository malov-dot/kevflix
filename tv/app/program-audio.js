// KevFLIX program player (W10): audible playback for the PROGRAM view of the 6c0dd972 player.
// The release player.js (unchanged bytes) starts the <video> muted (player.js:118, a PC-proof rule) so its automatic
// resume can start without a gesture. In the program view the viewer's own Play action makes playback audible: on that
// user gesture (click / Enter / Space on Play, Pause/Play, Restart) the saved sound choice is applied, which is
// unmuted at full volume unless the viewer chose otherwise. Later choices (the Sound button, or any volume/mute change
// of the element) are remembered in localStorage 'kevflix-audio' = {muted, volume}.
(function () {
  'use strict';
  var KEY = 'kevflix-audio', LEVELS = [1, 0.75, 0.5, 0.25];
  var video = document.getElementById('video');
  var button = document.getElementById('sound');
  if (!video) { return; }
  var applying = false, gestured = false;
  function load() {
    try {
      var saved = JSON.parse(localStorage.getItem(KEY) || '{}');
      var volume = typeof saved.volume === 'number' && saved.volume >= 0 && saved.volume <= 1 ? saved.volume : 1;
      return { muted: saved.muted === true, volume: volume };
    } catch (ignore) { return { muted: false, volume: 1 }; }
  }
  function save(prefs) { try { localStorage.setItem(KEY, JSON.stringify(prefs)); } catch (ignore) {} }
  function label() {
    if (!button) { return; }
    button.textContent = video.muted ? 'Sound: off' : 'Sound: ' + Math.round(video.volume * 100) + '%';
  }
  function apply() {
    var prefs = load();
    applying = true;
    video.defaultMuted = false;
    video.volume = prefs.volume;
    video.muted = prefs.muted;
    applying = false;
    gestured = true;
    label();
  }
  // Capture phase: runs inside the same user activation, before the release's own click handler calls video.play().
  ['play-pause', 'detail-play', 'detail-restart'].forEach(function (id) {
    var element = document.getElementById(id);
    if (element) { element.addEventListener('click', apply, true); }
  });
  if (button) {
    button.addEventListener('click', function () {
      var prefs = load();
      if (!gestured) { apply(); prefs = load(); }
      if (prefs.muted) { prefs = { muted: false, volume: LEVELS[0] }; }
      else {
        var index = LEVELS.indexOf(prefs.volume);
        prefs = index >= 0 && index < LEVELS.length - 1 ? { muted: false, volume: LEVELS[index + 1] }
          : index === LEVELS.length - 1 ? { muted: true, volume: prefs.volume } : { muted: false, volume: LEVELS[0] };
      }
      save(prefs);
      applying = true;
      video.defaultMuted = false;
      video.volume = prefs.volume;
      video.muted = prefs.muted;
      applying = false;
      label();
    });
  }
  // Remember volume/mute changes made any other way (media keys, browser controls) once the viewer has interacted.
  video.addEventListener('volumechange', function () {
    if (!applying && gestured) { save({ muted: video.muted, volume: video.volume }); }
    label();
  });
  label();
}());
