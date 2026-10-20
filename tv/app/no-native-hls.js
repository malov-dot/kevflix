// KevFLIX program player (W10): page shim loaded by tv/app/index.html BEFORE the unmodified 6c0dd972 player.js.
// Desktop Edge reports canPlayType('application/vnd.apple.mpegurl') === 'maybe' and routes the relay's EVENT playlist to
// its built-in HLS demuxer, which has been recorded failing on live/EVENT playlists. Answering '' for the HLS MIME types
// only makes the player's own decoder take its existing fallback branch (vendored hls.js 1.7.3 over MSE). Every other
// canPlayType answer is the browser's own. player.js and player.css are the 6c0dd972 release bytes.
(function () {
  'use strict';
  // TVs (Samsung Tizen, LG webOS, other smart-TV browsers) keep their native HLS engine.
  if (/Tizen|SMART-TV|SmartTV|Web0S|webOS|NetCast/i.test(navigator.userAgent || '')) { return; }
  var HLS = /^\s*(application\/(vnd\.apple\.mpegurl|x-mpegurl)|audio\/(x-)?mpegurl)\s*(;|$)/i;
  var P = HTMLMediaElement.prototype, original = P.canPlayType;
  Object.defineProperty(P, 'canPlayType', {
    configurable: true, writable: true,
    value: function canPlayType(type) { return HLS.test(String(type)) ? '' : original.call(this, type); }
  });
  Object.defineProperty(window, '__kevflixTrialNoNativeHls', { value: true });
}());
