"""YouTube things the TV's YouTube-style controls need from YouTube itself.

* up_next(video_id): the video YouTube would play next (autoplay), from the public watch-next data. No account involved.
* rate / rating: the signed-in viewer's like or dislike of a video. The PC's own YouTube page (headless Chrome, the profile the owner signed in to)
  makes the request from inside youtube.com/tv with the TV sign-in it already holds, exactly like the YouTube page does, so no password or token ever leaves it or is stored here.
Everything raises AccountError with a short message the TV can show; nothing here can affect playback."""
from __future__ import annotations
import json
import re
import urllib.request

INNERTUBE_KEY = "AIzaSyAO_FJ2SlqU8Q4STEHLGCilw_Y9_11qcW8"          # the public web client key every youtube.com page ships
CLIENT = {"clientName": "WEB", "clientVersion": "2.20240814.00.00", "hl": "en", "gl": "US"}
VIDEO_ID = re.compile(r"[A-Za-z0-9_-]{11}")
RATINGS = {"like": "like/like", "dislike": "like/dislike", "none": "like/removelike"}


class AccountError(RuntimeError):
    """The message is shown on the TV as is."""


def clean_video(value):
    value = str(value or "")
    if not VIDEO_ID.fullmatch(value):
        raise AccountError("That is not a YouTube video")
    return value


def _walk(node):
    """Every dict inside a decoded response, depth first."""
    stack = [node]
    while stack:
        item = stack.pop()
        if isinstance(item, dict):
            yield item
            stack.extend(item.values())
        elif isinstance(item, list):
            stack.extend(item)


def _text(value):
    if isinstance(value, dict):
        if value.get("simpleText"):
            return str(value["simpleText"])
        if value.get("content"):
            return str(value["content"])
        return "".join(str(run.get("text", "")) for run in value.get("runs") or [] if isinstance(run, dict))
    return str(value or "")


def find_up_next(data, current):
    """{id, title} of the autoplay video in a watch-next response, or None. The autoplay set names it; its title is looked up among the
    related videos of the same response (both the older videoRenderer and the newer lockupViewModel layouts)."""
    target = ""
    autoplay = ((data.get("contents") or {}).get("twoColumnWatchNextResults") or {}).get("autoplay") if isinstance(data, dict) else None
    for node in _walk(autoplay or {}):
        endpoint = node.get("watchEndpoint")
        if isinstance(endpoint, dict) and VIDEO_ID.fullmatch(str(endpoint.get("videoId") or "")) and endpoint["videoId"] != current:
            target = endpoint["videoId"]
            break
    if not target:
        return None
    title = ""
    rest = dict(data)                                               # the autoplay block only carries the heading 'Up next': look elsewhere for the title
    contents = dict(rest.get("contents") or {})
    two = dict(contents.get("twoColumnWatchNextResults") or {})
    two.pop("autoplay", None)
    contents["twoColumnWatchNextResults"] = two
    rest["contents"] = contents
    for node in _walk(rest):
        if node.get("videoId") == target or node.get("contentId") == target:
            title = _text(node.get("title")) or _text((((node.get("metadata") or {}).get("lockupMetadataViewModel") or {}).get("title")))
            if title.strip().lower() in ("", "up next", "autoplay"):          # headings of the autoplay block, not the video's own title
                title = ""
                continue
            break
    return {"id": target, "title": title[:200]}


def up_next(video_id, opener=urllib.request.urlopen, timeout=10):
    video_id = clean_video(video_id)
    body = json.dumps({"context": {"client": CLIENT}, "videoId": video_id}).encode()
    request = urllib.request.Request(
        "https://www.youtube.com/youtubei/v1/next?prettyPrint=false&key=" + INNERTUBE_KEY, body,
        {"Content-Type": "application/json", "Origin": "https://www.youtube.com", "X-Origin": "https://www.youtube.com",
         "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"})
    try:
        with opener(request, timeout=timeout) as response:
            return find_up_next(json.loads(response.read().decode("utf-8", "replace")), video_id)
    except (OSError, ValueError):
        raise AccountError("YouTube did not answer")


# ---- the signed-in viewer's rating, through the PC's own YouTube page --------------------------------------------------------------------
# One expression, evaluated inside youtube.com/tv: the TV page is signed in with an OAuth token (not web cookies) that it keeps in its own
# storage; the expression sends the request with that token, from inside the page, exactly as the page's own code does, and returns
# {signed_in, rating?, ok?, status?}. The token never leaves the page. `mode` is 'read' (the watch-next data, which carries the viewer's like state) or an action.
PAGE_JS = r"""(async function (videoId, mode, key, fallbackClient) {
  var tokens = null, entry = null;
  try { tokens = JSON.parse(localStorage.getItem('yt.leanback.default::cached-access-tokens')).data; } catch (e) { tokens = null; }
  Object.keys(tokens || {}).forEach(function (name) {
    var item = tokens[name];
    if (item && typeof item.accessToken === 'string' && item.accessToken && (!entry || (item.accessTokenServerExpiration || 0) > (entry.accessTokenServerExpiration || 0))) entry = item;
  });
  if (!entry) return { signed_in: false };
  var config = (window.ytcfg && ytcfg.data_) || {};
  var context = config.INNERTUBE_CONTEXT || { client: fallbackClient };
  var path = mode === 'read' ? 'next' : mode;
  var body = mode === 'read' ? { context: context, videoId: videoId } : { context: context, target: { videoId: videoId } };
  var response = await fetch('/youtubei/v1/' + path + '?prettyPrint=false&key=' + encodeURIComponent(config.INNERTUBE_API_KEY || key), {
    method: 'POST', headers: { 'Content-Type': 'application/json', 'Authorization': 'Bearer ' + entry.accessToken }, body: JSON.stringify(body) });
  var text = await response.text();
  if (!response.ok) return { signed_in: response.status !== 401, ok: false, status: response.status };
  if (mode === 'read') {
    var m = text.match(/"likeStatus"\s*:\s*"(LIKE|DISLIKE|INDIFFERENT)"/);
    return { signed_in: true, ok: true, rating: m ? (m[1] === 'LIKE' ? 'like' : m[1] === 'DISLIKE' ? 'dislike' : 'none') : null };
  }
  return { signed_in: true, ok: true };
})"""


def page_expression(video_id, mode):
    return "%s(%s, %s, %s, %s)" % (PAGE_JS, json.dumps(clean_video(video_id)), json.dumps(mode), json.dumps(INNERTUBE_KEY), json.dumps(CLIENT))


def _run(host, video_id, mode):
    if host is None or getattr(host, "state", "") != "running":
        raise AccountError("YouTube is not open on the PC")
    try:
        result = host.evaluate(page_expression(video_id, mode))
    except RuntimeError as error:
        raise AccountError(str(error) or "YouTube did not answer")
    if not isinstance(result, dict):
        raise AccountError("YouTube did not answer")
    return result


def rating(host, video_id):
    """{signed_in, rating: 'like'|'dislike'|'none'|None}. Read-only."""
    result = _run(host, video_id, "read")
    return {"signed_in": bool(result.get("signed_in")), "rating": result.get("rating") if result.get("ok") else None}


def rate(host, video_id, value):
    """Like, dislike or clear the viewer's rating of the video. Returns {signed_in, rating}."""
    if value not in RATINGS:
        raise AccountError("Choose like, dislike or none")
    result = _run(host, video_id, RATINGS[value])
    if not result.get("signed_in"):
        raise AccountError("YouTube on the PC is not signed in")
    if not result.get("ok"):
        raise AccountError("YouTube did not accept the rating (%s)" % result.get("status", "error"))
    return {"signed_in": True, "rating": value}
