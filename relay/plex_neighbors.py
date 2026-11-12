"""What the TV's Plex-style controls show about the item being played: show / season / episode / year and the previous and next episode.

`plex_info(plex, metadata)` is the `plex` block of a Plex playback session's status:
  {key, type: 'movie'|'episode', title, show, season, episode, year, prev: {key, title, season, episode}|None, next: ...|None}
The neighbours come from Plex itself (the season's episodes, then the adjacent season like Plex's own player does). Any Plex failure just
leaves them None: the buttons are hidden, playback is never affected."""
from __future__ import annotations
import json
import logging
import time


def _number(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _entry(item):
    return {"key": str(item["ratingKey"]), "title": str(item.get("title") or ""), "season": _number(item.get("parentIndex")),
            "episode": _number(item.get("index"))}


def _sorted(items):
    return sorted((item for item in items if item.get("ratingKey") is not None), key=lambda item: (_number(item.get("index")) or 0))


def neighbours(plex, metadata, timeout=None):
    """(previous, next) episode entries of the playing episode, each possibly None. timeout (s): passed to every Plex
    request (plex.children(key, timeout=...)); None keeps the client's own default."""
    def children(key):
        return plex.children(key) if timeout is None else plex.children(key, timeout=timeout)

    key, season_key, show_key = str(metadata["ratingKey"]), metadata.get("parentRatingKey"), metadata.get("grandparentRatingKey")
    episodes = _sorted(children(season_key)) if season_key else []
    position = next((i for i, item in enumerate(episodes) if str(item["ratingKey"]) == key), None)
    if position is None:
        return None, None
    previous = episodes[position - 1] if position > 0 else None
    following = episodes[position + 1] if position + 1 < len(episodes) else None
    if (previous is None or following is None) and show_key:
        seasons = [s for s in _sorted(children(show_key)) if s.get("type") in (None, "season")]
        current = next((i for i, s in enumerate(seasons) if str(s.get("ratingKey")) == str(season_key)), None)
        if current is not None:
            specials = _number(seasons[current].get("index")) == 0
            usable = [s for s in seasons if specials or _number(s.get("index")) != 0]     # Specials are not part of the run-through (like Plex)
            at = next((i for i, s in enumerate(usable) if str(s.get("ratingKey")) == str(season_key)), None)
            if previous is None and at:
                last = _sorted(children(usable[at - 1]["ratingKey"]))
                previous = last[-1] if last else None
            if following is None and at is not None and at + 1 < len(usable):
                first = _sorted(children(usable[at + 1]["ratingKey"]))
                following = first[0] if first else None
    return (_entry(previous) if previous else None), (_entry(following) if following else None)


def plex_info(plex, metadata, lookup=True):
    """lookup False: no Plex request at all (prev / next stay None until fill_neighbours() sets them)."""
    episode = metadata.get("type") == "episode"
    info = {"key": str(metadata["ratingKey"]), "type": "episode" if episode else "movie", "title": str(metadata.get("title") or ""),
            "year": _number(metadata.get("year")), "show": None, "season": None, "episode": None, "prev": None, "next": None}
    if episode:
        info.update(show=str(metadata.get("grandparentTitle") or "") or None, season=_number(metadata.get("parentIndex")),
                    episode=_number(metadata.get("index")))
        if lookup:
            fill_neighbours(plex, metadata, info)
    return info


def fill_neighbours(plex, metadata, info, timeout=None):
    """Set info["prev"] / info["next"] in place (the session's `plex` block; the next status carries them). Any Plex
    failure or time-out leaves them None: the buttons stay hidden, playback is never affected."""
    began = time.monotonic()
    try:
        info["prev"], info["next"] = neighbours(plex, metadata, timeout=timeout)
        logging.info("tv plex_neighbours %s", json.dumps({"key": info.get("key"), "prev": info["prev"] is not None,
                                                          "next": info["next"] is not None,
                                                          "ms": round((time.monotonic() - began) * 1000)}))
    except Exception as error:                          # Plex unreachable or an unusual library: no neighbour buttons
        logging.info("Plex neighbours unavailable (%s) after %d ms", str(error)[:120],
                     round((time.monotonic() - began) * 1000))
    return info
