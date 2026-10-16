"""Read-only Plex library navigation. Metadata and ordering remain Plex-owned."""
from __future__ import annotations

import threading
import time
from urllib.parse import parse_qsl, urlsplit


def array(value):
    return value if isinstance(value, list) else [value] if isinstance(value, dict) else []


class PlexBrowse:
    def __init__(self, plex):
        self.plex = plex
        self.cache = {}
        self.lock = threading.Lock()

    def cached(self, key, fetch):
        with self.lock:
            saved = self.cache.get(key)
        if saved and time.monotonic() - saved[0] < 30:
            return saved[1]
        value = fetch()
        with self.lock:
            if len(self.cache) > 80:
                self.cache.clear()
            self.cache[key] = (time.monotonic(), value)
        return value

    def sections(self):
        return [{"id": str(s["key"]), "title": s["title"], "type": s.get("type", "")}
                for s in self.plex.sections()]

    def details(self, section):
        if not str(section).isdigit():
            raise ValueError("Choose a Plex library")
        def fetch():
            container = self.plex.call(f"/library/sections/{section}", {"includeDetails": 1})["MediaContainer"]
            types = array(container.get("Type"))
            selected = next((t for t in types if t.get("active") in (True, 1, "1")), None)
            if not selected:
                sections = self.sections()
                kind = next((s["type"] for s in sections if s["id"] == str(section)), "")
                selected = next((t for t in types if t.get("type") == kind), None)
            if not selected:
                raise ValueError("Plex did not return browsing details for this library")
            return selected
        return self.cached(("details", str(section)), fetch)

    def key(self, section, key, fallback):
        # Server-supplied browse keys may contain a query; only this section is allowed.
        parsed = urlsplit(str(key or fallback))
        prefix = f"/library/sections/{section}/"
        path = parsed.path if parsed.path.startswith("/") else prefix + parsed.path
        if parsed.scheme or parsed.netloc or not path.startswith(prefix) or ".." in path:
            raise ValueError("Unexpected Plex library key")
        return path, dict(parse_qsl(parsed.query))

    def options(self, section, field):
        if field not in ("genre", "year", "collection"):
            raise ValueError("Unknown library filter")
        details = self.details(section)
        entry = next((f for f in array(details.get("Filter"))
                      if str(f.get("filter", "")).split(".")[-1] == field), None)
        if entry is None:
            return []
        path, values = self.key(section, entry.get("key"), field)
        response = self.plex.call(path, values)
        return [{"value": str(d["key"]), "label": str(d["title"])}
                for d in self.plex.entries(response, "Directory")]

    def parameters(self, section, query="", genre="", year="", collection="", sort="titleSort:asc"):
        allowed_sorts = ("titleSort:asc", "titleSort:desc", "year:desc", "year:asc",
                         "addedAt:desc", "addedAt:asc", "duration:desc", "rating:desc")
        if sort not in allowed_sorts:
            raise ValueError("Unknown library sort")
        details = self.details(section)
        path, values = self.key(section, details.get("key"), "all")
        values["sort"] = sort
        if query:
            values["title"] = str(query)[:200]
        for field, value in (("genre", genre), ("year", year), ("collection", collection)):
            if value:
                if len(str(value)) > 300:
                    raise ValueError("Library filter is too long")
                entry = next((f for f in array(details.get("Filter"))
                              if str(f.get("filter", "")).split(".")[-1] == field), None)
                if not entry:
                    raise ValueError("This library does not support the selected filter")
                values[str(entry["filter"])] = str(value)
        return path, values, details

    @staticmethod
    def item(item):
        kind = item.get("type", "")
        container = kind in ("show", "season", "artist", "album", "photoalbum", "collection")
        return {"id": str(item["ratingKey"]), "title": item.get("title", ""),
                "year": item.get("year"), "summary": item.get("summary", ""), "type": kind,
                "resume": float(item.get("viewOffset", 0)) / 1000,
                "duration": float(item.get("duration", 0)) / 1000,
                "watched": int(item.get("viewCount", 0)) > 0,
                "unwatched_count": max(0, int(item.get("leafCount", 0)) - int(item.get("viewedLeafCount", 0))),
                "folder": False, "container": container, "playable": kind in ("movie", "episode", "clip"),
                "plex": True, "poster": bool(item.get("thumb"))}

    def metadata(self, key):
        item = self.plex.metadata(key)
        result = self.item(item)
        result.update({name: item.get(name) for name in (
            "tagline", "originalTitle", "studio", "contentRating", "rating", "audienceRating",
            "originallyAvailableAt", "editionTitle", "grandparentTitle", "parentTitle", "index", "parentIndex")})
        result["genres"] = [g.get("tag", "") for g in array(item.get("Genre"))]
        result["directors"] = [g.get("tag", "") for g in array(item.get("Director"))]
        result["writers"] = [g.get("tag", "") for g in array(item.get("Writer"))]
        result["cast"] = [{"name": g.get("tag", ""), "role": g.get("role", "")} for g in array(item.get("Role"))]
        result["media"] = [{k: m.get(k) for k in ("videoResolution", "width", "height", "videoCodec", "audioCodec", "audioChannels", "container")}
                           for m in array(item.get("Media"))]
        return result

    def continuing(self, section):
        if not str(section).isdigit():
            raise ValueError("Choose a Plex library")
        result = self.plex.call(f"/hubs/sections/{section}/continueWatching/items",
                               {"X-Plex-Container-Start": 0, "X-Plex-Container-Size": 10})
        return [self.item(item) for item in self.plex.entries(result, "Metadata") if item.get("ratingKey")]

    def watched(self, key, enabled):
        if not str(key).isdigit():
            raise ValueError("Invalid Plex item")
        self.plex.call("/:/scrobble" if enabled else "/:/unscrobble",
                       {"key": str(key), "identifier": "com.plexapp.plugins.library"})
        return {"watched": bool(enabled)}

    def page(self, section, query="", offset=0, genre="", year="", collection="",
             sort="titleSort:asc", parent=""):
        path, values, details = self.parameters(section, query, genre, year, collection, sort)
        if parent:
            if not str(parent).isdigit():
                raise ValueError("Invalid Plex parent")
            metadata = self.plex.metadata(parent)
            if str(metadata.get("librarySectionID")) != str(section):
                raise ValueError("This item belongs to another Plex library")
            if metadata.get("type") not in ("show", "season", "artist", "album", "photoalbum", "collection"):
                raise ValueError("This Plex item is not a container")
            path = f"/library/metadata/{parent}/children"
            values = {"sort": "titleSort:asc" if metadata.get("type") in ("artist", "collection") else "index:asc"}
        offset = max(0, int(offset))
        values.update({"X-Plex-Container-Start": offset, "X-Plex-Container-Size": 100})
        result = self.plex.call(path, values)
        container = result.get("MediaContainer", {})
        items = self.plex.entries(result, "Metadata")
        count = int(container.get("size", len(items)))
        total = container.get("totalSize")
        filters = [str(f.get("filter", "")).split(".")[-1] for f in array(details.get("Filter"))]
        return {"items": [self.item(item) for item in items if item.get("ratingKey")],
                "offset": offset, "next_offset": offset + count,
                "more": offset + count < int(total) if total is not None else count == 100,
                "total": int(total) if total is not None else None,
                "filters": [f for f in filters if f in ("genre", "year", "collection")],
                "sorts": [str(s.get("key", "")).split(":")[0].split(".")[-1] for s in array(details.get("Sort"))]}

    def alphabet(self, section, query="", genre="", year="", collection=""):
        _, values, _ = self.parameters(section, query, genre, year, collection)
        result = self.plex.call(f"/library/sections/{section}/firstCharacter", values)
        offset, letters = 0, []
        for entry in self.plex.entries(result, "Directory"):
            count = int(entry.get("size", 0))
            if count:
                letters.append({"label": str(entry.get("title", "#")), "offset": offset, "count": count})
                offset += count
        return letters
