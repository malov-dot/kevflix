"""Generate reviewable candidate files only. Never activate or edit the source runtime."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, help="Existing Sem app directory containing electron/")
    parser.add_argument("--output", required=True, help="New isolated candidate directory")
    args = parser.parse_args()
    source, output = Path(args.source).resolve(), Path(args.output).resolve()
    package = Path(__file__).resolve().parent
    if output == source or source in output.parents or output in source.parents:
        raise SystemExit("Candidate must be separate from the source runtime")
    if output.exists():
        raise SystemExit("Use a new candidate directory; existing work is preserved")
    before = {}
    for name in ("main.cjs", "conversation-provider-tools.cjs"):
        before[name] = (source / "electron" / name).read_bytes()
    main_text = before["main.cjs"].decode("utf-8-sig")
    tools_text = before["conversation-provider-tools.cjs"].decode("utf-8-sig")
    anchor = "async function executeReliableVoiceToolCore(call = {}, execution = {}) {"
    branch = '  if (call.name === "generate_picture") {'
    catalog = "const CONVERSATION_PROVIDER_TOOLS = Object.freeze(["
    if main_text.count(anchor) != 1 or main_text.count(branch) != 1 or tools_text.count(catalog) != 1:
        raise SystemExit("Sem source anchors changed; review the new source before staging")
    if "getMovieDubRelay" in main_text or "movie-dub-relay-client" in tools_text:
        raise SystemExit("Integration already exists; review it before updating")
    main_text = main_text.replace(anchor, package.joinpath("main-integration.txt").read_text() + "\n" + anchor)
    main_text = main_text.replace(branch, "  if (call.name === 'manage_movie_dub') {\n"
        "    verifiedWisdomToolContext(call, execution);\n"
        "    return getMovieDubRelay().tool(call.arguments || {}, execution);\n  }\n" + branch)
    tools_text = tools_text.replace(catalog, catalog + "\n  require('./movie-dub-relay-client.cjs').TOOL,")
    (output / "electron").mkdir(parents=True)
    files = {"main.cjs": main_text.encode(), "conversation-provider-tools.cjs": tools_text.encode(),
             "movie-dub-relay-client.cjs": (package / "movie-dub-relay-client.cjs").read_bytes()}
    for name, content in files.items():
        (output / "electron" / name).write_bytes(content)
    (output / "STAGING.json").write_text(json.dumps({
        "activated": False, "tests": "Not run; owner deferred", "source": str(source),
        "source_sha256": {n: hashlib.sha256(b).hexdigest() for n,b in before.items()},
        "candidate_sha256": {n: hashlib.sha256(b).hexdigest() for n,b in files.items()},
        "promotion": "Requires source rebase if live hashes changed, continuity comparison and owner ordinary-conversation acceptance. Never edit accepted hashes to bypass the guard."
    }, indent=2))
    print("Candidate files saved in " + str(output) + "; no runtime changed")


if __name__ == "__main__":
    main()
