"""Seamless HLS segment joins for TVService.render() (pure, stdlib only; config key tv.seamless_joins).

Every segment is one ffmpeg run. Encoded independently, each AAC stream starts with a priming packet (1024 samples before
its first real sample) and ends with a zero-padded frame, so a player that decodes the segments in sequence gets 1,792
samples (37.3 ms) of silence at every 30 s join and overlapping timestamps. Here the audio of all segments lies on ONE
global grid of 1024-sample AAC frames:

    S     = segment_seconds * rate                      samples per segment
    B(n)  = (n * S // 1024) * 1024                      grid point at or before the start of segment n
    d(n)  = n * S - B(n)                                0, 256, 512 or 768 at 30 s / 48 kHz
    tail  = pre * 1024 + d(n)                           samples fed BEFORE the segment start (from the previous window)
    keep  = S - d(n + 1)                                samples fed from the segment's own window (last segment: all)
    count = (B(n + 1) - B(n)) // 1024                   packets kept

The encoder is fed [n*S - tail, n*S + keep), i.e. `pre` whole frames of real pre-roll plus the segment up to the next grid
point. The first `pre` packets (the encoder's priming packet and pre - 1 pre-roll packets) and the one packet after the
kept range are dropped in the bitstream (noise bsf), so the kept packets carry PTS B(n) - 1024 ... B(n + 1) - 2048:
segment n + 1 continues at B(n + 1) - 1024 with no gap, no overlap and no priming. The MDCT block of the first kept packet
spans [B(n) - 1024, B(n) + 1024): its first half is the same real audio the previous segment's last packet encoded, so the
decoder's overlap-add across the join cancels (measured: 8-10 LSB error at the join, the same as mid-segment; with
pre = 2 the join frame was wrong, 2,226 of 2,896).
"""
from __future__ import annotations

RATE = 48000
FRAME = 1024             # AAC-LC frame length
PRE_FRAMES = 4           # frames of real pre-roll that are encoded and dropped
TS_BASE_SECONDS = 10     # constant PTS base of every segment: no segment ever needs the muxer's negative-timestamp shift
TS_LEAD_SECONDS = 12.0   # TS-like sources: seek this much early and trim (an input seek lands mid-GOP there)
SOURCE_MARGIN = 12000    # samples (0.25 s) decoded before the first wanted sample of a seeked source audio input
TS_LIKE_FORMATS = ("mpegts", "mpeg", "mpegtsraw", "vob")
MUX_VERSION = {"software": "tv-mux-cpu-v5", "nvenc": "tv-mux-nvenc-v2"}


# The TV's own report of three joins (tv/app/player.js SemJoinWatch, diagnostic stage "joins"): the most each number may be.
JOIN_LIMITS = {"waiting": 0, "hls_stalled": 0, "hls_nudge": 0, "hls_hole": 0, "gap_ms": 50, "jump_ms": 150, "stall_ms": 150,
               "dropped": 6}
JOIN_DRIFT_MS = 150      # |media time advanced - wall time passed| over the whole watch
JOINS_WATCHED = 3


def joins_verdict(report):
    """(ok, reasons) for one "joins" report; ok is None when the report does not cover three joins.
    A number the player could not supply (None) is not judged."""
    if not isinstance(report, dict):
        return None, ["no report"]
    joins = report.get("joins")
    if isinstance(joins, bool) or not isinstance(joins, (int, float)) or joins < JOINS_WATCHED:
        return None, ["fewer than %d joins" % JOINS_WATCHED]
    reasons = []
    for key, limit in JOIN_LIMITS.items():
        value = report.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool) and value > limit:
            reasons.append("%s %s > %s" % (key, value, limit))
    drift = report.get("drift_ms")
    if isinstance(drift, (int, float)) and not isinstance(drift, bool) and abs(drift) > JOIN_DRIFT_MS:
        reasons.append("drift_ms %s beyond +-%s" % (drift, JOIN_DRIFT_MS))
    return not reasons, reasons


def ts_like(format_name):
    """True for containers whose input seek is not keyframe-exact (MPEG-TS / MPEG-PS): ffprobe format_name is a comma list."""
    return any(name in TS_LIKE_FORMATS for name in str(format_name or "").lower().split(","))


def audio_join_plan(index, total, segment_seconds, duration, rate=RATE, frame=FRAME, pre=PRE_FRAMES):
    """The sample plan of segment `index` of `total` (pure). All values are samples at `rate` unless named otherwise.

    first / last   segment 0 / the final (possibly short) segment
    tail           samples fed before the segment start (0 for segment 0)
    keep           samples fed from the segment's own window
    head_packets   leading packets dropped (0 for segment 0: its priming packet is the true start of the stream)
    kept_packets   packets kept after them (None for the last segment: everything to the end is kept)
    pts_shift      PTS of the first fed sample relative to the segment start (= -tail)
    """
    index, total = int(index), max(1, int(total))
    seg = int(segment_seconds) * int(rate)

    def grid(n):
        return (n * seg // frame) * frame

    first, last = index <= 0, index >= total - 1
    start = index * seg
    offset = start - grid(index)
    tail = 0 if first else pre * frame + offset
    if last:
        keep, kept = max(0, int(round(float(duration) * rate)) - start), None
    else:
        keep = seg - ((index + 1) * seg - grid(index + 1))
        kept = (grid(index + 1) - grid(index)) // frame
    return {"index": index, "first": first, "last": last, "rate": int(rate), "frame": int(frame), "pre": int(pre),
            "segment_samples": seg, "grid": grid(index), "offset": offset, "tail": tail, "keep": keep,
            "head_packets": 0 if first else int(pre), "kept_packets": kept, "pts_shift": -tail}


def drop_expression(plan):
    """noise bsf `drop` expression (None when nothing is dropped): leading pre-roll packets and the one after the kept range."""
    parts = []
    if plan["head_packets"]:
        parts.append("lt(n\\,%d)" % plan["head_packets"])
    if plan["kept_packets"] is not None:
        parts.append("gte(n\\,%d)" % (plan["head_packets"] + plan["kept_packets"]))
    return "+".join(parts) or None


def audio_graph(plan, pre, main, *, start, first_input, layout, gain=""):
    """(input arguments, filter_complex audio part ending in [a]) for one segment.

    pre / main: where the audio before the segment start / of the segment comes from:
        {"kind": "file", "path": <wav or flac of one whole window>, "prefix": <filters or "">}
        {"kind": "source", "path": <the movie>, "stream": <absolute stream index>, "prefix": <filters or "">}
    pre is None for segment 0. `prefix` are that window's own layout filters (SURROUND_REAR_OFF / SURROUND_UPMIX), without a
    trailing comma. layout: "stereo" or "5.1", the layout every branch is brought to before the join. gain: the job's
    loudness filter ("" for none), applied once after the join. start: segment start in seconds. first_input: index of
    the first audio input (1: the movie is input 0)."""
    rate, tail, keep = plan["rate"], plan["tail"], plan["keep"]
    inputs, chains = [], []

    def branch(spec, first_sample, samples):
        number = first_input + len(chains)
        if spec["kind"] == "source":
            want = int(round(start * rate)) + first_sample           # first wanted sample on the movie's timeline
            # The seek target is a whole millisecond: ffmpeg starts a seeked Matroska input on the millisecond below the
            # target (measured: 29.659333 s started 16 samples early, 44.123456 s 22 samples), which would shift this run
            # against its neighbour. Never "-ss 0": it dropped the first audio frame (1,536 samples) of the test MKV.
            seek_ms = max(0, (want - SOURCE_MARGIN) * 1000 // rate)
            margin = want - seek_ms * rate // 1000
            if seek_ms > 0:
                inputs.extend(["-ss", "%d.%03d" % divmod(seek_ms, 1000)])
            inputs.extend(["-i", str(spec["path"])])
            chain = "[%d:%d]aresample=%d,apad,atrim=start_sample=%d:end_sample=%d" % (
                number, int(spec["stream"]), rate, margin, margin + samples)
        else:
            inputs.extend(["-i", str(spec["path"])])
            if first_sample < 0:                                      # the end of the previous window's file
                chain = "[%d:a:0]aresample=%d,atrim=start_sample=%d" % (number, rate, plan["segment_samples"] + first_sample)
            else:
                chain = "[%d:a:0]aresample=%d,apad,atrim=end_sample=%d" % (number, rate, samples)
        if spec.get("prefix"):
            chain += "," + spec["prefix"]
        chains.append(chain + ",aformat=sample_rates=%d:channel_layouts=%s" % (rate, layout))

    if pre is None or tail == 0:
        branch(main, 0, keep)
    elif pre["kind"] == "source" and main["kind"] == "source" and (pre.get("prefix") or "") == (main.get("prefix") or ""):
        branch(main, -tail, tail + keep)                              # one continuous decode of the movie's own audio
    else:
        branch(pre, -tail, tail)
        branch(main, 0, keep)
    stamp = (gain + "," if gain else "") + "asetpts=(N-%d)/SR/TB[a]" % tail
    if len(chains) == 1:
        return inputs, chains[0] + "," + stamp
    return inputs, "%s[j0];%s[j1];[j0][j1]concat=n=2:v=0:a=1,%s" % (chains[0], chains[1], stamp)


def video_graph(stream, scale, duration, lead=0.0):
    """filter_complex video part ending in [v]: the frames whose source time is in [start, start + duration).
    lead > 0 (TS-like source): the input was opened `lead` seconds early."""
    if lead > 0:
        trim = "trim=start=%.6f:end=%.6f,setpts=PTS-%.6f/TB" % (lead, lead + duration, lead)
    else:
        trim = "trim=end=%.6f" % duration
    return "[0:%d]%s,%s[v]" % (int(stream), trim, scale)


def mux_arguments(*, ffmpeg, source, video_stream, start, duration, scale, codec, codec_options, threads, pix_fmt,
                  maxrate, bufsize, color, audio_rate, channel_args, plan, pre, main, layout, gain, ts_source, output,
                  input_options=(), pixel=None, container_args=None):
    """The complete ffmpeg command of one seamless segment (MPEG-TS unless container_args says otherwise).
    F19: input_options (hardware decode / filter devices) go before the picture input, the only input they apply to;
    pixel replaces ["-pix_fmt", pix_fmt]; container_args replaces the MPEG-TS tail (fMP4). The defaults give F07's list."""
    lead = min(TS_LEAD_SECONDS, float(start)) if ts_source else 0.0
    audio_inputs, audio = audio_graph(plan, pre, main, start=float(start), first_input=1, layout=layout, gain=gain)
    drop = drop_expression(plan)
    args = [ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", *input_options,
            "-ss", "%.6f" % (float(start) - lead), "-i", str(source), *audio_inputs,
            "-filter_complex", video_graph(video_stream, scale, float(duration), lead) + ";" + audio,
            "-map", "[v]", "-map", "[a]", "-enc_time_base:v", "1:90000",
            "-c:v", codec, *codec_options,
            "-threads", str(threads), "-filter_complex_threads", str(min(2, int(threads))),
            *(pixel if pixel is not None else ["-pix_fmt", pix_fmt]), "-maxrate", str(maxrate), "-bufsize", str(bufsize), *color,
            "-force_key_frames", "expr:gte(t,n_forced*2)",
            "-c:a", "aac", "-b:a", str(audio_rate), "-ar", str(plan["rate"]), *channel_args]
    if drop:
        args += ["-bsf:a", "noise=drop='%s'" % drop]
    if container_args is None:
        container_args = ["-output_ts_offset", "%.6f" % (float(start) + TS_BASE_SECONDS),
                          "-muxdelay", "0", "-muxpreload", "0", "-mpegts_flags", "+resend_headers+initial_discontinuity",
                          "-f", "mpegts"]
    args += ["-sn", "-dn", *container_args, "-y", str(output)]
    return args
