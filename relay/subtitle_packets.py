"""Read bounded embedded SubRip cues with their original container timestamps."""
from __future__ import annotations
import json,math,re


def decode_packet_text(dump):
    chunks=[];size=0;ended=False
    for row in str(dump).splitlines():
        if not row.strip():continue
        if not re.match(r'^[0-9a-fA-F]{8}: ',row):raise ValueError('Unexpected ffprobe data dump')
        column=row[10:49]
        if not re.fullmatch(r'[0-9a-fA-F ]*',column):raise ValueError('Invalid subtitle packet hex')
        # Each row's offset must continue the bytes already read, so duplicated,
        # reordered or missing dump rows cannot silently corrupt cue text.
        if ended or int(row[:8],16)!=size:raise ValueError('Subtitle packet dump rows are not contiguous')
        chunk=bytes.fromhex(column);size+=len(chunk);ended=len(chunk)<16
        if size>65536:raise ValueError('Subtitle cue exceeds bounded size')
        chunks.append(chunk)
    raw=b''.join(chunks)
    text=raw.decode('utf-8-sig').replace('\r\n','\n').strip()
    return re.sub(r'</?(?:i|b|u|font)(?:\s[^>]*)?>','',text,flags=re.I)


def embedded_subrip_window(config,owner,source,stream,begin,end,*,rejected=None):
    """Return cues with original container timestamps (never clipped/rebased).

    Packets without a positive duration or without text are not turned into
    cues; when ``rejected`` is a list, each is recorded there so a missing
    subtitle line is visible to the caller instead of silently lost."""
    if stream.get('codec_type')!='subtitle' or stream.get('codec_name')!='subrip':
        raise ValueError('Bounded packet reader supports declared SubRip only')
    if not all(math.isfinite(v) for v in (begin,end)) or not 0<=begin<end or end-begin>120:
        raise ValueError('Subtitle packet window must be bounded to120seconds')
    output=owner.run([config['ffprobe'],'-v','error','-read_intervals',f'{begin:.6f}%+{end-begin:.6f}',
        '-select_streams',str(stream['index']),'-show_packets','-show_data',
        '-show_entries','packet=pts_time,duration_time,data','-of','json',source],timeout=90)
    packets=json.loads(output)['packets'];cues=[]
    if len(packets)>2000:raise ValueError('Unexpected subtitle packet count')
    for packet in packets:
        start=float(packet['pts_time']);duration=float(packet.get('duration_time',0))
        if not math.isfinite(start) or not math.isfinite(duration) or duration<=0:
            if rejected is not None:rejected.append({'pts_time':start,'duration_time':duration,
                'reason':'nonpositive_or_missing_duration'})
            continue
        if start+duration<=begin or start>=end:continue
        text=decode_packet_text(packet.get('data',''))
        if not text and rejected is not None:
            rejected.append({'pts_time':start,'duration_time':duration,'reason':'empty_text'})
        if text:cues.append({'start':start,'end':start+duration,'text':text,'actor':None,
            'language':stream.get('tags',{}).get('language'),'source':'embedded-subrip-packet',
            'stream_index':stream['index']})
    return sorted(cues,key=lambda c:(c['start'],c['end']))
