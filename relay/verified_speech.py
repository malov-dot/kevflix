"""Verify a bounded actor take before it can enter a playback buffer."""
from __future__ import annotations
from collections import Counter
import re,time
from .language import word_error,words
from .media import file_hash


def critical_readback_changes(expected,heard):
    def tokens(text):
        text=str(text).lower().replace('’',"'")
        text=re.sub(r"\bcan't\b",'cannot',text)
        text=re.sub(r"\bcannot\b",'can not',text)
        text=re.sub(r"\bwon't\b",'will not',text)
        text=re.sub(r"n't\b",' not',text)
        return words(text)
    a,b=tokens(expected),tokens(heard)
    # 'nope', 'nah' and the recognizer's 'nu' (Russian ну) for 'no' are the same negation, not a new one
    negatives={'not','no','never','cannot','without','nothing','nobody','neither','nor','nope','nah','nu'}
    issues=[]
    def negation(ts):return Counter('no' if w in ('nope','nah','nu') else w for w in ts if w in negatives)
    if negation(a)!=negation(b):issues.append('negation_changed')
    names=('zero','one','two','three','four','five','six','seven','eight','nine','ten','eleven','twelve','thirteen','fourteen','fifteen','sixteen','seventeen','eighteen','nineteen','twenty')
    numbers={name:str(i) for i,name in enumerate(names)}
    numbers.update({'thirty':'30','forty':'40','fifty':'50','sixty':'60','seventy':'70','eighty':'80','ninety':'90','hundred':'100','thousand':'1000'})
    def numeral(w):
        if w in numbers:return numbers[w]
        if w.endswith('s') and w[:-1] in numbers:return numbers[w[:-1]]    # "ones", "twenties": the same number
        return w if w.isdecimal() else None
    def numeric(ts):return [n for n in map(numeral,ts) if n is not None]
    if numeric(a)!=numeric(b):issues.append('number_changed')
    return issues


def check_take(language,line,expected):
    started=time.perf_counter();beam=5 if line['seconds']<2 else 1
    heard,_=language.speech(line['audio'],'en',vad=line['seconds']>=2,beam_size=beam)
    text=' '.join(s['text'] for s in heard)
    error=word_error(expected,text);critical=critical_readback_changes(expected,text)
    if beam==1 and (error>.25 or critical):
        beam=5;heard,_=language.speech(line['audio'],'en',vad=True,beam_size=beam)
        text=' '.join(s['text'] for s in heard);error=word_error(expected,text)
        critical=critical_readback_changes(expected,text)
    return {'readback':text,'readback_word_error':error,'critical_readback_issues':critical,
            'readback_beam_size':beam,'verification_seconds':time.perf_counter()-started}


class SpeechQualityError(RuntimeError):
    def __init__(self,line):
        super().__init__('Generated actor line failed readback validation')
        self.line=line


class VerifiedSpeech:
    def __init__(self,renderer,language,owner,hints=()):
        self.renderer,self.language,self.owner=renderer,language,owner
        self.hints={}
        for hint in hints:
            heard=hint.get('heard',hint.get('readback',''))
            if word_error(hint.get('text',''),heard)!=0 or hint.get('reference_only'):continue
            if hint.get('model_sha256')!=renderer.model:continue
            if file_hash(hint['audio'])!=hint['sha256']:raise ValueError('Verified take hint changed')
            key=(hint['text'],hint['voice']['reference_sha256'])
            self.hints[key]=int(hint.get('seed',42))

    def render(self,cue,voice,directory,window):
        seed=self.hints.get((cue['english'],voice['reference_sha256']),42)
        attempts=[];best=None
        for choice in (seed,7 if seed!=7 else 42):
            self.owner.check()
            started=time.perf_counter()
            path,receipt=self.renderer.render(cue['english'],voice,directory,window,seed=choice,context_key=cue['id'])
            line={**receipt,'audio':str(path),'cue':cue['id'],'render_wall_seconds':time.perf_counter()-started}
            line.update(check_take(self.language,line,cue['english']))
            attempts.append({'audio':str(path),'sha256':line['sha256'],'seed':choice,
                             'wer':line['readback_word_error'],'critical':line['critical_readback_issues']})
            if best is None or line['readback_word_error']<best['readback_word_error']:best=line
            if line['readback_word_error']<=.25 and not line['critical_readback_issues']:
                line['verification_attempts']=attempts
                return line
        best['verification_attempts']=attempts
        raise SpeechQualityError(best)
