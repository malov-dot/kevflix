"""Lines that must keep their ORIGINAL audio in KevFLIX (never dubbed).

Owner rule (2026-10-20): singing (also chanted or sung prayers, choirs, hummed tunes), laughing, yelling WITHOUT words,
coughing, sneezing, sighing, groaning, crying/sobbing, breathing, whistling and similar non-word vocalizations, and lines
that are already English stay original. Yelled or whispered SENTENCES made of real words are speech and ARE dubbed.

Self-contained: standard library only. numpy is optional and used only by `audio_features()` (the extra pitch features
for the chant rule). Nothing here reads files, loads a model, or raises on odd input: every public function returns None
when unsure. Design goal: a wrongly KEPT line stays Russian (small harm), a wrongly DUBBED sound loses the real sound, but
a wrongly kept REAL SPEECH line is the worse outcome for the viewer, so every rule is written to leave real speech alone.

Public API
  nonverbal_text_reason(text)                     -> reason | None   (text only; ru / es / fr / en forms)
  nonverbal_audio_reason(features, events, ...)   -> reason | None   (acoustic + classifier rules)
  translation_artifact_reason(english, source)    -> reason | None   (translator meta text / transliteration spoken as English)
  nonverbal_reason(text, features, events, ...)   -> reason | None   (text first, then audio)
  audio_features(samples, rate)                   -> dict | None     (optional extra features; needs numpy)

Thresholds were derived from the 1048 dubbed lines of the full-film run of a 2014 Russian horror film and are explained next to each
constant. The reasons are short stable strings (see REASONS).
"""
from __future__ import annotations

import re
import unicodedata

VERSION = 'nonverbal-v2'

REASONS = (
    'laughter', 'vocalization', 'filler', 'interjection', 'breath', 'cough', 'sneeze', 'cry', 'whistle', 'singing_syllables',
    'sound_tag', 'music_mark', 'credits_hallucination', 'foreign_script', 'no_text', 'fragment', 'english_text',
    'english_transliteration',
    'singing_label', 'singing_held_note', 'chant_monotone', 'nonspeech_burst', 'no_signal', 'inaudible', 'scream',
    'english_audio', 'sparse_speech',
)

# --------------------------------------------------------------------------------------------------------------------
# text normalisation
# --------------------------------------------------------------------------------------------------------------------
_ZERO_WIDTH = re.compile('[\u200b\u200c\u200d\u2060\ufeff]')


def _fold(text):
    """NFKC, casefold, e-diaeresis -> e, Latin accents stripped (not the Cyrillic short-i breve)."""
    text = _ZERO_WIDTH.sub('', unicodedata.normalize('NFKC', str(text))).casefold().replace('\u0451', '\u0435').replace('\u2019', "'")
    out = []
    for ch in text:
        if ch.isascii() or not ch.isalpha() or not unicodedata.name(ch, '').startswith('LATIN'):
            out.append(ch)
        else:
            out.append(''.join(c for c in unicodedata.normalize('NFD', ch) if not unicodedata.combining(c)))
    return ''.join(out)


def _compact(text):
    """Only letters and digits of the folded text: 'А-а-а...' -> 'ааа', 'Ха -ха -ха!' -> 'хахаха'."""
    return ''.join(ch for ch in _fold(text) if ch.isalnum())


_UNIT_SPLIT = re.compile(r'[.!?…;:,¡¿«»"()\[\]{}—–/\\|]+')

# --------------------------------------------------------------------------------------------------------------------
# credits, tags, music marks (superset of relay/nonverbal.py)
# --------------------------------------------------------------------------------------------------------------------
_MUSIC_MARK = re.compile('[♪♫♬\U0001F3B5\U0001F3B6]')
_CREDITS = re.compile(
    r'(субтитры\s+(?:сделал|делал|создавал|подготовил|от)|редактор\s+субтитров|корректор\s+[а-яa-z]\.|'
    r'продолжение\s+следует|спасибо\s+за\s+просмотр|thanks\s+for\s+watching|subtitles?\s+by|amara\.org|'
    r'sous-titr(?:age|es)\s+(?:par|st)|subt[ií]tulos\s+(?:por|realizados)|dimatorzok|www\.|\.com\b|'
    r'подписывайтесь\s+на\s+(?:мой\s+)?канал|ставьте\s+лайк|like\s+and\s+subscribe|please\s+subscribe|'
    r'gracias\s+por\s+ver|merci\s+d.avoir\s+regard)', re.I)
# a bare credit fragment: Whisper splits "Субтитры сделал DimaTorzok" across cues; "Субтитры" alone was dubbed in the run
_CREDIT_FRAGMENT = re.compile(r'^(?:субтитры|субтитр|subtitles?|sous-titres|subtitulos|amara|сделал|подготовил)$')

_TAG_NOUNS = frozenset((
    # ru
    'смех', 'смеется', 'смеются', 'смеялся', 'хохот', 'хохочет', 'аплодисменты', 'музыка', 'музыку', 'песня', 'пение', 'поет',
    'поют', 'вздох', 'вздыхает', 'кашель', 'кашляет', 'чихает', 'плач', 'плачет', 'рыдает', 'всхлипывает', 'стон', 'стонет',
    'крик', 'кричит', 'визг', 'вопль', 'свист', 'свистит', 'дыхание', 'дышит', 'тишина', 'неразборчиво', 'шум', 'звон',
    'звучит', 'играет',
    # en
    'laugh', 'laughs', 'laughter', 'laughing', 'applause', 'music', 'singing', 'sings', 'sighs', 'sigh', 'coughs', 'coughing',
    'sneezes', 'cries', 'crying', 'sobbing', 'sobs', 'screams', 'screaming', 'yells', 'gasps', 'groans', 'breathing',
    'whistles', 'whistling', 'humming', 'hums', 'chuckles', 'giggles', 'silence', 'inaudible', 'indistinct', 'song', 'cough',
    'sneeze', 'sneezes', 'cry', 'scream', 'gasp', 'groan', 'yell', 'whistle', 'hum', 'chuckle', 'giggle', 'sob', 'pant', 'pants',
    'moan', 'moans', 'snore', 'snores', 'shout', 'shouts', 'shouting', 'cheering', 'singing', 'panting', 'groaning', 'gasping',
    # es
    'risas', 'risa', 'rie', 'aplausos', 'musica', 'canta', 'cantando', 'suspira', 'tos', 'tose', 'estornuda', 'llora', 'grita',
    'gritos', 'gime', 'silba', 'respira', 'silencio', 'cancion', 'suspiro', 'grito', 'llanto', 'estornudo', 'tosiendo', 'jadea',
    # fr
    'rires', 'rire', 'applaudissements', 'musique', 'chante', 'chantant', 'soupire', 'toux', 'tousse', 'eternue', 'pleure',
    'crie', 'cris', 'gemit', 'siffle', 'respire', 'chanson', 'soupir', 'cri', 'pleurs', 'sifflement', 'halete',
))
_TAG_ADJECTIVES = frozenset((
    'интригующая', 'напряженная', 'тревожная', 'мрачная', 'зловещая', 'грустная', 'веселая', 'торжественная', 'лирическая',
    'драматическая', 'спокойная', 'фоновая', 'мелодичная', 'печальная',
    'ритмичная', 'церковная', 'народная', 'мистическая', 'динамичная',
    'suspenseful', 'tense', 'eerie', 'sad', 'soft', 'upbeat', 'dramatic', 'ominous', 'gentle', 'dark', 'quiet', 'loud',
    'tensa', 'triste', 'suave', 'tendue', 'douce', 'sombre', 'lente', 'rapide',
))
_BRACKETED = re.compile(r'^\s*[\[\(\*<{♪♫]+([^\]\)\*>}]{0,60})[\]\)\*>}♪♫]*\s*$')
_BARE_TAGS = frozenset(('музыка', 'аплодисменты', 'смех', 'applause', 'music', 'laughter', 'risas', 'aplausos', 'musica',
                        'musique', 'rires', 'applaudissements'))


def _tag_reason(text):
    stripped = text.strip()
    words = [w for w in re.split(r'[^\w]+', _fold(stripped)) if w]
    if not words:
        return None
    inside = _BRACKETED.match(stripped)
    if inside:
        inner = [w for w in re.split(r'[^\w]+', _fold(inside.group(1))) if w]
        if inner and any(w in _TAG_NOUNS for w in inner):
            return 'sound_tag'
    # caption descriptor without brackets: ALL CAPS and nothing but descriptor words ("ИНТРИГУЮЩАЯ МУЗЫКА", "МУЗЫКА")
    letters = [c for c in stripped if c.isalpha()]
    if letters and len(letters) >= 4 and all(c.isupper() for c in letters):
        if all(w in _TAG_NOUNS or w in _TAG_ADJECTIVES for w in words) and any(w in _TAG_NOUNS for w in words):
            return 'sound_tag'
        # one all-caps caption adjective whose noun went into the next cue ("ИНТРИГУЮЩАЯ" / "МУЗЫКА" were separate lines)
        if len(words) == 1 and words[0] in _TAG_ADJECTIVES:
            return 'sound_tag'
    if len(words) == 1 and words[0] in _BARE_TAGS:
        return 'sound_tag'
    return None


# --------------------------------------------------------------------------------------------------------------------
# non-word vocalization patterns on the compact (letters only) form of ONE sentence-like unit
# --------------------------------------------------------------------------------------------------------------------
# laughter: >= 2 laugh syllables (so "ха" alone, "ухо", "хохма" never match); coverage rule for ragged transcripts
_LAUGH_RUN = re.compile(r'[аеиоуэы]?х?(?:(?:х[аеиоуэы])|(?:гы)){2,}х?')
_LAUGH_CYR = re.compile(r'^[аеиоуэы]?х?(?:(?:х[аеиоуэы])|(?:гы)){2,}[аоэ]?х?$|^(?P<w>[аеиоуэ])х(?:(?P=w)х){2,}(?P=w)?$|^хохот$')
_LAUGH_LAT = (
    re.compile(r'^(?:[mb]?[wo]?u?a)?(?:h[aeiou]|j[aeiou]){2,}h?$'),      # haha jaja jejeje hehe muahaha bwahaha
    re.compile(r'^(?:ah|eh|oh|ha|he|ho|ja|je|ji|jo){3,}$'),           # "ah ah ah" "ha ha ha" in other spellings
    re.compile(r'^a?(?:h[aeiou]){1}h+$'),                              # ahah
    re.compile(r'^te+he+$|^xd+$|^lol+$|^lmao+$|^mdr+$|^ptdr+$|^kkk+$|^rsrs+$|^hjhj+$'),
)
# long vowel / moan / scream / cry vowel: the same vowel >= 3 times ("А-а-а", "Ааа!", "Оооо", "Э-э-э", "aaah")
_VOWEL_STRETCH = re.compile(r'^(?P<v>[aeiouаеиоуыэюя])(?P=v){2,}[hхx]*$|^a+r*g+h+$|^a+rr+gh*$|^grr+$')
_GROAN_X = re.compile(r'^[аоуэиaou][хxh]{2,}$')                         # охх, аххх, ugh? (ahh)
# hesitation / hum / thinking sounds
_FILLER = re.compile(r'^(?:э+м*|э*м{2,}|м+х+м*|х+м+|г+м+|эм+|u+m+|u+h+|e+r+m*|h+m+|m+h+m+|e+u+h+|h+e+u+|h+u+m+|e+h+m+)$')
_HUM = re.compile(r'^(?:м{2,}|н{2,}|m{2,}|n{2,})$')
_TRILL = re.compile(r'^(?:б?р{3,}|бр{2,}|fr{2,}|br{2,}|пф+|гр{2,}|gr{2,}|р{3,})$')     # Брррр, Грррр (growl), Ррр
# breath / hush: Шшш, Тссс, Псст, shhh, pffff
_BREATH = re.compile(r'^(?:ш{2,}|с{3,}|тс{2,}|пс{2,}т*|фф+|хх+|s+h+|sss+|pss+t*|pff+|fff+|hhh+)$')
_COUGH = re.compile(r'^(?:кх[аеоэ]?)+м?$|^(?:cof+|kof+|ahem+|ehem+|ejem+)$|^(?:cof){2,}$')
_SNEEZE = re.compile(r'^а{0,3}п{1,2}[чт]?ч?х+и*$|^апчх|^[ая]чх+и+$|^a{1,3}t?ch?(?:oo+|ou)m?$|^ha?tch?(?:oo+|ou)m?$|^achis$|^achu$')
_CRY = re.compile(r'^(?:хныкь?|хнык(?:хнык)*|хныч|шмыг(?:шмыг)*|всхлип|бу+а+|буу+|уа+(?:уа+)*|уэ+|snif+|sniff+|sob+s?|boo+hoo+|bu+a+h?|wa+h+|wa{3,}|u+a+h*|hu+hu+)$')
_WHISTLE = re.compile(r'^(?:ф[ьъ]?ю+(?:ить|ть|т)?)+$|^fiu+$')
# pure interjection lines (a single sound word, no content word): Ох/Ах/Эх/Ух/Ой/Ай/О/Оу/Уф/Фух ... and en/es/fr equivalents
_INTERJ = re.compile(r'^(?:о|оу)$|^(?:ох+|ах+|эх+|ух+|уф+|фух+|фуф+|пф+|ой+|ай+)+$'     # NOT (о|оу)+: "ухо" is a word
                     r'|^(?:o+h+|a+h+|ay+|ugh+|phew|whew|oo+f+|ou+f+|uf+|pf+f*)$')
# singing syllables: la-la-la, ля-ля-ля, лу-лу, тра-ля-ля, na-na-na (>= 3 syllables so "лала" nickname never matches)
_SING_SYL = re.compile(r'^(?:тра)?(?:л[аяуюеи]){3,}$|^тра(?:л[аяу]){2,}$|^(?:la|na|lu){3,}$|^(?:doo|do|dum|bum){3,}$|^tra(?:la)+$|^tra(?:ra){2,}$')


def _laugh_cov(c):
    if len(c) < 4:
        return 0.0
    covered = sum(len(m.group(0)) for m in _LAUGH_RUN.finditer(c))
    return covered / len(c) if covered >= 6 else 0.0      # "хахаль" (a real word) covers only 4 letters


def _unit_reason(c):
    """Reason for ONE unit given its compact form, or None."""
    if not c:
        return None
    if _LAUGH_CYR.match(c) or any(p.match(c) for p in _LAUGH_LAT):
        return 'laughter'
    if len(c) >= 6 and _laugh_cov(c) >= 0.6:
        return 'laughter'                                   # "Ха-ха-ха .Т-хам-ха-хам." (ragged laugh transcripts)
    if _VOWEL_STRETCH.match(c) or _GROAN_X.match(c):
        return 'vocalization'
    if _COUGH.match(c):
        return 'cough'
    if _SNEEZE.match(c):
        return 'sneeze'
    if _CRY.match(c):
        return 'cry'
    if _WHISTLE.match(c):
        return 'whistle'
    if _SING_SYL.match(c):
        return 'singing_syllables'
    if _HUM.match(c):
        return 'vocalization'
    if _FILLER.match(c):
        return 'filler'
    if _TRILL.match(c):
        return 'vocalization'
    if _BREATH.match(c):
        return 'breath'
    if _INTERJ.match(c):
        return 'interjection'
    return None


# --------------------------------------------------------------------------------------------------------------------
# already-English text
# --------------------------------------------------------------------------------------------------------------------
# Function words that are English and not also common in ru/es/fr/Latin (so "no", "me", "a", "on", "son", "he" are excluded)
_EN_WORDS = frozenset((
    'the', 'and', 'you', 'your', 'are', 'was', 'were', 'been', "i'm", "i'll", "i've", "it's", "that's", 'this', 'these', 'those',
    'what', 'who', 'why', 'how', 'when', 'where', 'with', 'without', 'from', "don't", "doesn't", "didn't", "can't", "couldn't",
    'would', 'should', "won't", 'have', 'has', 'had', 'does', 'did', 'there', 'here', 'they', 'them', 'his', 'her', 'but', 'just',
    'now', 'well', 'yes', 'please', 'sir', 'thank', 'thanks', 'hello', 'okay', 'going', 'know', 'want', 'need', 'think', "let's",
    'get', 'got', 'come', 'tell', 'help', 'wait', 'stop', 'listen', 'look', 'really', 'very', 'good', 'right', 'not', 'for',
    'we', 'our', 'us', 'my', 'it', 'is', 'of', 'to', 'in', 'at', 'if', 'then', 'so', 'be', 'do', 'will', 'can', 'yeah', 'yep',
    "what's", "who's", "where's", "there's", "he's", "she's", "isn't", "aren't", "wasn't", "weren't", "haven't", "hasn't",
    "wouldn't", "shouldn't", "you're", "we're", "they're", "you'll", "we'll", "he'll", "she'll", "they'll", "you've", "we've",
    "they've", 'name', 'understand', 'nothing', 'something', 'anything', 'everything', 'because', 'about', 'people', 'never',
    'always', 'again', 'still', 'before', 'after', 'maybe', 'little', 'think', 'friend', 'home', 'time', 'god', 'run', 'go',
    'let', 'make', 'take', 'give', 'say', 'said', 'see', 'seen', 'love', 'life', 'man', 'woman', 'sorry', 'church', 'night',
    'day', 'back', 'out', 'up', 'down', 'over', 'into', 'than', 'like', 'only', 'also', 'too', 'much', 'many', 'more', 'some',
    'any', 'every', 'each', 'own', 'same', 'other', 'another', 'such', 'both', 'between', 'through', 'during', 'against'))
_EN_SINGLE = frozenset(('yeah', 'yes', 'okay', 'ok', 'hello', 'hi', 'bye', 'sir', 'thanks', 'please', 'sorry', 'wow', 'whoa', 'what',
                        'why', 'well', 'sure', 'wait', 'stop', 'help', 'right', 'really', 'nothing', 'nobody', 'listen'))
_TRANSLIT_EN = frozenset(('летсго', 'оукей', 'хеллоу', 'хэллоу', 'хелло', 'плиз', 'сорри', 'сенкс'))


def _english_text(text):
    toks = re.findall(r"[a-z]+(?:'[a-z]+)?", _fold(text))
    letters = [c for c in _fold(text) if c.isalpha()]
    if not letters:
        return None
    latin = sum(1 for c in letters if c.isascii())
    if latin / len(letters) < 0.9 or not toks:
        return None
    hits = sum(1 for t in toks if t in _EN_WORDS)
    if len(toks) == 1:
        return 'english_text' if toks[0] in _EN_SINGLE else None
    if hits >= 2 or (hits >= 1 and hits / len(toks) >= 0.3):
        return 'english_text'
    return None


# --------------------------------------------------------------------------------------------------------------------
# public text rule
# --------------------------------------------------------------------------------------------------------------------
_NON_LATIN_CYR_SCRIPTS = ('CJK', 'HIRAGANA', 'KATAKANA', 'HANGUL', 'ARABIC', 'HEBREW', 'THAI', 'DEVANAGARI', 'BENGALI', 'TAMIL',
                          'TELUGU', 'GEORGIAN', 'ARMENIAN')


def _foreign_script(text):
    seen = 0
    letters = 0
    for ch in text:
        if ch.isalpha():
            letters += 1
            if unicodedata.name(ch, '').split(' ')[0] in _NON_LATIN_CYR_SCRIPTS:
                seen += 1
    return letters > 0 and seen / letters >= 0.5


def nonverbal_text_reason(text, *, interjections=True, english=True):
    """None for ordinary speech, else a short reason (see REASONS). Pure text; never raises.

    interjections=False leaves lone "Ах", "Ох!", "О!" alone (the behaviour of relay/nonverbal.py and its tests);
    english=False skips the already-English text check (the pipeline also compares source and translation afterwards).

    Conservative by construction: a line is flagged only when EVERY sentence-like unit in it is a non-word sound, tag,
    credit or English phrase. A laugh followed by words ("Ха-ха, ну-ка.") is speech and is NOT flagged."""
    reason = _text_reason(text)
    if reason == 'interjection' and not interjections:
        return None
    if reason in ('english_text', 'english_transliteration') and not english:
        return None
    return reason


def _text_reason(text):
    try:
        if not isinstance(text, str) or not text.strip():
            return None
        if _MUSIC_MARK.search(text):
            return 'music_mark'
        if _CREDITS.search(text):
            return 'credits_hallucination'
        tag = _tag_reason(text)
        if tag:
            return tag
        if _foreign_script(text):
            return 'foreign_script'
        whole = _compact(text)
        if not whole:
            return 'no_text' if any(unicodedata.category(ch)[0] in 'PSN' for ch in text) else None
        if _CREDIT_FRAGMENT.match(whole):
            return 'credits_hallucination'
        if len(whole) == 1 and whole in 'вксВКС'.casefold():
            return 'fragment'                      # an orphaned preposition cut off from its sentence ("В", "К", "С")
        if whole in _TRANSLIT_EN:
            return 'english_transliteration'
        toks = [_compact(t) for t in re.split(r'[\s.!?,;:…]+', _fold(text)) if _compact(t)]
        if len(toks) >= 2 and all(_INTERJ.match(t) for t in toks):
            return 'interjection'                  # "Ох, ах, эх." is three sighs, not the laugh syllables о-ха-хэ-х
        reason = _unit_reason(whole)               # the whole line at once: "Ха! Ха! Ха!" -> хахаха
        if reason:
            return reason
        units = [u for u in _UNIT_SPLIT.split(_fold(text)) if any(ch.isalnum() for ch in u)]
        found = []
        for unit in units:
            reason = _unit_reason(_compact(unit))
            if reason is None:
                # a unit of several plain tokens ("ох ах эх"): every token must be a non-word sound
                toks = [t for t in re.split(r'\s+', unit.strip()) if t]
                reasons = [_unit_reason(_compact(t)) for t in toks]
                reason = reasons[0] if toks and all(reasons) else None
            if reason is None:
                return _english_text(text)         # real words present: speech, unless the whole line is English
            found.append(reason)
        return found[0] if found else _english_text(text)
    except Exception:       # a text check must never break dubbing
        return None


# --------------------------------------------------------------------------------------------------------------------
# translator artifacts spoken as if they were the translation
# --------------------------------------------------------------------------------------------------------------------
_META = re.compile(
    r"in the russian sentence|the russian sentence|there is no actual sentence|here is the translation|here's the translation|"
    r"please provide a valid|incomplete or empty input|as an ai\b|i (?:cannot|can't|am unable to) translate|"
    r"into natural, accurate english|keeping the meaning exactly", re.I)
_TR = dict(zip('абвгдеёжзийклмнопрстуфхцчшщъыьэюя', ['a', 'b', 'v', 'g', 'd', 'e', 'e', 'zh', 'z', 'i', 'y', 'k', 'l', 'm', 'n', 'o', 'p',
                                                    'r', 's', 't', 'u', 'f', 'kh', 'ts', 'ch', 'sh', 'shch', '', 'y', '', 'e', 'yu', 'ya']))


def translation_artifact_reason(english, source=None):
    """'translator_meta_text' when the English that would be spoken is the model's commentary ("In the Russian sentence
    provided, there is no actual sentence to translate"), 'transliteration_not_translation' when it is the Russian source
    spelled in Latin letters ("Zhuchen', ya khochu znat'."). None otherwise. Seen 4 + 2 times in the 1048 dubbed lines."""
    try:
        if not isinstance(english, str):
            return None
        if _META.search(english):
            return 'translator_meta_text'
        if re.search(r'[\u0400-\u04ff]', english):
            return 'cyrillic_in_english'
        if isinstance(source, str):
            src = re.sub(r'[\W_]+', '', ''.join(_TR.get(ch, ch) for ch in source.casefold()))
            eng = re.sub(r'[\W_]+', '', english.casefold())
            words = re.findall(r"[a-z]+(?:'[a-z]+)?", english.casefold())
            if len(src) >= 16 and eng and sum(1 for w in words if w in _EN_WORDS) <= 1:    # real English words => a translation
                import difflib
                if difflib.SequenceMatcher(None, src, eng).ratio() >= 0.8:
                    return 'transliteration_not_translation'
        return None
    except Exception:
        return None


# --------------------------------------------------------------------------------------------------------------------
# acoustic rules
# --------------------------------------------------------------------------------------------------------------------
# AST (AudioSet) labels, computed on the separated vocals. In this film the singing family is the ONLY classifier family that was
# usable: 10 of 1047 dubbed lines reach 0.15 and all 10 are sung/chanted material (the highest unlabelled line is 0.095, the first
# false positive appears at 0.08). Laughter, Cough, Screaming, Shout, Yell and Crying never exceed 0.21 on any dubbed line, so
# they cannot be thresholded on their own; Gasp >= 0.2 occurs on 38 dubbed lines and Sigh >= 0.2 on 13, nearly all ordinary speech.
SING_LABELS = ('Singing', 'Choir', 'Chant', 'Mantra', 'Male singing', 'Female singing', 'Child singing', 'Humming', 'Yodeling')
SING_LABEL_MIN = 0.15
BURST_LABELS = {
    'Sneeze': 'sneeze', 'Cough': 'cough', 'Throat clearing': 'cough', 'Sigh': 'sigh_groan', 'Groan': 'sigh_groan',
    'Grunt': 'sigh_groan', 'Snort': 'sigh_groan', 'Wail, moan': 'sigh_groan', 'Gasp': 'breath', 'Breathing': 'breath',
    'Pant': 'breath', 'Whimper': 'cry', 'Crying, sobbing': 'cry', 'Laughter': 'laughter', 'Giggle': 'laughter',
    'Chuckle, chortle': 'laughter', 'Belly laugh': 'laughter', 'Snicker': 'laughter', 'Whistling': 'whistle'}
BURST_LABEL_MIN = 0.20          # label probability
BURST_SPEECH_MAX = 0.10         # AST "Speech" at or below this: 14 of 1047 dubbed lines
BURST_VOICED_MAX = 0.10         # and almost no voiced frames (a spoken word has some)
HELD_NOTE_MIN_S = 1.0           # longest steady pitch; highest line judged speech is 0.74 s (shouted sentence), prayer chant reaches 0.84
HELD_NOTE_MIN_VOICED = 0.5
CHANT_HELD_MIN_S = 0.40         # monotone recitation with the existing features only
CHANT_VOICED_MIN = 0.70
CHANT_RANGE_MAX = 5.0           # pitch_range_semitones (10-90 percentile spread in the stem measurement)
CHANT_SPEECH_MAX = 0.35
TONE_SHARE_MIN = 0.75           # optional extra features (audio_features): share of voiced 50 ms blocks within +-1 semitone
TONE_RUN_MIN_S = 1.0            # of the most common pitch, longest run of such blocks
TONE_VOICED_MIN_S = 2.0
INAUDIBLE_DB = -66.0            # active level of the original vocals at or below this: nothing a viewer could hear (the film's quietest real
                                # lines sit at -57 dB; every cue at -66 dB or less in that run was a hallucination or an inaudible aside)
LOUD_VOWEL_DB = -25.0           # single-vowel line at least this loud: the loudest ordinary line in the film is -28 dB, "А" is -17


def _events_dict(events):
    """(labels, complete): a dict is the full label set, a list is only the top labels."""
    if isinstance(events, dict):
        return {str(k): float(v) for k, v in events.items() if isinstance(v, (int, float))}, True
    out = {}
    for item in events or ():
        try:
            out[str(item[0])] = float(item[1])
        except Exception:
            continue
    return out, False


def nonverbal_audio_reason(features, events, *, text=None, duration=None):
    """Reason from the measured features of the ORIGINAL vocals excerpt and the classifier labels, else None.

    features: dict with voiced_fraction, longest_held_note_s, pitch_range_semitones, rms_active_db (the export fields) and,
              optionally, tone_share / tone_run_s / voiced_s from audio_features().
    events:   [[label, prob], ...] (top labels) or {label: prob} (all labels). The full dict is better.
    text/duration: optional, used by two small rules (single loud vowel, empty cue)."""
    try:
        f = features or {}
        ev, complete = _events_dict(events)
        vf = float(f.get('voiced_fraction') or 0.0)
        hold = float(f.get('longest_held_note_s') or 0.0)
        rng = f.get('pitch_range_semitones')
        rms = f.get('rms_active_db')
        speech = ev.get('Speech')
        if speech is None and not complete and ev:
            speech = min(ev.values())          # Speech is not among the top labels: it is below all of them
        # 1 an empty cue (Whisper hallucination over silence): too short to measure (the export has no features for spans under
        #   ~0.1 s) or level exactly 0 and nothing voiced
        if duration is not None and not f and float(duration) < 0.15:
            return 'no_signal'
        if f and 'rms_active_db' in f and float(rms or 0.0) == 0.0 and vf == 0.0 and (duration is None or float(duration) < 0.3):
            return 'no_signal'
        if rms is not None and float(rms) <= INAUDIBLE_DB:
            return 'inaudible'
        # 2 classifier says singing / choir / chant / mantra / humming
        if max([ev.get(k, 0.0) for k in SING_LABELS] or [0.0]) >= SING_LABEL_MIN:
            return 'singing_label'
        # 3 a long held steady note
        if hold >= HELD_NOTE_MIN_S and vf >= HELD_NOTE_MIN_VOICED:
            return 'singing_held_note'
        # 4 recitative on one tone (extra features)
        if (float(f.get('tone_share') or 0) >= TONE_SHARE_MIN and float(f.get('tone_run_s') or 0) >= TONE_RUN_MIN_S
                and float(f.get('voiced_s') or 0) >= TONE_VOICED_MIN_S):
            return 'chant_monotone'
        # 5 recitative on one tone with the export features only (speech-like classifier score is low)
        if (hold >= CHANT_HELD_MIN_S and vf >= CHANT_VOICED_MIN and rng is not None and float(rng) <= CHANT_RANGE_MAX
                and speech is not None and speech <= CHANT_SPEECH_MAX):
            return 'chant_monotone'
        # 6 an unvoiced burst the classifier calls a cough / sneeze / sigh / laugh / breath and does not call speech
        if speech is not None and speech <= BURST_SPEECH_MAX and vf <= BURST_VOICED_MAX:
            best = max(((ev.get(k, 0.0), k) for k in BURST_LABELS), default=(0.0, None))
            if best[0] >= BURST_LABEL_MIN:
                return 'nonspeech_burst:' + BURST_LABELS[best[1]]
        # 7 one loud vowel ("А" at -17 dB): a scream or a shout without words
        if text is not None and rms is not None and float(rms) >= LOUD_VOWEL_DB:
            c = _compact(text)
            if len(c) == 1 and c in 'аоуэиыяюе':
                return 'scream'
        return None
    except Exception:
        return None


def nonverbal_reason(text, features=None, events=None, *, duration=None):
    """Text rule first, then the acoustic rule. None means: dub it."""
    return nonverbal_text_reason(text) or (nonverbal_audio_reason(features, events, text=text, duration=duration) if features is not None else None)


# --------------------------------------------------------------------------------------------------------------------
# optional extra acoustic features (numpy). Same definitions as used to derive the thresholds above.
# --------------------------------------------------------------------------------------------------------------------
def audio_features(samples, rate):
    """{'voiced_s','tone_share','tone_run_s','jumps'} of a mono excerpt, or None. needs numpy; any rate (resampled by
    decimation/interpolation to 16 kHz). tone_share = share of voiced 50 ms blocks within +-1 semitone of the most common
    pitch, tone_run_s = longest run of blocks at that pitch. Chanted recitation and sung notes have a high share and a long
    run, speech (even loud or flat speech) does not hold one pitch for a second."""
    try:
        import numpy as np
    except Exception:
        return None
    try:
        x = np.asarray(samples, dtype=np.float64).ravel()
        if rate != 16000:
            if rate % 16000 == 0:                      # 48 kHz -> 16 kHz: block average = crude anti-alias low-pass
                k = rate // 16000
                x = x[:len(x) // k * k].reshape(-1, k).mean(1)
            else:
                n = int(round(len(x) * 16000 / float(rate)))
                x = np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x)
        win, hop = 640, 160
        if len(x) < win * 2:
            return None
        n = 1 + (len(x) - win) // hop
        idx = np.arange(win)[None, :] + hop * np.arange(n)[:, None]
        fr = x[idx] * np.hanning(win)[None, :]
        db = 20 * np.log10(np.sqrt((fr ** 2).mean(1)) + 1e-6)
        spec = np.fft.rfft(fr, 1024, axis=1)
        ac = np.fft.irfft(spec * np.conj(spec), 1024, axis=1)[:, :win]
        lag = np.arange(win)[None, :]
        r = ac / np.maximum(ac[:, :1], 1e-12) * (win / np.maximum(win - lag, 1))
        lo, hi = int(16000 / 600), int(16000 / 70)
        sub = r[:, lo:hi + 1]
        k = np.argmax(sub, 1)
        peak = sub[np.arange(n), k]
        lagpk = k + lo
        f0 = np.zeros(n)
        for i in range(n):
            lg = lagpk[i]
            if 1 <= lg < win - 1:
                a, b, c = r[i, lg - 1], r[i, lg], r[i, lg + 1]
                d = a - 2 * b + c
                f0[i] = 16000.0 / (lg + (0.5 * (a - c) / d if d != 0 else 0.0))
        act = db > np.percentile(db, 95) - 30
        s = 12 * np.log2(np.maximum(f0, 1.0) / 55.0)
        blocks = []
        for j in range(0, n - 4, 5):
            ok = act[j:j + 5] & (peak[j:j + 5] > 0.55)
            if ok.sum() >= 3:
                blocks.append(float(np.median(s[j:j + 5][ok])))
        if len(blocks) < 6:
            return {'voiced_s': len(blocks) * 0.05, 'tone_share': 0.0, 'tone_run_s': 0.0, 'jumps': 0}
        b = np.array(blocks)
        best, centre = 0.0, b[0]
        for c in np.arange(b.min(), b.max() + 0.01, 0.25):
            share = float((np.abs(b - c) <= 1.0).mean())
            if share > best:
                best, centre = share, c
        run = mx = 0
        for v in b:
            run = run + 1 if abs(v - centre) <= 1.0 else 0
            mx = max(mx, run)
        return {'voiced_s': len(b) * 0.05, 'tone_share': best, 'tone_run_s': mx * 0.05, 'jumps': int((np.abs(np.diff(b)) > 3).sum())}
    except Exception:
        return None


SPARSE_MIN_SECONDS = 3.0        # a cue this long ...
SPARSE_SECONDS_PER_WORD = 2.5   # ... holding fewer words than one per 2.5 s: the recognizer's word times span silence or noise


def sparse_speech_reason(duration, words):
    """'sparse_speech' for a cue whose few words are stretched over a long span (one word over 11.6 s in that run: replacing
    it removed 10.8 s of film sound for the one word 'caught'); None otherwise or when the inputs are unusable."""
    try:
        duration, words = float(duration), int(words)
        if duration >= SPARSE_MIN_SECONDS and words >= 0 and duration / max(words, 1) > SPARSE_SECONDS_PER_WORD:
            return 'sparse_speech'
    except (TypeError, ValueError):
        pass
    return None


def is_nonverbal(text, features=None, events=None, *, duration=None):
    return nonverbal_reason(text, features, events, duration=duration) is not None
