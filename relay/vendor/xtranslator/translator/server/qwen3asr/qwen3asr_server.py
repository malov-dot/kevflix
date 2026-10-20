#!/usr/bin/env python3
import argparse
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import numpy as np
import base64
import io
import wave
import asyncio
from threading import Lock
from concurrent.futures import ThreadPoolExecutor
from qwen_asr import Qwen3ASRModel
import logging
import time
import uuid
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Qwen3ASR")

app = FastAPI(title="Qwen3 ASR Service")

model = None
startup_warmup = None

@app.get("/healthz")
async def healthz():
    if model is None or startup_warmup is None:
        raise HTTPException(status_code=503, detail="Qwen3-ASR is not loaded and warmed")
    return {"status": "ok", "backend": "qwen3-asr-vllm", "device": "cuda", "startup_warmup": startup_warmup}

session_cache = {}
gpu_lock = Lock()
poison_lock = Lock()
poisoned_session_ids = set()  # separate from mutable contexts touched by executor threads
executor = ThreadPoolExecutor(max_workers=10)

MAX_SESSION_STEPS = 60
OVERLAP_CHUNKS = 4
VAD_THRESHOLD = 0.001
ASR_REQUEST_TIMEOUT_SECONDS = 90.0  # client read deadline is this value +15s


class SessionPoisonedError(RuntimeError):
    pass


def reject_poisoned_session(session_id):
    with poison_lock:
        if session_id is not None and session_id in poisoned_session_ids:
            raise SessionPoisonedError('A prior request timed out; this session cannot be reused')

# Native Qwen names only; an unmappable label remains unknown, never guessed.
NATIVE_LANGUAGE_CODES = dict(zip(
    ('Chinese English Arabic German French Spanish Portuguese Indonesian Italian Korean Russian Thai Vietnamese '
     'Japanese Turkish Hindi Malay Dutch Swedish Danish Finnish Polish Czech Persian Greek Romanian Hungarian Macedonian').lower().split(),
    'zh en ar de fr es pt id it ko ru th vi ja tr hi ms nl sv da fi pl cs fa el ro hu mk'.split()))


def validated_language_context(code, basis_final_ids):
    names = {value: key.title() for key, value in NATIVE_LANGUAGE_CODES.items()}
    if code is None and not basis_final_ids:
        return {'hint': None, 'basis_final_ids': []}
    if code not in names or len(basis_final_ids) != 2 or len(set(basis_final_ids)) != 2 \
            or any(not isinstance(item, str) or re.fullmatch('[0-9a-f]{32}', item) is None for item in basis_final_ids):
        raise ValueError('Language context requires a canonical native code and two distinct final observation IDs')
    return {'hint': names[code], 'basis_final_ids': list(basis_final_ids)}


def language_context_prompt(context):
    # Canonical audio-derived label only; avoid adding instructional English
    # that could leak into the transcript. This is not the language= argument.
    return context['hint'] or ''


def init_context_state(context):
    return model.init_streaming_state(context=language_context_prompt(context), language=None,
                                     unfixed_chunk_num=2, unfixed_token_num=5, chunk_size_sec=2.0)


def recognition_result(text, state, context, observation_id):
    raw = getattr(state, 'language', None)
    raw = raw.strip() if isinstance(raw, str) and raw.strip() else None
    return {'text': text or '', 'language_raw': raw,
            'language_code': NATIVE_LANGUAGE_CODES.get(raw.lower()) if raw else None,
            'language_source': 'qwen3-asr-native' if raw else None,
            'language_observation_id': observation_id,
            'language_context': {'hint': context['hint'], 'basis_final_ids': list(context['basis_final_ids'])}}


def preview_text(text: str, limit: int = 120) -> str:
    value = (text or "").strip()
    if len(value) <= limit:
        return value
    return f"{value[:limit]}..."


def merge_text_prefix(prefix: str, current: str) -> str:
    left = (prefix or "").strip()
    right = (current or "").strip()
    if not left:
        return right
    if not right:
        return left
    max_overlap = min(len(left), len(right))
    for size in range(max_overlap, 0, -1):
        if left[-size:] == right[:size]:
            return left + right[size:]
    return left + right


def strip_prefix_text(text: str, prefix: str) -> str:
    value = (text or "").strip()
    base = (prefix or "").strip()
    if not base:
        return value
    if value.startswith(base):
        return value[len(base):].strip()
    if base.endswith(value):
        return ""
    return value

class RecognitionRequest(BaseModel):
    audio: str
    session_id: str = None
    is_final: bool = False
    sample_rate: int = 16000
    committed_text: str = ""
    language_context_code: str | None = Field(default=None, max_length=2)
    basis_final_ids: list[str] = Field(default_factory=list, max_length=2)


def rebuild_streaming_state(ctx: dict) -> None:
    logger.info(
        "rebuild state history_chunks=%s prefix=%s",
        len(ctx["history_chunks"]),
        preview_text(ctx.get("prefix_text", "")),
    )
    new_state = init_context_state(ctx['language_context'])
    for chunk in ctx["history_chunks"]:
        if chunk.size == 0:
            continue
        model.streaming_transcribe(chunk, new_state)
    logger.info("rebuild state done text=%s", preview_text(getattr(new_state, "text", "")))
    ctx["state"] = new_state
    ctx["step_count"] = len(ctx["history_chunks"])


def apply_committed_prefix(ctx: dict, committed_text: str) -> None:
    committed = (committed_text or "").strip()
    if not committed:
        return
    if committed == ctx["prefix_text"]:
        return
    logger.info(
        "apply commit old_prefix=%s new_prefix=%s state_text=%s",
        preview_text(ctx["prefix_text"]),
        preview_text(committed),
        preview_text(getattr(ctx["state"], "text", "")),
    )
    ctx["prefix_text"] = committed
    ctx["last_valid_text"] = ""
    model.finish_streaming_transcribe(ctx["state"])
    rebuild_streaming_state(ctx)

def run_inference(request: RecognitionRequest, audio_data: np.ndarray):
    global model, session_cache
    inference_started = time.perf_counter()
    
    # 如果是纯静音包，跳过推理以防止模型“胡思乱想”产生重复字
    is_speech = np.abs(audio_data).mean() > VAD_THRESHOLD
    
    with gpu_lock:
        try:
            reject_poisoned_session(request.session_id)
            if request.session_id:
                # 初始化 Session
                if request.session_id not in session_cache:
                    context = validated_language_context(request.language_context_code, request.basis_final_ids)
                    session_cache[request.session_id] = {
                        "state": init_context_state(context),
                        "language_context": context,
                        "language_observation_id": uuid.uuid4().hex,
                        "step_count": 0,
                        "history_chunks": [], 
                        "last_valid_text": "",
                        "prefix_text": "",
                    }
                
                ctx = session_cache[request.session_id]
                apply_committed_prefix(ctx, request.committed_text)
                
                # 记录音频到滑动窗口
                if audio_data.size > 0:
                    ctx["history_chunks"].append(audio_data)
                    if len(ctx["history_chunks"]) > OVERLAP_CHUNKS:
                        ctx["history_chunks"].pop(0)

                # 滑动窗口重置逻辑：达到上限时，平滑衔接
                if ctx["step_count"] >= MAX_SESSION_STEPS and not request.is_final:
                    logger.info(f"Session {request.session_id} 滑动重置：保留上下文重新注入")
                    ctx["prefix_text"] = merge_text_prefix(
                        ctx["prefix_text"],
                        ctx["state"].text,
                    )
                    model.finish_streaming_transcribe(ctx["state"])
                    rebuild_streaming_state(ctx)
                
                # 执行识别 (仅在有声音或强制结束时执行)
                if is_speech or request.is_final:
                    model.streaming_transcribe(audio_data, ctx["state"])
                    ctx["step_count"] += 1
                
                if request.is_final:
                    model.finish_streaming_transcribe(ctx["state"])
                    final_text = merge_text_prefix(
                        ctx["prefix_text"],
                        ctx["state"].text,
                    ).strip()
                    logger.info(
                        "return final prefix=%s state_text=%s merged=%s",
                        preview_text(ctx["prefix_text"]),
                        preview_text(getattr(ctx["state"], "text", "")),
                        preview_text(final_text),
                    )
                    result = recognition_result(final_text, ctx['state'], ctx['language_context'], ctx['language_observation_id'])
                    del session_cache[request.session_id]
                    return result
                else:
                    # 如果当前识别结果为空（由于静音过滤等），返回上一帧文本避免闪烁
                    full_text = merge_text_prefix(
                        ctx["prefix_text"],
                        ctx["state"].text,
                    ).strip()
                    current_text = strip_prefix_text(full_text, ctx["prefix_text"])
                    logger.info(
                        "return partial prefix=%s state_text=%s full=%s current=%s",
                        preview_text(ctx["prefix_text"]),
                        preview_text(getattr(ctx["state"], "text", "")),
                        preview_text(full_text),
                        preview_text(current_text),
                    )
                    if not current_text:
                        ctx["last_valid_text"] = ""
                        return recognition_result('', ctx['state'], ctx['language_context'], ctx['language_observation_id'])
                    if ctx["last_valid_text"] and current_text == ctx["last_valid_text"]:
                        return recognition_result(current_text, ctx['state'], ctx['language_context'], ctx['language_observation_id'])
                    ctx["last_valid_text"] = current_text
                    ctx['last_valid_result'] = recognition_result(current_text, ctx['state'], ctx['language_context'], ctx['language_observation_id'])
                    return dict(ctx['last_valid_result'])
            else:
                # 非流式处理
                context = validated_language_context(request.language_context_code, request.basis_final_ids)
                state = init_context_state(context)
                model.streaming_transcribe(audio_data, state)
                model.finish_streaming_transcribe(state)
                return recognition_result(state.text, state, context, uuid.uuid4().hex)
                
        except Exception as e:
            logger.exception("Inference internal error class=%s stage=run_inference elapsed_seconds=%.3f session_id=%s message=%r",
                             type(e).__name__, time.perf_counter() - inference_started, request.session_id, str(e))
            if request.session_id in session_cache:
                del session_cache[request.session_id]
            raise e

@app.post("/v1/recognize")
async def recognize(request: RecognitionRequest):
    started = time.perf_counter()
    stage = 'decode_audio'
    try:
        reject_poisoned_session(request.session_id)
        validated_language_context(request.language_context_code, request.basis_final_ids)
        if not request.audio.startswith("data:audio/wav;base64,"):
            raise ValueError("不支持的音频格式")
        
        base64_data = request.audio.split("data:audio/wav;base64,")[-1]
        wav_bytes = base64.b64decode(base64_data)
        
        with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
            frames = wf.readframes(wf.getnframes())
            sample_width = wf.getsampwidth()
            num_channels = wf.getnchannels()
            wav_sample_rate = wf.getframerate()
        
        if sample_width == 2:
            audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
        else:
            audio = np.frombuffer(frames, dtype=np.float32)
        
        if num_channels > 1:
            audio = audio.reshape(-1, num_channels).mean(axis=1)

        if wav_sample_rate != request.sample_rate:
            logger.warning(
                "sample rate mismatch request=%s wav=%s session_id=%s",
                request.sample_rate,
                wav_sample_rate,
                request.session_id,
            )
        
        loop = asyncio.get_running_loop()
        stage = 'executor_inference_wait'
        result = await asyncio.wait_for(
            loop.run_in_executor(executor, run_inference, request, audio),
            timeout=ASR_REQUEST_TIMEOUT_SECONDS
        )
        stage = 'response'
        logger.info("Recognition completed session_id=%s final=%s elapsed_seconds=%.3f",
                    request.session_id, request.is_final, time.perf_counter() - started)
        return {**result, "session_id": request.session_id, "is_final": request.is_final}
        
    except Exception as e:
        timed_out = isinstance(e, TimeoutError) and stage == 'executor_inference_wait'
        if timed_out and request.session_id is not None:
            with poison_lock:
                poisoned_session_ids.add(request.session_id)
        bad_request = stage == 'decode_audio' and isinstance(e, (ValueError, wave.Error, EOFError))
        status_code = (409 if isinstance(e, SessionPoisonedError) else 504 if timed_out
                       else 400 if bad_request else 500)
        detail = {'error_type': type(e).__name__, 'stage': stage,
                  'phase': 'inference' if stage == 'executor_inference_wait' else stage,
                  'elapsed_seconds': round(time.perf_counter() - started, 3),
                  'timeout_seconds': ASR_REQUEST_TIMEOUT_SECONDS,
                  'session_id': request.session_id, 'is_final': request.is_final,
                  'message': str(e) or type(e).__name__,
                  'executor_may_still_be_running': timed_out,
                  'session_poisoned': timed_out or isinstance(e, SessionPoisonedError)}
        if isinstance(e, SessionPoisonedError):
            detail['error_type'] = 'SessionPoisoned'
        logger.exception("Recognition failed: %s", detail)
        # wait_for cancellation cannot stop an already running executor thread.
        # The caller receives a terminal HTTP error; do not retry inference here.
        raise HTTPException(status_code=status_code, detail=detail) from e

def main():
    global model, ASR_REQUEST_TIMEOUT_SECONDS, startup_warmup
    parser = argparse.ArgumentParser(description="Qwen3 ASR服务器")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-ASR-1.7B")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--gpu-memory-utilization", type=float, required=True)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--kv-cache-memory-bytes", type=int, default=1024 * 1024 * 1024)
    parser.add_argument("--max-num-seqs", type=int, default=1)
    parser.add_argument("--max-num-batched-tokens", type=int, default=8192)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--dtype", choices=["bfloat16"], default="bfloat16")
    parser.add_argument("--request-timeout-seconds", type=float, default=90.)
    args = parser.parse_args()
    if not 30. <= args.request_timeout_seconds <= 90.:
        raise ValueError('ASR request timeout must be30..90 seconds')
    ASR_REQUEST_TIMEOUT_SECONDS = args.request_timeout_seconds
    if args.max_num_seqs != 1 or not args.enforce_eager or args.max_num_batched_tokens != args.max_model_len:
        raise ValueError("XT ASR requires one eager BF16 request chain with batched-token cap equal to max_model_len")
    
    model = Qwen3ASRModel.LLM(
        model=args.model,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        kv_cache_memory_bytes=args.kv_cache_memory_bytes,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        enforce_eager=args.enforce_eager,
        dtype=args.dtype,
        max_new_tokens=32,
        trust_remote_code=True
    )
    if args.host != "127.0.0.1":
        raise RuntimeError("Qwen3-ASR may bind only to loopback")
    # One synthetic startup decode, within the existing supervisor startup
    # deadline. Never insert this state into session_cache or publish its text.
    warmup_started = time.perf_counter()
    try:
        warm_state = model.init_streaming_state(unfixed_chunk_num=2, unfixed_token_num=5, chunk_size_sec=2.0)
        model.streaming_transcribe(np.zeros(2 * 16000, dtype=np.float32), warm_state)
        model.finish_streaming_transcribe(warm_state)
        del warm_state
    except Exception:
        logger.exception('Synthetic ASR startup warmup failed elapsed_seconds=%.3f', time.perf_counter() - warmup_started)
        raise
    startup_warmup = {'kind': 'synthetic-startup', 'input_seconds': 2., 'sample_rate': 16000,
                      'seconds': time.perf_counter() - warmup_started, 'output_discarded': True, 'movie_credit': False}
    logger.info('Synthetic ASR startup warmup completed: %s', startup_warmup)
    uvicorn.run(app, host=args.host, port=args.port, workers=1)

if __name__ == "__main__":
    main()
