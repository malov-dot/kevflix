"""Native cloning-only process. This module never imports Torch or ASR models."""
import argparse,json,sys,queue,threading,time
from pathlib import Path
from .actor_voice import ActorVoice
from .movie_faults import fault_record
from .resource_budget import ResourceBudget


def emit(value):
    print(json.dumps(value),flush=True)


def failure(error):
    """Failure envelope; explicit fault metadata (fault_kind/fault_code/detail) travels with it unchanged."""
    value={'error':str(error),'kind':type(error).__name__}
    fault=fault_record(error)
    if fault is not None:value['fault']=fault
    return value


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--manifest',required=True)
    parser.add_argument('--budget',required=True)
    parser.add_argument('--capacity',required=True,type=float)
    args=parser.parse_args();engine=None
    try:
        budget=ResourceBudget(**json.loads(Path(args.budget).read_text(encoding='utf-8')))
        engine=ActorVoice(args.manifest,backend='cuda',threads=2,max_reference_seconds=args.capacity,budget=budget)
        emit({'ready':True})
        import soundfile as sf
        requests=queue.Queue(maxsize=1)
        def receive():
            try:
                for value in sys.stdin:requests.put(value)
            finally:requests.put(None)
        threading.Thread(target=receive,daemon=True).start()
        idle_since=time.monotonic()
        while True:
            try:line=requests.get(timeout=.25)
            except queue.Empty:
                if budget.should_yield():
                    emit({'error':budget.reason,'kind':'ResourceYield',
                          'fault':{'kind':'resource','code':'actor_budget_yield','detail':str(budget.reason)}});break
                if time.monotonic()-idle_since>60:
                    emit({'error':'Actor worker released its idle model','kind':'ResourceYield',
                          'fault':{'kind':'resource','code':'actor_worker_idle_release',
                                   'detail':'Actor worker released its idle model'}});break
                continue
            if line is None:break
            if len(line)>32768:raise ValueError('Actor request exceeds bounded size')
            request=json.loads(line)
            if request.get('action')=='close':break
            try:
                audio,receipt=engine.synthesize(request['text'],request['reference'],request.get('reference_text',''),seed=int(request.get('seed',42)))
                output=Path(request['output']);temporary=output.with_suffix('.tmp.wav')
                sf.write(temporary,audio,48000,subtype='FLOAT');temporary.replace(output)
                emit({'receipt':receipt,'seconds':len(audio)/48000})
                idle_since=time.monotonic()
            except Exception as error:
                emit(failure(error))
                break
    except Exception as error:
        emit(failure(error))
    finally:
        if engine:engine.close()


if __name__=='__main__':main()
