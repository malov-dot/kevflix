"""Bounded owned-process IPC: the legacy actor voice model runs only in its child; failures keep their fault records."""
import json,os,queue,subprocess,sys,threading,time
from pathlib import Path
from .media import atomic_json
from .movie_faults import MovieConfigurationError,from_record
from .resource_budget import ResourceYield


class ActorWorkerYield(ResourceYield):
    """The actor worker reported a real resource hold (fault_kind 'resource'): the waiting_resources path, with its
    fault record."""
    fault_kind='resource'
    def __init__(self,code,detail):
        self.fault_code,self.detail=str(code),str(detail)
        super().__init__(self.fault_code+': '+self.detail)


def worker_error(value):
    """The exception for one worker failure envelope. Explicit fault records win; only a resource record or the
    legacy ResourceYield kind is a wait; everything else is terminal (never a soft wait)."""
    fault=value.get('fault')
    if value.get('kind')=='InterruptedError' and not isinstance(fault,dict):return InterruptedError(value['error'])
    if isinstance(fault,dict):
        error=from_record(fault)
        return ActorWorkerYield(error.fault_code,error.detail) if error.fault_kind=='resource' else error
    if value.get('kind')=='ResourceYield':return ActorWorkerYield('actor_worker_yield',value['error'])
    return MovieConfigurationError(value.get('code') or 'actor_worker_failed','%s: %s'%(value.get('kind'),value['error']))


class ActorClient:
    def __init__(self,manifest,capacity,budget,owner,folder,python=None):
        self.owner=owner;self.closed=False;self.responses=queue.Queue()
        owner.check()
        folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
        budget_file=folder/'actor-resource-budget.json'
        atomic_json(budget_file,dict(stage='voice-cloning',status_files=[str(p) for p in budget.status_files],
            coordination_file=str(budget.coordination_file) if budget.coordination_file else None,
            hold_file=str(budget.hold_file) if budget.hold_file else None,
            reserve_gpu_mib=budget.reserve_gpu_mib,required_gpu_mib=budget.required_gpu_mib,reserve_ram_mib=budget.reserve_ram_mib,
            process_names=sorted(budget.process_names),require_process_absence=budget.require_process_absence))
        env=dict(os.environ,PYTHONUTF8='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
        env.pop('CUDA_VISIBLE_DEVICES',None)
        self.log=(folder/'actor-native.log').open('ab')
        options={'creationflags':subprocess.CREATE_NO_WINDOW|subprocess.BELOW_NORMAL_PRIORITY_CLASS} if os.name=='nt' else {'start_new_session':True}
        self.child=subprocess.Popen([str(python or sys.executable),'-m','relay.actor_child','--manifest',str(manifest),
            '--budget',str(budget_file),'--capacity',str(capacity)],stdin=subprocess.PIPE,stdout=subprocess.PIPE,
            stderr=self.log,text=True,encoding='utf-8',bufsize=1,env=env,cwd=Path(__file__).resolve().parent.parent,**options)
        try:owner.register(self.child)
        except BaseException:
            owner.unregister(self.child)
            self.child.stdin.close();self.child.stdout.close();self.log.close()
            raise
        def read():
            try:
                for line in self.child.stdout:
                    try:self.responses.put(json.loads(line))
                    except ValueError:continue
            finally:self.responses.put({'error':'Actor process closed its response stream','kind':'RuntimeError',
                                        'code':'actor_worker_exited'})
        self.reader=threading.Thread(target=read,name='actor-responses',daemon=True);self.reader.start()
        try:
            if not self.receive(120).get('ready'):raise RuntimeError('Actor process did not initialize')
        except BaseException:self.close();raise

    def receive(self,timeout):
        deadline=time.monotonic()+timeout
        while time.monotonic()<deadline:
            self.owner.check()
            try:value=self.responses.get(timeout=.1)
            except queue.Empty:continue
            if value.get('error'):
                error=worker_error(value)
                if getattr(error,'fault_code',None)=='actor_worker_idle_release':
                    # Director decision: a wait only when the worker REALLY released its model and exited; a worker
                    # that reports idle release but keeps running is a broken worker (fatal).
                    try:self.child.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        raise MovieConfigurationError('actor_worker_idle_release_invalid',
                            'Actor worker reported an idle release but did not exit') from error
                raise error
            return value
        raise TimeoutError('Owned actor worker exceeded its request deadline')

    def synthesize_file(self,text,reference,reference_text,output,seed):
        self.owner.check()
        self.child.stdin.write(json.dumps({'text':text,'reference':str(reference),'reference_text':reference_text,
                                          'output':str(output),'seed':seed})+'\n')
        self.child.stdin.flush()
        return self.receive(180)

    def close(self):
        if self.closed:return
        self.closed=True
        try:
            if self.child.poll() is None:
                try:self.child.stdin.write('{"action":"close"}\n');self.child.stdin.flush()
                except (OSError,ValueError):pass
                try:self.child.wait(timeout=5)
                except subprocess.TimeoutExpired:self.owner.terminate(self.child)
        finally:
            self.owner.unregister(self.child)
            self.reader.join(timeout=2)
            self.child.stdin.close();self.child.stdout.close();self.log.close()
