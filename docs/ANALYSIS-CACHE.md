# Immutable analysis cache

Source implementation only. No import, compile, inference, or runtime test has
been performed for this change.

## Public interface

`AnalysisCache(root, namespace)` stores raw inference observations below `root`.
`namespace` is any JSON-compatible value identifying the effective model bytes,
configuration, backend/version, and source preprocessing policy. It is detached
when the cache is constructed. Callers must include every setting that changes
inference; the cache cannot infer an omitted model or policy revision.

`get_or_compute(operation, inputs, compute)` uses a nonempty operation name,
canonical JSON inputs, and the namespace to create a SHA-256 key. `compute` is a
zero-argument callable. The returned value is detached JSON data on both hits and
misses: dictionaries/lists/scalars, never a shared tensor, NumPy array, or mutable
object retained in memory. NumPy arrays/scalars become ordinary JSON values via
their conversion protocol without importing NumPy or a model library. Tuples
become lists. Mapping keys must be strings; NaN, infinity, circular structures,
unsupported objects, and more than 64 nested levels are rejected.

`stats()` returns a separate dictionary of integer counters: `hits`, `misses`,
`computations`, `compute_failures`, `conversion_failures`, `corrupt_reads`,
`read_failures`, `writes`, `write_failures`, `oversized_results`, `lock_bypasses`,
`nested_bypasses`.
These describe cache operations, not model correctness or movie coverage.

`file_identity(path)` hashes one regular file in 1 MiB blocks and returns
`kind`, `sha256`, and `size_bytes`. File symlinks resolve through the normal file
open. Directory input raises `IsADirectoryError`; identify the exact model/config
files or supply their independently pinned hashes. Location and timestamps are
not identity. Missing/unreadable files raise their normal filesystem exception.
The caller must keep source/model files stable while hashing and inference run.

`array_identity(array, sample_rate=16000)` requires NumPy-compatible `shape`,
`dtype`, and `tobytes(order="C")`. It returns `kind`, `sha256`, `shape`, `dtype`,
`sample_rate`, and `size_bytes`. The digest uses logical C-order sample bytes;
geometry, dtype/byte order, and sample rate are separate identity fields. It
rejects object/structured arrays and invalid geometry/sample rates. Callers must
not mutate the array during identity creation or the related computation.

## Key policy and integration

Cache immutable observations: decoded/source-policy audio identity, ASR text and
timestamps, diarization segments, raw embeddings in one explicitly identified
embedding space, and translation from immutable source text/settings. Include
language hints and relevant decode/inference options. Caption-derived text is
its own input with its own text/provenance policy.

Do not include mutable actor names, actor-bank revisions, current reference
selection, or current assignment thresholds in raw ASR/diarization/embedding
keys. Do not cache actor bindings as though they were raw model observations.
Rebinding an existing embedding to a changed actor bank is a cheap later step.
When a setting really changes a raw observation, it belongs in that operation's
namespace or inputs. Stock diarization and a different clustering policy must
therefore not share a diarization key.

An embedding consumer must convert the returned float list back to its desired
array type outside this helper. Tensor-valued model output must first be moved
and converted deliberately by the model owner; this helper never causes device
transfers or tensor/model imports implicitly.

## Concurrency, corruption, and failure behavior

The module retains 256 process-wide lock stripes. Every key is serialized within
the process, including across cache instances. Hash collisions can serialize
unrelated keys; no per-key lock dictionary or result/audio memory cache grows
with a movie. Nested cache calls can read completed entries. A nested miss
computes immediately without acquiring another lock or publishing, increments
`nested_bypasses`, and returns a detached value. This prevents a wrapper around
diarization from deadlocking when its backend also caches an embedding, including
when unrelated keys share a stripe. Avoid redundant cache wrappers if independent
reuse of the inner operation is wanted. Computations must terminate; a function
that recursively asks for its own unfinished result is a caller error.

Across processes, misses also use one of 256 OS file locks per cache root.
An exited/crashed process releases its OS lock. A waiter re-reads after acquiring
the lock. A lock unavailable for 200 milliseconds, an unsupported lock facility, or an
unwritable cache volume degrades to uncached computation and increments
`lock_bypasses`. This rare path may duplicate inference across processes; lease
ownership/resource limits remain the pipeline's responsibility. It never writes
without the disk lock. An already valid atomic entry remains readable even when
the cache volume is read-only.

Entries contain their key, format, and result checksum. A bounded read ignores
unknown, malformed, truncated, oversized, mismatched, or non-finite records and
computes again. Cache records are limited to 64 MiB. Failed computations propagate
their original exception and never create an entry. Invalid results propagate a
conversion error, also without caching. Filesystem write failures return the
successful inference result uncached. Valid entries are reused without rewriting.

Writes use a unique temporary file beside the destination, flush/fsync, then
atomic `os.replace`. Only this call's temporary file is removed in cleanup.
Interrupted temporary files are ignored on read. No partial record is published.
The cache does not evict disk entries or delete abandoned files automatically;
the application owns cache-root lifecycle and disk-capacity policy. In-memory
results are not retained after the call returns.

## Limits of this source change

This helper does not prove device choice, backend determinism, model quality,
speaker correctness, subtitle coverage, or throughput. Input/model fingerprints
must be supplied correctly by the caller. A process-locked miss waits for at most
200 milliseconds of lock contention before proceeding uncached; process cancellation/preemption is handled
by the owning analysis child. No runtime claim is made until an authorized
end-to-end run exercises the assembled pipeline.
