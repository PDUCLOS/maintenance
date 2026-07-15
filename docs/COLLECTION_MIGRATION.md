# Collection migration — embedding dimension change

> Procedural reference for the moment the embedding model (and therefore
> the vector dimension) changes. The ChromaDB collection stores vectors
> with a fixed dimension, so a model switch requires a new collection
> (you cannot change the dim of an existing collection).

## Why this matters

ChromaDB stores vectors in an HNSW index. The index dimension is fixed
at collection creation: it is set by the **first vector you upsert**,
and cannot be changed afterwards. If you try to upsert a vector with a
different dimension, Chroma will reject the upsert and the new vectors
will silently be missing from the index (or the API will 400).

This is what happens when the embedding model is swapped, for example:

| Migration | Old dim | New dim | Affected collections |
|-----------|--------:|--------:|----------------------|
| bge-small-en-v1.5 → bge-m3 | 384 | 1024 | All collections built with bge-small |
| bge-small → multilingual-e5-large | 384 | 1024 | All collections |
| bge-m3 → Stella-en-1.5B | 1024 | 4096 / 8192 | All collections |

## The ingest pipeline does NOT auto-reset

By design, `python -m src.ingestion.pipeline` (and `make ingest`) will
**not** reset the collection. If the dim is wrong, you'll get a 400 from
Chroma on the first upsert, with no data loss.

This is intentional: an auto-reset would silently destroy an existing
index. The rule is: **no destructive operation without an explicit
human command**.

## The 2 safe migration paths

### Path A — Drop and recreate (simple, destructive)

Use this when the old collection is genuinely disposable (e.g. you've
already validated the new pipeline end-to-end and don't need the old
baseline anymore).

```bash
# 1. Inspect current state
python scripts/manage_collection.py list
python scripts/manage_collection.py info cmapss_kb

# 2. Make sure you know what you're dropping
make collection-drop NAME=cmapss_kb
#   (Makefile will ask you to retype the name as a confirmation)

# 3. Re-ingest
make ingest
```

### Path B — New collection alongside (recommended, non-destructive)

Use this when you want to keep the old collection as a baseline (for
A/B comparison) or when you're not 100% sure the new model is better.

```bash
# 1. Inspect current state
python scripts/manage_collection.py list

# 2. Create the new collection (alongside the old one)
python scripts/manage_collection.py new cmapss_kb_bge_m3

# 3. Point the project at the new collection
python scripts/manage_collection.py use cmapss_kb_bge_m3
#   (updates .env — restart make api / make ui to pick up)

# 4. Re-ingest
make ingest

# 5. Verify the new collection has the right dim and count
python scripts/manage_collection.py info cmapss_kb_bge_m3

# 6. (Optional) Once you're satisfied, drop the old one
python scripts/manage_collection.py drop cmapss_kb --yes
#   OR keep it around for an A/B comparison
```

## Refusing to drop the active collection

`scripts/manage_collection.py drop` will refuse to drop the collection
currently set as active in `settings.chroma_collection`. Switch to
another collection first:

```bash
python scripts/manage_collection.py use cmapss_kb_bge_m3
python scripts/manage_collection.py drop cmapss_kb --yes   # now allowed
```

## Estimating the rebuild cost

The rebuild time scales linearly with the number of chunks. The
bottleneck is the embedding step (Metal-accelerated MPS for bge-small,
CPU for sentence-transformers) and the upsert (HTTP roundtrips to
Chroma).

| Chunks | Embedding (bge-small, MPS) | Embedding (bge-m3, MPS) | Upsert to Chroma |
|-------:|--------------------------:|------------------------:|-----------------:|
| 5 000  | ~30 s                     | ~1 min                  | ~30 s            |
| 10 000 | ~1 min                    | ~2 min                  | ~1 min           |
| 20 000 | ~2 min                    | ~4 min                  | ~2 min           |

These are wall-clock times on a MacBook Pro M5 Pro. The CPU-only path
for the embedding is 3-5x slower.

## Volume cost (disk)

| Collection | Dim | Chunks | Vectors only | HNSW overhead | Total |
|------------|----:|-------:|-------------:|--------------:|------:|
| bge-small  | 384 | 10 000 | 15 MB        | ~20 MB        | ~35 MB |
| bge-m3     | 1024 | 10 000 | 40 MB        | ~50 MB        | ~90 MB |

Keeping both for an A/B comparison costs ~125 MB on the Docker volume.
Drop the old one when you're done to reclaim the disk.

## Related commands

```bash
make collection-list                    # list all
make collection-info COLLECTION=name   # show details
make collection-new NAME=name          # create empty
make collection-drop NAME=name          # drop with confirm
make collection-use NAME=name          # set active in .env
```
