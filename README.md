# tsnl2: Inductive KG pipeline (CSR-backed)

This repo implements the inductive knowledge-graph system from the ILPC-style design doc using the exact forward CSR format produced by `gencsr.cpp`. Everything is C++20 with mmap-based IO; no Python dependencies.

## gencsr.cpp format (authoritative)
- `offsets.bin`: `uint32_t offsets[N+2]`, `offsets[v]` is the start of node `v`’s out-adjacency (1-based IDs, `offsets[0]=0`, `offsets[N+1]=M`).
- `csr.bin`: `uint32_t csr[M]`, out-neighbor node IDs.
- `rels.bin`: `uint16_t rels[M]`, relation IDs aligned with `csr`.
- `entities.bin`: `uint32_t entities[N]`, internal node `i` (1-based) maps to Wikidata Q-ID number `entities[i-1]`.
- `props.bin`: `uint16_t props[R]`, internal relation `r` (1-based) maps to P-ID number `props[r-1]`.
All loaders consume these files as-is; no reformatting occurs.

## Building
```
cmake -S . -B build
cmake --build build -j
```
Executables: `kg_train`, `kg_infer`, `kg_eval`, `kg_build_reverse`, and the test `small_sanity`.

## Reverse CSR (optional)
If you need reverse edges for in-degree features, generate them once:
```
./kg_build_reverse --input data --output data
```
This writes `offsets_rev.bin`, `csr_rev.bin`, `rels_rev.bin` alongside the forward files.

## Training (`kg_train`)
Binary triples must be laid out as `{uint32_t h, uint32_t r, uint32_t t}` using internal IDs. Example:
```
./kg_train --train train.bin --data data --reverse data \
  --epochs 5 --batch 512 --dim 128 --layers 2 --fanout1 20 --fanout2 10 \
  --negatives 5 --lambda_rel 1.0 --lr 0.001 --optimizer adam --checkpoint ckpt.bin
```
The encoder is GraphSAGE-style (1–2 layers), structural features only (log degrees plus optional noise), relation-aware aggregation, DistMult decoder + relation classifier, manual backprop, and SGD/Adam optimizers. Checkpoints store encoder/decoder weights, feature config, and optimizer moments.

## Inference (`kg_infer`)
Loads a checkpoint and precomputes an embedding cache (batch-wise, using the same fanouts as training). Query files are binary triples:
- Relation inference: use `(h, r, t)` with `r` optional; prints top-K relations and rank of `r` if provided.
- Tail prediction: uses `(h, r, t)`; prints top-K tails and the filtered rank of `t`.
Example:
```
./kg_infer --data data --reverse data --checkpoint ckpt.bin \
  --relation_queries relq.bin --tail_queries tailq.bin --topk 5
```

## Evaluation (`kg_eval`)
Computes filtered MRR/Hits@{1,3,10,100} for a set of triples:
```
./kg_eval --data data --reverse data --checkpoint ckpt.bin \
  --eval eval.bin --train train.bin
```
Filtering removes any tail that appears as a known true `(h,r,*)` in the union of train and eval triples.

## Tests
`ctest` runs `small_sanity`, which builds a tiny 3-node graph on disk, executes a forward/backward step, and checks that loss decreases after one SGD update.

## Notes
- IDs are 1-based; ID 0 is reserved.
- Reverse CSR is optional; if missing, in-degree features default to zero so checkpoints remain loadable.
- All binaries expect the on-disk format from `gencsr.cpp` without modification.
