# Source and runtime lineage

Recorded R016 performance was measured against:

- vLLM-Ascend baseline commit: `e8f47fc11c81f3eb3efeb9b40400db8b0fa6eef3`
- candidate reconstruction: apply `inputs/SFA-LSE-R013-C01-BA07.patch.gz` to that baseline
- vLLM reference commit: `ba07e4a48fc951300d97eb506217dd530583dea3`

Decision-relevant source hashes from the recorded run:

| Source | Baseline SHA256 | Candidate SHA256 |
|---|---|---|
| `vllm_ascend/attention/context_parallel/sfa_cp.py` | `17d80412c16270ac0da8a188bdd6d6b7d8b108fbbf92752a6ddf965898085e57` | `015e2bd76fe5910736c32ae46ef8bfe08c5589f467560515112b58e28f125795` |
| `vllm_ascend/ops/triton/sfa_cp.py` | `c5d5a2b4d516a81b53effa88ee4ba861800dd833a6a085908609e6b757e1fcef` | `9124bea2df152af8e3114dd42b6ffebee9dfb0223c9e1839d3f280f6eacb8eb9` |
| SFA producer `vllm_ascend/device/device_op.py` | `c64dc4f693da5c537315fb66c9dd121b16be4ef9a7eaaae45390d1871760ba1d` | same |

Recorded runtime:

- CANN 9.0.1
- Python 3.12.13
- PyTorch 2.10.0
- torch-npu 2.10.0.post2
- Linux aarch64
- two visible NPUs for the two-rank lane

The production PR was subsequently rebased onto newer upstream `main`. Performance
was not re-measured after every upstream movement because the decision-relevant
Local-SFA callsite/pack seam remained byte-identical, and focused post-rebase
host/NPU/HCCL/ACL-Graph regressions passed.
