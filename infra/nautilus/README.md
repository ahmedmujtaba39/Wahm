# WAHM judge training on NRP Nautilus

This uses the established `nautilus` Kubernetes context and `aiea-interns`
namespace. It contains no credentials.

The stages are separated deliberately:

1. `pvc.yaml` creates a WAHM-specific 20 GiB volume. It does not reuse the
   Shepherd-AI ReadWriteOnce claim.
2. `cache-job.yaml` is the networked stage. It checks out one immutable WAHM
   commit, downloads the pinned AraBERT revision, and records every file's size
   and SHA-256 checksum.
3. `train-job.yaml` verifies every cached file, requires CUDA, runs offline,
   trains on question-disjoint partitions, calibrates the threshold on
   validation, evaluates the untouched test partition, and writes the model and
   `judge_metadata.json` to the PVC.

Do not submit from an uncommitted or unpushed tree. Both Jobs contain
`__WAHM_GIT_COMMIT__`; replace it with the exact pushed commit.
The cache job reads the reviewed branch from the public
`cyberuniversal/Wahm` fork because the authenticated account cannot push
directly to `ahmedmujtaba39/Wahm`.

```powershell
kubectl config use-context nautilus
kubectl apply -f infra/nautilus/pvc.yaml

$commit = git rev-parse HEAD
(Get-Content infra/nautilus/cache-job.yaml -Raw).Replace(
  '__WAHM_GIT_COMMIT__', $commit
) | kubectl apply -f -

kubectl -n aiea-interns logs -f job/wahm-judge-cache
kubectl -n aiea-interns wait --for=condition=complete `
  job/wahm-judge-cache --timeout=2h

(Get-Content infra/nautilus/train-job.yaml -Raw).Replace(
  '__WAHM_GIT_COMMIT__', $commit
) | kubectl apply -f -

kubectl -n aiea-interns logs -f job/wahm-judge-train
kubectl -n aiea-interns wait --for=condition=complete `
  job/wahm-judge-train --timeout=24h
```

To retrieve the final model, start a temporary pod mounting
`wahm-judge-539948` and use `kubectl cp`. The durable path is:

`/workspace/results/wahm-judge/<commit>/arabert-gold-answer`

Preserve failed Job YAML, pod YAML, events, and logs before deleting or
resubmitting. Never merge smoke-test metrics into the paper results.
