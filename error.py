watch -n 5 'ps -ef | grep -E "docker build|buildkit|containerd" | grep -v grep; echo "---"; docker system df; echo "---"; docker images | grep nemotron_finetuned'

Every 5.0s: ps -ef | grep -E "docker build|...  cx-speech: Mon Jul  6 11:06:02 2026

root         405       1  0 10:57 ?        00:00:01 /usr/bin/containerd
root       20402    1741  0 11:03 pts/2    00:00:00 docker build --progress=plain -
---
TYPE            TOTAL     ACTIVE    SIZE      RECLAIMABLE
Images          4         1         164.9GB   91.02GB (55%)
Containers      1         0         345.2MB   345.2MB (100%)
Local Volumes   0         0         0B        0B
Build Cache     29        16        17.54GB   14.46GB
---
WARNING: This output is designed for human readability. For machine-readable output
, please use --format.
