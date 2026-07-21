# Vendored benchmark model source

These are third-party model repositories, committed here as **vendored source** so the
benchmarks are reproducible: each carries local modifications that are NOT in upstream, so a
plain `git clone` will not reproduce them.

Their nested `.git` directories were removed so the files could be tracked by this repo
(git cannot commit files inside a nested repository). The upstream base each was taken from is
recorded below — diff against that commit to see the local changes.

| model | upstream | base commit | branch | commits ahead | locally modified files |
| --- | --- | --- | --- | --- | --- |
| GEARS | https://github.com/snap-stanford/GEARS.git | `1202ffe65e` | master | 1 | 5 |
| GenePert | https://github.com/zou-group/GenePert | `31c0034211` | main | 0 | 7 |
| scGPT | https://github.com/bowang-lab/scGPT.git | `cebd6fae65` | main | 0 | 94 |
| scLAMBDA | https://github.com/gefeiwang/scLAMBDA.git | `a5be849797` | main | 0 | 10 |
| scouter | https://github.com/PancakeZoy/scouter.git | `bf763aaf87` | master | 0 | 17 |

Licenses remain those of the upstream projects (see each subdirectory).

A copy of the original `.git` histories was kept outside the repository at
`CIPHER_models_git_backup/` at the time of vendoring.
