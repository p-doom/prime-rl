# Downstream maintenance

This guide is the source of truth for the changes intentionally carried on
`downstream/main`. It is for maintainers who synchronize that branch with
`main`, which mirrors upstream PrimeRL. It does not cover unmerged feature
branches or changes that happen to be absent because `downstream/main` has not
yet been synchronized.

## Branch roles

- `main` mirrors upstream. Do not merge downstream-only changes into it.
- `downstream/main` is the integration branch for the deployed fork.
- Fork fixes should be developed in a short-lived branch based on
  `downstream/main` and merged back through a pull request.
- Upstream updates flow from `main` into `downstream/main`, never in the other
  direction.

A direct two-dot diff from `main` to `downstream/main` mixes the downstream
patches with every upstream commit that the downstream branch has not absorbed.
Use the merge base and the three-dot range when auditing the fork:

```bash
git fetch origin main downstream/main
fork_base=$(git merge-base origin/main origin/downstream/main)
git log --left-right --graph --oneline origin/main...origin/downstream/main
git diff --stat "$fork_base"..origin/downstream/main
git diff origin/main...origin/downstream/main
```

The current intentional overlay consists only of the six changes below. The
asset-root requirement for projects that embed PrimeRL is tracked separately
because it is not a change to this fork.

## Advertise inference to external environment servers

Source: [downstream PR #2](https://github.com/p-doom/prime-rl/pull/2)

Files:

- `packages/prime-rl-configs/src/prime_rl/configs/inference.py`
- `src/prime_rl/entrypoints/rl.py`

HAICORE runs environment servers outside the node that hosts a single-node
PrimeRL launcher. A model-client URL such as `http://localhost:8000` works for
processes on the inference node, but not for those external environment
servers. Setting the following option makes the launcher advertise a reachable
node address to rollout clients:

```toml
[inference.server]
advertise = true
```

The default is `false`; address rewriting must remain opt-in. When enabled on
the local/single-node launch path, the implementation:

1. derives the advertised host from a specific non-loopback
   `inference.server.host`, or from `SLURMD_NODENAME` and then
   `socket.gethostname()` when the server binds to an unspecified address;
2. replaces only loopback and unspecified hosts in
   `orchestrator.model.client.base_url`;
3. leaves already-remote URLs unchanged;
4. preserves scheme, credentials, port, path, query, fragment, and IPv6
   bracket syntax; and
5. saves the original local URL in `admin_base_url` when that field was not
   explicitly configured, so launcher-local administrative traffic remains
   local.

The launcher must reject advertisement from an explicitly loopback-only bind
address. Rewriting must happen before the orchestrator subconfiguration is
serialized. Multi-node deployments already construct routable inference and
router endpoints and must not be routed through this single-node behavior.

This area is a semantic conflict hotspot even when Git merges it cleanly.
Upstream changes to router ownership, model-client URL construction,
`write_subconfigs()`, or the local/Slurm launch split can silently move the
point at which rewriting must occur. After a sync, verify both the URL seen by
external environment servers and the administrative URL used on the inference
node.

Retire this patch only after upstream provides equivalent opt-in behavior for
external environment servers in single-node deployments. Remove the config
field, launcher helpers, and downstream config overrides together.

## Pin Mooncake for the HAICORE glibc baseline

Source: [downstream PR #3](https://github.com/p-doom/prime-rl/pull/3)

Files:

- `pyproject.toml`
- `uv.lock`

The upstream Mooncake releases selected by
`mooncake-transfer-engine>=0.3.10.post2` require glibc 2.35 or newer. That ABI
baseline is not available on the HAICORE compute environment, so the fork pins
the last validated compatible release:

```toml
mooncake-transfer-engine==0.3.8.post1 ; sys_platform == 'linux'
```

Upstream declares Mooncake in the `gpu` optional dependency group, so the pin
lives there too.

The manifest and lockfile must agree on that exact version. During a merge,
take upstream dependency updates for unrelated packages, restore the exact
Mooncake constraint, and regenerate the lockfile with `uv`; do not resolve a
large `uv.lock` conflict by retaining either side wholesale.

Do not remove or raise this pin based only on package metadata or a successful
install on a login node. Validate the candidate wheel's glibc requirements and
the relevant Mooncake runtime path on every target compute architecture through
Slurm. Once the cluster baseline or upstream wheel compatibility makes the pin
unnecessary, update `pyproject.toml` and `uv.lock` in the same change.

## Use the W&B Workspaces GraphQL client

Files:

- `src/prime_rl/monitors/wandb/overview.py`

The declared W&B dependency range allows newer SDK releases, including 0.28.1,
that no longer provide the `wandb_gql` module or the legacy
`wandb.Api().client.execute()` interface. A consuming project with such a lock
therefore failed while importing the monitor, before a training job could
start. `list_views()` now sends its query through
`wandb_workspaces._graphql.execute_graphql()`, the helper shipped with the
Workspaces package that already owns the saved-view workflow.

Preserve the query variables and the existing result parsing when resolving
W&B or Workspaces changes. This code intentionally passes the query as a string
because the Workspaces helper parses it using the GraphQL implementation that
matches the installed W&B SDK. Do not restore a direct `wandb_gql` dependency
or reach back into the removed API client interface.

The Workspaces helper is an internal API, so dependency updates are a conflict
hotspot even when the import still succeeds. After changing either W&B
dependency, verify that the monitor imports in the consuming environment and
that saved-view discovery still works. Retire this patch when upstream uses a
supported Workspaces API for discovering saved project views.

## Preserve component configs in shared W&B runs

Files:

- `src/prime_rl/monitors/wandb/monitor.py`
- `tests/unit/utils/test_wandb_monitor.py`

RL launches use one shared W&B run with the orchestrator as the primary writer
and the trainer as a secondary writer. Upstream passes each component's config
to `wandb.init()` without a namespace, and the secondary trainer config is not
retained in the shared run. Namespace shared configs by their writer label and
explicitly publish them after attaching, so W&B stores both
`orchestrator.optim.lr` and `trainer.optim.lr` without changing which value the
trainer uses.

Retire this patch when upstream preserves namespaced configs from every writer
in shared W&B runs.

## Install NIXL with an external Python environment

Source: [downstream PR #10](https://github.com/p-doom/prime-rl/pull/10)

Files:

- `scripts/install_nixl_from_source.sh`
- `docs/advanced.md`

Projects that install PrimeRL as an editable dependency use the parent
project's virtual environment rather than `prime-rl/.venv`. Set
`PRIME_RL_VENV_BIN` to that environment's `bin` directory before running the
NIXL installer. The installer validates the selected interpreter and passes it
explicitly to `uv pip`, while UCX and the built wheel remain under the PrimeRL
checkout.

Keep the default `prime-rl/.venv/bin` behavior for standalone checkouts. Do not
replace this explicit override with environment discovery.

## Re-parse the single-node `rl.json`

Source: upstream [PR #3924](https://github.com/PrimeIntellect-ai/prime-rl/pull/3924),
carried until it merges

Files:

- `src/prime_rl/entrypoints/rl.py`
- `tests/unit/test_configs.py`

A single-node Slurm launch writes the resolved config to `rl.json`, and the job
runs `rl @ rl.json`. Upstream PR #3788 made `rl` reject a set
`weight_broadcast.inference_world_size`, which that dump always contains for an
NCCL or NIXL broadcast, so every such job failed before it started.
`write_config` now leaves the field out. Drop this patch at the first sync that
contains upstream's fix.

## Preserve native assets when embedding PrimeRL

This is a consuming-project integration contract, not part of the
`downstream/main` overlay. The NIXL and llm-d installers write native assets to
the PrimeRL checkout's `third_party/` directory. If a parent project overrides
`slurm.project_dir` and carries custom Slurm templates, those templates must
still resolve UCX and llm-d from `deps/prime-rl/third_party` (or pass an
equivalent explicit asset root). Pointing them at the parent project's
`third_party/` silently ignores the installed assets.

## History that is not part of the current fork

The downstream merge graph retains commits whose changes were later removed.
Do not infer the live fork delta from downstream-only commit subjects alone.

- The early site-specific `uv sync` argument work added launcher, template,
  documentation, skill, utility, and test changes. The Mooncake fix replaced
  that workaround with the exact dependency pin and removed the sync-argument
  mechanism. Do not resurrect it during conflict resolution unless there is a
  new, independently reviewed site-wide requirement.
- [Downstream PR #4](https://github.com/p-doom/prime-rl/pull/4) removed
  `#SBATCH --exclusive` from the single-node RL template, but downstream PR #6
  explicitly reverted that change. The current policy remains an exclusive
  single-node RL allocation.
- [Downstream PR #6](https://github.com/p-doom/prime-rl/pull/6) taught the
  TOML config serializer to omit null fields of nested dataclasses, such as
  OSWorld's `DesktopPoolConfig`. The 2026-10-07 sync retired it: upstream now
  writes resolved configs as JSON, which keeps nulls, so there is no TOML
  serializer left to patch.
- [Downstream PR #9](https://github.com/p-doom/prime-rl/pull/9) synchronized
  upstream as a squash commit, which dropped upstream's ancestry and left the
  merge base at 2026-07-20. The 2026-10-07 sync is a true merge of upstream
  `main` whose tree is upstream plus the overlay above, which restores that
  ancestry. Keep later syncs as merges.

## Upstream synchronization checklist

1. Fetch both maintained branches and start from a clean checkout of the
   downstream branch.

   ```bash
   git fetch origin main downstream/main --prune
   git switch --create chore/sync-upstream-YYYYMMDD origin/downstream/main
   git status --short
   ```

2. Audit both sides of the divergence. Read upstream release notes and breaking
   configuration changes as well as the file diff.

   ```bash
   git log --left-right --cherry-pick --oneline origin/main...HEAD
   git diff --stat origin/main...HEAD
   git merge-tree --write-tree HEAD origin/main
   ```

   `git merge-tree` exits nonzero when it predicts conflicts; it does not change
   the checkout.

3. Merge upstream into the sync branch. Preserve merge ancestry rather than
   rebasing the shared downstream integration branch.

   ```bash
   git merge --no-ff origin/main
   ```

4. Resolve the known hotspots deliberately:

   - In `pyproject.toml`, accept unrelated upstream dependency changes but keep
     `mooncake-transfer-engine==0.3.8.post1` until its retirement criteria are
     met. Regenerate `uv.lock` after the manifest is resolved.
   - In `inference.py` and `rl.py`, integrate upstream router and launch-flow
     changes while retaining opt-in single-node advertisement before
     subconfiguration serialization.
   - In `monitors/wandb/overview.py`, keep saved-view queries on the GraphQL
     client provided by the installed Workspaces package, and in
     `monitors/wandb/monitor.py`, keep shared-run configs namespaced by writer.
   - In `install_nixl_from_source.sh`, retain `PRIME_RL_VENV_BIN` and ensure
     every Python package operation targets the selected interpreter.
   - Treat changes to environment/source schemas, model-client URL ownership,
     and generated Slurm configs as integration changes even if Git reports no
     textual conflict.

5. Update submodules and synchronize the environment before validation, never
   from inside a live Slurm job.

   ```bash
   git submodule update --init --recursive
   uv lock
   uv sync --all-extras
   ```

6. Run CPU validation on the login node; do not run training, GPU tests, heavy
   preprocessing, or benchmarks there. At minimum:

   - run `git diff --check` and `uv run ruff check`;
   - run the configuration serializer tests with
     `uv run pytest -q tests/unit/test_configs.py`;
   - exercise URL rewriting for loopback, unspecified, already-remote, and IPv6
     inputs, including preservation of `admin_base_url`;
   - write and reparse a resolved OSWorld configuration containing a
     `DesktopPoolConfig` with both null and non-null optional paths;
   - run a downstream RL `--dry-run` and inspect the generated orchestrator TOML
     and Slurm script; and
   - confirm `pyproject.toml` and `uv.lock` both resolve Mooncake to
     `0.3.8.post1`.

7. Submit runtime, GPU, or allocation-level validation with `sbatch`. Record the
   job IDs, configs, logs, output paths, and tested node architectures in the
   pull request. Check jobs with `squeue -u "$USER"` and `sacct -j <jobid>`.

8. Review the final downstream overlay again. Every remaining change relative
   to the upstream merge base should map to one of the six fork sections above
   or be explained in this guide before merge.

9. Push the sync branch and open a draft pull request targeting
   `downstream/main`. Never use an upstream sync pull request to update `main`,
   and do not merge until the required Slurm validation has completed.
