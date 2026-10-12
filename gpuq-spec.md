# `gpuq` — Lightweight GPU Job Scheduler: Implementation Spec

## Overview

A lightweight, single-node GPU job scheduler for multi-user Linux servers.
No containers, no Kubernetes. Users submit jobs from their own shell session;
jobs run as the submitting user. A central daemon acts purely as a traffic
controller — it never impersonates or switches users.

---

## Architecture

```
User shell                Background waiter            Daemon (root)
    |                           |                           |
    | gpuq submit "cmd"         |                           |
    |-------------------------->| fork, parent returns       |
    |  (shell free immediately) |                           |
    |                           |-- register(job) --------->|
    |                           |   (blocks, waiting)       | (queues job)
    |                           |                           |
    |                           |                           | GPU becomes free
    |                           |                           | fairness check
    |                           |<-- go(CUDA_VISIBLE_DEVICES=N) ---|
    |                           |                           |
    |                           | redirect stdout/stderr    |
    |                           | to log file               |
    |                           | exec(cmd) as self         |
    |                           | ...job runs...            |
    |                           |-- done(job_id, exit_code)->|
    |                           | exit                      | (GPU freed)
```

**Key property:** The daemon never spawns or owns user processes.
The background waiter process is forked from the user's shell — it is
already running as the correct user before it ever contacts the daemon.

---

## Components

### 1. Daemon — `gpuqd`

- Runs as `root` via systemd
- Owns a Unix domain socket at `/var/run/gpuq/gpuq.sock` (group-writable
  so all users can connect)
- Maintains job state in SQLite at `/var/lib/gpuq/queue.db`
- Polls `nvidia-smi` to track GPU availability
- Applies fairness scheduling (see below)
- Never touches user processes after sending "go"

### 2. CLI + Waiter — `gpuq`

Single Python script installed system-wide (e.g. `/usr/local/bin/gpuq`).

Subcommands:
- `gpuq submit <cmd> [--name <jobname>] [--gpus <n>]` — queue a job
- `gpuq status [--mine]` — show queue and running jobs (all users, or own only)
- `gpuq mine` — shorthand for `gpuq status --mine`, plus a summary line
- `gpuq logs <job_id> [-f]` — print (or tail) log file for a job
- `gpuq cancel <job_id> [--force]` — cancel a queued job, or kill a running one
  (confirms first unless `--force`)
- `gpuq cancel --all` — cancel all of the current user's queued jobs
- `gpuq priority <job_id> (<n> | --top)` — reorder one of the user's own queued
  jobs (`submit` takes `--priority <n>` / `--top` too)
- `gpuq history [--limit N] [--mine]` — show recently completed/failed jobs

On `submit`:
1. Determine log file path (see Log File Naming below)
2. Print job ID, queue position, and log file path to terminal
3. **Fork immediately** — parent exits (shell returns to user)
4. Child connects to daemon socket, sends `register` message, blocks

### 3. Systemd Service — `gpuq.service`

Starts `gpuqd` at boot, restarts on failure.

---

## IPC Protocol

All messages are newline-delimited JSON over the Unix socket.
Each connection represents one job lifecycle or one status query.

### Job submission (waiter → daemon)
```json
{
  "type": "register",
  "job_id": "uuid4",
  "user": "alice",
  "cmd": "python train.py --lr 0.001",
  "gpus_requested": 1,
  "log_file": "/home/alice/.gpuq/logs/python_train.py_20260516_143022.log",
  "submitted_at": "2026-05-16T14:30:22Z"
}
```

### Daemon grants execution
```json
{ "type": "go", "gpu_ids": [2] }
```

### Job completion (waiter → daemon)
```json
{ "type": "done", "job_id": "uuid4", "exit_code": 0 }
```

The daemon answers `done` with `{ "type": "done_ack" }`; a waiter that sees
the connection close without it reattaches and reports again.

### Reattach after a daemon restart (waiter → daemon)
```json
{ "type": "reattach", "version": 2, "job_id": "uuid4", "user": "alice", "state": "running", "exit_code": null }
```
Sent on a *new* connection by a waiter whose connection dropped without an
explicit `error` (the daemon restarted). `state` is the job as the waiter sees
it:

| `state` | Daemon's answer | Then |
|---------|-----------------|------|
| `queued` | `reattached` | waiter waits for `go` (sent at once if the DB already says `running`: the old daemon recorded the placement but its `go` was lost) |
| `running` | `reattached` | daemon waits for `done`; `terminate` works as before |
| `finished` (+ `exit_code`) | `reattached` | daemon records the exit code and closes; also acknowledged if already recorded |

Refused with `error` (`not_found`, `permission_denied`, `not_reattachable`)
when the job is unknown, belongs to someone else, is no longer active, or
already has a waiter attached. A refused waiter stops tracking; it never kills
a running command on its own.

Waiters retry with backoff (at most 5s apart) for up to 24h.

### Protocol versioning
`register` and `reattach` carry `"version"` and `status_response` carries
`"protocol_version"` (absent = 1). A waiter keeps running the code it was
started with across daemon upgrades, so the daemon must keep accepting every
version still in use: messages only ever change additively.

### Requester identity
The `user` / `requesting_user` fields are claims. The daemon reads the peer's
uid from the socket (`SO_PEERCRED`) and uses that user instead, unless the peer
is root or runs as the daemon's own uid — those may act for any user.

### Status query (any client → daemon)
```json
{ "type": "status" }
```

### Status response (daemon → client)
```json
{
  "type": "status_response",
  "jobs": [
    {
      "job_id": "uuid4",
      "user": "alice",
      "name": "baseline_run",
      "cmd": "python train.py --lr 0.001",
      "status": "running",
      "gpu_ids": [2],
      "queue_position": null,
      "submitted_at": "2026-05-16T14:30:22Z",
      "started_at": "2026-05-16T14:35:10Z",
      "finished_at": null,
      "exit_code": null,
      "log_file": "/home/alice/.gpuq/logs/baseline_run_20260516_143022.log"
    }
  ]
}
```

Fields explained:
- `name` — the `--name` value if provided, otherwise null
- `queue_position` — 1-based position in queue for `queued` jobs, null otherwise
- `started_at` / `finished_at` — null until the relevant event occurs
- `exit_code` — null until finished; 0 = success, non-zero = failure

### Cancel request
```json
{ "type": "cancel", "job_id": "uuid4", "requesting_user": "alice", "force": false }
```
`force` must be `true` to cancel a *running* job; the daemon rejects an
unforced running-job cancel with `cancel_requires_force`, so a client cannot
skip the confirmation.

### Terminate (daemon -> waiter)
```json
{ "type": "terminate", "job_id": "uuid4" }
```
Sent on the running job's existing waiter connection. The waiter `SIGTERM`s the
command's process group, waits 10s, then `SIGKILL`s it, and reports the exit
code back through the normal `done` message.

### Cancel all request
```json
{ "type": "cancel_all", "requesting_user": "alice" }
```
Cancels all `queued` jobs belonging to `requesting_user`.
Returns a response listing how many jobs were cancelled.

### Prioritize request
```json
{ "type": "prioritize", "job_id": "uuid4", "requesting_user": "alice", "priority": 5, "top": false }
```
Sets a `queued` job's priority (integer, default 0, higher first, bounded to
±1000000). With `top: true` the daemon ignores `priority` and picks one more
than the highest priority among the owner's other queued jobs. `register`
accepts the same two fields, and the `queued` ack echoes the resulting
`priority`.

```json
{ "type": "prioritize_response", "priority": 5, "previous": 0, "queue_position": 1, "name": "baseline_run" }
```

Prioritize rules:
- A user may only reprioritize their own jobs; root may reprioritize any job
- Only `queued` jobs (`not_queued` otherwise)
- Priority orders a user's own jobs only — see Fairness Scheduling

Cancel rules:
- A user may only cancel their own jobs
- Root may cancel any job
- Queued jobs are cancelled outright; running jobs are killed, but only with
  `force` (the CLI confirms first)
- A killed job is recorded `cancelled`, not `failed` — its non-zero exit came
  from gpuq's own signal, not the user's command
- `cancel_all` only ever touches queued jobs

---

## Fairness Scheduling

Goal: prevent any single user from monopolizing all GPUs.

**Algorithm (simple weighted fair queuing):**

When a GPU becomes free and there are queued jobs:

1. Group queued jobs by user
2. For each user, count how many jobs they currently have **running**
3. Select the job belonging to the user with the **fewest running jobs**
4. Ties broken by submission time (FIFO)

**Priority (within one user only):** each queued job contributes one FIFO slot
`(submitted_at, job_id)` to its owner. A user's queued jobs are assigned to that
user's slots in `(-priority, submitted_at, job_id)` order, and step 4 compares
slots. With equal priorities this is plain FIFO. Because a priority only
permutes jobs among slots the user already holds, it cannot move a job ahead of
another user's job.

This is simple to implement and understand, and naturally balances load
without complex configuration.

**Optional future extension:** per-user GPU quotas configurable in
`/etc/gpuq/quotas.conf`.

---

## GPU Detection

Use `nvidia-smi` to detect available GPUs:

```bash
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits
```

A GPU is considered **free** if:
- No job in the DB has it marked as in use, AND
- Its memory usage is below a configurable idle threshold
  (default: < 500 MB used, to catch orphaned processes)

On startup the daemon rebuilds its GPU reservations from the `running` rows in
the DB and waits for the waiters to reattach (see "Reattach after a daemon
restart").

---

## Log File Naming

Log files are stored in `~/.gpuq/logs/` (created if not exists).

**Default naming (from command):**
```
<sanitized_cmd>_<timestamp>.log
```
Sanitization: lowercase, replace non-alphanumeric chars with `_`,
truncate to 60 chars, deduplicate consecutive underscores.

Examples:
```
python_train.py_lr_0.001_20260516_143022.log
./run_experiment.sh_20260516_143022.log
```

**With `--name`:**
```
<name>_<timestamp>.log
```
Example: `gpuq submit "python train.py" --name baseline_run`
→ `baseline_run_20260516_143022.log`

Timestamp format: `YYYYMMDD_HHMMSS` (local time).

Log file contains:
- Header comment lines with job metadata (job_id, cmd, user, gpu assigned)
- Full stdout + stderr of the job (merged, with timestamps optional)

---

## SQLite Schema

```sql
CREATE TABLE jobs (
    job_id         TEXT PRIMARY KEY,
    user           TEXT NOT NULL,
    name           TEXT,                -- optional --name label
    cmd            TEXT NOT NULL,
    gpus_requested INTEGER NOT NULL DEFAULT 1,
    gpu_ids        TEXT,               -- JSON array, null until running
    status         TEXT NOT NULL,      -- queued | running | done | failed | cancelled
    log_file       TEXT NOT NULL,
    submitted_at   TEXT NOT NULL,      -- ISO8601
    started_at     TEXT,
    finished_at    TEXT,
    exit_code      INTEGER,
    priority       INTEGER NOT NULL DEFAULT 0  -- order among the owner's queued jobs
);

CREATE INDEX idx_status ON jobs(status);
CREATE INDEX idx_user   ON jobs(user);
```

---

## File Layout

```
/usr/local/bin/gpuq          # CLI + waiter (single script)
/usr/local/sbin/gpuqd        # daemon
/etc/gpuq/gpuq.conf          # daemon config (optional)
/var/run/gpuq/gpuq.sock      # Unix socket (runtime)
/var/lib/gpuq/queue.db       # SQLite DB
/var/log/gpuq/daemon.log     # daemon's own log
~/.gpuq/logs/                # per-user job output logs
```

---

## Configuration — `/etc/gpuq/gpuq.conf`

```ini
[daemon]
socket_path = /var/run/gpuq/gpuq.sock
db_path = /var/lib/gpuq/queue.db
log_path = /var/log/gpuq/daemon.log
poll_interval_seconds = 5
reattach_grace_seconds = 30

[gpus]
# Comma-separated list of GPU indices to manage.
# Defaults to all GPUs detected by nvidia-smi.
# managed_gpus = 0,1,2,3

# Memory threshold below which a GPU is considered idle (MB)
idle_memory_threshold_mb = 500

[fairness]
# Maximum GPUs any single user can hold simultaneously (0 = unlimited)
max_gpus_per_user = 0
```

---

## Systemd Unit — `gpuq.service`

```ini
[Unit]
Description=GPU Job Queue Daemon
After=network.target

[Service]
Type=simple
ExecStart=/usr/local/sbin/gpuqd
Restart=on-failure
RestartSec=5
StandardOutput=append:/var/log/gpuq/daemon.log
StandardError=append:/var/log/gpuq/daemon.log

[Install]
WantedBy=multi-user.target
```

---

## Socket Permissions

The socket must be accessible by all users who will submit jobs.

Recommended approach:
1. Create a group `gpuq` and add all allowed users to it
2. Daemon creates socket with group ownership `gpuq` and mode `0660`

OR for an open server where all local users are trusted, mode `0666`.

---

## Implementation Notes

- **Language:** Python 3.8+, stdlib only (no pip dependencies)
  - `sqlite3`, `socket`, `subprocess`, `os`, `json`, `threading`
- **Daemon concurrency:** One thread per connected client socket; a
  scheduler thread that wakes on job completion events or poll interval
- **Waiter process:** After fork, the child should `setsid()` to detach
  from the terminal fully, so it survives the user closing their shell
- **Shutdown:** Daemon catches SIGTERM and exits without touching job state;
  queued and running rows stay as they are for the next daemon
- **Startup reconciliation:** On daemon start, restore GPU reservations for
  `running` rows, hold the queue, and wait `reattach_grace_seconds` for the
  waiters of all active rows to reattach. Rows still unclaimed afterwards are
  finalised: `queued` → `cancelled`, `running` → `failed`

---

## Example User Workflow

```bash
# Submit a job, shell returns immediately
$ gpuq submit "python train.py --epochs 50 --lr 0.001" --name baseline
Job 7a3f queued (position 2)
Logs: /home/alice/.gpuq/logs/baseline_20260516_143022.log

# Check all active jobs (queued + running, all users)
$ gpuq status
ID      USER    STATUS    POS  GPU  SUBMITTED         STARTED           NAME/CMD
7a3f    alice   queued    2    -    2026-05-16 14:30  -                 baseline
8c9e    alice   queued    3    -    2026-05-16 14:31  -                 python_eval.py
3b1c    bob     running   -    0    2026-05-16 14:15  2026-05-16 14:20  ./finetune.sh
9d2a    carol   running   -    1    2026-05-16 14:22  2026-05-16 14:25  python eval.py

# Check only your own jobs, with summary
$ gpuq mine
Your jobs: 0 running, 2 queued

ID      STATUS    POS  GPU  SUBMITTED         STARTED  NAME/CMD
7a3f    queued    2    -    2026-05-16 14:30  -        baseline
8c9e    queued    3    -    2026-05-16 14:31  -        python_eval.py

# Tail logs once a job starts
$ gpuq logs 7a3f -f

# Cancel a single job
$ gpuq cancel 7a3f
Cancelled job 7a3f (baseline)

# Kill a job that is already running (confirms first)
$ gpuq cancel 2a1b
Job 2a1b (long_run) is running on GPU 1:
  python train.py
Kill it? [y/N] y
Killed job 2a1b (long_run)

# Cancel all your queued jobs at once
$ gpuq cancel --all
Cancelled 1 queued job(s).

# View recently completed jobs
$ gpuq history --mine --limit 5
ID      STATUS  GPU  SUBMITTED         STARTED           FINISHED          EXIT  NAME/CMD
2a1b    done    1    2026-05-16 12:00  2026-05-16 12:05  2026-05-16 13:42  0     long_run
```

---

## Out of Scope (for now)

- Multi-GPU jobs (`--gpus 2`) — schema supports it but scheduling logic
  can be simplified to single-GPU only in v1
- Web dashboard
- Email/notification on job completion
- Per-user quotas config file
- Job priorities / explicit priority levels
