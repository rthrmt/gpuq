# gpuq — User Guide

`gpuq` queues your GPU jobs so they run automatically when a GPU is free. You
submit a command; it waits in line; when a GPU opens up, gpuq sets
`CUDA_VISIBLE_DEVICES` for you, runs your command **as you**, saves the output
to a log, and frees the GPU when it finishes.

You don't install or start anything — an admin runs the daemon. If `gpuq status`
works, you're ready to go.

---

## 30-second start

```bash
gpuq submit "python train.py" --name myrun --gpus 1   # queue a job; your shell returns at once
gpuq status        # the queue (everyone's jobs)
gpuq mine          # just your jobs, with a summary
gpuq logs <id> -f  # follow output (Ctrl-C to stop)
gpuq cancel <id>   # drop a queued job, or kill a running one (asks first)
gpuq priority <id> --top   # run this queued job before your other queued jobs
gpuq history       # recently finished jobs
```

`<id>` is the short 4-character job id shown in `gpuq status` / `gpuq submit`.

---

## Try it — a test job

Run this to confirm gpuq works for you end-to-end. It just prints the GPU it was
given and sleeps 20 seconds, so it's safe to run on a shared machine:

```bash
gpuq submit "python3 -c 'import os,time; print(\"got GPU:\", os.environ.get(\"CUDA_VISIBLE_DEVICES\")); time.sleep(20)'" --name test --gpus 1
```

You'll see something like `Job <id> queued (position 1)` and a log path. Then:

```bash
gpuq status            # your job, RUNNING, with a GPU number assigned
gpuq logs <id> -f      # use your real id; should print "got GPU: <n>", Ctrl-C to stop
gpuq history --mine    # after ~20s: STATUS done, EXIT 0
```

If the log shows `got GPU: <some number>` and history reports exit `0`, you're
all set. (If `python3` isn't your interpreter, swap in the right path — see
[Using your own environment](#using-your-own-environment-venv--conda).)

---

## Commands

### `submit` — queue a job

```
gpuq submit "<command>" [--name NAME] [--gpus N] [--mem SIZE] [--share] [--priority N | --top]
                        [--email ADDR | --no-email]
```

Your shell returns immediately. A background process holds your place in line
and runs the command when gpuq grants a GPU.

| Flag | Default | Meaning |
|------|---------|---------|
| `--name NAME` | derived from command | Label for the job and its log file |
| `--gpus N` | `1` | GPUs required — **all N at once or none** |
| `--mem SIZE` | none | Per-GPU memory, e.g. `8G`, `512M`, `1.5G` (unit required) |
| `--share` | off | Allow co-locating with other `--share` jobs (requires `--mem`) |
| `--priority N` | `0` | Order among **your own** queued jobs; higher runs first |
| `--top` | off | Queue ahead of all your other queued jobs |
| `--email ADDR` | your default | Email `ADDR` (comma-separated for several) when the job ends |
| `--no-email` | off | No email for this job, even if you set a default |

```bash
gpuq submit "python train.py --lr 0.001" --name baseline
gpuq submit "python ddp_train.py" --name ddp --gpus 2 --mem 40G
gpuq submit "python eval.py" --name eval --gpus 1 --mem 8G --share
```

Validation errors (bad `--mem`, `--share` without `--mem`, asking for more GPUs
than exist, daemon down) are reported in your shell right away and nothing is
queued.

#### Email when a job ends

`--email you@example.org` sends a short mail when the job finishes, fails or is
cancelled while running: exit code, runtime, GPUs, log path and the last 20
lines of the log. To get it for every job, set a default once:

```ini
# ~/.config/gpuq/config
[notify]
email = you@example.org
```

`--email` overrides it for one job, `--no-email` turns it off for one job. The
mail comes from your own account via the server's mail system. If it can't be
sent, a `# gpuq: could not send email ...` line is added to the job's log.

### `status` / `mine`

```bash
gpuq status        # all users
gpuq mine          # your jobs only, plus "N running, M queued"
```

```
ID      USER     STATUS    POS  PRIO  GPU  SUBMITTED         STARTED           NAME/CMD
3b1c    bob      running   -    -     0    2026-05-16 14:15  2026-05-16 14:20  finetune
7a3f    alice    queued    1    -     -    2026-05-16 14:30  -                 baseline
```

Queued jobs are listed in the order they are expected to start (`POS`), which
follows the fairness rule rather than submission time. `PRIO` shows a queued
job's priority when it isn't the default `0`.

### `logs` — view output

```bash
gpuq logs 7a3f       # print the log so far
gpuq logs 7a3f -f    # follow live (like tail -f), Ctrl-C to stop
```

Logs live in `~/.gpuq/logs/` and contain a metadata header (id, command, user,
assigned GPUs) followed by merged stdout + stderr. Works for finished jobs too.

### `cancel` — drop a queued job, or kill a running one

You can cancel only your own jobs.

```bash
gpuq cancel 7a3f     # cancel one
gpuq cancel --all    # cancel all your queued jobs (never kills running ones)
```

If the job is already **running**, gpuq shows you what you're about to kill and
waits for an answer:

```
Job 7a3f (trainjob) is running on GPU 0:
  python train.py
Kill it? [y/N]
```

Answer `y` and the job gets `SIGTERM`, then `SIGKILL` 10 seconds later if it
hasn't exited — so if your script cleans up on `SIGTERM`, it gets the chance.
Anything your command started is killed too, so nothing is left holding the GPU.
The job shows up in `gpuq history` as `cancelled`.

In a script (no terminal to prompt on), pass `--force`:

```bash
gpuq cancel 7a3f --force
```

### `priority` — run one of your queued jobs sooner (or later)

Just submitted something more important than what you already have waiting?
Move it to the front of **your** queue — nothing gets cancelled:

```bash
gpuq priority 7a3f --top   # ahead of all your other queued jobs
gpuq priority 7a3f 5       # or set a number: higher runs first, default is 0
gpuq priority 7a3f -1      # send it behind your other jobs
```

Or do it in one go when submitting:

```bash
gpuq submit "python urgent.py" --name urgent --top
```

This only reorders **your own** jobs: the promoted job takes the place in line
that your oldest queued job had, and the rest of yours shift back. It never
puts you ahead of anyone else, and it only works on jobs that are still queued.
(`gpuq prio` is a shorter alias.)

### `history` — recently finished

```bash
gpuq history                  # last 20 finished/failed/cancelled (all users)
gpuq history --mine           # your jobs only
gpuq history --limit 5 --mine
```

---

## Using your own environment (venv / conda)

`gpuq` runs on its own Python — your job uses **your** environment, not gpuq's.
The catch: your environment is captured **when you submit**, not when the job
later starts. So the reliable patterns are:

```bash
# 1. Activate first, then submit (simplest)
source ~/myproject/.venv/bin/activate     # or: conda activate myproject
gpuq submit "python train.py" --name baseline

# 2. Or make it explicit in the command (runs in a shell, so && works)
gpuq submit "source ~/myproject/.venv/bin/activate && python train.py" --name baseline

# 3. Or call the interpreter by absolute path
gpuq submit "~/myproject/.venv/bin/python train.py" --name baseline
```

> Note: if `python` on this machine is Python 2, use `python3` (or an absolute
> path to your interpreter) to avoid surprises.

---

## How your job gets scheduled (the short version)

- **Fairness:** when a GPU frees up, the user with the fewest currently-running
  jobs goes first; ties broken oldest-first. No one can hog every GPU.
- **Priority:** only decides which of *your own* queued jobs goes next; it
  can't be used to jump ahead of other users.
- **Best-fit:** gpuq picks the smallest GPU(s) that fit, keeping big cards free
  for jobs that need them.
- **Skip, don't block:** if the next-in-line job can't fit right now (e.g. it
  needs 2 GPUs but only 1 is free), gpuq runs a smaller job that fits instead.
  A large/exclusive job can therefore wait a while behind a stream of small
  ones — including your own lower-priority ones; a priority orders your jobs,
  it doesn't hold a GPU back for the big one.
- **Exclusive vs shared:** by default a job owns its GPU(s) entirely. Add
  `--share --mem SIZE` to let gpuq pack your job onto a GPU alongside other
  `--share` jobs when the memory fits.

---

## Things to know

- **`--mem` is advisory, not enforced.** gpuq uses it to *place* jobs; it does
  not cap your VRAM. A `--share` job that overruns its `--mem` (or an external
  process on the same card) can OOM a neighbour. Need hard isolation? Use the
  default exclusive mode.
- **`--gpus N` is all-or-nothing** — a 2-GPU job waits until 2 GPUs are free
  together.
- **Daemon restarts and upgrades don't lose your jobs.** Queued jobs keep
  their place and running jobs stay tracked; a job that finishes while the
  daemon is down reports its result when it comes back. This relies on the
  small background process `gpuq submit` leaves behind for each job — don't
  kill it. If it does die, a queued job is `cancelled` and a running one is
  marked `failed` (its process may keep going, but gpuq no longer tracks it).

---

## Troubleshooting

| Symptom | Cause / fix |
|---------|-------------|
| `cannot reach gpuqd at /run/gpuq/gpuq.sock` | The daemon isn't running — ask an admin (`sudo systemctl status gpuq`). |
| `gpuq: command not found` | `/usr/local/bin` not on your `PATH`, or open a fresh shell. |
| `Permission denied` on the socket | An admin set a restrictive `socket_mode`; ask them. Normally the socket is open to all local users. |
| Job stuck `queued` for a long time | GPUs are busy, or your job needs more/bigger GPUs than are currently free. Check `gpuq status` and GPU use with `nvidia-smi`. |

For the full reference (install, daemon, config, internals) see `README.md`.
