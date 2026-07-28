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
gpuq submit "<command>" [--name NAME] [--gpus N] [--mem SIZE] [--share]
```

Your shell returns immediately. A background process holds your place in line
and runs the command when gpuq grants a GPU.

| Flag | Default | Meaning |
|------|---------|---------|
| `--name NAME` | derived from command | Label for the job and its log file |
| `--gpus N` | `1` | GPUs required — **all N at once or none** |
| `--mem SIZE` | none | Per-GPU memory, e.g. `8G`, `512M`, `1.5G` (unit required) |
| `--share` | off | Allow co-locating with other `--share` jobs (requires `--mem`) |

```bash
gpuq submit "python train.py --lr 0.001" --name baseline
gpuq submit "python ddp_train.py" --name ddp --gpus 2 --mem 40G
gpuq submit "python eval.py" --name eval --gpus 1 --mem 8G --share
```

Validation errors (bad `--mem`, `--share` without `--mem`, asking for more GPUs
than exist, daemon down) are reported in your shell right away and nothing is
queued.

### `status` / `mine`

```bash
gpuq status        # all users
gpuq mine          # your jobs only, plus "N running, M queued"
```

```
ID      USER     STATUS    POS  GPU  SUBMITTED         STARTED           NAME/CMD
7a3f    alice    queued    1    -    2026-05-16 14:30  -                 baseline
3b1c    bob      running   -    0    2026-05-16 14:15  2026-05-16 14:20  finetune
```

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
- **Best-fit:** gpuq picks the smallest GPU(s) that fit, keeping big cards free
  for jobs that need them.
- **Skip, don't block:** if the next-in-line job can't fit right now (e.g. it
  needs 2 GPUs but only 1 is free), gpuq runs a smaller job that fits instead.
  A large/exclusive job can therefore wait a while behind a stream of small
  ones — there's no priority bump in v1.
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
- **If the daemon restarts while your job is running**, gpuq marks it `failed`
  and stops tracking it (your process may keep running and logging on its own,
  but its GPU slot won't be auto-freed). Re-check with `gpuq status`.

---

## Troubleshooting

| Symptom | Cause / fix |
|---------|-------------|
| `cannot reach gpuqd at /run/gpuq/gpuq.sock` | The daemon isn't running — ask an admin (`sudo systemctl status gpuq`). |
| `gpuq: command not found` | `/usr/local/bin` not on your `PATH`, or open a fresh shell. |
| `Permission denied` on the socket | An admin set a restrictive `socket_mode`; ask them. Normally the socket is open to all local users. |
| Job stuck `queued` for a long time | GPUs are busy, or your job needs more/bigger GPUs than are currently free. Check `gpuq status` and GPU use with `nvidia-smi`. |

For the full reference (install, daemon, config, internals) see `README.md`.
