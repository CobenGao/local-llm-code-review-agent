# Troubleshooting: GitLab Runner on a private, non-Docker AI server

These are the two hardest, least-documented issues we hit connecting a
GitLab server to a GitLab Runner installed directly on an AI inference
box (shell executor, no containers). Both look like unrelated
infrastructure bugs at first glance — they only show up once you combine
a shell executor, a non-root execution user, and a customized shell
environment, which is exactly the setup most private on-prem GitLab + AI
server deployments end up with.

## Issue 1 — Runner keeps cloning from the wrong (stale) IP

**Symptom.** The GitLab server's real IP had changed, but the
`external_url` baked into GitLab's own database/web UI still pointed at
the old address. Without database admin rights to fix that at the
source, every job the Runner picked up tried to clone from the stale
URL and failed.

**Fix — override at the Runner and pipeline level, without touching
GitLab's database.**

`/etc/gitlab-runner/config.toml`:
```toml
[[runners]]
  url = "http://<your-gitlab-real-ip>:<port>"
  clone_url = "http://<your-gitlab-real-ip>:<port>"
```

In the pipeline, ignore the auto-injected (and stale) `CI_SERVER_URL`
and define a known-good address instead:
```yaml
variables:
  GITLAB_ACTUAL_URL: "http://<your-gitlab-real-ip>:<port>"
```
`ci/ai_agent_review.py` reads `GITLAB_ACTUAL_URL` first and only falls
back to `CI_SERVER_URL` if it isn't set.

## Issue 2 — the job dies during "prepare environment", no useful error

**Symptom.** `prepare environment: exit status 1`. The shell executor
runs as root but switches into a regular user via `su - <user>` in
*login shell* mode for every job. If that user's shell profile has
anything unfriendly to a non-interactive session — a blocking Conda
init hook in `.bashrc`, or `clear_console` in `.bash_logout` — the
session gets killed before your script ever runs.

**Fix — three layers of defense.**

1. Tell the Runner to skip automatic profile loading, in the pipeline:
```yaml
variables:
  FF_DISABLE_AUTOMATIC_PROFILE_LOADING_ON_SHELL_EXECUTORS: "true"
```
2. In `~/.profile`, make init hooks fail-safe instead of fatal:
```bash
[ -f "$HOME/.cargo/env" ] && . "$HOME/.cargo/env" || true
```
3. In `~/.bash_logout`, remove the console-clearing block that runs on
   logout:
```bash
if [ "$SHLVL" = 1 ]; then
    [ -x /usr/bin/clear_console ] && /usr/bin/clear_console -q
fi
```
If PAM resource limits also interfere, comment out
`session required pam_limits.so` in `/etc/pam.d/su`.

## Why the shell executor (not Docker)?

The agent needs to call directly into a Conda-managed Python
interpreter and reach `localhost:11434` (Ollama) with minimal latency.
A container layer only adds friction for a job that's already running
on trusted, single-tenant hardware.
