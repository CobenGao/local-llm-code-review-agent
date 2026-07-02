# Local LLM Code Review Agent (GitLab CI + Ollama + LangGraph)

> 🚀 **Featured on Towards AI**
>
> **Building a Zero-Trust AI Code Review Agent with GitLab, LangGraph, and Qwen3-Coder**
>
> https://medium.com/towards-artificial-intelligence/building-a-zero-trust-ai-code-review-agent-with-gitlab-langgraph-and-qwen3-coder-4dd17dbca145

Companion code for the published Towards AI article.

> 🔒 Every byte of code reviewed by this agent stays inside your LAN.
> No cloud API, no external network call — built for industries
> (automotive, embedded, fintech) where source code can never
> leave the building.

## What this is

A GitLab CI/CD pipeline that automatically reviews every Merge Request's
Python diff using a **fully local** LLM (`qwen3-coder:30b` served by
[Ollama](https://ollama.com)), orchestrated by a
[LangGraph](https://github.com/langchain-ai/langgraph) state machine, and
posts inline review comments (with one-click "Apply suggestion" fixes)
directly on the MR — exactly like a senior reviewer would.

## Repository structure

```
.
├── .gitlab-ci.yml                 # CI pipeline definition (review stage)
├── ci/
│   └── ai_agent_review.py         # LangGraph agent: diff → review → post
└── src/utils/
    └── data_processor.py          # intentionally-buggy demo file used to
                                    # trigger and validate the 4 rule categories
```

## How it works (short version)

1. A Merge Request is opened/updated → GitLab Runner (shell executor)
   picks up the `ai_agent_review` job.
2. `ci/ai_agent_review.py` pulls the MR's Python diff via the GitLab
   REST API v4.
3. A 3-node LangGraph state machine (`analyze → execute_review →
   validate_format`, with an automatic retry loop) sends the diff plus a
   strict system prompt to the local Ollama endpoint (`/api/chat`).
4. The model must return a JSON array of findings; a hand-rolled
   sanitizer/validator repairs common JSON formatting issues before
   parsing, and retries (max 3×) on failure.
5. Each finding is posted back as an **inline discussion** on the exact
   line, using GitLab's `suggestion` block syntax so reviewers get a
   one-click "Apply suggestion" button.

Full write-up and reasoning behind the `num_ctx=65536` context window
are in the article — link above. For the two nastiest GitLab Runner
DevOps pitfalls we hit (stale `external_url` + `su -` shell profile
crash) and their fixes, see **[TROUBLESHOOTING.md](./TROUBLESHOOTING.md)**.

## Reproducing this locally

```bash
# 1. Pull and configure the model
curl -fsSL https://ollama.com/install.sh | sh
ollama pull qwen3-coder:30b

cat > Modelfile <<'EOF'
FROM qwen3-coder:30b
PARAMETER num_ctx 65536
PARAMETER num_predict 8192
EOF
ollama create qwen3-coder-review -f Modelfile

# 2. Install the agent's dependencies
pip install langgraph requests

# 3. Set the required GitLab CI variables (see ci/ai_agent_review.py
#    "环境校验" / validate_env section for the full list), then run:
python ci/ai_agent_review.py
```

## GitLab project setup (web UI)

Before the pipeline can run, configure these three things on your
GitLab project:

1. **Settings → CI/CD → Runners** — register/enable a Runner and give it
   a tag (e.g. `ai-review`) that matches the `tags:` field in
   `.gitlab-ci.yml`.
2. **Settings → CI/CD → Variables** — add `AI_REVIEW_TOKEN`, a GitLab
   Personal/Project Access Token with `api` scope, so the agent can post
   comments back to the MR. Mark it **masked** (and **protected** if
   your pipeline only runs on protected branches).
3. **Settings → CI/CD → Job token permissions** — allow this project's
   CI job token to access the GitLab API for the current project (needed
   for the agent's REST calls to succeed under the built-in
   `CI_JOB_TOKEN`, in addition to `AI_REVIEW_TOKEN`).

## Disclaimer

The IP addresses, project IDs and tokens shown in the article are
illustrative placeholders taken from a private lab environment — replace
them with your own GitLab instance's values. **Never commit real
`PRIVATE-TOKEN` / Runner registration tokens to source control**; this
agent reads all secrets from CI/CD environment variables at runtime.

## License

MIT — see [LICENSE](./LICENSE).
