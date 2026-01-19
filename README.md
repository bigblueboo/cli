# CLI Tools for AI Agents

A collection of 24 self-documenting command-line tools designed for AI agents. Each tool features comprehensive `--help` documentation, typed error handling with specific exit codes, and environment variable configuration.

## Quick Start

```bash
# Clone and setup
git clone https://github.com/bigblueboo/cli.git
cd cli
python -m venv .venv && source .venv/bin/activate
pip install -r <tool>/requirements.txt

# Run any tool
python -m <tool>.<tool> --help
```

## Tool Index

| Tool | Purpose | Primary Env Var |
|------|---------|-----------------|
| [calendar_schedule](#calendar_schedule) | Google Calendar management | `GOOGLE_APPLICATION_CREDENTIALS` |
| [code_review](#code_review) | AI-powered code review | `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` |
| [consult_pro](#consult_pro) | Long-running GPT-5.2 Pro queries | `OPENAI_API_KEY` |
| [data_extractor](#data_extractor) | Extract structured data from text | `OPENAI_API_KEY` |
| [db_query](#db_query) | Read-only SQL queries | Connection string |
| [doc_generator](#doc_generator) | Generate docs from code | `OPENAI_API_KEY` |
| [email_user](#email_user) | Send notification emails | `RESEND_API_KEY` |
| [gemini_query](#gemini_query) | Analyze media with Gemini | `GOOGLE_API_KEY` |
| [git_changelog](#git_changelog) | Generate changelogs from commits | None |
| [github_ops](#github_ops) | GitHub issues/PRs/repos | `GITHUB_TOKEN` |
| [google_docs](#google_docs) | Google Docs & Sheets | `GOOGLE_APPLICATION_CREDENTIALS` |
| [llm_query](#llm_query) | Query multiple LLM providers | `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` |
| [log_analyzer](#log_analyzer) | Parse and analyze logs | `OPENAI_API_KEY` |
| [mcp_bridge](#mcp_bridge) | Model Context Protocol servers | None |
| [prompt_manager](#prompt_manager) | Store/manage prompt templates | None |
| [s3_sync](#s3_sync) | Cloud storage operations | `AWS_ACCESS_KEY_ID` |
| [screenshot_url](#screenshot_url) | Capture web page screenshots | None |
| [semantic_search](#semantic_search) | Semantic file search | None |
| [shell_assist](#shell_assist) | Natural language to shell | `OPENAI_API_KEY` |
| [slack_notify](#slack_notify) | Send Slack messages | `SLACK_BOT_TOKEN` / `SLACK_WEBHOOK_URL` |
| [test_generator](#test_generator) | Generate tests from code | `OPENAI_API_KEY` |
| [text_transform](#text_transform) | Text/data transformation | None |
| [vector_search](#vector_search) | Semantic document search | None |
| [webhook_dispatch](#webhook_dispatch) | Send HTTP webhooks | None |

---

## Tools Reference

### calendar_schedule

Google Calendar management: create, update, delete events, check availability, list upcoming.

```bash
# List upcoming events
calendar_schedule list --days 7

# Create an event
calendar_schedule create --title "Team Sync" --start "2024-01-20T10:00:00" --duration 60

# Check free/busy
calendar_schedule freebusy --start "2024-01-20" --end "2024-01-21"
```

**Use when:** Scheduling meetings, checking availability, managing calendar events programmatically.

---

### code_review

AI-powered code review using OpenAI, Anthropic, or Google models with configurable focus areas.

```bash
# Review a file
code_review review src/main.py --provider openai

# Review with specific focus
code_review review src/auth.py --focus security,performance

# Review from stdin
git diff | code_review review - --format markdown
```

**Use when:** Automating PR reviews, checking code quality, identifying bugs before commit.

---

### consult_pro

Query GPT-5.2 Pro with background mode for complex tasks taking 5-50+ minutes.

```bash
# Submit a complex query
consult_pro "Analyze this codebase architecture and suggest improvements..."

# From file, maximum reasoning
consult_pro -f analysis_request.txt --effort xhigh

# Fire and forget (returns task ID)
consult_pro "Deep research task" --no-wait
```

**Use when:** Complex analysis requiring deep reasoning, research tasks, comprehensive reports.

---

### data_extractor

Extract structured data from unstructured text using LLMs with JSON Schema validation.

```bash
# Extract from file with schema
data_extractor extract document.txt --schema schema.json

# Extract from URL
data_extractor extract https://example.com/page --schema person.json

# Output as CSV
data_extractor extract data.txt --schema schema.json --format csv
```

**Use when:** Parsing invoices, extracting entities from documents, structuring messy data.

---

### db_query

Execute read-only SQL queries against PostgreSQL, MySQL, or SQLite databases.

```bash
# Query PostgreSQL
db_query "SELECT * FROM users LIMIT 10" --db "postgresql://user:pass@host/db"

# Query SQLite with JSON output
db_query "SELECT * FROM orders" --db "sqlite:///data.db" --format json

# From file
db_query -f report.sql --db "$DATABASE_URL"
```

**Use when:** Fetching data for reports, investigating database state, read-only analytics.

---

### doc_generator

Generate documentation from source code using LLMs in Markdown, RST, or HTML format.

```bash
# Generate docs for a file
doc_generator generate src/utils.py --format markdown

# Generate for directory
doc_generator generate src/ --output docs/ --recursive

# With specific provider
doc_generator generate api.py --provider anthropic --format rst
```

**Use when:** Auto-generating API docs, creating README sections, documenting legacy code.

---

### email_user

Send notification emails via Resend API to charliedeck@gmail.com.

```bash
# Send a notification
email_user -s "Build Complete" -b "All tests passed successfully"

# Send HTML email with tags
email_user -s "Deploy Alert" -b "<h1>Production deployed</h1>" --html --tag deploy

# Pipe content
echo "Task finished" | email_user -s "Status Update"
```

**Use when:** Notifying yourself of task completion, sending alerts, automated status updates.

---

### gemini_query

Analyze PDFs, images, and audio files using Google Gemini multimodal models.

```bash
# Analyze an image
gemini_query image photo.jpg "What objects are in this image?"

# Analyze a PDF
gemini_query pdf document.pdf "Summarize the key findings"

# Analyze audio
gemini_query audio recording.mp3 "Transcribe and summarize"
```

**Use when:** Extracting text from images, summarizing documents, transcribing audio.

---

### git_changelog

Generate changelogs from git commit history with conventional commit parsing.

```bash
# Generate changelog for last 10 commits
git_changelog --count 10

# Generate for version range
git_changelog --from v1.0.0 --to v2.0.0 --format markdown

# Group by type
git_changelog --since "2024-01-01" --group-by type
```

**Use when:** Preparing release notes, documenting changes, generating CHANGELOG.md.

---

### github_ops

GitHub operations: issues, pull requests, repositories, and comments.

```bash
# Create an issue
github_ops issue create --repo owner/repo --title "Bug report" --body "Description"

# List open PRs
github_ops pr list --repo owner/repo --state open

# Comment on PR
github_ops pr comment --repo owner/repo --number 123 --body "LGTM!"
```

**Use when:** Automating GitHub workflows, managing issues, CI/CD integrations.

---

### google_docs

Read, create, and modify Google Docs and Sheets.

```bash
# Read a document
google_docs read --doc-id "1abc123..."

# Append to document
google_docs append --doc-id "1abc123..." --text "New content"

# Read spreadsheet
google_docs sheets read --sheet-id "1xyz789..." --range "A1:D10"
```

**Use when:** Generating reports in Docs, updating spreadsheets, reading shared documents.

---

### llm_query

Unified interface for querying OpenAI, Anthropic, and Google GenAI models.

```bash
# Query OpenAI
llm_query "Explain quantum computing" --provider openai --model gpt-4

# Query Anthropic with system prompt
llm_query "Write a poem" --provider anthropic --system "You are a poet"

# JSON output with schema
llm_query "List 3 colors" --provider openai --json-schema schema.json
```

**Use when:** Quick LLM queries, testing prompts across providers, scripted AI interactions.

---

### log_analyzer

Parse and analyze log files with auto-detection of formats (nginx, apache, syslog, JSON).

```bash
# Find errors in logs
log_analyzer errors /var/log/app.log --last 1000

# Summarize log patterns
log_analyzer summarize /var/log/nginx/access.log

# Timeline analysis
log_analyzer timeline /var/log/syslog --start "2024-01-01" --end "2024-01-02"
```

**Use when:** Debugging production issues, analyzing traffic patterns, incident investigation.

---

### mcp_bridge

Manage Model Context Protocol (MCP) servers for tool integration.

```bash
# Start an MCP server
mcp_bridge start --config server.json

# List available tools
mcp_bridge tools --server localhost:8080

# Call a tool
mcp_bridge call --server localhost:8080 --tool search --params '{"query": "test"}'
```

**Use when:** Connecting AI agents to external tools, building MCP integrations.

---

### prompt_manager

Store and manage reusable prompt templates with variable substitution.

```bash
# Save a template
prompt_manager save code_review "Review this {{language}} code: {{code}}" --category dev

# Use a template
prompt_manager render code_review --vars '{"language": "Python", "code": "..."}'

# List templates
prompt_manager list --category dev
```

**Use when:** Standardizing prompts, building prompt libraries, team prompt sharing.

---

### s3_sync

Cloud storage operations for S3, Google Cloud Storage, and Azure Blob Storage.

```bash
# Upload a file
s3_sync upload local.txt s3://bucket/remote.txt

# Download a file
s3_sync download s3://bucket/file.txt ./local.txt

# Sync directory
s3_sync sync ./local-dir s3://bucket/prefix/ --delete
```

**Use when:** Backing up files, deploying assets, syncing data between local and cloud.

---

### screenshot_url

Capture screenshots of web pages using Playwright with device emulation.

```bash
# Basic screenshot
screenshot_url https://example.com -o screenshot.png

# Full page capture
screenshot_url https://example.com -o full.png --full-page

# Mobile emulation
screenshot_url https://example.com -o mobile.png --device "iPhone 14"
```

**Use when:** Visual regression testing, generating previews, archiving web pages.

---

### semantic_search

Semantic file search using local embeddings (sentence-transformers) and FAISS.

```bash
# Index a directory
semantic_search index ./docs --output index.faiss

# Search indexed files
semantic_search search "authentication flow" --index index.faiss --top 5

# One-shot search
semantic_search query ./src "error handling patterns"
```

**Use when:** Finding relevant code/docs by meaning, exploring unfamiliar codebases.

---

### shell_assist

Natural language to shell command conversion with safety checks.

```bash
# Convert natural language to command
shell_assist "find all Python files modified in the last week"

# Explain a command
shell_assist explain "find . -name '*.py' -mtime -7"

# Interactive mode
shell_assist interactive
```

**Use when:** Learning shell commands, quick command lookup, safely executing suggestions.

---

### slack_notify

Send Slack messages via Bot API or webhooks with rich formatting support.

```bash
# Send via webhook
slack_notify "Build complete!" --webhook "$SLACK_WEBHOOK_URL"

# Send to channel via Bot API
slack_notify "Deployment started" --channel "#deploys" --bot

# Send with blocks
slack_notify --blocks-file message.json --channel "#alerts"
```

**Use when:** CI/CD notifications, alerting, automated status updates to Slack.

---

### test_generator

Generate comprehensive tests from source code using LLMs.

```bash
# Generate tests for a file
test_generator generate src/utils.py --output tests/test_utils.py

# Specify framework
test_generator generate api.py --framework pytest --coverage 80

# Generate with mocks
test_generator generate service.py --mock-externals
```

**Use when:** Bootstrapping test suites, increasing coverage, testing legacy code.

---

### text_transform

Text and data transformation: encoding, format conversion, templating.

```bash
# Base64 encode
text_transform base64 encode "Hello World"

# JSON to YAML
text_transform convert data.json --from json --to yaml

# JQ-style query
text_transform jq '.users[].name' data.json

# Render template
text_transform template template.j2 --vars '{"name": "World"}'
```

**Use when:** Data format conversion, encoding/decoding, quick text manipulation.

---

### vector_search

Semantic document search with ChromaDB, Pinecone, Weaviate, or Qdrant backends.

```bash
# Index documents
vector_search index ./documents --backend chroma --collection docs

# Search
vector_search search "machine learning basics" --backend chroma --collection docs

# With specific embedding model
vector_search index ./docs --backend pinecone --model all-MiniLM-L6-v2
```

**Use when:** Building search over documents, RAG applications, semantic retrieval.

---

### webhook_dispatch

Send HTTP webhook requests with authentication, retries, and multiple content types.

```bash
# POST JSON
webhook_dispatch https://api.example.com/webhook --data '{"event": "deploy"}'

# With authentication
webhook_dispatch https://api.example.com/hook --bearer "$TOKEN" --data @payload.json

# With retries
webhook_dispatch https://api.example.com/hook --data '{}' --retries 3 --retry-delay 5
```

**Use when:** Triggering webhooks, integrating services, CI/CD event dispatch.

---

## Exit Codes

All tools follow a consistent exit code convention:

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | Configuration error (missing env vars, invalid config) |
| 2 | API/Service error (auth failure, rate limit, server error) |
| 3 | Input validation error (bad arguments, invalid data) |
| 4 | Network error (connection failed, timeout) |

## License

MIT
