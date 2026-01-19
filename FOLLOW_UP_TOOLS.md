# Follow-up CLI Tools for AI Agents

Ten additional tools that would complement the existing email_user, gemini_query, and google_docs tools.

---

## 1. `slack_notify`
**Purpose:** Send messages to Slack channels or users via webhook or API

**Use cases:**
- Alert team channels about completed deployments
- Send error notifications to on-call channels
- Post daily/weekly status summaries

**Key features:**
- Support webhooks (simple) and Bot API (rich formatting)
- Attachments, blocks, and threading support
- Channel and DM targeting

```bash
slack_notify -c "#deploys" -m "Build v1.2.3 deployed successfully"
slack_notify --webhook $WEBHOOK_URL -m "Alert: CPU > 90%"
```

---

## 2. `github_ops`
**Purpose:** Interact with GitHub repos, issues, and PRs

**Use cases:**
- Create issues from error reports
- Add comments to PRs with analysis results
- Create releases after successful builds
- Manage labels and milestones programmatically

**Key features:**
- Issue CRUD operations
- PR comments and reviews
- Release creation with asset uploads
- Repository dispatch events (trigger workflows)

```bash
github_ops issue create --repo owner/repo --title "Bug: X" --body "Details"
github_ops pr comment 123 --body "LGTM, tests pass"
github_ops release create v1.0.0 --notes "$(cat CHANGELOG.md)"
```

---

## 3. `db_query`
**Purpose:** Execute read-only SQL queries against databases

**Use cases:**
- Generate reports from production data
- Validate data migrations
- Extract metrics for analysis

**Key features:**
- Support Postgres, MySQL, SQLite
- Read-only mode enforced
- Output as JSON, CSV, or table
- Query from file or stdin

```bash
db_query --db postgres://... -q "SELECT count(*) FROM users"
db_query --db sqlite:///app.db -f report.sql --format csv
```

---

## 4. `s3_sync`
**Purpose:** Upload/download files to cloud storage (S3, GCS, Azure Blob)

**Use cases:**
- Archive processed files
- Stage data for ML pipelines
- Backup generated reports

**Key features:**
- Multi-provider support
- Streaming for large files
- Presigned URL generation
- Sync directories with filters

```bash
s3_sync upload report.pdf s3://bucket/reports/
s3_sync download gs://bucket/data/*.csv ./local/
s3_sync presign s3://bucket/file.zip --expires 3600
```

---

## 5. `calendar_schedule`
**Purpose:** Interact with Google Calendar / Outlook

**Use cases:**
- Schedule follow-up meetings after task completion
- Check availability before proposing times
- Create reminders for deadline tracking

**Key features:**
- Create/update/delete events
- Check free/busy times
- Meeting room booking
- Recurring event support

```bash
calendar_schedule create "Project Review" --start "2024-01-15 14:00" --duration 60m
calendar_schedule free-busy --start tomorrow --end "+3 days"
```

---

## 6. `webhook_dispatch`
**Purpose:** Send HTTP requests to arbitrary webhooks/APIs

**Use cases:**
- Trigger external integrations (Zapier, IFTTT, n8n)
- Call internal microservices
- Push data to analytics platforms

**Key features:**
- GET/POST/PUT/PATCH/DELETE methods
- Header and auth configuration
- Request/response logging
- Retry with backoff

```bash
webhook_dispatch POST https://api.example.com/events \
  --json '{"event": "complete", "id": 123}'
webhook_dispatch GET https://api.example.com/status --header "Auth: Bearer $TOKEN"
```

---

## 7. `screenshot_url`
**Purpose:** Capture screenshots of web pages

**Use cases:**
- Document UI state for bug reports
- Generate thumbnails for content
- Visual regression testing artifacts

**Key features:**
- Full page or viewport capture
- Wait for selectors/network idle
- Multiple output formats (PNG, JPEG, PDF)
- Mobile viewport emulation

```bash
screenshot_url https://example.com -o page.png --full-page
screenshot_url https://app.com/dashboard --wait-for ".loaded" --viewport 1920x1080
```

---

## 8. `text_transform`
**Purpose:** Process and transform text with various encodings/formats

**Use cases:**
- Convert between formats (JSON, YAML, TOML, CSV)
- Encode/decode (base64, URL encoding)
- Extract/restructure data

**Key features:**
- Format conversion (json↔yaml↔toml)
- JQ-style JSON queries
- Template rendering (Jinja2)
- Encoding operations

```bash
text_transform json-to-yaml < config.json > config.yaml
text_transform jq '.data[] | .name' < response.json
text_transform base64-encode < binary.dat
```

---

## 9. `git_changelog`
**Purpose:** Generate changelogs from git history

**Use cases:**
- Release notes generation
- Sprint summary reports
- Audit trail documentation

**Key features:**
- Conventional commit parsing
- Version range selection
- Category grouping (feat, fix, docs)
- Multiple output formats

```bash
git_changelog --from v1.0.0 --to v1.1.0 --format markdown
git_changelog --since "2 weeks ago" --group-by author
```

---

## 10. `llm_query`
**Purpose:** Query various LLM providers with unified interface

**Use cases:**
- Chain multiple model calls in workflows
- Compare outputs across providers
- Structured data extraction with schema validation

**Key features:**
- Support OpenAI, Anthropic, Gemini, local models
- Streaming output
- JSON schema output validation
- System prompt management

```bash
llm_query -m claude-3-5-sonnet -p "Summarize this text" < document.txt
llm_query -m gpt-4o -p "Extract entities" --json-schema entities.json < text.txt
echo "Translate to Spanish:" | llm_query -m gemini-2.5-flash --stdin
```

---

## Implementation Priority

| Priority | Tool | Rationale |
|----------|------|-----------|
| High | `slack_notify` | Common notification need, complements email_user |
| High | `github_ops` | Essential for dev automation workflows |
| High | `db_query` | Data access is fundamental for reports |
| Medium | `s3_sync` | File staging is common requirement |
| Medium | `webhook_dispatch` | Enables arbitrary integrations |
| Medium | `llm_query` | Useful for chaining AI operations |
| Medium | `text_transform` | Data manipulation utility |
| Lower | `screenshot_url` | Specialized but valuable |
| Lower | `calendar_schedule` | Nice-to-have for scheduling |
| Lower | `git_changelog` | Useful for release automation |
