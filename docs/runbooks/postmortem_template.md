# Blameless Postmortem Template

- **Incident ID:** INC-YYYYMMDD-XXX  
- **Date/Time:** <start> to <end> (UTC)  
- **Severity:** <SEV1|SEV2|SEV3>  
- **Services/Components:** <list>  
- **On-call:** <name/rotation>  
- **Summary (2-3 lines):** <what happened and user impact>  

## Timeline (UTC)
- T0: <event>
- T+?m: <detection>
- T+?m: <mitigation>
- T+?m: <recovery>

## Impact
- Users affected (%/count):
- Transactions/errors:
- SLO/SLA breached? <yes/no> (reference SLI/SLO doc)

## Root Cause
- <5-whys / contributing factors / guardrails that failed>

## Resolution
- <short-term mitigation>
- <long-term fix>

## Detection & Alerting
- Was detection timely? <yes/no>
- Alert quality (noise/fatigue?): <notes>
- Gaps to fix: <list>

## Action Items (tracked in backlog; include owners & due dates)
- [ ] <fix> (Owner, Due)
- [ ] <test/coverage> (Owner, Due)
- [ ] <runbook/update> (Owner, Due)
- [ ] <observability/alert tuning> (Owner, Due)

## Lessons & Prevention
- What worked:
- What to improve:
- Policy/process changes:

## Attachments
- Logs/traces/metrics: <links>
- Related tickets/PRs: <links>
