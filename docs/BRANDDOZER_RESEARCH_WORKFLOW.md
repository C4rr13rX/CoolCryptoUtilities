# Brand Dozer Archival Research Workflow

Brand Dozer research projects produce versioned, auditable research papers
instead of software patches. Select **Archival research paper** in the Brand
Dozer delivery form, enter the research goal, and choose the desired
publication settings.

## Execution model

1. A research director converts the goal into a protocol and independent Scrum
   work packages.
2. Literature-review agents execute those packages concurrently.
3. Candidate source URLs are discovered independently and ranked toward
   authoritative scientific and engineering domains.
4. Every cited source is fetched under bounded, robots-aware rules. Its content
   is hashed and the agent's purported supporting passage must occur verbatim
   in the fetched document.
5. A methods reviewer records inclusion/exclusion decisions, bias risks,
   validity limits, unresolved questions, and synthesis rules.
6. A senior writer produces a complete journal-style paper using only verified
   citation keys.
7. A hostile citation auditor evaluates every claim against its cited source
   keys.
8. An independent peer-review agent checks methods, evidence, uncertainty,
   counterevidence, novelty, and overclaiming.
9. Deterministic publication-readiness gates evaluate structure, word count,
   source count and diversity, source authority, citation identity, claim
   support, and peer-review blockers.
10. A failed draft is retained and the entire paper is rewritten. The loop
    continues up to the configured revision bound. Only a passing revision is
    marked `validated`.

The workflow performs archival synthesis only. It must not describe simulated,
proposed, or unperformed experiments as observed results.

## Evidence and provenance

Each paper retains:

- the exact research question and target journal or discipline;
- the final Markdown and PDF;
- every revision and its validation report;
- source identity, URL, DOI when independently present, retrieval time, content
  hash, authority tier, and verified passage;
- claim text, paper section, supporting citation keys, disposition, and
  rationale;
- agent-session logs, Scrum backlog, sprint records, and the final delivery
  artifact.

PDF evidence is extracted with `pypdf`; HTML and text evidence use the bounded
research harvester. Inaccessible, non-public, robots-excluded, missing, or
passage-mismatched sources are rejected rather than silently accepted.

## Paper library

The Brand Dozer page contains a research-paper library. It searches titles,
questions, abstracts, keywords, and paper text. Selecting a paper opens a modal
with:

- validation-gate results;
- the research question and abstract;
- source, audited-claim, and revision counts;
- the full paper;
- PDF, Markdown, and evidence-JSON downloads.

The evidence JSON is the authoritative machine-readable audit package.

## Publication-readiness boundary

`validated` means the configured evidence, traceability, structure, citation,
claim, and independent-review gates passed. It does not guarantee that a
specific external journal will accept the manuscript. Journal-specific
formatting, human subject-matter review, legal/ethical review, and submission
requirements remain external acceptance steps.
