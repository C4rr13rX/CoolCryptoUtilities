# Wizard Vision operations controls

The six desktop shortcuts installed by
`scripts/operations/install_desktop_controls.ps1` are the recovery surface for
the complete local stack:

- **Start Everything - Wizard Vision** idempotently ensures the private AWS
  programming-brain connection, crypto brain, Django/Waitress, ghost-only
  production manager, brain feeder, and protected-fold market evolution.
- **Monitor Everything - Wizard Vision** shows listening processes, the
  production heartbeat and live gate, current held-out market metrics, and the
  AWS curriculum owner counts.
- The separate crypto and senior-software-brain start/tail shortcuts provide
  narrower recovery and monitoring without stopping services when a tail
  window closes.

The Django dashboard is `http://127.0.0.1:8001/`. The local crypto brain stays
on `127.0.0.1:8090`. The AWS senior-software-engineer brain is registered as an
independent Wizard brain and selected for the operations/C0D3R purpose; normal
Wizard chat can keep its independently selected brain.

The AWS brain itself remains private and loopback-bound. Because the current
IAM identity permits SSM commands but not `ssm:StartSession`,
`scripts/aws/programming_brain_proxy.py` in W1z4rDV1510n provides a
loopback-only relay on `127.0.0.1:18096`. It accepts only health and chat routes,
caps requests at 12 KiB, and never logs request bodies. Do not replace it with
a public AWS listener merely for convenience.

Live trading remains disabled until the existing promotion gates pass. A
running production manager or a profitable-looking in-sample candidate is not
permission to promote it.

## Repository hygiene

Runtime datasets, logs, databases, cloud identity folders, environment files,
wallet/key material, and common credential bundles are ignored. Stage files
explicitly and run a staged secret scan before every commit. Never use `git add
.` for operational changes in a dirty worktree.
