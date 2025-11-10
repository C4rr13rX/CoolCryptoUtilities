## Runtime Logging Map

All runtime logs are centralized under the top-level `logs/` directory:

| Path | Description |
| --- | --- |
| `logs/system.log` | Aggregated feed of every `log_message()` call (guardian, production manager, wallet console, etc.). |
| `logs/services/<source>.log` | Per-source streams automatically generated from the same logging bus (e.g. `guardian.log`, `production.log`). |
| `logs/django.log` | Django request/exception log from the project logging config. |
| `logs/console.log` | Raw stdout/stderr from `main.py` when the console/production manager starts via the UI. |
| `logs/waallet_lig.log` | Wallet instrumentation/debugging (legacy but still referenced by diagnostics). |

To give a new component its own log, call `services.logging_utils.log_message("component-name", "...")`. The bus writes the entry to `logs/system.log` **and** creates/updates `logs/services/component-name.log`, so the file structure stays consistent without any extra wiring.
