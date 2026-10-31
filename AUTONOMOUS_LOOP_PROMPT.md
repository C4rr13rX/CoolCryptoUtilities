AUTONOMOUS LIVE-TRADING LOOP. Goal: live PROFITABLE trades in D:\Projects\CoolCryptoUtilities. Do not stop until step 10 (PROFIT) passes.

1. Run: cd /d/Projects/CoolCryptoUtilities && ./.venv/Scripts/python.exe scripts/live_path_check.py
2. Find the FIRST failing link. That is the only thing to work on — everything after it is unknowable.
3. Diagnose it with real measurements (query the DB, read the code, check on-chain). Never assume; verify. If a number looks too good, it is probably an artifact — check it against an independent source before believing it.
4. Fix the root cause, add a test that pins the behaviour, run the relevant tests, and commit.
5. Restart production if needed (PowerShell: stop python.exe matching *start_production*, then nohup ./.venv/Scripts/python.exe -u main.py --action start_production --stay-alive > data/production.log 2>&1 &).
6. Re-run live_path_check.py to confirm the link now passes.
7. Report concisely: which link failed, the evidence, the fix, and what the next link is.

If every link passes and live trades are profitable, say so plainly and stop.
If a live trade LOSES, verify the demotion guards fired (STRATEGY_DEMOTE_MAX_LIVE_LOSSES=2) and report the P/L honestly — never hide a loss.

Wallet: 0x291c854811e92906a658Fb94Aa511bF919f968ad on base. Keep ENABLE_GAS_REFILL=0 (the refill drained the wallet three times).