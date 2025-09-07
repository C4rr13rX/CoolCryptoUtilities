import json
from typing import Dict

from cache import CacheTransfers, CacheBalances
from router_wallet import (
    UltraSwapBridge,
    normalize_snapshot_numbers, enrich_portfolio_with_0x, CHAINS,
)
from balances import MultiChainTokenPortfolio
from filter_scams import FilterScamTokens

# ======================================================================
# DEMO
# ======================================================================
if __name__ == "__main__":

    # NEW: create a transfers cache and pass it into the router
    ct = CacheTransfers() # honors PORTFOLIO_CACHE_DIR env if set
    router = UltraSwapBridge(cache_transfers=ct) # pulls MNEMONIC/PRIVATE_KEY from .env
    wallet = router.get_address()
    print(f"Wallet: {wallet}")

    # 1) Discover & annotate in balances.py's canonical format
    annotated = router.discover_tokens() # defaults to style="portfolio"
    print(f"Discovered {len(annotated)} tokens")
    if annotated:
        print("Sample:", annotated[:8])
        
    # 2) Filter scam tokens 
    filt = FilterScamTokens()
    try:
        res = filt.filter(annotated)
        safe_tokens = res.tokens
        # --- CACHES (create after you know wallet/tokens) ---
        cb = CacheBalances() 
        ct = CacheTransfers() # optional args; defaults are fine
        
        # --- Portfolio using caches ---
        tp = MultiChainTokenPortfolio(
            wallet_address=wallet,
            tokens=safe_tokens, # <- use the filtered list
            cache_balances=cb, # <- balance cache
            cache_transfers=ct, # <- transfer movement probe (fast path)
            # keep transfers off for speed unless you need them:
            max_transfers_per_token=0,
            verbose=False # turn on to see cache hits/misses
        )
        
        snapshot = tp.build()
        print("INPUT :", json.dumps(annotated, indent=2))
        print("OUTPUT:", json.dumps(safe_tokens, indent=2))
# ---- NEW: clean up scientific notation (formatting only)
        normalize_snapshot_numbers(snapshot, qty_dp=18, usd_dp=8)
    
        # ---- NEW: optional 0x fill-in for tokens with usd_amount == 0 (balances untouched)
        # Build address->chain map from the *filtered* list
        addr_to_chain: Dict[str, str] = {}
        for tag in safe_tokens:
            if ":" in tag and tag.split(":",1)[0] in CHAINS:
                ch, addr = tag.split(":",1)
            elif "@" in tag and tag.split("@",1)[1] in CHAINS:
                addr, ch = tag.split("@",1)
            else:
                continue
            addr_to_chain[addr.lower()] = ch.lower()
        try:
            enrich_portfolio_with_0x(snapshot, addr_to_chain, qty_dp=18, usd_dp=8)
        except Exception as e:
            print(f"[pricing] 0x enrich skipped due to error: {e}")
    
        # 4) Print raw snapshot JSON (balances + txs; USD may remain 0 for illiquid tokens)
        print(json.dumps(snapshot, indent=2))
    
        # 5) Pretty total (optional)
        total = 0.0
        for info in snapshot.values():
            try:
                total += float(info.get("usd_amount") or 0)
            except Exception:
                pass
        print("\n=== Portfolio snapshot ===")
        print(f"TOTAL ≈ ${total:.2f}")
    
        if res.flagged:
            print("\n-- Flagged (dropped) --")
            for a, why in res.reasons.items():
                print(a, "=>", why)
    except Exception as e:
        print("Filter error:", type(e).__name__, str(e))