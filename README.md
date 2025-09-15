I am currently working on a lot of updates in the main branch. When I am done, I'll fix this ReadMe file cleaned up and with info for the updates, and then won't work in here any longer. Only merge when something is done in another branch.

These are python utilities for creating an index of the top 2000 crypto currencies, making an assignment file out of the index, and downloading the historical OHLCV data for 2000 top cryptos for the past 3 years at 5 minute granularity. I also uploaded a requirements.txt file from pip freeze to install the necessary libraries for these scripts and more.

You will need to get an API key from both ANKR and TheGraph.
ANKR: https://www.ankr.com/web3-api/
TheGraph: https://thegraph.com/studio/

Run files in this order.
- python make2000index.py
- python makeServiceAssignment.py
- python download2000.py

This will leave you with 2000 .json files, one for each crypto, that can be over a half gig large that you can train an AI model with. All of this was designed to run on the free tier. download2000.py makes heavy use of multithreading for efficiency. 

WARNING: It will take approximately 250 days to get this data from ANKR on the free tier running 24/7.

Update: This is getting serious. Switched all of the API Keys to .env, which isn't included in the git pushes. Here is what needs to be in it with your own values.:

###############################################
# 🚨 SECURITY
# - Do NOT keep real seed phrases in .env on shared or cloud machines.
# - Prefer hardware wallets or ephemeral test mnemonics.
###############################################

# ===== Wallet (ONLY for test/dev — never prod) =====
MNEMONIC=

DERIVATION_PATH=m/44'/60'/0'/0/0


# ===== Core RPCs (set at least one per chain) =====
# Option A: put FULL RPC URLs here (these override everything else)
#RPC_ETHEREUM=

#RPC_BASE=

#RPC_ARB=

#RPC_OP=

#RPC_POLY=

#RPC_ZORA=

#RPC_REI=

# Option B: supply provider keys and we’ll build URLs in code
ALCHEMY_API_KEY=

ALCHEMY_KEY_BASE=

INFURA_API_KEY=


# (Optional) Pre-built provider URLs your code already checks
ALCHEMY_ETH_URL=https://eth-mainnet.g.alchemy.com/v2/${ALCHEMY_API_KEY}

ALCHEMY_BASE_URL=https://base-mainnet.g.alchemy.com/v2/${ALCHEMY_API_KEY}

ALCHEMY_ARB_URL=https://arb-mainnet.g.alchemy.com/v2/${ALCHEMY_API_KEY}

ALCHEMY_OP_URL=https://opt-mainnet.g.alchemy.com/v2/${ALCHEMY_API_KEY}

ALCHEMY_POLY_URL=https://polygon-mainnet.g.alchemy.com/v2/${ALCHEMY_API_KEY}


INFURA_ETH_URL=https://mainnet.infura.io/v3/${INFURA_API_KEY}

INFURA_ARB_URL=https://arbitrum-mainnet.infura.io/v3/${INFURA_API_KEY}

INFURA_OP_URL=https://optimism-mainnet.infura.io/v3/${INFURA_API_KEY}

INFURA_POLY_URL=https://polygon-mainnet.infura.io/v3/${INFURA_API_KEY}


# ===== Aggregators / indexers =====
ANKR_API_KEY=3

# LI.FI (quotes + calldata). Optional but recommended for routing.
LIFI_API_KEY=


# Covalent (balances). Optional but speeds up multi-chain balance discovery.
#COVALENT_KEY=


# The Graph (Gateway) — speeds up Uniswap subgraph queries dramatically
THEGRAPH_API_KEY=

# If you want to hardcode gateway endpoints per chain (fastest):
UNISWAP_SUBGRAPH_ETHEREUM=https://gateway.thegraph.com/api/${THEGRAPH_API_KEY}/subgraphs/id/5zvR82QoaXYFyDEKLZ9t6v9adgnptxYpKpSbxtgVENFV

UNISWAP_SUBGRAPH_BASE=https://gateway.thegraph.com/api/${THEGRAPH_API_KEY}/subgraphs/id/FUbEPQw1oMghy39fwWBFY5fE6MXPXZQtjncQy2cXdrNS

UNISWAP_SUBGRAPH_ARBITRUM=

UNISWAP_SUBGRAPH_OPTIMISM=

UNISWAP_SUBGRAPH_POLYGON=


# Uniswap Trade API (optional; for instant quotes like the Uniswap app)
#UNISWAP_API_KEY=


# GoPlus token security (honeypot/tax checks). Optional but helpful.
GOPLUS_APP_KEY=

GOPLUS_APP_SECRET=


# (Optional) CoinGecko Pro key (free paths don’t need it)
#COINGECKO_API_KEY=


# ===== Pricing behavior / performance knobs =====
# Order is handled in code, but these toggle heavy paths
PRICE_UNISWAP=1 # 1 = use Uniswap subgraph batch if URLs set; 0 = skip

PRICE_TTL_SEC=300 # cache token prices for N seconds

PRICE_MAX_TOKENS=25 # per-chain price cap per run

PRICE_WORKERS=8 # parallel HTTP workers for price calls

CG_TIMEOUT=6 # CoinGecko request timeout (s)

DS_TIMEOUT=6 # Dexscreener request timeout (s)

HTTP_TIMEOUT_SEC=8 # generic request timeout (s)


# ===== Scam filtering (set to taste) =====
FILTER_SCAMS=1 # 1=enable; 0=disable

SCAM_STRICT=1 # 1=drops high-tax/proxy/mintable; 0=only explicit honeypots

SCAM_MIN_LIQ_USD=2000 # Dexscreener min liquidity to keep (when enabled)

SCAM_MAX_TAX_PCT=25 # drop if buy/sell tax > this (when GoPlus returns it)

SCAM_USE_GOPLUS=1 # 1=use GoPlus checks; 0=skip

SCAM_WHITELIST= # comma-separated token addresses you always keep


# Include unverified tokens from Blockscout tokenlist? (1=yes, 0=no)
BLOCKSCOUT_INCLUDE_UNVERIFIED=1


# Reprice Covalent balances with your own price stack (for consistency)
REPRICE_COVALENT=0


# ===== Optional canonical token overrides (addresses) =====
# Use these if you want to force canonical stables per chain (price routing helpers use them)
ETH_USDC_ADDR=0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48

BASE_USDC_ADDR=0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913

ARB_USDC_ADDR=0xAf88d065e77c8cC2239327C5EDB3A432268E5831

OP_USDC_ADDR=0x0b2C639c533813f4Aa9D7837CAf62653d097Ff85

POLY_USDC_ADDR=0x3c499c542cEF5E3811e1192ce70d8cC03d5c3359 # native USDC

POLY_USDCe_ADDR=0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174 # bridged USDC.e


# ===== Debug toggles (optional) =====
DEBUG_PRICING=0

DEBUG_FILTER=0

