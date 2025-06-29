These are python utilities for creating an index of the top 2000 crypto currencies, making an assignment file out of the index (even though that didn't exactly get used for its full purpose, but it can be. Right now it is used as a state file for starting the script where it last left off downloading historical OHLCV data for 2000 cryptos for the past 5 years.), and downloading the historical OHLCV data for 2000 top cryptos for the past 5 years. I also uploaded a requirements.txt file from pip freeze to install the necessary libraries for these scripts and more.

You will need to get an API key from both ANKR and TheGraph.
ANKR: https://www.ankr.com/web3-api/
TheGraph: https://thegraph.com/studio/

Run files in this order.
- python make2000index.py
- python makeServiceAssignment.py
- python download2000.py

This will leave you with 2000 .json files, one for each crypto, that can be over a half gig large that you can train an AI model with.
