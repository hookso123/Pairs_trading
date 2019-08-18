# Pairs_trading

This contains a small expereimnt I did into pairs trading with GSK and PFE stock. The idea is to fir a markov model to the ratio of the stock prices.

GSK.csv and PFE.csv are both downlaoded form Yahoo fincance. 

The trading algorithm then takes a position depending on its belief about how the ratio will change over the next 10 days. 

It doesn't work very well! I think because the ratio has some big long term changes that are not accounted for in my model of strategy. 

Hopefully this is an interesting starting point for something more sophistocated. 
