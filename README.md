## Pricing Equity Autocallable structures using Local Stochastic Volatility model

### Product overview

Autocallable Note 1Y, quarterly observations:

- Underlying = SPX vol surface as of 30/06/2025
- Autocall barrier H = 100%
- Coupon barrier B = 80%
- Coupon 7% p.a.
- Key features: Early redemption if above 100%, coupon paid if above 80%, otherwise final redemption linked to SPX performance

### Model framework
#### Local Volatility
- SVI/SSVI fit to IV to clean arbitrage
- Dupire equation to derive LV grid
- Serves as benchmark model

#### Local Stochastic Volatility
- Heston model combined with a leverage function
- Calibration using Particle method: bins Monte Carlo with 300,000 paths and 20 bins

### LV vs LSV smile fit

![lsv_iv_sp500](https://github.com/user-attachments/assets/ba29aeda-2895-466b-8d97-a1715453cd5a)

- LSV creates higher forward smiles compared to LV
- Wings of short maturities are harder to fit


### Autocall pricing

![abrc_price](https://github.com/user-attachments/assets/a107a222-ec31-4233-b6ea-7bbb1577d69a)

- LSV price < LV price due to stronger downside skew and reduced probability of early autocall

### Greeks

![abrc_delta](https://github.com/user-attachments/assets/a1bc55ce-11ef-45d3-aba3-0390cadac80c)
![abrc_vega](https://github.com/user-attachments/assets/572642ab-0f9b-44b4-9017-09c7c1d9367e)

Autocallables typically exhibit:
- short Vega for investor
- long forward skew
- sensitivity concentrated near barrier levels

### Extensions (in progress)

Worst-of basket Autocallable
- Multi-asset LSV
- Correlation structure
- Probability of crossing barriers

