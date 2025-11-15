## Pricing Equity Autocallable structures using Local Stochastic Volatility model

### Product parameters

- Underlying = SPX vol surface as of 30/06/2025
- 1Y maturity
- 4 observation dates
- Autocall barrier H = 100%
- Coupon barrier B = 80%
- Coupon 7%

### Fitting the vols, IV from LV vs LSV

LSV using Particle method: bins Monte Carlo with 300,000 paths and 20 bins

![lsv_iv_sp500](https://github.com/user-attachments/assets/ba29aeda-2895-466b-8d97-a1715453cd5a)

-> Vols from LSV creates higher forward smiles compared to LV
-> Wings of short maturities are harder to fit


### Pricing the Autocall

![abrc_price](https://github.com/user-attachments/assets/a107a222-ec31-4233-b6ea-7bbb1577d69a)


### Greeks

![abrc_delta](https://github.com/user-attachments/assets/a1bc55ce-11ef-45d3-aba3-0390cadac80c)
![abrc_vega](https://github.com/user-attachments/assets/572642ab-0f9b-44b4-9017-09c7c1d9367e)

### Barrier

Brownian Bridge for continuously monitored barrier

### To do: Worst-of basket of multiple indexes

