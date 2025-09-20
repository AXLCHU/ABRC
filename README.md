## Pricing equity autocallable structures using Local Stochastic Volatility model

### Product parameters

- Underlying = SPX vol surface as of 30/06/2025
- 1Y maturity
- 4 observation dates
- Autocall barrier H = 100%
- Coupon barrier B = 80%
- Coupon 7%

### Fitting the vols

Particle method, bins Monte Carlo method

![lsv_iv_sp500](https://github.com/user-attachments/assets/4acbe621-c966-4136-9119-a084c58e936b)

### Pricing the Autocall

![abrc_price](https://github.com/user-attachments/assets/a107a222-ec31-4233-b6ea-7bbb1577d69a)

-> Vols from LSV always higher than LV ones + better fit 

#### Greeks

![abrc_delta](https://github.com/user-attachments/assets/a1bc55ce-11ef-45d3-aba3-0390cadac80c)
![abrc_vega](https://github.com/user-attachments/assets/572642ab-0f9b-44b4-9017-09c7c1d9367e)

#### Barrier

Brownian Bridge for continuously monitored barrier

#### To do: Worst-of basket of multiple indexes

