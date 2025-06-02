## Pricing equity autocallable structures using LSV model

### Product parameters

- Underlying = SPX
- 1Y maturity
- 4 observation dates
- Autocall barrier H = 100%
- Coupon barrier B = 80%
- Coupon 7%

### Inputs

- SPX vol surface as of 24/07/2024

![iv_spx](https://github.com/user-attachments/assets/7217789e-74c7-41cb-8b05-7319450b8e8f)

- SSVI calibration to remove arbitrage

### Pricing under LSV

QMC bins method using randomized Sobol sequence

![abrc_price](https://github.com/user-attachments/assets/a107a222-ec31-4233-b6ea-7bbb1577d69a)

#### Greeks

![abrc_delta](https://github.com/user-attachments/assets/a1bc55ce-11ef-45d3-aba3-0390cadac80c)
![abrc_vega](https://github.com/user-attachments/assets/572642ab-0f9b-44b4-9017-09c7c1d9367e)

#### Barrier

Brownian Bridge for continuously monitored barrier

### To do: Worst-of case basket of multiple indexes

