## Pricing equity autocallable structures using LSV model

### Product parameters

- 1Y maturity
- 4 observation dates
- Autocall barrier H = 1
- Coupon barrier B = 0.8
- Coupon 7%

### Inputs

- SPX vol surface as of 24/07/2024

![iv_spx](https://github.com/user-attachments/assets/8a2e96bc-aa05-455d-ba74-0f810c41b7d6)

- SSVI calibration to remove arbitrage

### Pricing under LSV vs LV & SV

QMC using randomized Sobol sequence

#### Barrier

Brownian Bridge for continuously monitored barrier

### Worst-of case

Basket of 2 indexes

