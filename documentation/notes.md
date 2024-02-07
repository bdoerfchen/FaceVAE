## Tries
- Dropout layers
    - dr 0.4: KL war extrem hoch, MSE wurde komplett vernachlässigt => Alles gleiches Durchschnittsbild
    - dr 0.1: KL * 1/100, dadurch einigermaßen ok, => Bild aber eher verschwommen

## Implementation
- Herleitung warum MSE + KL anstatt ELBO ok ist: https://www.youtube.com/watch?v=vEPQNwxd1Y4