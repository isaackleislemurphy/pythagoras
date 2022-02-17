# Pythagoras

### Overview
- `R` contains files from in-class -- notably, those used to make visuals and pull in game-by-game data from 2016-2021. 

- `src` contains a Python script that trains historical [PECOTA](https://www.baseballprospectus.com/pecota-projections/) projections (2016-2020) against observed run production and prevention, using the Weibull to parameterize both. Specifically, Weibull is parameterized via a small neural network, where the OPS'/VORPs/WARPs/DRAs of each team's (ranked) players are the features. Beware this was done on the fly, likely contains bugs, and I don't at all stand by it.


Game data courtesy of Bill Petti and the `baseballr` [package](https://billpetti.github.io/baseballr/). Projection data courtesy of Baseball Prospectus. 