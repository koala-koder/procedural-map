# Procedural Map Generator
This repository contains code that uses perlin noise to create seemingly realistic maps.

## How To Use

To get the generator, open your preferred folder to store the code and open with terminal. Enter:

`git clone https://github.com/koala-koder/procedural-map.git`

Navigate into the clone:

`cd procedural-map`

And run:

---
**Mac**: `python3 main.py`

**Linux**: `python3 main.py`

**Windows**: `py main.py`

---

After that, you should see progress bars, and after some time, there will be a file called `map.png`. This is the generate map, with a resolution of **300 megapixels**, unless you change the parameters in the python file.

## How To Edit

All you need to edit the map is to tamper with the parameters part in the code:

```
# Parameters - play around with these!
ratio = (3, 4)                       # Ratio
multiplier = 5000                   # Multiplier for ratio
shape = (ratio[0]*multiplier,
         ratio[1]*multiplier)        # Image size
scale = 0.29 * multiplier            # Avoid abrupt changes between adjacent coordinates
octaves = 16                         # Number of layers of noise (adds finer details)
persistence = 0.5                    # How quickly AMPLITUDE DECREASES for subsequent octaves
lacunarity = 2.0                     # How quickly FREQUENCY INCREASES for subsequent octaves
seed = 1                             # Seed for noise
```

## Future Changes

I plan to change/add the following features in the future:

- Circular gradient to create a centralised, more realistic archipelago
- Enhanced code optimisation
- Moisture noise maps for biome generation
- 3D conversion based on depth