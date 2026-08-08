import noise
import numpy as np
from PIL import Image


# Parameters - play around with these!
ratio = (3, 4)                       # Ratio
multiplier = 1000                    # Multiplier for ratio
shape = (ratio[0]*multiplier,
         ratio[1]*multiplier)        # Image size
scale = 0.29 * multiplier             # Avoid abrupt changes between adjacent coordinates
octaves = 16                         # Number of layers of noise (adds finer details)
persistence = 0.5                    # How quickly AMPLITUDE DECREASES for subsequent octaves
lacunarity = 2.0                     # How quickly FREQUENCY INCREASES for subsequent octaves
seed = 1                             # Seed for noise (change this to completely change the noise!)

# Create empty world
world = np.zeros(shape)

# Change image to noise, pixel by pixel
for x in range(shape[0]):
    for y in range(shape[1]):
        world[x][y] = noise.pnoise2(
            x=x/scale,
            y=y/scale,
            octaves=octaves,
            persistence=persistence,
            lacunarity=lacunarity,
            repeatx=shape[0],
            repeaty=shape[1],
            base=seed
        )


world_min, world_max = world.min(), world.max()
world_normalized = (world - world_min) / (world_max - world_min)

# Color definitions (RGB)
ocean = [65, 105, 225]
beach = [238, 214, 175]
green = [34, 139, 34]
mountain = [120, 120, 120]
snow = [255, 255, 255]

# Thresholds (0.0 to 1.0)
ocean_thr = 0.45
beach_thr = 0.54
green_thr = 0.7
mountain_thr = 0.8

def add_color(world_map):
    color_world = np.zeros(world_map.shape + (3,), dtype=np.uint8)

    # Apply thresholds
    color_world[world_map < ocean_thr] = ocean
    color_world[(world_map >= ocean_thr) & (world_map < beach_thr)] = beach
    color_world[(world_map >= beach_thr) & (world_map < green_thr)] = green
    color_world[(world_map >= green_thr) & (world_map < mountain_thr)] = mountain
    color_world[world_map >= mountain_thr] = snow

    return color_world

# Generate colored image
color_world = add_color(world_normalized)
img = Image.fromarray(color_world, mode='RGB')

# Save!
img.save("map.png")