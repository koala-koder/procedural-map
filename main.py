import multiprocessing
import numpy as np
from PIL import Image
from numba import njit, prange
from tqdm import tqdm


# --- Fast Numba Perlin Generator ---
@njit(fastmath=True)
def fade(t):
    return t * t * t * (t * (t * 6 - 15) + 10)


@njit(fastmath=True)
def lerp(t, a, b):
    return a + t * (b - a)


@njit(fastmath=True)
def grad(hash_val, x, y):
    h = hash_val & 7
    u = x if h < 4 else y
    v = y if h < 4 else x
    return (u if (h & 1) == 0 else -u) + (v if (h & 2) == 0 else -v)


@njit(parallel=True, fastmath=True)
def generate_perlin_rows(
    world, start_row, end_row, scale, octaves, persistence, lacunarity, p
):
    width = world.shape[1]

    for i in prange(start_row, end_row):
        for j in range(width):
            total = 0.0
            frequency = 1.0 / scale
            amplitude = 1.0

            for _ in range(octaves):
                x = i * frequency
                y = j * frequency

                X = int(np.floor(x)) & 255
                Y = int(np.floor(y)) & 255

                xf = x - np.floor(x)
                yf = y - np.floor(y)

                u = fade(xf)
                v = fade(yf)

                n00 = grad(p[p[X] + Y], xf, yf)
                n01 = grad(p[p[X] + Y + 1], xf, yf - 1)
                n10 = grad(p[p[X + 1] + Y], xf - 1, yf)
                n11 = grad(p[p[X + 1] + Y + 1], xf - 1, yf - 1)

                x1 = lerp(u, n00, n10)
                x2 = lerp(u, n01, n11)

                total += lerp(v, x1, x2) * amplitude
                amplitude *= persistence
                frequency *= lacunarity

            world[i, j] = total


def generate_perlin_world_with_progress(
    shape, scale, octaves, persistence, lacunarity, seed, chunk_size=500
):
    height, width = shape
    world = np.zeros((height, width), dtype=np.float32)

    # Generate permutation table once based on seed
    np.random.seed(seed)
    p = np.arange(256, dtype=np.int32)
    np.random.shuffle(p)
    p = np.concatenate((p, p))

    # Process in row chunks to update progress bar
    with tqdm(total=height, desc="Generating Noise", unit="row") as pbar:
        for start_row in range(0, height, chunk_size):
            end_row = min(start_row + chunk_size, height)
            generate_perlin_rows(
                world,
                start_row,
                end_row,
                scale,
                octaves,
                persistence,
                lacunarity,
                p,
            )
            pbar.update(end_row - start_row)

    return world


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


def add_color_with_progress(world_map, chunk_size=1000):
    height, width = world_map.shape
    color_world = np.zeros((height, width, 3), dtype=np.uint8)

    # Process coloring in row chunks to display progress and prevent RAM overloads
    with tqdm(total=height, desc="Applying Colors", unit="row") as pbar:
        for start_row in range(0, height, chunk_size):
            end_row = min(start_row + chunk_size, height)

            sub_map = world_map[start_row:end_row]
            sub_color = color_world[start_row:end_row]

            sub_color[sub_map < ocean_thr] = ocean
            sub_color[(sub_map >= ocean_thr) & (sub_map < beach_thr)] = beach
            sub_color[(sub_map >= beach_thr) & (sub_map < green_thr)] = green
            sub_color[(sub_map >= green_thr) & (sub_map < mountain_thr)] = (
                mountain
            )
            sub_color[sub_map >= mountain_thr] = snow

            pbar.update(end_row - start_row)

    return color_world


# --- Main Execution Guard for Executable Bundling ---
if __name__ == "__main__":
    # Prevents infinite sub-process spawning with Numba parallel mode in PyInstaller
    multiprocessing.freeze_support()

    # Parameters
    ratio = (3, 4)  # Aspect Ratio
    multiplier = 1000  # Multiplier for ratio
    shape = (
        ratio[0] * multiplier,
        ratio[1] * multiplier,
    )  # Image size
    scale = 0.29 * multiplier
    octaves = 16
    persistence = 0.5
    lacunarity = 2.0
    seed = 11

    # Generate noise using Numba with progress tracking
    world = generate_perlin_world_with_progress(
        shape=shape,
        scale=scale,
        octaves=octaves,
        persistence=persistence,
        lacunarity=lacunarity,
        seed=seed,
        chunk_size=500,
    )

    # Normalize map values between 0.0 and 1.0
    world_min, world_max = world.min(), world.max()
    world -= world_min
    world /= world_max - world_min

    # Generate colored image with progress tracking
    color_world = add_color_with_progress(world, chunk_size=1000)
    img = Image.fromarray(color_world, mode="RGB")

    # Save output
    img.save("map.png", compress_level=1)
    print("\nMap successfully saved to 'map.png'!")

    # Keep terminal open so progress bars remain visible
    input("\nProcess finished! Press Enter to exit...")