import multiprocessing
from matplotlib.pyplot import grid
import numpy as np
from PIL import Image
from numba import njit, prange
from tqdm import tqdm
import pyvista as pv


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
green_thr = 0.75
mountain_thr = 0.90


def apply_elevation_curve(noisevalue):
    """
    Applies a curve to the noise value to adjust elevation distribution.
    This is because, naturally, low land areas (greenery and beaches) 
    rise more slowly than mountains, which rise more sharply.
    """

    thr_points = [0, ocean_thr, beach_thr, green_thr, mountain_thr, 1.0]

    # Define custom Z values for each threshold
    elevation_points = [0.0, 0.0, 0.02, 0.20, 0.60, 1.0]

    return np.interp(noisevalue, thr_points, elevation_points)

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

def render_3d(elevation_map, color_world, z_scale=250.0):
    """
    Renders 3D terrain with planar UV projection and horizontal alignment fix.
    """
    print("\nRendering 3D Terrain...")
    length, width = elevation_map.shape

    # Downsample geometry grid for performance
    step = max(1, length // 1000)
    elev_sub = elevation_map[::step, ::step]

    sub_height, sub_width = elev_sub.shape
    x = np.arange(0, sub_width) * step
    y = np.arange(0, sub_height) * step
    x_grid, y_grid = np.meshgrid(x, y)

    z_grid = elev_sub * z_scale

    # 1. Build heightmap mesh geometry
    grid = pv.StructuredGrid(x_grid, y_grid, z_grid)

    # 2. Anchor UV plane bounds to match grid coordinates exactly
    x_max = (sub_width - 1) * step
    y_max = (sub_height - 1) * step

    grid = grid.texture_map_to_plane(
        origin=(0, 0, 0),
        point_u=(x_max, 0, 0),
        point_v=(0, y_max, 0)
    )

    # 3. Flip texture horizontally to correct X-axis orientation
    texture = pv.numpy_to_texture(np.flipud(color_world))

    print("Opening 3D viewer window...")
    plotter = pv.Plotter()
    plotter.add_mesh(grid, texture=texture, smooth_shading=True)
    plotter.show()


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
    octaves = 64
    persistence = 0.5
    lacunarity = 2.0
    seed = 1

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

    # Apply elevation slopes
    elevation_map = apply_elevation_curve(world)

    # Generate colored image with progress tracking
    color_world = add_color_with_progress(world, chunk_size=1000)
    img = Image.fromarray(color_world, mode="RGB")

    # Save output
    img.save("map.png", compress_level=1)
    print("\nMap successfully saved to 'map.png'!")

    # Render 3D terrain
    render_3d(elevation_map, color_world, z_scale=multiplier * 0.2)

    # Keep terminal open so progress bars remain visible
    input("\nProcess finished! Press Enter to exit...")