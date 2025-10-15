# -*- coding: utf-8 -*-
"""
Created on Wed Oct  8 17:25:02 2025

@author: joshua
"""

# %%

from manim import *

class RecursiveCubes(ThreeDScene):
    def add_cubes(self, sizes, spacing=2.8, depth=0.125, arrow_text="Reduction", arrow_gap=0.2):
        """
        Adds cubes and connecting arrows recursively from a list of sizes.

        Parameters
        ----------
        sizes : list[int]
            List of cube side sizes (e.g. [128, 64, 32])
        spacing : float
            Distance between cube centers (reduce for tighter layout)
        depth : float
            Depth compression ratio for the cube's z dimension
        arrow_text : str | list[str]
            Text for the arrows (single string or list of strings)
        arrow_gap : float
            Gap between the cube faces and arrow start/end
        """
        cubes, labels, arrows, arrow_labels = [], [], [], []
        colors = color_gradient([BLUE, GREEN, RED, PURPLE, ORANGE], len(sizes))

        for i, s in enumerate(sizes):
            # Create cube
            cube = Cube(
                side_length=2.0 / (2 ** i),
                fill_color=colors[i],
                fill_opacity=0.4,
                stroke_color=WHITE,
            )
            cube.stretch(depth, dim=2)
            cube.rotate(90 * DEGREES, axis=Y_AXIS)
            cube.shift(RIGHT * spacing * (i - (len(sizes) - 1) / 2))
            cubes.append(cube)

            # Label under cube
            lbl = Text(f"{s}×{s}×16", font_size=28, color=colors[i]).next_to(cube, DOWN)
            labels.append(lbl)

            # Add arrow between cubes
            if i > 0:
                prev = cubes[i - 1]

                # Compute start/end points with a small gap
                start = prev.get_right() + RIGHT * arrow_gap
                end = cube.get_left() - RIGHT * arrow_gap

                arrow = Arrow3D(start=start, end=end, color=YELLOW, stroke_width=5)
                arrows.append(arrow)

                # Handle arrow label (string or list)
                if isinstance(arrow_text, list):
                    text = arrow_text[i - 1] if i - 1 < len(arrow_text) else "→"
                else:
                    text = arrow_text

                lbl_arrow = Text(text, font_size=26, color=YELLOW).next_to(arrow, DOWN)
                arrow_labels.append(lbl_arrow)

        # Add all elements
        for c, l in zip(cubes, labels):
            self.add(c, l)
        for a, la in zip(arrows, arrow_labels):
            self.add(a, la)

    def construct(self):
        self.set_camera_orientation(phi=-15 * DEGREES, theta=-90 * DEGREES)
        self.add_cubes(
            [128, 64, 32, 16],
            spacing=2.8,                # reduced distance between cubes
            arrow_gap=0.25,             # small padding for arrows
            arrow_text=["Downsample", "Feature Extract", "Flatten"]
        )
        
        # self.begin_ambient_camera_rotation(rate=0.1)
        # self.wait(1)
        # self.begin_ambient_camera_rotation(rate=-0.2)
        # self.wait(1)

# %%

if __name__ == "__main__":
    from os import system
    system(f"manim -pql {__file__} RecursiveCubes")

