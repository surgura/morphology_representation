import cairo

from revolve2.core.modular_robot import ActiveHinge, Body, Brick, Core, Module
from graphviz import Digraph
from collections import deque
from typing import Tuple, Optional


def render_modular_robot2d(body: Body, file_name: str) -> None:
    SCALE = 100

    grid, core_pos = body.to_grid()
    assert len(grid[0][0]) == 1, "Can only render 2d robots."

    surface = cairo.ImageSurface(
        cairo.FORMAT_ARGB32, len(grid) * SCALE, len(grid[0]) * SCALE
    )
    context = cairo.Context(surface)
    context.scale(SCALE, SCALE)

    for x, yz_grid in enumerate(grid):
        for y, z_grid in enumerate(yz_grid):
            module = z_grid[0]
            if isinstance(module, Core):
                context.rectangle(x, y, 1, 1)
                context.set_source_rgb(255, 255, 0)
                context.fill_preserve()
                context.set_source_rgb(0, 0, 0)
                context.set_line_width(0.01)
                context.stroke()
            elif isinstance(module, Brick):
                context.rectangle(x, y, 1, 1)
                context.set_source_rgb(0, 0, 1)
                context.fill_preserve()
                context.set_source_rgb(0, 0, 0)
                context.set_line_width(0.01)
                context.stroke()
            elif isinstance(module, ActiveHinge):
                context.rectangle(x, y, 1, 1)
                context.set_source_rgb(1, 0, 0)
                context.fill_preserve()
                context.set_source_rgb(0, 0, 0)
                context.set_line_width(0.01)
                context.stroke()
            elif module is not None:
                raise NotImplementedError()

    surface.write_to_png(file_name)


def _mod_name(mod: Optional[Module]) -> str:
    if mod is None:
        return "X"
    elif isinstance(mod, Core):
        return "C"
    elif isinstance(mod, ActiveHinge):
        return "H"
    elif isinstance(mod, Brick):
        return "B"
    else:
        raise NotImplementedError()


def render_modular_robot_radial(body: Body, file_name: str) -> None:
    queue: deque[Tuple[Module, str, int]] = deque()
    tree = Digraph(engine="neato")

    i = 0

    tree.node(f"{i}", _mod_name(body.core))
    for n in range(len(body.core.children)):
        queue.append((body.core, f"{i}", n))

    i += 1

    while len(queue) > 0:
        (pmod, pname, child_n) = queue.pop()
        cmod = pmod.children[child_n]
        if cmod is not None:
            tree.node(f"{i}", _mod_name(cmod))
            tree.edge(pname, f"{i}", f"{child_n}")
            if cmod is not None:
                for n in range(len(cmod.children)):
                    queue.append((cmod, f"{i}", n))
            i += 1

    tree.render(outfile=file_name, format="png", cleanup=True)
