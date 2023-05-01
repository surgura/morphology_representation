import cairo

from revolve2.core.modular_robot import ActiveHinge, Body, Brick, Core


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
