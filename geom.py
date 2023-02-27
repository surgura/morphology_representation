import torch
from torch import tensor
from torch_geometric.data import HeteroData
from revolve2.core.modular_robot import Body, Brick, ActiveHinge
from revolve2.standard_resources.modular_robots import gecko

"""
head
brick
hinge

head_brick: slot
head_hinge slot
brick_hinge slot
hinge_brick
"""


def body_to_hetero(body: Body) -> HeteroData:
    core = body.core
    bricks = body.find_bricks()
    active_hinges = body.find_active_hinges()

    data = HeteroData()
    data["core"].x = tensor([[1.0]])
    data["brick"].x = [[] for brick in bricks]
    # data["active_hinge"] = [[] for active_hinge in active_hinges]

    # data["core", "core_to_brick", "brick"].edge_index = (
    #     tensor(
    #         [
    #             [0, bricks.index(child)]
    #             for child in core.children
    #             if child is not None and isinstance(child, Brick)
    #         ]
    #     )
    #     .t()
    #     .contiguous()
    # )

    # data["core", "core_to_active_hinge", "brick"].edge_index = (
    #     tensor(
    #         [
    #             [0, active_hinges.index(child)]
    #             for child in core.children
    #             if child is not None and isinstance(child, ActiveHinge)
    #         ]
    #     )
    #     .t()
    #     .contiguous()
    # )

    # data["brick", "brick_to_brick", "brick"].edge_index = (
    #     tensor(
    #         [
    #             [bricks.index(brick), bricks.index(child)]
    #             for brick in bricks
    #             for child in brick.children
    #             if child is not None and isinstance(child, Brick)
    #         ]
    #     )
    #     .t()
    #     .contiguous()
    # )

    # data["brick", "brick_to_active_hinge", "active_hinge"].edge_index = (
    #     tensor(
    #         [
    #             [bricks.index(brick), active_hinges.index(child)]
    #             for brick in bricks
    #             for child in brick.children
    #             if child is not None and isinstance(child, ActiveHinge)
    #         ]
    #     )
    #     .t()
    #     .contiguous()
    # )

    # data["active_hinge", "active_hinge_to_brick", "brick"].edge_index = (
    #     tensor(
    #         [
    #             [active_hinges.index(active_hinge), bricks.index(child)]
    #             for active_hinge in active_hinges
    #             for child in active_hinge.children
    #             if child is not None and isinstance(child, Brick)
    #         ]
    #     )
    #     .t()
    #     .contiguous()
    # )

    # data["active_hinge", "active_hinge_to_brick", "active_hinge"].edge_index = (
    #     tensor(
    #         [
    #             [active_hinges.index(active_hinge), active_hinges.index(child)]
    #             for active_hinge in active_hinges
    #             for child in active_hinge.children
    #             if child is not None and isinstance(child, ActiveHinge)
    #         ]
    #     )
    #     .t()
    #     .contiguous()
    # )

    # # edge data
    # # data["core", "core_to_brick", "brick"].edge_attr = tensor(
    # #     [
    # #         [child.rotation]
    # #         for child in core.children
    # #         if child is not None and isinstance(child, Brick)
    # #     ]
    # # )

    # data["core", "core_to_active_hinge", "brick"].edge_attr = tensor(
    #     [
    #         [child.rotation]
    #         for child in core.children
    #         if child is not None and isinstance(child, ActiveHinge)
    #     ]
    # )

    # data["brick", "brick_to_brick", "brick"].edge_attr = tensor(
    #     [
    #         [child.rotation]
    #         for brick in bricks
    #         for child in brick.children
    #         if child is not None and isinstance(child, Brick)
    #     ]
    # )

    # data["brick", "brick_to_active_hinge", "active_hinge"].edge_attr = tensor(
    #     [
    #         [child.rotation]
    #         for brick in bricks
    #         for child in brick.children
    #         if child is not None and isinstance(child, ActiveHinge)
    #     ]
    # )

    # data["active_hinge", "active_hinge_to_brick", "brick"].edge_attr = tensor(
    #     [
    #         [child.rotation]
    #         for active_hinge in active_hinges
    #         for child in active_hinge.children
    #         if child is not None and isinstance(child, Brick)
    #     ]
    # )

    # data["active_hinge", "active_hinge_to_brick", "active_hinge"].edge_attr = tensor(
    #     [
    #         [child.rotation]
    #         for active_hinge in active_hinges
    #         for child in active_hinge.children
    #         if child is not None and isinstance(child, ActiveHinge)
    #     ]
    # )

    return data


data = body_to_hetero(gecko())
print(data)
print(data.has_isolated_nodes())
