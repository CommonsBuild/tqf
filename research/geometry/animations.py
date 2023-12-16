import param as pm

# Eigen::Vector2d Animations::moveable_points[5];
# Eigen::Vector2d Animations::unfiltered_points[5];

movable_points = dict(
    x=[1, 2, 3, 4, 5],
    y=[3, 6, 1, 5, 4],
    radius=1,
    alpha=0.7,
    color='green',
)

unfiltered_points = dict(
    x=[2, 3, 1, -5, -2], y=[5, -6, 2, 9, 1], radius=1, alpha=0.7, color='blue'
)


class DualLines:
    mode = pm.Selector(
        default='MODE_PT_ONLY',
        objects=[
            'MODE_PT_ONLY',
            'MODE_LINE_ONLY',
            'MODE_PT_AND_LINE',
            'MODE_CIRCLE',
            'MODE_PERP',
            'MODE_END',
        ],
    )

    t = param.Number(0)

    p1 = movable_points[0]


class Points(pm.Parameterized):
    points = pm.DataFrame()


points = pm.DataFrame()
