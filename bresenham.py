import numpy as np

def bresenham_points(start, target, scale=1, do_floor=True):
    """Get the points along a line using Bresenham's Algorithm.
    Using Bresenham's algorithm, this function returns a list of points foom the
    starting location to the target location. An optional scale argument can be
    passed to compute points at a fractional grid resolution. For example, if
    requesting the points between start=[0.0, 0.9] and target=[5.0, 1.9], we
    might want the line to "jump" earlier. Using scale=2 allows us to accomplish
    this:
    >>> bresenham_points(start=[0.0, 0.9], target=[5.0, 1.9], \
                         scale=1, do_floor=True)
    array([[0, 1, 2, 3, 4, 5],
           [0, 0, 0, 1, 1, 1]])
    >>> bresenham_points(start=[0.0, 0.9], target=[5.0, 1.9], \
                         scale=2, do_floor=True)
    array([[0, 1, 2, 3, 4, 5],
           [0, 1, 1, 1, 1, 1]])
    If the values are not "floored", all the sub-pixel point are returned:
    >>> bresenham_points(start=[0.0, 0.9], target=[5.0, 1.9], \
                         scale=2, do_floor=False)
    array([[ 0. ,  0.5,  1. ,  1.5,  2. ,  2.5,  3. ,  3.5,  4. ,  4.5,  5. ],
           [ 0.5,  0.5,  0.5,  1. ,  1. ,  1. ,  1. ,  1. ,  1.5,  1.5,  1.5]])
    Args:
        start (2x float): 2D starting point
        target (2x float): 2D ending point
        scale (int): Optional sub-pixel scale argument
        do_floor (Bool): Optional argument to return integer coordinates
            or fractional coordinages (for scale != 1)
    Returns: 2xN numpy array of N coordinates corresponding to points given by
        Bresenham's algorithm. (This includes the endpoints start and target.)
    """

    # Convert to integers
    upscaled_start_int = [int(scale * start[0]), int(scale * start[1])]
    upscaled_target_int = [int(scale * target[0]), int(scale * target[1])]

    upscaled_point = upscaled_start_int

    dx = upscaled_target_int[0] - upscaled_start_int[0]
    xstep = 1
    if dx < 0:
        dx = -dx
        xstep = -1

    dy = upscaled_target_int[1] - upscaled_start_int[1]
    ystep = 1
    if dy < 0:
        dy = -dy
        ystep = -1

    if dx == 0:
        # Vertical
        upsampled_points = np.zeros([2, dy + 1])
        for ii in range(dy + 1):
            upsampled_points[0, ii] = upscaled_point[0]
            upsampled_points[1, ii] = upscaled_point[1]
            upscaled_point[1] += ystep
    elif dy == 0:
        # Horizontal
        upsampled_points = np.zeros([2, dx + 1])
        for ii in range(dx + 1):
            upsampled_points[0, ii] = upscaled_point[0]
            upsampled_points[1, ii] = upscaled_point[1]
            upscaled_point[0] += xstep
    elif dx > dy:
        n = dx
        dy += dy
        e = dy - dx
        dx += dx

        upsampled_points = np.zeros([2, n + 1])
        for ii in range(n + 1):
            upsampled_points[0, ii] = upscaled_point[0]
            upsampled_points[1, ii] = upscaled_point[1]
            if e >= 0:
                upscaled_point[1] += ystep
                e -= dx
            e += dy
            upscaled_point[0] += xstep
    else:
        n = dy
        dx += dx
        e = dx - dy
        dy += dy

        upsampled_points = np.zeros([2, n + 1])
        for ii in range(n + 1):
            upsampled_points[0, ii] = upscaled_point[0]
            upsampled_points[1, ii] = upscaled_point[1]
            if e >= 0:
                upscaled_point[0] += xstep
                e -= dy
            e += dx
            upscaled_point[1] += ystep

    # Now return the collision state and the current pose
    points = 1.0 * upsampled_points / scale

    if do_floor is True:
        points = points.astype(int)
        indices = np.unique(points, axis=1, return_index=True)[1]
        points = np.array([points[:, ind] for ind in sorted(indices)]).T

    return points
