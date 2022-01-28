import numpy as np
import matplotlib.pyplot as plt 
from bresenham import bresenham_points
from baseline_utils import pose_to_coords_frame
import math

def manhattan_distance(coord_a, coord_b):
    return (abs(coord_a[0] - coord_b[0]) + abs(coord_a[1] - coord_b[1]))

class LocalNavigator:
    def __init__(self, pose_range, coords_range):
        self.pose_range = pose_range
        self.coords_range = coords_range
        self.mode_follow_line = True

    def new_episode(self, start_pose, goal_pose):
        self.start = pose_to_coords_frame((start_pose[0], start_pose[1]), self.pose_range, self.coords_range, cell_size=.1)
        self.goal = pose_to_coords_frame((goal_pose[0], goal_pose[1]), self.pose_range, self.coords_range, cell_size=.1)
        # The line now extends *beyond* the goal (important for checking intersection)
        term = 5 * list(self.goal) - list(4 * self.start)
        self.line = bresenham_points(self.start, term)

    ## predict the next subgoal of habitat agent
    def local_nav(self, grid, current_pose):
        # convert pose to coordinates 
        current_pos = pose_to_coords_frame((current_pose[0], current_pose[1]), self.pose_range, self.coords_range, cell_size=.1)
        current_pos = (current_pos[0], current_pos[1], current_pose[2])

        # stop criteria
        if manhattan_distance(current_pos, self.goal) < 5:
            return "STOP"

        if self.mode_follow_line:
            # Follow the line until we cannot
            next_action = self.follow_line(grid, current_pos)
            #if did_encounter_obstacle:
            #    self.mode_follow_line = False
        '''
        else:
            # Follow the object until we encounter
            robot.follow_object(grid)
            if robot.is_on_line(line):
                self.mode_follow_line = True
        '''

        return next_action

    # return the next position, orientation not included
    def follow_line(self, grid, current_pos):
        manhattan_dist = (np.abs(self.line[0] - current_pos[0]) +
                          np.abs(self.line[1] - current_pos[1]))
        ind = np.argwhere(manhattan_dist < 0.5)[0]

        # If reached the end, stop
        #if ind == self.line.shape[1] - 1:
        #    return False

        # predict next subgoal
        next_pos = self.line[:, ind+1].T[0]

        # convert subgoal to action according to position and orientation
        stg_x, stg_z = next_pos
        angle_st_goal = math.degrees(math.atan2(stg_x - current_pos[0],
                                                stg_z - current_pos[1]))
        angle_agent = (start_o) % 360.0
        if angle_agent > 180:
            angle_agent -= 360

        relative_angle = (angle_agent - angle_st_goal) % 360.0
        if relative_angle > 180:
            relative_angle -= 360

        if relative_angle > 30 / 2.:
            action = "TURN_RIGHT"  # Right
        elif relative_angle < -30 / 2.:
            action = "TURN_LEFT"  # Left
        else:
            action = "MOVE_FORWARD"  # Forward


        return action

        '''
        orientation = next_pos.T - self.position
        self.orientation = np.array(
            [round(orientation[0]), round(orientation[1])])
        if grid[next_pos[0], next_pos[1]] == OBSTACLE:

            return True
        else:
            self._move_forward()
            return False
        '''

    

