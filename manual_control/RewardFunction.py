import math
import numpy as np

# Global variable to keep track of static count
static_counter = 0

class reward_function:
    def __init__(self, prev_location, location, target, rotation, lasers):
        """
        Initialize the reward function.

        Parameters:
        - prev_location: Previous location of the agent.
        - location: Current location of the agent.
        - target: Target location.
        - rotation: Rotation of the agent.
        - lasers: Laser readings.
        """
        self.prev_location = prev_location
        self.location = location
        self.target = target
        self.rotation = rotation
        self.lasers = lasers

    def outside_box(self):
        """
        Check if the agent is outside the predefined box.

        Returns:
        - 1 if outside the box, 0 otherwise.
        """
        box_center = (200, 200, -250)
        box_dimensions = (120, 120, 120)
        box_x_min = box_center[0] - box_dimensions[0] / 2
        box_x_max = box_center[0] + box_dimensions[0] / 2
        box_y_min = box_center[1] - box_dimensions[1] / 2
        box_y_max = box_center[1] + box_dimensions[1] / 2
        box_z_min = box_center[2] - box_dimensions[2] / 2
        box_z_max = box_center[2] + box_dimensions[2] / 2

        is_outside_box = (
            self.location[0] < box_x_min or self.location[0] > box_x_max or
            self.location[1] < box_y_min or self.location[1] > box_y_max or
            self.location[2] < box_z_min or self.location[2] > box_z_max
        )

        return 1 if is_outside_box else 0

    def distance_to_target(self):
        """
        Check if the agent has moved closer to the target.

        Returns:
        - 1 if moved closer, 0 otherwise.
        """
        prev_distance = math.dist(self.prev_location, self.target)
        distance = math.dist(self.location, self.target)
        return 1 if prev_distance - distance > 0.02 else 0

    def collision(self):
        """
        Check for collisions using laser readings.

        Returns:
        - 1 if a collision is detected, 0 otherwise.
        """
        return 1 if any(self.lasers <= 0) else 0
    
    def incline(self, roll, pitch):
        """
        Calculate penalty based on roll and pitch angles.

        Returns:
        - Penalty value.
        """
        penalty = 0
        if 15 < roll < 180:
            penalty += roll - 15
        elif 180 < roll < 345:
            penalty += 345 - roll
        if 0 < pitch < 170:
            penalty += 170 - pitch
        elif 190 < pitch < 360:
            penalty += pitch - 190
        return penalty * 0.001
    
    def static(self):
        """
        Check if the agent is static for a certain duration.

        Returns:
        - 1 if static for a specified duration, 0 otherwise.
        """
        global static_counter
        displacement = math.dist(self.prev_location, self.location)
        if displacement < 0.01:
            static_counter += 1
        else:
            static_counter = 0
        return 1 if static_counter >= 50 else 0
    
    def near_miss(self):
        """
        Check for a near miss using laser readings.

        Returns:
        - 1 if a near miss is detected, 0 otherwise.
        """
        return 1 if np.any(self.lasers < 1) else 0
    
    def reach_target(self):
        """
        Check if the agent has reached the target.

        Returns:
        - 1 if the target is reached, 0 otherwise.
        """
        distance = math.dist(self.location, self.target)
        return 1 if distance < 2 else 0

    def calculate_reward(self):
        """
        Calculate the total reward based on different criteria.

        Returns:
        - Total reward value.
        """
        self.reward = 0
        self.reward -= 100*self.outside_box()
        self.reward -= 30*self.collision()
        self.reward -= 5*self.near_miss()
        self.reward -= 1*self.incline(self.rotation[0], self.rotation[1])
        self.reward -= 1*self.static()
        self.reward += 1*self.distance_to_target()
        self.reward += 100*self.reach_target()
        return self.reward
