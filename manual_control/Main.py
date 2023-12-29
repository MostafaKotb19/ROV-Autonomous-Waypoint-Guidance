import numpy as np
from RewardFunction import reward_function
from CustomEnvironment import custom_environment
from KeyboardController import KeyboardController
from scenario import scenario

# Global constants
SCENARIO = scenario
N_TARGETS = 10
N_OBSTACLES = 50

if __name__ == "__main__":

    # Initialize the environment
    env = custom_environment(SCENARIO, N_TARGETS, N_OBSTACLES)
    
    # Initialize variables
    achieved_targets = 0
    state = np.array(env.observation_space)
    total_reward = 0

    # Reset the environment
    env.reset()

    # Initialize keyboard controller
    controller = KeyboardController()

    while True:
        # Exit the loop if 'q' is pressed
        if 'q' in controller.pressed_keys:
            break
        
        # Get control command from pressed keys
        command = controller.parse_keys()

        # Perform a simulation step
        states = env.tick(command)
        env.update_state(states)

        # Update state and calculate rewards
        next_state = env.observation_space
        done = False
        print("Target:", env.get_current_target())
        print("Reward:", total_reward)
        
        reward_f = reward_function(env.prev_location, env.location, env.get_current_target(), 
                                    env.rotation, env.lasers)
        reward = reward_f.calculate_reward()

        # Update previous location
        env.prev_location = env.location

        # Check if the target is reached
        if reward_f.reach_target():              
            achieved_targets += 1

            # Finish the game if all targets are reached
            if achieved_targets == N_TARGETS:
                print("Game Completed")
                done = True
                reward += 1000
                total_reward += reward
                state = next_state
                break
            
            env.set_current_target(env.choose_next_target())
            env.draw_targets()  
        
        total_reward += reward
        state = next_state
