import numpy as np
from PPOAgent import PPO_agent
from RewardFunction import reward_function
from CustomEnvironment import custom_environment
from scenario import scenario

# Global constants
SCENARIO = scenario
ACTION_SPACE_SIZE = 5
OBSERVATION_SPACE_SIZE = 36
N_TARGETS = 10
N_OBSTACLES = 50
LAST_EPISODE = 0
N_EPISODES = 100000
REWARD_THRESHOLD = -1000
MAX_STEPS = int(1e4)

if __name__ == "__main__":

    # Initialize the environment, and PPO agent
    env = custom_environment(SCENARIO, N_TARGETS, N_OBSTACLES)
    ppo_agent = PPO_agent(ACTION_SPACE_SIZE, OBSERVATION_SPACE_SIZE)
    ppo_agent.load_model(LAST_EPISODE)

    for episode in range(LAST_EPISODE, N_EPISODES):
        
        # Initialize variables
        achieved_targets = 0
        state = np.array(env.observation_space)
        total_reward = 0
        episode_states, episode_actions, episode_rewards, episode_dones, episode_probs = [], [], [], [], []

        # Reset the environment
        env.reset()

        for i in range(MAX_STEPS):
            #Select action based on the ppo agent weights
            action_probs = ppo_agent.policy(np.array([state]))[0].numpy()
            action = ppo_agent.select_action(state)
            print("Episode: ", episode)
            print("Selected action:", action)

            # Perform a simulation step
            states = env.tick(action)
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
                    episode_states.append(state)
                    episode_actions.append(action)
                    episode_rewards.append(reward)
                    episode_dones.append(done)
                    episode_probs.append(action_probs)

                    total_reward += reward
                    state = next_state
                    break
                
                env.set_current_target(env.choose_next_target())
                env.draw_targets()
            
            #Append state, selected action, gained reward, done, and action probabilities
            episode_states.append(state)
            episode_actions.append(action)
            episode_rewards.append(reward)
            episode_dones.append(done)
            episode_probs.append(action_probs)

            total_reward += reward
            state = next_state
            if done or total_reward < REWARD_THRESHOLD:
                break
        
        # Convert episode data to arrays for processing
        episode_states = np.array(episode_states)
        episode_actions = np.array(episode_actions)
        episode_rewards = np.array(episode_rewards)
        episode_dones = np.array(episode_dones)
        episode_probs = np.array(episode_probs)

        # Calculate advantages and normalize them
        values = ppo_agent.value_network(episode_states).numpy().flatten()
        advantages = ppo_agent.compute_advantages(episode_rewards, values, episode_dones)
        advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)
        
        # Update policy and value networks
        ppo_agent.update_policy(episode_states, episode_actions, advantages, episode_probs, episode)
        discounted_rewards = ppo_agent.discounted_rewards(episode_rewards)
        ppo_agent.update_value_network(episode_states, discounted_rewards, episode)
        ppo_agent.log_episode_reward(episode, total_reward)

        # Save the model periodically
        if episode % 20 == 0:
            ppo_agent.save_model(episode)