import csv
import matplotlib.pyplot as plt

def log_to_csv(filename):
    # Read the log file
    with open(f'{filename}.txt', 'r') as file:
        lines = file.readlines()

    # Create a CSV file
    with open(f'{filename}.csv', 'w', newline='') as csvfile:
        # Define the CSV header
        fieldnames = ['Episode Number', 'Policy Loss', 'Value Loss', 'Total Reward', 'Achieved Targets']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        # Write the header to the CSV file
        writer.writeheader()

        # Process each line in the log file
        for i in range(0, len(lines), 4):
            episode_number = int(lines[i].split(',')[0].split(' ')[1])
            policy_loss = float(lines[i].split(',')[1].split(': ')[1])
            value_loss = float(lines[i + 1].split(',')[1].split(': ')[1])
            achieved_targets = int(lines[i + 2].split(',')[1].split(': ')[1])
            total_reward = float(lines[i + 3].split(',')[1].split(': ')[1])

            # Write the data to the CSV file
            writer.writerow({'Episode Number': episode_number,
                            'Policy Loss': policy_loss,
                            'Value Loss': value_loss,
                            'Total Reward': total_reward,
                            'Achieved Targets': achieved_targets})
            
def graph(filename):
    # Read data from the CSV file
    with open(f'{filename}.csv', 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        data = list(reader)

    # Extract values for each metric
    episode_numbers = [int(row['Episode Number']) for row in data]
    policy_loss = [float(row['Policy Loss']) for row in data]
    value_loss = [float(row['Value Loss']) for row in data]
    total_reward = [float(row['Total Reward']) for row in data]
    achieved_targets = [int(row['Achieved Targets']) for row in data]

    # Create three subplots
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, sharex=True, figsize=(10, 10))

    # Plot Policy Loss vs Episode Number
    ax1.plot(episode_numbers, policy_loss, label='Policy Loss', color='blue')
    ax1.set_ylabel('Policy Loss')
    ax1.legend()

    # Plot Value Loss vs Episode Number
    ax2.plot(episode_numbers, value_loss, label='Value Loss', color='orange')
    ax2.set_ylabel('Value Loss')
    ax2.legend()

    # Plot Total Reward vs Episode Number
    ax3.plot(episode_numbers, total_reward, label='Total Reward', color='green')
    ax3.set_ylabel('Total Reward')
    ax3.legend()

    # Plot Achieved Targets vs Episode Number
    ax4.plot(episode_numbers, achieved_targets, label='Achieved Targets', color='yellow')
    ax4.set_xlabel('Episode Number')
    ax4.set_ylabel('Achieved Targets')
    ax4.legend()

    # Adjust layout and save the plots    
    plt.tight_layout()
    plt.savefig('results.png')