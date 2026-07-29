from quanser.hardware import HIL
import numpy as np
import math
import torch
import time
import argparse
from read_angle import EncoderReader
from swingup.swing import SwingUpController
from stabilization.model import QModel, LR, select_action, optimize_model
from buffer import ReplayBuffer
from plot_csv import init_logger

N_EPISODES = 100000
MAX_STEPS_PER_EPISODE = 500
BATCH_SIZE = 64

ACTIONS = np.linspace(-5.8, 5.8, 21)

FREQUENCY = 300
TS = 1/FREQUENCY

UPRIGHT_THRESHOLD = np.deg2rad(20) 
UPRIGHT_VEL = 2.0       

# general parameter
RUN_TIME = 15
FREQUENCY = 300
W1, W2, W3 = 0.1, 0.02, 0.001
TAU = .005


def modulo(angle):
    return (angle + np.pi) % (2*np.pi) - np.pi


def reset(card, reader, swing, analog_channels_write):
    """
    Bring the pendulum to the upright position using swing-up control

    Args:
        card: Quanser HIL card interface
        reader: EncoderReader instance
        swing: SwingUpController instance
        analog_channels_write: analog channel index array for motor output

    Returns:
        state: numpy array [alpha_mod, theta, alpha_dot, theta_dot]
    """

    while True:

        theta, theta_dot, alpha, alpha_dot = reader.read()

        alpha_mod = (alpha + np.pi) % (2*np.pi) - np.pi


        if abs(alpha_dot) > 45:


            for _ in range(100):

                theta, theta_dot, alpha, alpha_dot = reader.read()

                u = -0.5 * alpha_dot
                u = np.clip(u, -2, 2)

                card.write_analog(
                    analog_channels_write,
                    1,
                    np.array([u], dtype=np.float64)
                )

                time.sleep(TS)


            # stop moteur
            card.write_analog(
                analog_channels_write,
                1,
                np.array([0.0], dtype=np.float64)
            )

            continue   


        if (
            abs(abs(alpha_mod)-np.pi) < np.deg2rad(15)
            and abs(alpha_dot) < 7
        ):
            break


        # Swing-up
        u = swing.get_control(
            theta,
            alpha,
            alpha_dot
        )

        card.write_analog(
            analog_channels_write,
            1,
            np.array([u], dtype=np.float64)
        )

        time.sleep(TS)


    card.write_analog(
        analog_channels_write,
        1,
        np.array([0.0], dtype=np.float64)
    )


    state = np.array([
        alpha_mod,
        theta,
        alpha_dot,
        theta_dot
    ], dtype=np.float32)

    # Return the initial upright state after reset
    return state


def step(card,
         reader,
         action_voltage,
         analog_channels_write, 
         filters):
    """
    Execute one environment step and compute reward

    Args:
        card: Quanser HIL card interface
        reader: EncoderReader instance
        action_voltage: motor voltage command
        analog_channels_write: analog channel index array
        filters: list storing filtered velocities [alpha_dot_f, theta_dot_f]

    Returns:
        state: next state array.
        reward: scalar reward.
        done: boolean
    """

    alpha_dot_f, theta_dot_f = filters
    card.write_analog(
        analog_channels_write,
        1,
        np.array([action_voltage], dtype=np.float64)
    )

    time.sleep(TS)

    theta, theta_dot, alpha, alpha_dot = reader.read()

    # filter
    alpha_dot_f = 0.9 * alpha_dot_f + 0.1 * alpha_dot
    theta_dot_f = 0.9 * theta_dot_f + 0.1 * theta_dot

    alpha_dot = alpha_dot_f
    theta_dot = theta_dot_f

    alpha_mod = (alpha + np.pi) % (2*np.pi) - np.pi

    state = np.array([
        alpha_mod,
        theta,
        alpha_dot,
        theta_dot
    ], dtype=np.float32)
    alpha_error = state[0]

    stability_bonus = 2.0 if abs(alpha_error) < np.deg2rad(20) else 0.0
    reward = (
    10*np.cos(alpha_mod)
    -0.02*theta**2
    +stability_bonus
    -0.05*alpha_dot**2
    -0.001*theta_dot**2
    -0.0005*action_voltage**2
)
    if abs(abs(alpha_mod)-np.pi) < np.deg2rad(15):
        reward += 80
    if abs(theta)>math.radians(90):
        reward-=220
    if abs(alpha_dot) > 5:
        reward -= 100

    done = abs(abs(alpha_mod)-np.pi) > np.deg2rad(40) or abs(theta) > np.pi

    filters[0] = alpha_dot_f
    filters[1] = theta_dot_f
    return state, reward, done





def train():
    """
    Train the DQN agent on the real Quanser environment

    This function initializes hardware, builds the policy and target networks,
    collects transitions, and trains the agent online
    """

    print("Start")
    time.sleep(2)


    card = HIL("qube_servo3_usb", "0")

    analog_channels = np.array([0], dtype=np.uint32)
    digital_channels = np.array([0], dtype=np.uint32)
    other_channels = np.array([11000,11001,11002], dtype=np.uint32)
    encoder_channels = np.array([0,1], dtype=np.uint32)

    card.set_encoder_counts(
        encoder_channels,
        len(encoder_channels),
        np.array([0,0], dtype=np.int32)
    )

    card.write_digital(
        digital_channels,
        len(digital_channels),
        np.array([1], dtype=np.int8)
    )

    card.write_other(
        other_channels,
        len(other_channels),
        np.array([0,0,1], dtype=np.float64)
    )


    reader = EncoderReader(card)
    swing = SwingUpController()



    policy_net = QModel(
        obs_dim=4,
        n_actions=len(ACTIONS)
    )

    target_net = QModel(
        obs_dim=4,
        n_actions=len(ACTIONS)
    )

    # policy_net.load_state_dict(torch.load("stabilizer_BESTB.pt"))

    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = torch.optim.Adam(
        policy_net.parameters(),
        lr=LR
    )

    memory = ReplayBuffer(100000)

    step_done = 0
    alpha_dot_f = 0.0
    theta_dot_f = 0.0

    maxr = -np.inf

    log_file, log_writer = init_logger(
        "training_log.csv"
    )
    try:

        for episode in range(N_EPISODES):

            print(f"\nEpisode {episode}")

            # Start each episode by swinging the pendulum upright
            state = reset(
                card,
                reader,
                swing,
                analog_channels
            )

            done = False
            episode_reward = 0
            filters = [0.0, 0.0]

            episode_step = 0
            episode_start_time = time.time()
            while not done:

                state_tensor = torch.tensor(
                    state,
                    dtype=torch.float32
                )

                # Choose an action using the policy network and epsilon-greedy
                action_index = select_action(
                    state_tensor,
                    policy_net,
                    step_done,
                    len(ACTIONS)
                )

                step_done += 1

                u = ACTIONS[action_index]

                next_state, reward, done = step(
                    card,
                    reader,
                    u,
                    analog_channels,
                    filters
                )
                theta_log, theta_dot_log, alpha_log, alpha_dot_log = reader.read()

                alpha_mod_log = modulo(alpha_log)

                alpha_error_log = (
                    (alpha_mod_log - np.pi) + np.pi
                ) % (2*np.pi) - np.pi

                episode_reward += reward

                log_writer.writerow([
                    episode,
                    episode_step,
                    time.time() - episode_start_time,

                    alpha_log,
                    alpha_error_log,

                    theta_log,

                    alpha_dot_log,
                    theta_dot_log,

                    u,

                    reward,
                    episode_reward,
                    maxr,

                    done
                ])

                log_file.flush()
                episode_step += 1

                memory.push(
                    state,
                    action_index,
                    reward,
                    next_state,
                    done
                )

                if len(memory) > BATCH_SIZE:

                    # Update the policy network from a sampled batch
                    optimize_model(
                        policy_net,
                        target_net,
                        optimizer,
                        memory
                    )
                    # Soft-update the target network.
                    for target_param, policy_param in zip(
                            target_net.parameters(),
                            policy_net.parameters()):

                        target_param.data.copy_(
                            TAU * policy_param.data +
                            (1 - TAU) * target_param.data
                        )

                state = next_state


            if episode_reward>maxr:
                maxr = episode_reward
                print("NEW BEST : ", maxr, "\n")
                # Save the best performing policy weights.
                torch.save(policy_net.state_dict(), "stabilizer_BEST_tolog.pt")

            print(
                "Reward  :",
                episode_reward, 
                "     | BEST:", maxr
            )

    finally:
        torch.save(policy_net.state_dict(), "stabilizer_tolog.pt")

        card.write_analog(
            analog_channels,
            1,
            np.array([0.0], dtype=np.float64)
        )

        card.close()
        log_file.close()

def controle():
    """
    Run the trained controller on the Quanser hardware.

    This function loads a saved policy and alternates between swing-up
    and DQN stabilization modes
    """

    print("START")
    time.sleep(2)

    card = HIL("qube_servo3_usb", "0")

    analog_channels = np.array([0], dtype=np.uint32)
    digital_channels = np.array([0], dtype=np.uint32)
    other_channels = np.array([11000,11001,11002], dtype=np.uint32)

    encoder_channels = np.array([0,1], dtype=np.uint32)

    card.set_encoder_counts(
        encoder_channels,
        len(encoder_channels),
        np.array([0,0],dtype=np.int32)
    )

    card.write_digital(
        digital_channels,
        len(digital_channels),
        np.array([1],dtype=np.int8)
    )

    # LED
    card.write_other(
        other_channels,
        len(other_channels),
        np.array([0,0,1],dtype=np.float64)
    )

    reader = EncoderReader(card)

    swing = SwingUpController()

    theta, theta_dot, alpha, alpha_dot = reader.read()
    policy_net = QModel(
    obs_dim=4,
    n_actions=len(ACTIONS)
    )

    policy_net.load_state_dict(torch.load("stabilizer_BESTnice.pt"))
    policy_net.eval()

    try:

        # Start in swing-up mode and switch to DQN once upright
        mode = "SWING"

        filters = [0.0, 0.0]

        while True:

            state = reset(card, reader, swing, analog_channels)

            episode_reward = 0
            done = False

            while not done:

                state_tensor = torch.tensor(state, dtype=torch.float32)

                with torch.no_grad():
                    action = policy_net(state_tensor).argmax().item()

                u = ACTIONS[action]

                next_state, reward, done = step(
                    card,
                    reader,
                    u,
                    analog_channels,
                    filters
                )

                episode_reward += reward

                print(f"Reward : {reward:.2f} | Return : {episode_reward:.2f}")

                state = next_state

            print("End of episode:", episode_reward)
    except KeyboardInterrupt:

        print("Stop")

    finally:

        card.write_analog(
            analog_channels,
            len(analog_channels),
            np.array([0.0],dtype=np.float64)
        )

        card.write_digital(
            digital_channels,
            len(digital_channels),
            np.array([0],dtype=np.int8)
        )

        card.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run Quanser QUBE controller or training mode")
    parser.add_argument("--mode", choices=["train", "control"], default="control",
                        help="Select mode: train to run training, control to run the controller")
    args = parser.parse_args()

    if args.mode == "train":
        train()
    else:
        controle()
