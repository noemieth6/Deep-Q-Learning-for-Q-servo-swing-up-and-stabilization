from quanser.hardware import HIL
import numpy as np
import math
import time

def encoder_to_rad(counts):
    return 2 * math.pi / (512 * 4) * counts


class EncoderReader:

    def __init__(self, card):

        self.card = card

        self.encoder_channels = np.array([0, 1], dtype=np.uint32)
        self.encoder_buffer = np.zeros(2, dtype=np.int32)

        self.card.set_encoder_counts(
            self.encoder_channels,
            len(self.encoder_channels),
            np.array([0, 0], dtype=np.int32)
        )

        self.theta_prev = 0.0
        self.alpha_prev = 0.0

        self.theta_dot = 0.0
        self.alpha_dot = 0.0

        self.t_prev = time.perf_counter()

        self.beta = 0.2

    def read(self):

        self.card.read_encoder(
            self.encoder_channels,
            len(self.encoder_channels),
            self.encoder_buffer
        )

        theta = encoder_to_rad(self.encoder_buffer[0])
        alpha = encoder_to_rad(self.encoder_buffer[1])

        t = time.perf_counter()
        Ts = t - self.t_prev
        self.t_prev = t

        theta_dot_raw = (theta - self.theta_prev) / Ts
        alpha_dot_raw = (alpha - self.alpha_prev) / Ts

        self.theta_dot = (
            (1 - self.beta) * self.theta_dot
            + self.beta * theta_dot_raw
        )

        self.alpha_dot = (
            (1 - self.beta) * self.alpha_dot
            + self.beta * alpha_dot_raw
        )

        self.theta_prev = theta
        self.alpha_prev = alpha

        return theta, self.theta_dot, alpha, self.alpha_dot
    
    def continuous_read():
        card = HIL("qube_servo3_usb", "0")

        reader = EncoderReader(card)

        try:

            while True:

                theta, theta_dot, alpha, alpha_dot = reader.read()

                print(
                    f"theta={theta:6.3f} | "
                    f"alpha={alpha:6.3f} | "
                    f"theta_dot={theta_dot:6.2f} | "
                    f"alpha_dot={alpha_dot:6.2f}"
                )

        except KeyboardInterrupt:
            pass

        finally:
            card.close()