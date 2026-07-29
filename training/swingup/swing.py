import numpy as np
from quanser.hardware import HIL
import time
import math
import traceback


# general parameter
RUN_TIME = 15
FREQUENCY = 500

# qube servo specification
L = 0.129
M = 0.024
J = (1/3)*M*L**2
G = 9.81
U_MAX = 1.85
K = 20
THETA_LIMIT =  math.pi


def encoder_to_rad(counts):
    """
    Convert encoder counts to radians

    Args:
        counts: encoder count value

    Returns:
        Angle in radians
    """
    return 2 * math.pi / 512 / 4 * counts

class SwingUpController:
 
    def __init__(self,
                 J = J,
                 m = M,
                 l = L,
                 g=9.81,
                 k=K,
                 u_max=5.0):
        """
        Initialize the swing-up controller

        Args:
            J: moment of inertia
            m: pendulum mass
            l: pendulum length
            g: gravity constant
            k: swing gain
            u_max: maximum control voltage
        """
        self.J = J
        self.m = m
        self.l = l
        self.g = g
        self.k = k
        self.u_max = u_max
        self.E_d = 2*m*g*l

    def get_energy(self, theta, theta_dot):
        """
        Compute the energy of the pendulum

        Args:
            theta: pendulum angle in radians
            theta_dot: angular velocity

        Returns:
            Scalar energy of the pendulum
        """
        # Energy measured relative to the pendulum hanging downward
        return 0.5* self.J * theta_dot**2 - M*G*L*(np.cos(theta)-1)
    


    def get_u(self, alpha, alpha_dot):
        """
        Compute the swing-up voltage from the energy error

        Args:
            alpha: pendulum angle in radians
            alpha_dot: pendulum angular velocity

        Returns:
            Control voltage value for swing-up
        """
        if abs(alpha_dot) > 30:   # rad/s
            # High velocity safety clamp for fast motion.
            return -2

        E = self.get_energy(alpha, alpha_dot)
        error = self.E_d - E

        # Scale the gain based on the energy error
        ratio = np.clip(abs(error) / self.E_d, 0.1, 1.0)
        K_eff = self.k * ratio

        u = K_eff * error * alpha_dot * np.cos(alpha)
        if abs(alpha_dot) > 30:   # rad/s
            # Very high velocity: reduce the command magnitude
            return -5*u

        return u
    

    def swing_up(self):
        """
        Run a swing-up sequence on the Quanser hardware

        This function reads encoder values, computes swing-up control signals,
        and writes voltages to the motor for a fixed run time
        """

        time.sleep(2)

        card = HIL("qube_servo3_usb", "0")
        task = None

        encoder_channels_read = np.array([0, 1], dtype=np.uint32)
        analog_channels_write = np.array([0], dtype=np.uint32)
        digital_channels_write = np.array([0], dtype=np.uint32)
        other_channels_write = np.array([11000, 11001, 11002], dtype=np.uint32)

        analog_buffer = np.zeros(1, dtype=np.float64)
        encoder_buffer = np.zeros(len(encoder_channels_read), dtype=np.int32)
        digital_buffer = np.zeros(3, dtype=np.int8)
        other_buffer = np.zeros(2, dtype=np.float64)
        

        try:
           
            card.set_encoder_counts(encoder_channels_read, len(encoder_channels_read),
                                    np.array([0, 0], dtype=np.int32))

            card.write_other(other_channels_write, len(other_channels_write),
                            np.array([0, 0, 1], dtype=np.float64))  # LED

            card.write_analog(analog_channels_write, len(analog_channels_write),
                           np.array([0], dtype=np.float64))
            card.write_digital(digital_channels_write, len(digital_channels_write),
                                np.array([1], dtype=np.int8))

            samples_in_buffer = 1000
            samples = 2 ** 32 - 1

            # Create and start the encoder reader task
            task = card.task_create_reader(
                samples_in_buffer,
                None, 0,
                encoder_channels_read, len(encoder_channels_read),
                None, 0,
                None, 0)

            card.task_start(task, 0, FREQUENCY, samples)

            n_steps = RUN_TIME * FREQUENCY
            alpha_prev = 0   
            Ts = 1.0 / FREQUENCY
            
            for i in range(n_steps):
                # Read current encoder values from the hardware task
                card.task_read(task, 1, analog_buffer, encoder_buffer, digital_buffer, other_buffer)

                theta = encoder_to_rad(encoder_buffer[0])
                alpha = encoder_to_rad(encoder_buffer[1])
                print(f"alpha:  {alpha}\n")

                # Estimate angular velocity by finite difference.
                alpha_dot = (alpha - alpha_prev) / Ts
                alpha_prev = alpha

                if abs(alpha_dot) < 0.02 and abs(alpha) < 0.05:
                    # Small initial push when the pendulum is still near down
                    u = 1.0
                else:
                    # Compute swing-up control based on energy
                    u = self.get_u(alpha, alpha_dot)
                u = self.get_u(alpha, alpha_dot)
                u -= 4 * theta

                # Saturate the voltage to safe limits
                if u > U_MAX:
                    u = U_MAX
                if u < -U_MAX:
                    u = -U_MAX

                # Send the computed voltage to the motor
                card.write_analog(analog_channels_write, len(analog_channels_write),
                                np.array([u], dtype=np.float64))

            card.write_analog(analog_channels_write, len(analog_channels_write),
                            np.array([0], dtype=np.float64))
            card.write_other(other_channels_write, len(other_channels_write),
                            np.array([1, 0, 0], dtype=np.float64))
            card.write_digital(digital_channels_write, len(digital_channels_write),
                                np.array([0], dtype=np.int8))

            card.task_stop(task)
            card.task_delete(task)
            card.close()

        except Exception:
            traceback.print_exc()
            if task:
                try:
                    card.task_stop(task)
                    card.task_delete(task)
                except Exception:
                    pass
            try:
                card.write_analog(analog_channels_write, len(analog_channels_write),
                                np.array([0], dtype=np.float64))
                card.write_digital(digital_channels_write, len(digital_channels_write),
                                    np.array([0], dtype=np.int8))
                card.close()
            except Exception:
                pass

    def get_control(self, theta, alpha, alpha_dot):
        """
        Return a control voltage for the current state.

        Args:
            theta: base angle.
            alpha: pendulum angle.
            alpha_dot: pendulum angular velocity.

        Returns:
            Clipped motor voltage.
        """
        # If pendulum is nearly still and down, give a small push
        if abs(alpha_dot) < 0.02 and abs(alpha) < 0.05:
            u = 1.0
        else:
            u = self.get_u(alpha, alpha_dot)

        # Add a base correction for the arm angle
        u -= 2 * theta
        return np.clip(u, -U_MAX, U_MAX)

    
    

# SU = SwingUpController()
# SU.swing_up()