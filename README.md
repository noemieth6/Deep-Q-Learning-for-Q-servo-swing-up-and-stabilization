# DDO Learning Double Project

This project contains code for controlling and training a Quanser QUBE servo 3 system using reinforcement learning (Double Deep Q-Learning ).

## Project structure

- `training/controller.py` - main Python script for controlling the hardware and running the trained policy.
- `training/read_angle.py` - encoder reading helper for the Quanser hardware.
- `training/buffer.py` - replay buffer used for training.
- `training/stabilization/model.py` - DQN model and training utilities.
- `training/swingup/swing.py` - swing-up controller for the pendulum.
- `training/*.pt` - saved model weights.

## Requirements

- Python 3.x
- `numpy`
- `torch`
- `quanser` hardware API
```

The `quanser` package is required for the hardware interface and may need a special installation provided by Quanser.

## How to run

1. Open a terminal in the project root.
2. Make sure the Quanser QUBE hardware is connected.
3. Run the controller script:

```bash
python training/controller.py
```
  
or for training mode:

  
```bash
python training/controller.py --train
```

This script starts the hardware control loop and loads the saved policy from `training/stabilizer_BEST.pt`.

## Notes

- The current entry point is `training/controller.py`, which runs the `controle()` function when executed.
- Training code exists in `training/controller.py` and `training/stabilization/model.py`, but the default run path only starts the controller.
- If you need to modify or retrain the model, update the script and run the appropriate training function.

## Hardware

This project is designed for a Quanser QUBE servo system with encoder input and analog output control.

> Use caution when running on real hardware and verify connections before starting the script.
