# Shared_Autonomy_Driving
Course project to learn if doing overtaking maneuver is safe and to allow human inputs to take over.

To train the Torch policy you will need PyTorch installed:

```bash
pip install torch
```

Run the automated overtaking demo with:

```bash
python main.py
```

Run the keyboard-playable version with arrow keys for throttle and steering:

```bash
python play.py
```

Run the keyboard-playable version with safety indicator from trained policy:

```bash
python play.py --policy artifacts/overtaking_safety_policy.pt --stack-size 4
```

The safety indicator shows whether the trained classifier predicts it is safe to overtake at each timestep.


Train the Torch safety policy classifier:

```bash
python train_overtaking_policy.py
```

The trainer stacks consecutive environment snapshots so the policy can infer motion from short temporal context.
