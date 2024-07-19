import os
from getpass import getpass


def get_checkpoint_dir(cfg):
    if cfg.slurm_checkpoint:
        return f"/checkpoint/{getpass.getuser()}/{os.environ.get('SLURM_JOB_ID')}"
    return cfg.checkpoint_dir


def load_checkpoint(trainer):
    cfg = trainer.cfg
    checkpoint_dir = get_checkpoint_dir(cfg)
    steps_done_path = os.path.join(checkpoint_dir, 'steps_done.txt')
    has_checkpoint = os.path.exists(steps_done_path)
    if has_checkpoint:
        with open(steps_done_path, 'r') as f:
            steps_done = f.read()
        print(f'Loading checkpoint from step {steps_done}')
        checkpoint_path = os.path.join(checkpoint_dir, 'checkpoint_' + steps_done)
        trainer.agent.load(os.path.join(checkpoint_path, 'agent'))
        trainer.buffer.load(os.path.join(checkpoint_path, 'buffer'))
        return int(steps_done)
    else:
        print(f'No checkpoint found at {checkpoint_dir}')
        return 0

def save_checkpoint(step, trainer):
    cfg = trainer.cfg
    checkpoint_dir = get_checkpoint_dir(cfg)
    print(f'Saving checkpoint at step {step} to {checkpoint_dir}')
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_{step}')
    os.makedirs(checkpoint_path, exist_ok=True)
    trainer.agent.save(os.path.join(checkpoint_path, 'agent'))
    trainer.buffer.save(os.path.join(checkpoint_path, 'buffer'))
    with open(os.path.join(checkpoint_dir, 'steps_done.txt'), 'w') as f:
        f.write(str(step))
    print(f'Saved checkpoint at step {step}')
    prev_checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_{step - cfg.checkpoint_interval}')
    if os.path.exists(prev_checkpoint_path):
        os.system(f'rm -rf {prev_checkpoint_path}')