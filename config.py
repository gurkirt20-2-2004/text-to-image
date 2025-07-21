# config.py

import torch

class CFG:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed = 42
    generator = torch.Generator(device).manual_seed(seed)

    # Image generation settings
    image_gen_steps = 35
    image_gen_model_id = "stabilityai/stable-diffusion-2"
    image_gen_size = (400, 400)
    image_gen_guidance_scale = 9

    # Prompt generation settings
    prompt_gen_model_id = "gpt2"
    prompt_dataset_size = 6
    prompt_max_length = 12

    # Training and fine-tuning settings
    batch_size = 2
    learning_rate = 5e-6
    num_epochs = 3

    # Paths
    dataset_path = "./data/text_image_pairs"  # Update to your dataset path
    output_dir = "./output"  # Where generated images and checkpoints will be saved
    logs_dir = "./logs"  # Log files path

    # For reproducibility in torch, numpy, random
    @staticmethod
    def set_global_seed(seed_value=42):
        import random
        import numpy as np

        random.seed(seed_value)
        np.random.seed(seed_value)
        torch.manual_seed(seed_value)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed_value)

# Call once at the top level to ensure reproducibility
CFG.set_global_seed(CFG.seed)
