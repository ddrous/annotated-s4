#%%

import jax
import jax.numpy as np
import hydra
from omegaconf import DictConfig, OmegaConf
from functools import partial

# --- Flax/Model Imports ---
from flax.training import checkpoints
import orbax.checkpoint as ocp  # <--- ADD THIS FOR ORBAX CHECKPOINTS
from s4 import (
    BatchStackedModel,
    S4Layer,
    sample_image_prefix_celeba,
    init_recurrence,
)
from data import Datasets

# --- 1. SET YOUR PARAMETERS HERE ---

# Point this to the checkpoint directory (can be with or without .orbax-checkpoint-tmp-* suffix)
CHECKPOINT_PATH = "/home/gb21553/Projects/Annotated-S4/checkpoints/celeba/s4-d_model=340-lr=0.001-bsz=32/checkpoint_0.orbax-checkpoint-tmp-0"

# Set any new prefix size you want!
NEW_PREFIX_SIZE = 500

# Set the batch size for how many images to generate
BATCH_SIZE = 8

# --- 2. LOAD CHECKPOINT FUNCTION ---
def load_orbax_checkpoint(checkpoint_path):
    """
    Load an Orbax checkpoint.
    """
    print(f"[*] Loading Orbax checkpoint from {checkpoint_path}...")
    
    # Create an Orbax checkpointer
    checkpointer = ocp.PyTreeCheckpointer()
    
    # Restore the checkpoint
    # For Orbax checkpoints, we just restore directly
    restored = checkpointer.restore(checkpoint_path)
    
    return restored


# --- 3. MAIN FUNCTION ---
@hydra.main(version_base=None, config_path="", config_name="config")
def main(cfg: DictConfig) -> None:
    print("[*] Loading configuration...")
    OmegaConf.set_struct(cfg, False)

    # Override with our specific training config
    cfg.model.d_model = 340
    cfg.model.n_layers = 4
    cfg.model.layer.N = 64
    cfg.dataset = "celeba"
    cfg.train.bsz = BATCH_SIZE

    # Get model class
    layer_cls = S4Layer
    cfg.model.layer.l_max = 3072  # H*W*C

    # Define the model class (must be in decode mode)
    model_cls = partial(
        BatchStackedModel,
        layer_cls=layer_cls,
        d_output=256,  # n_classes
        classification=False,
        decode=True,  # CRITICAL: Must be True for sampling
        **cfg.model,
    )
    
    print(f"[*] Defining model with d_model={cfg.model.d_model}, n_layers={cfg.model.n_layers}...")
    model = model_cls(training=False)

    # --- 4. LOAD THE CHECKPOINT ---
    try:
        state_dict = load_orbax_checkpoint(CHECKPOINT_PATH)
        
        # Orbax checkpoints from Flax save the entire TrainState
        # Extract params from the state
        if hasattr(state_dict, 'params'):
            params = state_dict.params
        elif 'params' in state_dict:
            params = state_dict['params']
        else:
            # The state_dict itself might be the params
            params = state_dict
            
        print("[*] Model parameters loaded successfully.")
        
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        print("\nTrying alternative loading method...")
        
        # Alternative: Use Flax's restore_checkpoint on the parent directory
        import os
        parent_dir = os.path.dirname(CHECKPOINT_PATH)
        step = int(os.path.basename(CHECKPOINT_PATH).split('_')[1].split('.')[0])
        
        try:
            state_dict = checkpoints.restore_checkpoint(
                parent_dir,
                target=None,
                step=step,
            )
            
            if hasattr(state_dict, 'params'):
                params = state_dict.params
            elif 'params' in state_dict:
                params = state_dict['params']
            else:
                params = state_dict
                
            print("[*] Model parameters loaded successfully using alternative method.")
        except Exception as e2:
            print(f"Alternative method also failed: {e2}")
            return

    # --- 5. GET DATA TO USE AS A PROMPT ---
    print(f"[*] Loading '{cfg.dataset}' dataset for prompts...")
    _, testloader, _, _, _ = Datasets[cfg.dataset](bsz=cfg.train.bsz)

    # --- 6. RUN SAMPLING WITH THE NEW PREFIX ---
    print(f"[*] Running sampling with new prefix: {NEW_PREFIX_SIZE}")
    rng = jax.random.PRNGKey(42)

    samples, examples = sample_image_prefix_celeba(
        params=params,
        model=model,
        rng=rng,
        dataloader=testloader,
        prefix=NEW_PREFIX_SIZE,
        n_batches=1,  # Only run on one batch
        save=True,
    )

    # --- 7. CALCULATE METRICS ---
    print(f"[*] Calculating metrics...")
    
    # Stack samples
    samples_array = np.stack(samples)
    examples_array = np.stack(examples)
    
    # Flatten images (B, H, W, C) -> (B, L*C)
    samples_flat = samples_array.reshape(samples_array.shape[0], -1)
    examples_flat = examples_array.reshape(examples_array.shape[0], -1)
    
    # Get suffix only (after the prefix)
    samples_suffix = samples_flat[:, NEW_PREFIX_SIZE:]
    examples_suffix = examples_flat[:, NEW_PREFIX_SIZE:]
    
    # Calculate MSE on suffix
    mse_suffix = np.mean(((samples_suffix - examples_suffix) / 255) ** 2)
    
    print(f"\n{'='*50}")
    print(f"[*] Results with Prefix={NEW_PREFIX_SIZE}:")
    print(f"    Suffix MSE: {mse_suffix:.6f}")
    print(f"{'='*50}\n")
    
    # --- 8. CALCULATE BPD ---
    print(f"[*] Calculating Suffix BPD...")
    
    # We need to manually define validate_suffix_bpd since we can't import from train.py
    # Copy the function here:
    from tqdm import tqdm
    
    def cross_entropy_loss(logits, label):
        """Calculate cross-entropy loss."""
        one_hot_label = jax.nn.one_hot(label, num_classes=logits.shape[-1])
        return -np.sum(one_hot_label * logits, axis=-1)
    
    def validate_suffix_bpd_local(params, model_cls, rng, testloader, prefix):
        """Calculate NLL and BPD on suffix only."""
        model = model_cls(decode=True, training=False)
        batch_losses = []
        
        # Initialize model to get cache structure
        init_batch = np.array(next(iter(testloader))[0].numpy())
        init_rng, _ = jax.random.split(rng, 2)
        variables = model.init(init_rng, init_batch)
        prime = variables["prime"]
        
        print("[*] Calculating Suffix BPD over test set...")
        for batch_idx, (inputs, labels) in enumerate(tqdm(testloader)):
            inputs = np.array(inputs.numpy())  # (B, L, 1)
            
            # Initialize fresh cache
            batch_cache = jax.tree.map(np.zeros_like, variables["cache"])
            
            # Prime with prefix
            _, vars = model.apply(
                {"params": params, "prime": prime, "cache": batch_cache},
                inputs[:, np.arange(0, prefix)],
                mutable=["cache"],
            )
            primed_cache = vars["cache"]
            
            # Get logits for suffix
            suffix_inputs = inputs[:, np.arange(prefix, inputs.shape[1])]
            suffix_logits, _ = model.apply(
                {"params": params, "prime": prime, "cache": primed_cache},
                suffix_inputs,
                mutable=["cache"],
            )
            
            # Get suffix labels
            suffix_labels = inputs[:, prefix:, 0]
            
            # Calculate loss
            # Vectorize cross_entropy_loss over batch and sequence
            loss_fn = jax.vmap(jax.vmap(cross_entropy_loss, in_axes=(0, 0)), in_axes=(0, 0))
            losses = loss_fn(suffix_logits, suffix_labels)
            loss = np.mean(losses)
            batch_losses.append(loss)
        
        avg_nll_loss = np.mean(np.array(batch_losses))
        avg_bpd = avg_nll_loss / np.log(2)
        return avg_nll_loss, avg_bpd
    
    suffix_nll, suffix_bpd = validate_suffix_bpd_local(
        params=params,
        model_cls=model_cls,
        rng=rng,
        testloader=testloader,
        prefix=NEW_PREFIX_SIZE,
    )
    
    print(f"    Suffix NLL (nats): {suffix_nll:.6f}")
    print(f"    Suffix BPD (bits): {suffix_bpd:.6f}")
    print(f"{'='*50}\n")

    print(f"[*] Done! Generated {len(samples)} images.")
    print(f"[*] Check your directory for 'im_celeba_0_...png' files.")


if __name__ == "__main__":
    main()
