import torch
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend for headless rendering

import matplotlib.pyplot as plt

import os

def plot_cls_heatmap(student_logits, teacher_logits, save_path, step, name="rpn"):
    """
    Visualize teacher vs. student classification outputs as heatmaps.
    Assumes logits are of shape [B, N, 1]
    """
    os.makedirs(save_path, exist_ok=True)
    with torch.no_grad():
        student_probs = torch.sigmoid(student_logits.squeeze(-1).detach().cpu())
        teacher_probs = torch.sigmoid(teacher_logits.squeeze(-1).detach().cpu())

        for b in range(student_probs.shape[0]):
            fig, axs = plt.subplots(1, 2, figsize=(10, 4))
            axs[0].imshow(student_probs[b].view(64, -1), cmap='hot', aspect='auto')
            axs[0].set_title('Student')

            axs[1].imshow(teacher_probs[b].view(64, -1), cmap='hot', aspect='auto')
            axs[1].set_title('Teacher')

            plt.suptitle(f"{name.upper()} Heatmap - Step {step} - Batch {b}")
            plt.tight_layout()
            plt.savefig(os.path.join(save_path, f'{name}_heatmap_step{step}_b{b}.png'))
            plt.close()


def plot_regression_delta(student_reg, teacher_reg, save_path, step, title="rpn"):
    """
    Visualize histogram of mean absolute regression delta.
    """
    os.makedirs(save_path, exist_ok=True)

    delta = (student_reg - teacher_reg).abs().mean(dim=-1).detach().cpu()

    # 🔧 Flatten to 1D to avoid matplotlib "multiple datasets" issue
    delta = delta.view(-1)

    plt.figure(figsize=(6, 4))
    plt.hist(delta.numpy(), bins=40, color='skyblue')
    plt.title(f"{title.upper()} Regression Δ @ Step {step}")
    plt.xlabel("Mean Abs Difference")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f'{title}_reg_delta_step{step}.png'))
    plt.close()

