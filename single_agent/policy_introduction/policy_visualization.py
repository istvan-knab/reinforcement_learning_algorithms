from matplotlib import pyplot as plt
import seaborn as sns
class PolicyVisualization:
    def __init__(self):
        print("Initializing Policy Visualization...")

    def plot_policy(self, probs_or_qvals, frame, action_meanings=None):
        if action_meanings is None:
            action_meanings = {0: 'U', 1: 'R', 2: 'D', 3: 'L'}
        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        max_prob_actions = probs_or_qvals.argmax(axis=-1)
        probs_copy = max_prob_actions.copy().astype(object)
        for key in action_meanings:
            probs_copy[probs_copy == key] = action_meanings[key]
        sns.heatmap(max_prob_actions, annot=probs_copy, fmt='', cbar=False, cmap='coolwarm',
                    annot_kws={'weight': 'bold', 'size': 12}, linewidths=2, ax=axes[0])
        axes[1].imshow(frame)
        axes[0].axis('off')
        axes[1].axis('off')
        plt.suptitle("Policy", size=18)
        plt.tight_layout()

    def plot_values(self, state_values, frame):
        f, axes = plt.subplots(1, 2, figsize=(10, 4))
        sns.heatmap(state_values, annot=True, fmt=".2f", cmap='coolwarm',
                    annot_kws={'weight': 'bold', 'size': 12}, linewidths=2, ax=axes[0])
        axes[1].imshow(frame)
        axes[0].axis('off')
        axes[1].axis('off')
        plt.tight_layout()
