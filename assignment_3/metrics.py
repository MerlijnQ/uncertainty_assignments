
#Implement metrics
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
from scipy.stats import entropy
import seaborn as sns
import torch


class Metric():
    def __init__(self):
        print("Metrics initialized. Note that each metric expetcs softmax predictions. We assume all predictions are .cpu().numpy()")

    def auroc(self, ID_score, OOD_score, file_name, entropy_pred=True):
        y_true = np.concatenate(np.zeros_like(len(ID_score)), np.ones_like(len(OOD_score)))
        if entropy_pred:
            ID_score, OOD_score = entropy(ID_score.T), entropy(OOD_score.T)
        scores = np.concatenate(ID_score, OOD_score, axis = 0)

        false_possitive_rate, true_positive_rate, _ = roc_curve(y_true, scores)
        auroc = roc_auc_score(y_true, scores)

        plt.figure(figsize=(6, 5))
        plt.plot(false_possitive_rate, true_positive_rate, label=f"AUROC = {auroc:.2f}")
        plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
        plt.xlabel("False positive rate")
        plt.ylabel("True positive rate")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{file_name}.pdf")
        plt.close()

    def calibration_plot(self, predictions, labels, file_name, bins=10): 
        #We need to plot confidence against accuracy
        y_pred = np.argmax(predictions, axis=1)
        y_conf = np.max(predictions, axis=1)
        correct = (y_pred == labels).astype(int)

        bin_points = np.linspace(0, 1 , bins+1)
        assigned_bins = np.digitize(y_conf, bin_points, right=True) #Assign values to a bin

        bin_accuracy = []
        bin_confidence = []

        for i in range(0, bins+1):
            mask = assigned_bins == i
            bin_accuracy.append(np.mean(correct[mask]))
            bin_confidence.append(np.mean(y_conf[mask]))

        plt.figure()
        plt.plot(bin_confidence, bin_accuracy, marker='o')
        plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
        plt.xlabel("Confidence")
        plt.ylabel("Accuracy")
        plt.grid(False)
        plt.tight_layout()
        plt.savefig(f"{file_name}.pdf")
        plt.close()

    def confidence_OOD_ID(self, ID_pred, OOD_pred, file_name, bins=10):
        id_conf = np.max(ID_pred, axis=1)
        ood_conf = np.max(OOD_pred, axis=1)

        plt.hist(id_conf, bins=bins, alpha=0.6, label='In-Distribution', color='blue')
        plt.hist(ood_conf, bins=bins, alpha=0.6, label='Out-of-Distribution', color='red')
        plt.xlabel('Max Softmax Confidence')
        plt.ylabel('Count')
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{file_name}.pdf")
        plt.close()


    def entropy_plot(self, ID_pred, OOD_pred, file_name):

        OOD_entropy = entropy(OOD_pred.T)
        ID_entropy = entropy(ID_pred.T)

        sns.kdeplot(ID_entropy, label='In-Distribution', fill=True, color='blue')
        sns.kdeplot(OOD_entropy, label='Out-of-Distribution', fill=True, color='red')
        plt.xlabel("Entropy score")
        plt.ylabel("Density")
        plt.legend()
        plt.tight_layout()        
        plt.savefig(f"{file_name}.pdf")
        plt.close()

    def plot_images(self, ID_img, ID_entropy, OOD_img, OOD_entropy, file_name):
        n = len(ID_img)
        _, axes = plt.subplots(2, n, figsize=(n * 1.5, 3))
        
        for i in range(n):
            # Plot ID
            axes[0, i].imshow(ID_img[i].squeeze(), cmap='gray')
            axes[0, i].set_title(f"{ID_entropy[i]:.4f}", fontsize=8)
            axes[0, i].axis('off')
            
            # Plot OOD
            axes[1, i].imshow(OOD_img[i].squeeze(), cmap='gray')
            axes[1, i].set_title(f"{OOD_entropy[i]:.4f}", fontsize=8)
            axes[1, i].axis('off')
        
        axes[0, 0].set_ylabel("ID", fontsize=10)
        axes[1, 0].set_ylabel("OOD", fontsize=10)
        plt.savefig(f"{file_name}.pdf")
        plt.close()
            

    def qualitative(self, ID_pred, ID_loader, OOD_pred, OOD_loader, file_name, k=10):

        ID_pred = entropy(ID_pred.T)
        OOD_pred = entropy(OOD_pred.T)


        ID_images = []
        OOD_images = []

        for ID_image_batch, OOD_image_batch in zip(ID_loader, OOD_loader):
            ID_image, _ = ID_image_batch
            OOD_image, _ = OOD_image_batch
            ID_images.append(ID_image)
            OOD_images.append(OOD_image)
        ID_images = torch.cat(ID_images)
        OOD_images = torch.cat(OOD_images)

        idx = np.random.choice(len(ID_images), size=k, replace=False)
        self.plot_images(ID_images[idx], ID_pred[idx], OOD_images[idx], ID_pred[idx], file_name)