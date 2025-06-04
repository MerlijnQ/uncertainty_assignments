
#Implement metrics
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
from scipy.stats import entropy
import seaborn as sns


class metric():
    def __init__(self):
        pass

    def auroc(self, ID_score, OOD_score):
        y_true = np.concatenate(np.zeros_like(len(ID_score)), np.ones_like(len(OOD_score)))
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
        plt.savefig("roc_curve_uncertainty.png")
        plt.close()

    def calibration_plot(self, predictions, labels, bins=10): 
        #We need to plot confidence against accuracy
        y_pred = np.argmax(predictions, axis=1)
        y_conf = np.max(predictions, axis=1)
        correct = (y_pred == labels).astype(int)

        bin_points = np.linspace(0, 1 , bins+1)
        assigned_bins = np.digitize(y_conf, bin_points, right=True) #Assign values to a bin

        bin_accuracy = []
        bin_confidence = {}

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
        plt.savefig("reliability_plot.png")
        plt.close()

    def confidence_OOD_ID(self, OOD_pred, ID_pred, bins=10):
        id_conf = np.max(ID_pred, axis=1)
        ood_conf = np.max(OOD_pred, axis=1)

        plt.hist(id_conf, bins=bins, alpha=0.6, label='In-Distribution', color='blue')
        plt.hist(ood_conf, bins=bins, alpha=0.6, label='Out-of-Distribution', color='red')
        plt.xlabel('Max Softmax Confidence')
        plt.ylabel('Count')
        plt.legend()
        plt.grid(True)
        plt.savefig("confidence_histogram.png")
        plt.close()


    def entropy_plot(self, OOD_pred, ID_pred):

        OOD_entropy = entropy(OOD_pred.T)
        ID_entropy = entropy(ID_pred.T)

        sns.kdeplot(ID_entropy, label='In-Distribution', fill=True, color='blue')
        sns.kdeplot(OOD_entropy, label='Out-of-Distribution', fill=True, color='red')
        plt.xlabel("Entropy score")
        plt.ylabel("Density")
        plt.legend()
        plt.tight_layout()        
        plt.savefig("Entropy_density.png")
        plt.close()
        

    def qualitative():
        pass