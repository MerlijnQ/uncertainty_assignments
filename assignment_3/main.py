import os
import torch

from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from metrics import Metric
from bbb import BayesianCNN
from basicCNN import basicCNN
from train_test_models import TrainInference
from tempScaling import Calibrate

BATCH_SIZE = 64
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def get_data(val_ratio=0.1):
    transform = transforms.Compose([
        transforms.ToTensor(),  # Converts to tensor and normalize to [0, 1]
    ])

    # Load full training datasets
    mnist_full_train = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    mnist_train, mnist_val = train_test_split(mnist_full_train, test_size=val_ratio)

    # Load test datasets
    mnist_test = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    fmnist_test = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

    # Create data loaders
    mnist_train_loader = DataLoader(mnist_train, batch_size=BATCH_SIZE, shuffle=True)
    mnist_val_loader = DataLoader(mnist_val, batch_size=BATCH_SIZE, shuffle=False)
    mnist_test_loader = DataLoader(mnist_test, batch_size=BATCH_SIZE, shuffle=False)

    fmnist_test_loader = DataLoader(fmnist_test, batch_size=BATCH_SIZE, shuffle=False)

    return mnist_train_loader, mnist_val_loader, mnist_test_loader, fmnist_test_loader


def main():
    # Get data
    mnist_train_loader, mnist_val_loader, mnist_test_loader, fmnist_test_loader = get_data()
    assert len(fmnist_test_loader) == len(mnist_test_loader)
    
    # Train models (or load if alreaddy trained)
    inference_functions = TrainInference()
    os.makedirs("models", exist_ok=True)
    if os.path.exists(os.path.join("models", "BCNN.pth")):
        bayesian_cnn = BayesianCNN()
        model_state_dict = torch.load(os.path.join("models", "BCNN.pth"))
        bayesian_cnn.load_state_dict(model_state_dict)
    else:
        bayesian_cnn = inference_functions.train_bcnn(mnist_train_loader, mnist_val_loader)
        torch.save(bayesian_cnn.state_dict(), (os.path.join("models", "BCNN.pth")))
    
    if os.path.exists(os.path.join("models", "CNN_2.pth")):
        ensemble_cnn = []
        for i in range(3):
            model = basicCNN()
            model_state_dict = torch.load(os.path.join("models", f"CNN_{i}.pth"))
            model.load_state_dict(model_state_dict)
            ensemble_cnn.append(model)
    else:
        ensemble_cnn = inference_functions.train_ensemble(mnist_train_loader, mnist_val_loader)
        for i, model in enumerate(ensemble_cnn):
            torch.save(model.state_dict(), os.path.join("models", f"CNN_{i}.pth"))
    
    # Calibration plot (1)
    metric_maker = Metric()
    predictions, labels = inference_functions.bayesian_inference(mnist_val_loader, bayesian_cnn)
    metric_maker.calibration_plot(predictions, labels, "calibration_bcnn_1")
    predictions, labels = inference_functions.ensemble_inference(mnist_val_loader, ensemble_cnn)
    metric_maker.calibration_plot(predictions, labels, "calibration_ens_1")

    # Calibration    
    calibrator = Calibrate(device=DEVICE)
    calibrated_bayesian_cnn = calibrator.optimize(mnist_val_loader, bayesian_cnn)
    calibrated_ensemble_cnn = [calibrator.optimize(mnist_val_loader, ensemble_cnn[i]) for i in range(len(ensemble_cnn))]
    
    # Calibration plot (2)
    predictions, labels = inference_functions.bayesian_inference(mnist_val_loader, calibrated_bayesian_cnn)
    metric_maker.calibration_plot(predictions, labels, "calibration_bcnn_2")
    predictions, labels = inference_functions.ensemble_inference(mnist_val_loader, calibrated_ensemble_cnn)
    metric_maker.calibration_plot(predictions, labels, "calibration_ens_2")

    # Test models
    mnist_results_bayesian = inference_functions.bayesian_inference(mnist_test_loader, calibrated_bayesian_cnn)
    mnist_results_ensemble = inference_functions.ensemble_inference(mnist_test_loader, calibrated_ensemble_cnn)
    fmnist_results_bayesian = inference_functions.bayesian_inference(fmnist_test_loader, calibrated_bayesian_cnn)
    fmnist_results_ensemble = inference_functions.ensemble_inference(fmnist_test_loader, calibrated_ensemble_cnn)

    # Create final plots
    metric_maker.auroc(mnist_results_bayesian, fmnist_results_bayesian, "AUROC_bayesian")
    metric_maker.auroc(mnist_results_ensemble, fmnist_results_ensemble, "AUROC_ensemble")

    metric_maker.confidence_OOD_ID(mnist_results_bayesian, fmnist_results_bayesian, "confidence_bayesian")
    metric_maker.confidence_OOD_ID(mnist_results_ensemble, fmnist_results_ensemble, "confidence_ensemble")
    
    metric_maker.entropy_plot(mnist_results_bayesian, fmnist_results_bayesian, "entropy_bayesian")
    metric_maker.entropy_plot(mnist_results_ensemble, fmnist_results_ensemble, "entropy_ensemble")
    
    metric_maker.qualitative(mnist_results_bayesian, mnist_test_loader, fmnist_results_bayesian, fmnist_test_loader, "qualitative_bayesian")
    metric_maker.qualitative(mnist_results_ensemble, mnist_test_loader, fmnist_results_ensemble, fmnist_test_loader, "qualitative_ensemble")


if __name__ == "__main__":
    main()