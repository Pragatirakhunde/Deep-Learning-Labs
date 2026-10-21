# Assignment 9 - ViT vs CNN on a Small Dataset

Uses scikit-learn's built-in Digits dataset (1,797 small 8×8 grayscale digit images). The dataset is included with scikit-learn, so no large CIFAR-10 download is needed. Images are converted to RGB and resized for ImageNet-pretrained models.

## VS Code setup (Windows PowerShell)
```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python main.py
```
The first run downloads pretrained ViT-B/16 and ResNet18 weights; these model weights are separate from the dataset and can be large. Keep internet access enabled for this first run. If your internet/data allowance is limited, run only the CNN with `python main.py --models cnn`; a pretrained ViT requires its weights to be downloaded unless already cached.

Default: 800 train samples, 300 test samples, one epoch, batch size 8. For a lighter run: `python main.py --models cnn --train-samples 200 --test-samples 100 --batch-size 4`. To compare both with the same settings: `python main.py --train-samples 400 --test-samples 150 --batch-size 4`.

Outputs: `outputs/comparison_results.csv`, `outputs/training_history.csv`, `outputs/comparison_plot.png`.

Note: this is a small demonstration dataset, not a large benchmark. The pretrained backbones are frozen and only the new 10-class heads are trained. 
