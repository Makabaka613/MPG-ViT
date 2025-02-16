# Multimodal Prompt-Guided Vision Transformer For Precise Image Manipulation Localization
This study proposes a new image tampering localization method, which combines multimodal cue guidance with Vision Transformer (ViT) to improve the accuracy and generalization ability of tampering area localization.
## Requirements
To run this project, you'll need to have the following libraries installed:
- Python 3.8
- PyTorch
- albumentations
- matplotlib
- numpy
- scikit_learn
- timm
- fvcore
- tensorboard

You can install the required libraries using pip:
```bash
pip install torch, albumentations, matplotlib, numpy, scikit-learn, timm, fvcore, tensorboard
```
## Code

This project uses a pre-trained large-scale model to convert images into semantic text and locate image tampering. The model we use is the **LLaMA 3.1 8B Vision model**, which is available on Hugging Face and can effectively convert image content into meaningful text descriptions.

You can access and use the model at this link: [LLaMA 3.1 8B Vision](https://huggingface.co/qresearch/llama-3.1-8B-vision-378).

## Datasets

Here are several datasets used in this project, which you can access and download via the links provided:

- **NIST16**: [NIST16 Dataset](https://mig.nist.gov/MFC/PubData/Resources.html)
- **Defacto**: [Defacto Dataset](https://defactodataset.github.io/)
- **CASIAv1**: [CASIAv1 Dataset](https://www.kaggle.com/datasets/sophatvathana/casia-dataset?select=CASIA1)
- **CASIAv2**: [CASIAv2 Dataset](https://www.kaggle.com/datasets/sophatvathana/casia-dataset?select=CASIA2)
- **Columbia**: [Columbia Dataset](https://www.ee.columbia.edu/ln/dvmm/downloads/authsplcuncmp/)
- **Coverage**: [Coverage Dataset](https://stefan.winklerbros.net/Publications/icip2016b.pdf)

## Citation

If you use this code or data in your research, please cite our paper:

```bibtex
@article{Xiao2025,
  title={Multimodal Prompt-Guided Vision Transformer For Precise Image Manipulation Localization},
  journal={The Visual Computer},
}

