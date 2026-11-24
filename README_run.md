# Assignment 2 — brain MRI classification and explanation

## Layout

```
ass2-indi/
  DataSet/                     original clone (evidence.png is the download transcript)
  xai-series/
    data/brain_mri/training/   copied from DataSet/.../Training (lower-case, 4 class folders)
    data/brain_mri/testing/    copied from DataSet/.../Testing
    src/model.py               model, datasets, preprocessing, seeds
    src/train_cnn.py           training and evaluation
    src/checks_dataset.py      inventory, readability and duplicate audit
    src/calibration_and_checks.py  ECE, reliability diagram, duplicate-aware accuracy
    src/xai_lrp_gradcam.py     LRP, Grad-CAM, case figures, deletion test
    src/make_loader_evidence.py    folder + loader transcript (Figure 2)
    05_lrp_starter.py          supplied starter script, kept unmodified for reference
  figures/                     every figure in the report
  results/                     metrics, predictions, checkpoint, raw heatmaps
  report/                      docx builder and section text
  dev/                         one-off diagnostic scripts used while developing
```

## Environment

Python 3.13 with PyTorch 2.10 (CPU), torchvision 0.25, NumPy, scikit-learn, Matplotlib,
Pillow. No GPU and no pretrained weights were used. The report builder needs
python-docx and runs under Python 3.12.

## Run order

```
cd xai-series
python src/checks_dataset.py               # inventory + duplicate audit
python src/train_cnn.py --epochs 16 --lr 0.002     # ~24 min on 8 CPU threads
python src/make_loader_evidence.py         # Figure 2
python src/xai_lrp_gradcam.py              # LRP, Grad-CAM, case figures, deletion test
python src/calibration_and_checks.py       # ECE, reliability diagram, clean-subset accuracy
cd ../report
python build_report.py                     # writes the docx
```

Set `XAI_FIGURES_ONLY=1` to rerun only the case figures (skips the 60-image deletion test).Training writes `../results/brain_mri_cnn.pt` together with the history, the test predictions and the confusion matrix. The XAI script reads that checkpoint, so it must run after training. Everything is seeded (42); the only known source of non-determinism is the order of floating point reductions under different thread counts.

## Notes on the starter script

`05_lrp_starter.py` is the file supplied with the xai-series repository, kept as downloaded. Differences between it and the implementation used here are listed in Table 4 of the report. The three that change results are the test-set root (the starter loads the training folder for both splits), the model, and the input
normalisation.
