# Object Detection with YOLO-ResNet

This project implements an object detection system using a hybrid YOLO-ResNet model. It supports training, evaluation, and inference on custom datasets in YOLO format.

---

## 🚀 Features

- Custom object detection using YOLO format datasets
- Hybrid backbone: YOLO + ResNet
- Training, evaluation, and inference scripts
- Easily adaptable to new datasets
- Pre-trained model checkpoint included

---

## 🧠 Model Architecture

The architecture combines the speed of YOLO with the feature extraction strength of ResNet, delivering efficient and accurate object detection results.

---

## 📁 Project Structure

```
.
├── train.py                    # Model training script
├── evaluate.py                # Model evaluation script
├── demo.py                    # Inference/demo script
├── yolo_resnet.pth            # Pre-trained model weights
├── requirements.txt           # Python dependencies
├── Experience Report.txt      # Report detailing implementation
├── datasets/
│   └── test.v4i.yolov5pytorch/
│       ├── data.yaml          # Dataset configuration
│       ├── test/
│           ├── images/        # Test images
│           └── labels/        # Corresponding YOLO labels
```

---

## ⚙️ Installation

1. Clone the repository or unzip the package.

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. (Optional) Setup a virtual environment for isolation.

---

## 🏋️ Training

Ensure your dataset is in YOLO format. Update `data.yaml` with correct paths.

Run training with:
```bash
python train.py
```

---

## 📈 Evaluation

Evaluate your model's performance on the test set:
```bash
python evaluate.py
```

---

## 🎯 Inference / Demo

To test the trained model on new images:
```bash
python demo.py
```

Ensure the model checkpoint (`yolo_resnet.pth`) is in the project root.

---

## 📊 Dataset Format

The dataset follows the YOLOv5 standard format with:
- `images/`: Contains input `.jpg` files
- `labels/`: Contains label files with YOLO annotations

`data.yaml` defines:
```yaml
train: path/to/train/images
val: path/to/val/images
nc: <number_of_classes>
names: ['class1', 'class2', ...]
```

---

## 📝 Experience Report

A detailed explanation of the model design, training strategy, and results is included in the `Experience Report.txt`.

---

## 📌 License

This project is for academic and research use only. For commercial use, contact the author.

---

## 🙌 Acknowledgments

- YOLOv5 by Ultralytics
- ResNet by Microsoft Research