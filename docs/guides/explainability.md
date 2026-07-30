# Grad-CAM & Explainability

## Grad-CAM

Visualise which image regions activate the model's prediction.

```python
from fujicv.eval.gradcam import GradCAM, GradCAMPlusPlus, overlay_heatmap
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt

# Load image
transform = T.Compose([T.Resize(224), T.CenterCrop(224), T.ToTensor()])
image = transform(Image.open("dog.jpg")).unsqueeze(0)   # (1, 3, 224, 224)

# Attach to the last conv block
cam = GradCAM(model, target_layer=model.backbone.layer4[-1])
heatmap = cam.generate(image, target_class=None)   # None = predicted class

# Overlay
result = overlay_heatmap(image.squeeze(), heatmap, alpha=0.5)
result.save("cam.png")
```

## Grad-CAM++

```python
cam_pp = GradCAMPlusPlus(model, target_layer=model.backbone.layer4[-1])
heatmap_pp = cam_pp.generate(image)
```

## Confusion matrix

```python
from fujicv.eval.confusion import plot_confusion_matrix, per_class_metrics

y_true = [0, 1, 2, 0, 1]
y_pred = [0, 2, 2, 0, 1]

fig = plot_confusion_matrix(y_true, y_pred, class_names=["cat", "dog", "bird"])
fig.savefig("confusion.png")

df = per_class_metrics(y_true, y_pred, class_names=["cat", "dog", "bird"])
print(df)
```

## Temperature Scaling (calibration)

```python
from fujicv.eval.calibration import TemperatureScaling, compute_ece

cal = TemperatureScaling()
cal.fit(model, val_loader)         # fits temperature T on validation set
print(f"ECE before: {compute_ece(logits, labels):.4f}")
calibrated_logits = cal.calibrate(logits)
print(f"ECE after:  {compute_ece(calibrated_logits, labels):.4f}")
```
