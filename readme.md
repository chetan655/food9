# Food Classification with VGG16 and Custom CNN

This project is a deep learning pipeline for classifying images of 9 different food categories using PyTorch. It demonstrates both transfer learning with VGG16 and training a custom CNN from scratch.

---

## Features

- **Data Preparation:**

  - Unzips and splits the dataset into train/test folders.
  - Applies data augmentation and normalization.

- **Models:**

  - Custom CNN (`Food9`)
  - Transfer learning with VGG16 and ResNet18

- **Training:**

  - Training and evaluation loops with accuracy and loss tracking
  - Checkpoint saving and loading

- **Inference:**
  - Test the trained model on real images

---

## Folder Structure

```
foodcnn1/
├── code.ipynb
├── data/
│   ├── fruits9.zip
│   ├── split/
│   │   ├── train/
│   │   └── test/
├── checkpoint/
│   └── model_0.pth
└── ...
```

---

## Usage

1. **Prepare Data:**

   - Place your `fruits9.zip` dataset in the `data/` folder.

2. **Run the Notebook:**

   - Open `code.ipynb` in Jupyter or VS Code.
   - Run all cells to:
     - Extract and split data
     - Define transforms and dataloaders
     - Train the model (VGG16 or custom CNN)
     - Save checkpoints

3. **Test on Real Images:**
   - Place your test image in the project folder.
   - Update the `img_path` variable in the inference cell.
   - Run the inference cell to see the predicted class.

---

## Customization

- To use your own dataset, update the data paths and number of classes.
- You can switch between VGG16, ResNet18, or the custom `Food9` model by changing the model and optimizer in the training cell.

---

## Example Inference

```python
from PIL import Image

img_path = "path/to/your/image.jpg"
img = Image.open(img_path).convert('RGB')
img_tensor = test_transform(img).unsqueeze(0).to(device)

vgg16.eval()
with torch.no_grad():
    output = vgg16(img_tensor)
    pred_class = torch.argmax(output, dim=1).item()

class_names = train_data.classes
print("Predicted class:", class_names[pred_class])
```

---

## License

This project is for educational purposes.
