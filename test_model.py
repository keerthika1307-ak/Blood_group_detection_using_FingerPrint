import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Load the model
model = tf.keras.models.load_model("model/bloodgroup_cnn_model.keras")

# Prepare test data
test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    'dataset',
    target_size=(64, 64),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

print("Evaluating model on full dataset...")
print(f"Total samples: {test_generator.samples}")
print(f"Class indices: {test_generator.class_indices}")

# Get predictions
predictions = model.predict(test_generator, verbose=1)
predicted_classes = np.argmax(predictions, axis=1)

# Get true labels
true_classes = test_generator.classes

# Get class names
class_names = list(test_generator.class_indices.keys())

# Print classification report
print("\nClassification Report:")
print("="*60)
print(classification_report(true_classes, predicted_classes, target_names=class_names))

# Create confusion matrix
cm = confusion_matrix(true_classes, predicted_classes)

# Plot confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
print("\n✅ Confusion matrix saved as 'confusion_matrix.png'")

# Calculate per-class accuracy
print("\nPer-Class Accuracy:")
print("="*60)
for i, class_name in enumerate(class_names):
    class_correct = cm[i, i]
    class_total = cm[i, :].sum()
    accuracy = (class_correct / class_total) * 100 if class_total > 0 else 0
    print(f"{class_name}: {accuracy:.2f}% ({class_correct}/{class_total})")

# Overall accuracy
overall_accuracy = (predicted_classes == true_classes).mean() * 100
print(f"\nOverall Accuracy: {overall_accuracy:.2f}%")
