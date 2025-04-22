import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# List of image file paths
image_paths = [
    "/Users/mehjabeen/Desktop/LSTM figs/arima.png", "/Users/mehjabeen/Desktop/LSTM figs/translstm.png", "/Users/mehjabeen/Desktop/LSTM figs/transformer.png", 
    "/Users/mehjabeen/Desktop/LSTM figs/randomforest.png", "/Users/mehjabeen/Desktop/LSTM figs/finalseed.png", "/Users/mehjabeen/Desktop/LSTM figs/clear.png"
]

# List of custom titles for each image
image_titles = [
    "ARIMA Model", "Transformer-LSTM", "Transformer", 
    "Random Forest", "Final Seed", "LSTM"
]

# Create a figure with 2 rows and 3 columns
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Loop through each image and display it
for i, ax in enumerate(axes.flatten()):
    img = mpimg.imread(image_paths[i])  # Read the image
    ax.imshow(img)  # Display the image
    ax.axis('off')  # Hide axes for better visualization
    ax.set_title(image_titles[i])  # Set custom title for each image

# Adjust layout
plt.tight_layout()
plt.show()
