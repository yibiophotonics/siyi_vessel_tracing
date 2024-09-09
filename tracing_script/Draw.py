import matplotlib.pyplot as plt
import numpy as np

# Create a fixed-size figure
fig, ax = plt.subplots(figsize=(6, 6))  # Adjust the size as needed

# Generate a random image
image = np.random.rand(10, 10)

# Display the image
ax.imshow(image, cmap='gray', extent=[0, 10, 0, 10])

# Draw an arrow that goes out of the image
ax.arrow(-2, -2, 4, 4, width=0.1, color='red')

# Set aspect ratio to be equal
ax.set_aspect('equal')

# Adjust subplot boundaries to make room for the arrow
plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)

# Show the plot
plt.show()
