import os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm

# Path to dataset
dataset_path = "raw_data"

# Count images per folder (country)
country_counts = {country: len(os.listdir(os.path.join(dataset_path, country)))
                  for country in os.listdir(dataset_path)
                  if os.path.isdir(os.path.join(dataset_path, country))}

# Sort countries by image count (descending)
sorted_countries = sorted(country_counts.items(), key=lambda x: x[1], reverse=True)

# Group countries below a threshold into "Others"
threshold = 2.0 
total_images = sum(count for _, count in sorted_countries)

# Separate top countries and group others
top_countries = [(country, count) for country, count in sorted_countries if (count / total_images) * 100 > threshold]
others_total = sum(count for _, count in sorted_countries if (count / total_images) * 100 <= threshold)

# Combine data
labels = [country for country, _ in top_countries] + ["Others"]
sizes = [count for _, count in top_countries] + [others_total]

# Create spacing between slices (explode "Others" slice)
explode = [0.05 if label != "Others" else 0.1 for label in labels]

# Create color palette
colors = cm.viridis_r(np.linspace(0, 1, len(sizes)))

# Plot the pie chart
plt.figure(figsize=(10, 8))
wedges, texts, autotexts = plt.pie(
    sizes, labels=labels, autopct='%1.1f%%', startangle=140, explode=explode,
    colors=colors, textprops={'fontsize': 15, 'weight': 'bold'}
)

# Update the percentage text appearance
for autotext in autotexts:
    autotext.set_color('black') 
    autotext.set_fontsize(16)    


# Add a legend outside the chart
plt.legend(labels, loc="upper left", bbox_to_anchor=(1, 1), fontsize=10)

# Add a title
plt.title("Image Distribution Across Countries (Top Contributors and 'Others')", fontsize=14)

# Adjust layout
plt.tight_layout()
plt.show()
