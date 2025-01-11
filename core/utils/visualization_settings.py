import matplotlib.pyplot as plt
import seaborn as sns

# Set global Seaborn theme
sns.set_theme(
    style="white",  # White background
    context="notebook",  # Default context; adjust font sizes for notebooks
    rc={
        "axes.spines.top": False,  # Remove top spine
        "axes.spines.right": False,  # Remove right spine
        "axes.grid": False,  # Disable gridlines
        "xtick.bottom": True,  # Enable bottom ticks
        "ytick.left": True,  # Enable left ticks
        "xtick.major.size": 6,  # Length of major x-axis ticks
        "ytick.major.size": 6,  # Length of major y-axis ticks
        "axes.titlesize": 13,  # Font size for titles
        "axes.labelsize": 11,  # Font size for axis labels
        "legend.fontsize": 11,  # Font size for legends
    },
)

# Set Matplotlib defaults
plt.rc("font", family="sans-serif")  # Use a sans-serif font
plt.rc("axes", titlesize=13, labelsize=11)  # Font sizes for axes and titles
plt.rc("xtick", labelsize=11)  # Font size for x-tick labels
plt.rc("ytick", labelsize=11)  # Font size for y-tick labels
plt.rc("figure", figsize=(6, 5))  # Default figure size
