import io
import os
from collections import Counter

import cairosvg  # Install with: pip install cairosvg
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from PIL import Image


def count_languages():
    with open("ALGORITHMS.md", encoding="utf-8") as file:
        content = file.read()

    # Skip the header row and extract table rows
    rows = content.split("\n")[2:]  # Skip header and separator rows

    languages = []
    for row in rows:
        # Check if this is a table row (starts with |)
        if row.strip().startswith("|"):
            columns = row.split("|")
            if len(columns) > 1:
                # Get the language name from the first column and strip whitespace
                language = columns[1].strip()
                if language:  # Skip empty lines
                    languages.append(language)

    # Count occurrences using Counter
    counts = Counter(languages)

    # Print the results in descending order of frequency
    for language, count in counts.most_common():
        print(f"{language}: {count}")

    # Create a bar chart with logos
    create_bar_chart_with_logos(counts)


def find_logo(language):
    """Find a logo file for the given language in public/media/logos directory"""
    # Define special case mappings for language names with special characters
    special_cases = {
        "C++": [
            "c++",
            "cplusplus",
            "cplus",
            "cpp",
            "c-plus-plus",
        ],  # Added more variations
        "C#": ["c#", "csharp"],
        "F#": ["f#", "fsharp"],
        "RAPIDS (Python)": ["rapids", "rapids_python"],
        "RAPIDS (C++)": ["rapids", "rapids_cpp", "rapids_cplusplus"],
        "JavaScript": ["javascript", "js"],
        "TypeScript": ["typescript", "ts"],
        "pandas": ["pandas", "python_pandas"],
        "q": ["kx-logo"],
    }

    # Special debug for C++
    is_cpp = language == "C++"
    if is_cpp:
        print("Searching for C++ logo specifically...")

    # Get normalized variations of the language name
    name_variations = []
    base_name = language.lower()

    # Add special case variations if applicable
    if language in special_cases:
        name_variations.extend(special_cases[language])
    else:
        # Remove any parentheses and content within
        cleaned_name = base_name.split("(")[0].strip()
        name_variations.append(cleaned_name)

    # Add standard variations
    name_variations.extend(
        [
            base_name,
            base_name.replace(" ", ""),
            base_name.replace(" ", "_"),
            base_name.replace(" ", "-"),
            base_name.replace("+", "plus"),
            base_name.replace("#", "sharp"),
        ]
    )

    # Remove duplicates
    name_variations = list(set(name_variations))

    if is_cpp:
        print(f"C++ variations to try: {name_variations}")

    # List all files in the logos directory
    logo_dir = "public/media/logos/"
    try:
        all_logo_files = os.listdir(logo_dir)
        if is_cpp:
            print(f"Available logo files: {all_logo_files}")
    except Exception as e:
        print(f"Error listing logo directory: {e}")
        all_logo_files = []

    # Search for exact matches first
    for variation in name_variations:
        for file in all_logo_files:
            file_lower = file.lower()
            # Try exact match or with _logo suffix
            if (
                file_lower == f"{variation}.png"
                or file_lower == f"{variation}.svg"
                or file_lower == f"{variation}_logo.png"
                or file_lower == f"{variation}_logo.svg"
            ):
                if is_cpp:
                    print(f"Found C++ logo match: {file}")
                return os.path.join(logo_dir, file)

    # If no exact match, try substring matches
    for variation in name_variations:
        for file in all_logo_files:
            file_lower = file.lower()
            if variation in file_lower:
                if is_cpp:
                    print(f"Found C++ logo via substring: {file}")
                return os.path.join(logo_dir, file)

    # Last resort for C++: try with any file that has cpp, c++, or cplusplus anywhere in the name
    if is_cpp:
        for file in all_logo_files:
            file_lower = file.lower()
            if "cpp" in file_lower or "c++" in file_lower or "cplusplus" in file_lower:
                print(f"Found C++ logo via last resort: {file}")
                return os.path.join(logo_dir, file)

    print(
        f"No logo found for {language} (tried variations: {', '.join(name_variations)})"
    )
    return None


def resize_image_to_20x20(img_path):
    """Resize an image to exactly 20x20 pixels, handling both PNG and SVG formats"""
    try:
        if img_path.lower().endswith(".svg"):
            # For SVG files, convert to PNG first using cairosvg
            png_data = cairosvg.svg2png(url=img_path, output_width=20, output_height=20)
            img = Image.open(io.BytesIO(png_data))
            return np.array(img)

        # For other image formats (PNG, JPG, etc.)
        with Image.open(img_path) as img:
            # Convert to RGBA to handle transparency properly
            if img.mode != "RGBA":
                img = img.convert("RGBA")

            # Create a white background
            background = Image.new("RGBA", img.size, (255, 255, 255, 255))

            # Paste the image on the background using alpha composite
            background.paste(img, (0, 0), img)

            # Convert back to RGB (remove alpha channel)
            img = background.convert("RGB")

            # Resize to 20x20 with antialiasing
            img_resized = img.resize((20, 20), Image.LANCZOS)

            # Convert PIL image to numpy array for matplotlib
            return np.array(img_resized)
    except Exception as e:
        print(f"Error processing image {img_path}: {e}")
        return None


def create_bar_chart_with_logos(counts):
    # Get data sorted by count (descending)
    languages, counts_values = zip(*counts.most_common())

    # Create the bar chart with a clean, borderless style
    _, ax = plt.subplots(figsize=(16, 10))

    # Remove the plot frame/border
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)

    # Create the bars
    bars = ax.bar(languages, counts_values, color="skyblue")

    # Count found logos
    found_logos = 0

    # Add logos to the top of each bar
    for i, (language, bar) in enumerate(zip(languages, bars)):
        logo_path = find_logo(language)
        if logo_path:
            try:
                # Process the image based on its type (SVG or other)
                img_resized = resize_image_to_20x20(logo_path)

                if img_resized is not None:
                    # Create an OffsetImage with the resized logo
                    imagebox = OffsetImage(img_resized, zoom=1.0)

                    # Position the logo above the bar
                    xy = (
                        i,
                        bar.get_height() + max(counts_values) * 0.03,
                    )

                    # Add the logo to the plot
                    ab = AnnotationBbox(imagebox, xy, frameon=False)
                    ax.add_artist(ab)

                    found_logos += 1
                    print(f"Added logo for {language} from {logo_path}")
            except Exception as e:
                print(f"Error adding logo for {language}: {e}")

    print(f"Successfully added {found_logos} logos out of {len(languages)} languages")

    # Set chart labels and properties
    plt.xlabel("Programming Languages")
    plt.ylabel("Frequency")
    plt.title("Programming Language Frequency in ALGORITHMS.md")
    plt.xticks(rotation=90)  # Rotate language names for readability

    # Remove top and right ticks, keep only bottom and left
    ax.tick_params(top=False, right=False)

    # Tight layout to adjust spacing
    plt.tight_layout()

    # Reduce padding above title (smaller top margin)
    plt.subplots_adjust(top=0.95)  # Increased from 0.85 to 0.95 to reduce space

    # Save the chart
    plt.savefig("language_counts_with_logos.png", dpi=300)
    print("Bar chart saved as 'language_counts_with_logos.png'")

    # Show the chart
    plt.show()


if __name__ == "__main__":
    count_languages()
