# Pokemon Card Recognition System

A system for recognizing Pokemon cards using deep learning and image matching techniques.

## Features

- Card detection and cropping
- Deep learning feature extraction with GPU acceleration
- Fast nearest neighbor search with HNSW
- ORB feature matching verification
- Handles size difference between database and query images

## Installation

### Prerequisites

- Python 3.8+ 
- CUDA and cuDNN (for GPU acceleration)

### Setup

1. Clone the repository:
\`\`\`bash
git clone https://github.com/RichardFelix999/pokemon_card_embedding/tree/intergrate_v0.git
cd pokemon-card-recognition
\`\`\`

2. Install dependencies:

For Linux/Mac:
\`\`\`bash
pip install -r requirements.txt
\`\`\`

For Windows:
\`\`\`bash
pip install -r requirements-windows.txt
\`\`\`

3. Create required directories:
\`\`\`bash
mkdir -p card_images input
\`\`\`

4. Prepare your data:
   - Place your Pokemon card images in the `card_images` directory
   - Name them according to their product IDs (e.g., `sm10-60.jpg`)
   - Place your card data JSON file as `test_data.json` in the root directory

## Usage

### Building the Database

Build the feature database from your card images:

\`\`\`bash
python main.py build
\`\`\`

This will:
- Process all card images in the `card_images` directory
- Extract features using the EfficientNetV2S model
- Build a search index for fast retrieval
- Save the database artifacts to disk

### Searching for a Single Card

Search for a specific card:

\`\`\`bash
python main.py search --image input/your_card.jpg
\`\`\`

This will:
- Detect and crop the card from the input image
- Extract features from the card
- Find the most similar cards in the database
- Display the results

### Batch Searching

Process multiple query images at once:

\`\`\`bash
python main.py batch --dir input
\`\`\`

This will:
- Process all images in the specified directory
- Find matches for each image
- Display a summary of the results

## GPU Acceleration

The system automatically uses GPU acceleration if available:

- GPU memory growth is enabled to prevent out-of-memory errors
- Mixed precision (FP16) is used for faster computation
- Batch processing is used for better GPU utilization

To check if GPU is being used:
\`\`\`bash
python -c "import tensorflow as tf; print('GPU available:', tf.config.list_physical_devices('GPU'))"
\`\`\`

## Tips for Best Results

- Ensure good lighting when taking photos of cards
- Try to capture the card straight-on with minimal glare
- Include the entire card in the frame
- For best results, use a solid background

## Troubleshooting

### GPU Issues

If you encounter GPU-related errors:

1. Make sure you have the correct CUDA and cuDNN versions installed for your TensorFlow version
2. Try running with CPU only by setting the environment variable:
   \`\`\`bash
   export CUDA_VISIBLE_DEVICES=-1
   \`\`\`
   or on Windows:
   \`\`\`
   set CUDA_VISIBLE_DEVICES=-1
   \`\`\`

3. If you get out-of-memory errors, reduce the batch size in `main.py`

### Card Detection Issues

If cards are not being detected properly:

1. Check the lighting and background of your photos
2. Adjust the aspect ratio tolerance in `detect_and_crop_card()` function
3. Try preprocessing your images to improve contrast

## Project Structure

- `main.py`: Main script for database building and searching
- `utils.py`: Utility functions for image processing and feature extraction
- `requirements.txt`: Dependencies for Linux/Mac
- `requirements-windows.txt`: Dependencies for Windows
