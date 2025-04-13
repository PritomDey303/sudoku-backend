# 🧠 Sudoku OCR Solver API

A FastAPI-powered web service for recognizing Sudoku puzzles from images using a Convolutional Neural Network (CNN). The service detects the Sudoku grid, removes lines, splits the board into 81 cells, and uses a trained CNN model to predict digits.

---

## 🔍 Features

- 🧩 **Sudoku grid detection** via image preprocessing and contour analysis
- 🧼 **Grid line removal** for clean cell segmentation
- 🔢 **Digit recognition** using a pre-trained CNN model for accurate Sudoku solving
- 📦 **API endpoint** to send an image and receive a digit grid in JSON format


---

## 🚀 How It Works

1. **Image Preprocessing**: The input image is preprocessed to enhance the grid detection using OpenCV.
2. **Grid Detection & Line Removal**: The Sudoku grid is detected and grid lines are removed for better cell segmentation.
3. **Cell Segmentation**: The board is split into 81 smaller cells (9x9 grid).
4. **CNN-based Digit Prediction**: Each cell is passed through a Convolutional Neural Network (CNN) trained to predict digits (0-9).
5. **Response**: The predicted digits are returned in JSON format, representing the Sudoku board.

---

## 🛠️ Tech Stack

- **Backend**: FastAPI
- **Image Processing**: OpenCV
- **Digit Recognition**: Keras-based CNN model
- **Model**: Pre-trained MNIST-based CNN for digit classification
- **Deployment**: Render (cloud hosting)

