# 🧠 Sudoku OCR Solver API

A FastAPI-powered web service for recognizing Sudoku puzzles from images using OpenCV and Tesseract OCR. The service detects the Sudoku grid, removes lines, splits the board into 81 cells, and uses `pytesseract` to extract digits.

---

## 🔍 Features

- 🧩 Sudoku grid detection via image preprocessing and contour analysis
- 🧼 Grid line removal for clean cell segmentation
- 🔢 OCR-powered digit extraction using `pytesseract`
- 📦 API endpoint to send image and get digit grid in JSON format
- ⚙️ Deployable on [Render](https://render.com) (supports automatic Tesseract installation)

---
