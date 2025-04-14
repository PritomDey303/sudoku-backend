from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import cv2
import numpy as np
from keras.models import load_model
from PIL import Image
import shutil
import uuid
import os
import os
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"], 
)

model_path = os.path.join(os.getcwd(), 'public', 'model', 'mnist_cnn_model.h5')
print("Model Path:", model_path)

model = load_model(model_path)

###########
def resize_image(image, width=450, height=450):
    return cv2.resize(image, (width, height))
def add_padding_to_image_array(image, top=5, bottom=5, left=5, right=5, padding_color=(255,255,255)):

    if image is None:
        raise ValueError("Input image is None. Please provide a valid image array.")

    padded_image = cv2.copyMakeBorder(
        image,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=padding_color
    )

    return padded_image
def load_image(image_path):
    return cv2.imread(image_path)

def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    return blur

def threshold_image(blurred_image):
    return cv2.adaptiveThreshold(blurred_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY_INV, 11, 2)

def find_contours(thresh_image):
    contours, _ = cv2.findContours(thresh_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def find_sudoku_contour(contours):
    max_area = 0
    sudoku_contour = None
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
        if len(approx) == 4:
            area = cv2.contourArea(approx)
            if area > max_area:
                max_area = area
                sudoku_contour = approx
    if sudoku_contour is None:
        raise Exception("No Sudoku board found.")
    return sudoku_contour

def order_points(pts):
    pts = pts.reshape(4, 2)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    return np.array([
        pts[np.argmin(s)],
        pts[np.argmin(diff)],
        pts[np.argmax(s)],
        pts[np.argmax(diff)]
    ], dtype="float32")

def straighten_image(image, sudoku_contour):
    pts = order_points(sudoku_contour)
    (tl, tr, br, bl) = pts
    width = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
    height = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))
    dst = np.array([[0, 0], [width-1, 0], [width-1, height-1], [0, height-1]], dtype="float32")
    M = cv2.getPerspectiveTransform(pts, dst)
    return cv2.warpPerspective(image, M, (width, height))

def detect_grid_lines(gray_img):
    blurred = cv2.GaussianBlur(gray_img, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 120)
    edges = cv2.dilate(edges, np.ones((2,2), np.uint8), iterations=1)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=150, minLineLength=100, maxLineGap=20)
    return lines

def draw_grid_lines(gray_img, lines):
    if len(gray_img.shape) == 2:
        img = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)
    else:
        img = gray_img
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(img, (x1, y1), (x2, y2), (0, 0, 0), 8)
    return img

def split_sudoku_array(image_array):
    if len(image_array.shape) == 3:
        image_array = np.mean(image_array, axis=2).astype(np.uint8)
    h, w = image_array.shape
    cell_h, cell_w = h // 9, w // 9
    cells = np.zeros((9, 9, cell_h, cell_w), dtype=image_array.dtype)
    for i in range(9):
        for j in range(9):
            top, left = i * cell_h, j * cell_w
            cells[i, j] = image_array[top:top + cell_h, left:left + cell_w]
    return cells

def is_blank_cell(cell_img, white_pixel_threshold=25):
    if len(cell_img.shape) == 3:
        cell_img = cv2.cvtColor(cell_img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(cell_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    white_pixels = np.sum(binary == 255)
    return white_pixels < white_pixel_threshold

def predict_with_cnn(cell_img, model):
    if len(cell_img.shape) == 3:
        cell_img = cv2.cvtColor(cell_img, cv2.COLOR_BGR2GRAY)
    if np.mean(cell_img) > 127:
        cell_img = 255 - cell_img
    _, cell_img = cv2.threshold(cell_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    h, w = cell_img.shape
    scale = 1.2
    new_size = int(max(h, w) * scale)
    padded = np.zeros((new_size, new_size), dtype=np.uint8)
    y = (new_size - h) // 2
    x = (new_size - w) // 2
    padded[y:y+h, x:x+w] = cell_img
    resized = cv2.resize(padded, (28, 28), interpolation=cv2.INTER_AREA)
    processed = resized.astype(np.float32) / 255.0
    processed = np.expand_dims(processed, axis=(0, -1))
    pred = model.predict(processed, verbose=0)[0]
    digit = np.argmax(pred)
    return digit, np.max(pred)

def predict_sudoku_grid_cnn(cells, model):
    predicted_grid = np.zeros((9, 9), dtype=int)
    for i in range(9):
        for j in range(9):
            cell_img = cells[i, j]
            if is_blank_cell(cell_img):
                predicted_grid[i, j] = 0
                continue
            digit, _ = predict_with_cnn(cell_img, model)
            predicted_grid[i, j] = digit if digit != 0 else 0
    return predicted_grid.tolist()

@app.post("/predict/")
async def predict_sudoku(file: UploadFile = File(...)):
    filename = f"temp_{uuid.uuid4().hex}.jpg"
    with open(filename, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        image = load_image(filename)
        image=add_padding_to_image_array(image)
        blurred = preprocess_image(image)
        thresh = threshold_image(blurred)
        contours = find_contours(thresh)
        sudoku_contour = find_sudoku_contour(contours)
        straightened = straighten_image(image, sudoku_contour)
        processed = threshold_image(preprocess_image(straightened))
        lines = detect_grid_lines(processed)
        grid_img = draw_grid_lines(processed, lines)
        cells = split_sudoku_array(grid_img)
        predicted_grid = predict_sudoku_grid_cnn(cells, model)
        return JSONResponse(content={"grid": predicted_grid})
    finally:
        os.remove(filename)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)