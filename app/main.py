from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
from PIL import Image
import pytesseract
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
from io import BytesIO
from fastapi.responses import JSONResponse

app = FastAPI()

# Enable CORS for all domains
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all domains
    allow_credentials=True,
    allow_methods=["*"],  # Allows all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)

# ✅ Step 0: Resize the image
def resize_image(image, width=450, height=450):
    try:
        resized_image = cv2.resize(image, (width, height))
        return resized_image
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error resizing image: {str(e)}")

# ✅ Step 1: Load the image
def load_image(image_bytes):
    try:
        image = np.array(Image.open(BytesIO(image_bytes)))
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR (OpenCV format)
        return image
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading image: {str(e)}")

# ✅ Step 2: Convert image to grayscale and blur
def preprocess_image(image):
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (7, 7), 0)
        return blur
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error preprocessing image: {str(e)}")

# ✅ Step 3: Adaptive thresholding
def threshold_image(blurred_image):
    try:
        thresh = cv2.adaptiveThreshold(blurred_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 11, 2)
        return thresh
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error thresholding image: {str(e)}")

# ✅ Step 4: Find contours
def find_contours(thresh_image):
    try:
        contours, _ = cv2.findContours(thresh_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error finding contours: {str(e)}")

# ✅ Step 5: Find the largest contour with 4 points (Sudoku grid)
def find_sudoku_contour(contours, image):
    try:
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
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error finding Sudoku contour: {str(e)}")

# ✅ Step 6: Order corners for warping
def order_points(pts):
    try:
        pts = pts.reshape(4, 2)
        s = pts.sum(axis=1)
        diff = np.diff(pts, axis=1)
        return np.array([ 
            pts[np.argmin(s)],    # top-left
            pts[np.argmin(diff)], # top-right
            pts[np.argmax(s)],    # bottom-right
            pts[np.argmax(diff)]  # bottom-left
        ], dtype="float32")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error ordering points: {str(e)}")

# ✅ Step 7: Warp image
def straighten_image(image, sudoku_contour):
    try:
        pts = order_points(sudoku_contour)
        (tl, tr, br, bl) = pts
        width = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
        height = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))
        dst = np.array([[0, 0], [width-1, 0], [width-1, height-1], [0, height-1]], dtype="float32")
        M = cv2.getPerspectiveTransform(pts, dst)
        warped = cv2.warpPerspective(image, M, (width, height))
        return warped
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error warping image: {str(e)}")

# ✅ Detect and draw grid lines
def detect_grid_lines(gray_img):
    try:
        blurred = cv2.GaussianBlur(gray_img, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 120)
        edges = cv2.dilate(edges, np.ones((2,2), np.uint8), iterations=1)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=150, minLineLength=100, maxLineGap=20)
        return lines
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error detecting grid lines: {str(e)}")

def draw_grid_lines(gray_img, lines):
    try:
        if len(gray_img.shape) == 2:
            img = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)
        else:
            img = gray_img
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(img, (x1, y1), (x2, y2), (0, 0, 0), 5)
        return img
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error drawing grid lines: {str(e)}")

# ✅ Split Sudoku into 81 cells
def split_sudoku_array(image_array):
    try:
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
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error splitting Sudoku grid: {str(e)}")

def predict_sudoku_grid_ocr(cells):
    predicted_grid = np.zeros((9, 9), dtype=int)

    def improved_preprocess(cell):
        try:
            if len(cell.shape) == 3:
                cell = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)  # Ensure grayscale

            cell = cv2.resize(cell, (100, 100), interpolation=cv2.INTER_CUBIC)  # Resize for better OCR accuracy
            cell = cv2.GaussianBlur(cell, (3, 3), 0)  # Reduce noise
            _, cell_thresh = cv2.threshold(cell, 150, 255, cv2.THRESH_BINARY)  # Binary thresholding

            return cell_thresh
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error preprocessing cell: {str(e)}")

    def ocr_predict(cell):
        try:
            processed = improved_preprocess(cell)
            config = r'--oem 3 --psm 10 -c tessedit_char_whitelist=0123456789'
            digit_str = pytesseract.image_to_string(processed, config=config).strip()

            try:
                digit = int(digit_str) if digit_str else 0
            except (ValueError, IndexError):
                digit = 0

            return digit
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error OCR predicting digit: {str(e)}")

    try:
        for i in range(9):
            for j in range(9):
                cell_img = cells[i, j]
                digit = ocr_predict(cell_img)
                predicted_grid[i, j] = digit

        return predicted_grid
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in Sudoku OCR prediction: {str(e)}")

# ✅ Main function (with FastAPI endpoint)
@app.post("/predict_sudoku")
async def predict_sudoku(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        image = load_image(image_bytes)

        blurred = preprocess_image(image)
        thresh = threshold_image(blurred)
        contours = find_contours(thresh)
        sudoku_contour = find_sudoku_contour(contours, image)
        straightened = straighten_image(image, sudoku_contour)
        processed = threshold_image(preprocess_image(straightened))
        #resized = resize_image(processed)
        lines = detect_grid_lines(processed)
        grid_img = draw_grid_lines(processed, lines)
        inverted = cv2.bitwise_not(grid_img)

        cells = split_sudoku_array(inverted)
        predicted_grid = predict_sudoku_grid_ocr(cells)

        return JSONResponse(content={"predicted_grid": predicted_grid.tolist()})

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
