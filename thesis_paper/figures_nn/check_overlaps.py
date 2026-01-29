import cv2
import numpy as np
import os
import argparse
import subprocess
import glob
import sys

def convert_pdf_to_image(pdf_path, dpi=300):
    """
    Converts PDF to PNG using pdftoppm.
    Returns the path to the generated image (first page).
    """
    base_name = os.path.splitext(pdf_path)[0]
    prefix = base_name + "_temp"
    
    try:
        cmd = ["pdftoppm", "-png", "-r", str(dpi), "-f", "1", "-l", "1", pdf_path, prefix]
        subprocess.check_call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        expected_out = prefix + "-1.png"
        if os.path.exists(expected_out):
            return expected_out
        
        candidates = glob.glob(prefix + "*.png")
        if candidates:
            return candidates[0]
            
        return None
    except Exception as e:
        print(f"Error converting PDF {pdf_path}: {e}")
        return None

def line_rect_intersection(p1, p2, rect):
    """
    Cohen-Sutherland like clipping or simple parametric intersection.
    Returns list of intersection points.
    """
    x1, y1 = p1
    x2, y2 = p2
    rx, ry, rw, rh = rect
    rx2, ry2 = rx + rw, ry + rh
    
    # Parametric form: P(t) = P1 + t*(P2-P1), 0 <= t <= 1
    dx = x2 - x1
    dy = y2 - y1
    
    t_min, t_max = 0.0, 1.0
    
    p = [-dx, dx, -dy, dy]
    q = [x1 - rx, rx2 - x1, y1 - ry, ry2 - y1]
    
    for i in range(4):
        if p[i] == 0:
            if q[i] < 0:
                return False # Parallel and outside
        else:
            t = q[i] / p[i]
            if p[i] < 0:
                if t > t_max: return False
                if t > t_min: t_min = t
            else:
                if t < t_min: return False
                if t < t_max: t_max = t
                
    if t_min < t_max:
        return True # Intersection exists within segment
    return False

def detect_overlaps(image_path, debug_dir=None):
    """
    Detects if lines/arrows interfere with text regions.
    Ignores intersections that occur at the very tips of the lines (valid connections).
    """
    img = cv2.imread(image_path)
    if img is None:
        return -1, None
        
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    
    # Extract Text Regions
    kernel_text = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 3))
    dilated_text = cv2.dilate(binary, kernel_text, iterations=1)
    contours, _ = cv2.findContours(dilated_text, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    text_boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        if 50 < area < 100000:
            # Shrink box slightly to allow lines to touch the outer border without triggering
            # (Simulate "content" box vs "node" box)
            pad = 2
            text_boxes.append((x+pad, y+pad, w-2*pad, h-2*pad))
    
    # Detect Lines
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=30, minLineLength=30, maxLineGap=5)
    
    interferences = []
    debug_img = img.copy()
    
    # Draw Text Boxes (Blue)
    for (x,y,w,h) in text_boxes:
        cv2.rectangle(debug_img, (x,y), (x+w, y+h), (255, 0, 0), 1)
        
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            
            # Check intersection with all text boxes
            for (tx, ty, tw, th) in text_boxes:
                # Check if endpoints are inside (penetration)
                p1_in = (tx < x1 < tx+tw and ty < y1 < ty+th)
                p2_in = (tx < x2 < tx+tw and ty < y2 < ty+th)
                
                # If BOTH endpoints are inside, it's likely text strokes detected as lines -> Ignore
                if p1_in and p2_in:
                    continue
                    
                # If ONE endpoint is inside, it might be a valid arrow ending inside the box node.
                # But in TikZ, arrows stop at border. If it's inside the *shrunk* box, it's too deep.
                # However, Hough lines are imprecise.
                # Let's check for "crossing through": intersection exists but neither endpoint is deep inside?
                
                # Robust check: Check the MIDPOINT of the line. 
                # If the middle of the line is inside a text box, it's definitely crossing it.
                mid_x = (x1 + x2) / 2
                mid_y = (y1 + y2) / 2
                
                mid_in = (tx < mid_x < tx+tw and ty < mid_y < ty+th)
                
                if mid_in:
                    interferences.append(((x1,y1,x2,y2), (tx,ty,tw,th)))
                    cv2.line(debug_img, (x1,y1), (x2,y2), (0, 0, 255), 2)
                    cv2.rectangle(debug_img, (tx,ty), (tx+tw, ty+th), (0, 0, 255), 2)
                    break
            
            # Draw all lines (Green)
            cv2.line(debug_img, (x1,y1), (x2,y2), (0, 255, 0), 1)

    debug_path = None
    if debug_dir and len(interferences) > 0:
        fname = os.path.basename(image_path)
        debug_path = os.path.join(debug_dir, "debug_" + fname)
        cv2.imwrite(debug_path, debug_img)
        
    return len(interferences), debug_path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--save_debug", action="store_true")
    args = parser.parse_args()
    
    target = os.path.abspath(args.path)
    if os.path.isfile(target):
        files = [target]
    else:
        files = glob.glob(os.path.join(target, "**/*.pdf"), recursive=True)
        
    print(f"Checking {len(files)} files...")
    
    for pdf_file in files:
        if "debug_" in pdf_file: continue
        print(f"Checking {os.path.basename(pdf_file)}...", end="", flush=True)
        
        img_path = convert_pdf_to_image(pdf_file, args.dpi)
        if not img_path: continue
            
        count, debug_path = detect_overlaps(img_path, debug_dir=os.path.dirname(pdf_file) if args.save_debug else None)
        
        if os.path.exists(img_path): os.remove(img_path)
            
        if count > 0:
            print(f" FAIL ({count} overlaps)")
            if debug_path: print(f"  -> {debug_path}")
        else:
            print(" OK")

if __name__ == "__main__":
    main()
