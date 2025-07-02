from flask import Flask, request, send_file, render_template_string
from PIL import Image, ImageOps
import io
import numpy as np
import cv2

app = Flask(__name__)

INDEX_HTML = open('index.html').read()


def solve_maze(image: np.ndarray):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    # Ensure paths are white (255) and walls are black (0)
    if np.sum(binary[0]) < np.sum(binary[-1]):
        binary = cv2.bitwise_not(binary)

    h, w = binary.shape

    start = None
    for x in range(w):
        if binary[0, x] == 255:
            start = (x, 0)
            break

    end = None
    for x in range(w - 1, -1, -1):
        if binary[h - 1, x] == 255:
            end = (x, h - 1)
            break

    if start is None or end is None:
        return None

    from collections import deque
    queue = deque([start])
    visited = np.zeros_like(binary, dtype=bool)
    visited[start[1], start[0]] = True
    parent = {start: None}

    while queue:
        x, y = queue.popleft()
        if (x, y) == end:
            break
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < w and 0 <= ny < h and binary[ny, nx] == 255 and not visited[ny, nx]:
                visited[ny, nx] = True
                parent[(nx, ny)] = (x, y)
                queue.append((nx, ny))

    if end not in parent:
        return None

    solved = image.copy()
    node = end
    while node:
        cv2.circle(solved, node, radius=0, color=(255, 0, 0), thickness=1)
        node = parent[node]

    return solved


@app.route('/')
def index():
    return render_template_string(INDEX_HTML)


@app.route('/solve-maze', methods=['POST'])
def solve():
    file = request.files['mazeImage']
    img = Image.open(file.stream).convert('RGB')
    np_img = np.array(img)
    solved = solve_maze(np_img)

    if solved is None:
        solved = np_img

    result_img = ImageOps.invert(Image.fromarray(solved))

    buf = io.BytesIO()
    result_img.save(buf, format='PNG')
    buf.seek(0)
    return send_file(buf, mimetype='image/png')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

