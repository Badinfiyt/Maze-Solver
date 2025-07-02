from flask import Flask, request, send_file
from PIL import Image, ImageOps
import io
import numpy as np
import cv2

app = Flask(__name__)

def solve_maze(image):
    # Convert to grayscale and binary
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

    # Find start and end (simplified)
    start = (0, 0)
    end = (binary.shape[1]-1, binary.shape[0]-1)

    # BFS Maze solving (you can replace with A* or DFS)
    from collections import deque
    visited = np.zeros_like(binary)
    queue = deque([start])
    parent = {start: None}
    while queue:
        x, y = queue.popleft()
        if (x, y) == end:
            break
        for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < binary.shape[1] and 0 <= ny < binary.shape[0] and binary[ny, nx] == 255 and not visited[ny, nx]:
                visited[ny, nx] = 1
                parent[(nx, ny)] = (x, y)
                queue.append((nx, ny))

    # Draw path
    path = []
    node = end
    while node in parent:
        path.append(node)
        node = parent[node]
    for x, y in path:
        image[y, x] = (255, 0, 0)  # Mark path in red

    return image

@app.route('/solve-maze', methods=['POST'])
def solve():
    file = request.files['mazeImage']
    img = Image.open(file.stream).convert("RGB")
    np_img = np.array(img)
    solved = solve_maze(np_img)
    result_img = Image.fromarray(solved)
    result_img = ImageOps.invert(result_img)

    buf = io.BytesIO()
    result_img.save(buf, format='PNG')
    buf.seek(0)
    return send_file(buf, mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True)