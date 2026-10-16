
import cv2
import numpy as np
from ultralytics import YOLO

# Load YOLO model
model = YOLO("yolov8n.pt")

# Define table regions (x, y coordinates)
TABLES = {
    "Table 1": [(236, 262), (315, 258), (323, 284), (244, 291)],
    "Table 2": [(444, 178), (496, 181), (510, 202), (451, 207)],
    "Table 3": [(264, 225), (337, 230), (347, 249), (276, 254)],
    "Table 4": [(41, 190), (97, 183), (110, 205), (51, 216)],
    "Table 5": [(168, 139), (202, 139), (213, 154), (173, 158)],
    "Table 6": [(264, 139), (303, 139), (314, 154), (270, 158)]
}

# Open webcam (0) or replace 0 with a video filename
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam or video.")
    raise SystemExit

while True:
    ret, frame = cap.read()

    if not ret:
        break

    height, width = frame.shape[:2]

    # Scale table coordinates to the frame size
    sx = width / 640
    sy = height / 360

    # Detect people using YOLO (class 0 = person)
    results = model.predict(
        frame, classes=[0], conf=0.35, verbose=False
    )

    people = []

    for result in results:
        if result.boxes is not None:
            for box in result.boxes:
                x1, y1, x2, y2 = map(
                    int, box.xyxy[0].tolist()
                )
                people.append((x1, y1, x2, y2))

    occupied_count = 0

    # Check occupancy of each table
    for name, points in TABLES.items():

        polygon = np.array([
            (int(x * sx), int(y * sy))
            for x, y in points
        ], dtype=np.int32)

        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(mask, [polygon], 255)

        occupied = False

        for x1, y1, x2, y2 in people:
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(width, x2)
            y2 = min(height, y2)

            if x2 <= x1 or y2 <= y1:
                continue

            person_area = (x2 - x1) * (y2 - y1)
            overlap = cv2.countNonZero(
                mask[y1:y2, x1:x2]
            )

            if overlap / person_area > 0.08:
                occupied = True
                break

        if occupied:
            color = (0, 0, 255)  # Red = Occupied
            status = "Occupied"
            occupied_count += 1
        else:
            color = (0, 255, 0)  # Green = Free
            status = "Free"

        # Draw table boundary and status
        cv2.polylines(
            frame, [polygon], True, color, 2
        )

        x, y = polygon[0]

        cv2.putText(
            frame,
            f"{name}: {status}",
            (x, max(20, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2
        )

    # Display occupancy dashboard
    total_tables = len(TABLES)

    cv2.rectangle(
        frame, (10, 10), (320, 55),
        (30, 30, 30), -1
    )

    cv2.putText(
        frame,
        f"Occupied: {occupied_count}/{total_tables}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.imshow("AI Table Occupancy Detection", frame)

    # Press Q to exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

