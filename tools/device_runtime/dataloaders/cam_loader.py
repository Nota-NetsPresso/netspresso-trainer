import cv2

class LoadCamera:
    def __init__(self) -> None:
        self.cap = cv2.VideoCapture(0)

    def __iter__(self):
        return self

    def __next__(self):
        success, img = self.cap.read()
        if success:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return img
        else:
            raise IOError("Failed to read camera frame")

    def __len__(self):
        return 1e12 # Set a large number to make it long enough
