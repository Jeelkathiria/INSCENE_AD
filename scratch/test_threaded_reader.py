import cv2
import queue
import threading
import time

class ThreadedFrameReader:
    def __init__(self, cap: cv2.VideoCapture, max_queue_size: int = 48):
        self.cap = cap
        self.queue = queue.Queue(maxsize=max_queue_size)
        self.stopped = False
        self.thread = threading.Thread(target=self._reader_worker, daemon=True)
        self.thread.start()

    def _reader_worker(self):
        while not self.stopped:
            ret, frame = self.cap.read()
            if not ret:
                self.queue.put((False, None))
                break
            self.queue.put((True, frame))

    def read(self):
        return self.queue.get()

    def stop(self):
        self.stopped = True
        while not self.queue.empty():
            try:
                self.queue.get_nowait()
            except queue.Empty:
                break

cap = cv2.VideoCapture('data/uploads/9e44283a_CITY_VIDEO.mp4')
reader = ThreadedFrameReader(cap, max_queue_size=32)

t0 = time.time()
count = 0
while True:
    ret, frame = reader.read()
    if not ret or frame is None:
        break
    count += 1
    if count >= 100:
        break

reader.stop()
cap.release()
print(f"Successfully read {count} frames in {time.time() - t0:.2f}s with ThreadedFrameReader!")

