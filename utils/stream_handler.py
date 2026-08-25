import cv2
import threading
import time
import logging

class BufferlessVideoCapture:
    """
    A threaded, bufferless wrapper for cv2.VideoCapture.
    Ensures that the pipeline always processes the most recent frame,
    preventing latency buildup when inference time > frame time.
    """
    def __init__(self, source=0):
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            logging.error(f"Failed to open video source: {source}")
            raise ValueError(f"Unable to open video source: {source}")
            
        self.lock = threading.Lock()
        self.running = True

        
        # Read the first frame to initialize
        self.ret, self.frame = self.cap.read()
        
        # Start the background thread
        self.thread = threading.Thread(target=self._reader_thread, daemon=True)
        self.thread.start()
        logging.info(f"Bufferless stream initialized for source: {source}")

    def _reader_thread(self):
        """Continuously pulls frames from the camera and stores only the latest one."""
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                self.running = False
                break
                
            # Acquire lock to safely update the latest frame
            with self.lock:
                self.ret = ret
                self.frame = frame
                
            # Tiny sleep to yield thread execution
            time.sleep(0.005) 

    def read(self):
        """Returns the most recently captured frame."""
        with self.lock:
            # Return a copy to prevent the thread from overwriting it during inference
            return self.ret, self.frame.copy() if self.ret else None

    def release(self):
        """Stops the thread and releases the camera."""
        self.running = False
        self.thread.join()
        self.cap.release()
        logging.info("Video stream released.")

    def isOpened(self):
        return self.cap.isOpened()
