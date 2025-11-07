import unittest
import numpy as np
import time
from Input import FrameBuffer, VideoCapture, VideoPlayer, VideoStreamListener

# třída pro testování frame bufferu 
class TestFrameBuffer(unittest.TestCase):
    def setUp(self):
        self.buffer = FrameBuffer(capacity=5, frame_shape=(480, 640, 3))
    
    def test_add_and_retrieve(self):
        """Test přidávání a získávání snímků"""
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        self.buffer.add_frame(frame)
        
        self.assertEqual(len(self.buffer), 1)
        retrieved = self.buffer[0]
        np.testing.assert_array_equal(frame, retrieved)
    
    def test_buffer_overflow(self):
        """Test přetečení bufferu"""
        for i in range(10):
            frame = np.ones((480, 640, 3), dtype=np.uint8) * i
            self.buffer.add_frame(frame)
        
        # Buffer má kapacitu 5, měl by obsahovat snímky 5-9
        self.assertEqual(len(self.buffer), 5)
        self.assertTrue(np.all(self.buffer[0] == 5))  # Nejstarší
        self.assertTrue(np.all(self.buffer[4] == 9))  # Nejnovější
    
    def test_iteration(self):
        """Test iterace přes buffer"""
        for i in range(3):
            frame = np.ones((480, 640, 3), dtype=np.uint8) * i
            self.buffer.add_frame(frame)
        
        frames = list(self.buffer)
        self.assertEqual(len(frames), 3)

# mock listener pro testování
class MockListener(VideoStreamListener):
    """Mock listener pro testování"""
    def __init__(self):
        self.frames_received = []
    
    def on_frame(self, frame: np.ndarray) -> None:
        self.frames_received.append(frame.copy())

# testování videocapture
class TestVideoCapture(unittest.TestCase):
    def test_capture_initialization(self):
        """Test inicializace kamery"""
        try:
            stream = VideoCapture(device_index=0, full_buffer_size=5)
            self.assertIsNotNone(stream)
            
            # Čekej na pár snímků
            time.sleep(1)
            
            # Kontrola bufferu
            self.assertGreater(len(stream.full_buffer), 0)
            self.assertGreater(len(stream.scaled_buffer), 0)
            
            stream.stop()
        except RuntimeError as e:
            self.skipTest(f"Kamera není dostupná: {e}")
    
    def test_listener_notification(self):
        """Test notifikace listenerů"""
        try:
            stream = VideoCapture(device_index=0)
            listener = MockListener()
            stream.add_listener(listener)
            
            # Čekej na snímky
            time.sleep(2)
            
            # Listener by měl obdržet snímky
            self.assertGreater(len(listener.frames_received), 0)
            
            stream.stop()
        except RuntimeError as e:
            self.skipTest(f"Kamera není dostupná: {e}")

#testování přehrávání souboru
class TestVideoPlayer(unittest.TestCase):
    def test_player_with_sample_video(self):
        """Test přehrávání video souboru"""
        # Vytvoř testovací video
        import cv2
        video_path = 'test_video.mp4'
        
        # Vytvoř malé testovací video
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(video_path, fourcc, 20.0, (640, 480))
        
        for i in range(50):
            frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            out.write(frame)
        out.release()
        
        try:
            # Test přehrávání
            player = VideoPlayer(video_path, full_buffer_size=10)
            time.sleep(1)
            
            self.assertGreater(len(player.full_buffer), 0)
            
            player.stop()
        except Exception as e:
            self.fail(f"Chyba při přehrávání videa: {e}")
        finally:
            import os
            if os.path.exists(video_path):
                os.remove(video_path)

# funkce pro spuštění kamerys
def run_interactive_test():
    """Interaktivní test s live zobrazením"""
    print("Spouštím interaktivní test kamery..."); print("Stiskni 'q' pro ukončení")
    try:
        import cv2
        stream = VideoCapture(device_index=0, scaled_shape=(320, 240))
        frame_count = 0
        start_time = time.time()
        while True:
            if len(stream.full_buffer) > 0:
                frame = stream.last_frame
                # Zobraz FPS
                frame_count += 1
                elapsed = time.time() - start_time
                fps = frame_count / elapsed if elapsed > 0 else 0
                # Přidej text na snímek
                display_frame = frame.copy()
                cv2.putText(display_frame, f'FPS: {fps:.1f}', (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(display_frame, f'Buffer: {len(stream.full_buffer)}', (10, 70),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                cv2.imshow('Video Stream Test', display_frame)
                
                # Zobraz i scaled verzi
                if len(stream.scaled_buffer) > 0:
                    scaled = stream.scaled_buffer[-1]
                    cv2.imshow('Scaled Stream', scaled)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        print(f"\nStatistiky:")
        print(f"Celkové snímky: {frame_count}")
        print(f"Průměrné FPS: {fps:.1f}")
        print(f"Buffer size: {len(stream.full_buffer)}")
        stream.stop()
        cv2.destroyAllWindows()
    except Exception as e:
        print(f"Chyba: {e}")

if __name__ == '__main__':
    import sys    
    if len(sys.argv) > 1 and sys.argv[1] == '--interactive':
        run_interactive_test()
    else:
        unittest.main()