## nutné si udělat venv
import cv2
import time
import numpy as np
from Input import RemoteRaspberryPiStreamProvider, VideoStream, VideoStreamFormatterStrategy, VideoStreamListener

def main():
    try:
        strategy = VideoStreamFormatterStrategy.resize_strategy((160, 90), interpolation=cv2.INTER_LINEAR)
        strategy.append_chain(VideoStreamFormatterStrategy.gray_scale_strategy())
        stream_provider = RemoteRaspberryPiStreamProvider(raspberry_ip="raspberrypi.local", raspberry_user="imang", raspberry_password="imang", stream_port=8554, resolution=(640, 480))
        stream = VideoStream(stream_provider, buffer_size=10, format_strategy=strategy)
    
        print("Připojeno! Zobrazuji stream...")
        print("Stiskni 'q' pro ukončení")
        # Čti a zobraz snímky
        while True:
            frame = stream_provider.read()
            
            if frame is None:
                print("\nStream skončil nebo selhalo čtení snímku")
                break
            
            cv2.imshow('Remote Raspberry Pi Camera - Basic Test', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except Exception as e:
        print(f"\nChyba: {e}")    
    finally:
        stream_provider.release()
        cv2.destroyAllWindows()
        print("\n\nTest ukončen")
        
if __name__ == "__main__":
    main()