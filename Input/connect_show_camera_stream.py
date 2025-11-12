## nutné si udělat venv
import cv2
import paramiko
import time
import threading

# Konfigurace
RASPBERRY_IP = "raspberrypi.local"  # přes ip to nefunguje
RASPBERRY_USER = "imang"
RASPBERRY_PASSWORD = "imang"  # nebo použij SSH klíč
STREAM_PORT = 8554
STREAM_COMMAND = "rpicam-vid -t 0 --inline --listen -n -o tcp://0.0.0.0:8554"

def start_stream_on_raspberry():
    """Připojí se přes SSH a spustí stream na Raspberry Pi"""
    try:
        # Vytvoř SSH klienta
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())        
        print(f"Připojuji se k Raspberry Pi ({RASPBERRY_IP})...")
        ssh.connect(RASPBERRY_IP, username=RASPBERRY_USER, password=RASPBERRY_PASSWORD)
        print("Spouštím stream na Raspberry Pi...") # Spusť příkaz (non-blocking)
        stdin, stdout, stderr = ssh.exec_command(STREAM_COMMAND)
        # Počkej chvíli, než se stream inicializuje
        time.sleep(3)
        print("Stream by měl být připraven!")
        return ssh, stdin, stdout, stderr
    except Exception as e:
        print(f"Chyba při připojení k Raspberry Pi: {e}")
        return None, None, None, None

def monitor_stream_output(stdout, stderr):
    """Monitoruje výstup ze streamu (běží v samostatném vlákně)"""
    for line in stdout:
        print(f"[RPi stdout]: {line.strip()}")
    for line in stderr:
        print(f"[RPi stderr]: {line.strip()}")

def connect_to_stream(stdin, ssh):
    # Připoj se ke streamu
    stream_url = f"tcp://{RASPBERRY_IP}:{STREAM_PORT}"
    print(f"\nPřipojuji se ke streamu: {stream_url}")
    
    cap = cv2.VideoCapture(stream_url, cv2.CAP_FFMPEG)
    
    if not cap.isOpened():
        print("Stream se nepodařilo otevřít!")
        ssh.close()
        return
    
    print("Stream úspěšně připojen! Stiskni 'q' pro ukončení.")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Nepodařilo se načíst snímek.")
                break
            
            cv2.imshow('Raspberry Pi Camera Stream', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    except KeyboardInterrupt:
        print("\nUkončuji...")
    
    finally:
        # Ukliď
        cap.release()
        cv2.destroyAllWindows()        
        # Ukonči stream na Raspberry Pi (Ctrl+C)
        if stdin:
            stdin.close()
        ssh.close()
        print("Připojení ukončeno.")

def main():
    # Spusť stream na Raspberry Pi
    ssh, stdin, stdout, stderr = start_stream_on_raspberry()
    
    if ssh is None: print("Nepodařilo se spustit stream."); return
    
    # Spusť monitoring výstupu v samostatném vlákně
    monitor_thread = threading.Thread(target=monitor_stream_output, args=(stdout, stderr))
    monitor_thread.daemon = True
    monitor_thread.start()

    connect_to_stream(stdin, ssh)
    
if __name__ == "__main__":
    main()