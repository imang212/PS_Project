import requests
import threading
from datetime import datetime

def _send_request(url, data):
    """Pomocná funkce pro odeslání v jiném vlákně."""
    try:
        requests.post(url, json=data, timeout=0.2)
    except:
        pass

def post_vehicle_data(label, confidence):
    url = "http://127.0.0.1:8000/detection"
    data = {
        "label": label,
        "confidence": float(confidence),
        "timestamp": datetime.now().isoformat()
    }
    
    # Vytvoříme vlákno, které se postará o odeslání, zatímco AI pokračuje dál
    thread = threading.Thread(target=_send_request, args=(url, data))
    thread.start()