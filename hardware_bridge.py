import socket
import struct
import numpy as np
import time
import json
import serial 

# Configuration
SERIAL_PORT = "COM3"  # Change to your ESP32 Port
BAUD_RATE = 115200

def parse_esp32_line(line):
    """
    Parses a line of CSV/JSON text from ESP32 CSI Toolkit.
    Format varies by firmware, but usually looks like:
    "CSI_DATA, [len], [RSSI], [data_array...]"
    """
    try:
        # Placeholder parser - depends heavily on your specific ESP32 firmware output
        # This is a generic example assuming JSON output
        data = json.loads(line)
        if "csi" in data:
            # Output: real,imag interleaved
            raw = np.array(data["csi"]) 
            # Reshape to [1, 1, Subcarriers, 2, 1]
            return raw
    except:
        return None

def run_serial_bridge():
    print(f"Listening on {SERIAL_PORT}...")
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.1)
    except Exception as e:
        print(f"Could not open serial port {SERIAL_PORT}: {e}")
        print("Hardware bridge requires a connected ESP32.")
        return

    # UDP Target (The Main App)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    
    while True:
        line = ser.readline().decode('utf-8').strip()
        if line:
            csi_data = parse_esp32_line(line)
            if csi_data is not None:
                # Send to Main App via UDP localhost
                # We need to serialize this numpy array
                msg = csi_data.tobytes()
                sock.sendto(msg, ("127.0.0.1", 8080))

if __name__ == "__main__":
    run_serial_bridge()
