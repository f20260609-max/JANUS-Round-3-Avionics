import queue
import threading
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

# SIMULATION_MODE = True runs with simulated packets
# Set to False when you plug in your actual XBee module
SIMULATION_MODE = True

SERIAL_PORT = "COM3"  # Change to your XBee COM port if using real hardware
BAUD_RATE = 9600

data_queue = queue.Queue()
lats, lons, alts = [], [], []


def parse_telemetry_packet(raw_data: str):
    """
    Format: Timestamp, State, Temp, Pressure, Altitude, Batt_V, Batt_I, Latitude, Longitude, Prev_CMD_echo
    """
    try:
        tokens = [token.strip() for token in raw_data.strip().split(",")]
        if len(tokens) >= 9:
            alt = float(tokens[4])
            lat = float(tokens[7])
            lon = float(tokens[8])
            data_queue.put((lat, lon, alt))
    except Exception as err:
        print(f"Packet parse error: {err}")


def dummy_telemetry_generator():
    """Generates synthetic telemetry packets for offline testing."""
    t = 0
    while True:
        sim_packet = (
            f"{t},1,24.5,1013.2,{700.0 - t * 2.5},"
            f"12.4,1.1,{17.3850 + t * 0.0001},{78.4867 + t * 0.0001},ACK"
        )
        parse_telemetry_packet(sim_packet)
        t += 1
        time.sleep(0.3)


def update_plot(frame):
    updated = False
    while not data_queue.empty():
        lat, lon, alt = data_queue.get()
        lats.append(lat)
        lons.append(lon)
        alts.append(alt)
        updated = True

    if updated and len(lats) > 0:
        ax.cla()
        ax.plot(lons, lats, alts, color="tab:blue", lw=2, label="Flight Path")
        ax.scatter([lons[-1]], [lats[-1]], [alts[-1]], color="red", s=50, label="Current Position")

        ax.set_xlabel("Longitude (°)")
        ax.set_ylabel("Latitude (°)")
        ax.set_zlabel("Altitude (m)")
        ax.set_title("Real-Time Telemetry 3D Tracking (Digi XBee)")
        ax.legend(loc="upper left")
        ax.grid(True)


fig = plt.figure(figsize=(9, 7))
ax = fig.add_subplot(111, projection="3d")

if SIMULATION_MODE:
    print("Running in SIMULATION MODE. Generating dummy telemetry...")
    sim_thread = threading.Thread(target=dummy_telemetry_generator, daemon=True)
    sim_thread.start()
else:
    from digi.xbee.devices import XBeeDevice

    def xbee_callback(xbee_message):
        raw_str = xbee_message.data.decode("utf-8")
        parse_telemetry_packet(raw_str)

    xbee = XBeeDevice(SERIAL_PORT, BAUD_RATE)
    xbee.open()
    xbee.add_data_received_callback(xbee_callback)
    print(f"Connected to XBee on {SERIAL_PORT} at {BAUD_RATE} baud.")

ani = animation.FuncAnimation(fig, update_plot, interval=200)
plt.show()