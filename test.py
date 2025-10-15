import time
from adafruit_bno055 import BNO055_I2C
import board
import busio
from adafruit_servokit import ServoKit

# Initialize I2C
i2c = busio.I2C(board.SCL, board.SDA)

# Initialize BNO055 sensor
sensor = BNO055_I2C(i2c)

# Initialize ServoKit for 16 channels
kit = ServoKit(channels=16)

# Helper function to safely map angles to servo range
def angle_to_servo_position(angle):
    # Clamp angle between -90 and 90
    angle = max(min(angle, 90), -90)
    # Map to 0-180 degrees
    return int((angle + 90) * (180 / 180))

try:
    while True:
        # Get euler angles (heading, roll, pitch)
        euler = sensor.euler

        if euler is not None:
            heading, roll, pitch = euler

            # Convert pitch and roll to servo angles
            servo_0_angle = angle_to_servo_position(pitch)

            # Write to servos
            kit.servo[15].angle = servo_0_angle
            print(f"Pitch: {pitch:.2f}°, Roll: {roll:.2f}° → Servo0: {servo_0_angle}")
        else:
            print("Waiting for sensor data...")

        time.sleep(0.1)

except KeyboardInterrupt:
    print("Exiting program.")