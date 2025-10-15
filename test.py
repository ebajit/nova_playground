import time
import board
import busio
from adafruit_bno055 import BNO055_I2C
from adafruit_servokit import ServoKit
from adafruit_motor import servo

# Initialize I2C and BNO055 sensor
i2c = busio.I2C(board.SCL, board.SDA)
sensor = BNO055_I2C(i2c)

# Initialize ServoKit for 16 channels
kit = ServoKit(channels=16)

# Replace standard servo with continuous rotation servo on channel 15
continuous_servo = servo.ContinuousServo(kit._pca.channels[15])

try:
    print("Starting continuous rotation...")
    continuous_servo.throttle = 1.0  # Full speed forward
    while True:
        time.sleep(0.1)

except KeyboardInterrupt:
    print("Stopping...")
    continuous_servo.throttle = 0  # Stop the motor