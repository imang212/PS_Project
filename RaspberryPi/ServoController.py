import sys
import time
import lgpio

class ServoController:
    def __init__(self, gpio_pin):
        self.gpio_pin = gpio_pin
        self.handle = lgpio.gpiochip_open(0)
        lgpio.gpio_claim_output(self.handle, gpio_pin, 1500)  # Set initial position to 1500us

    def set_angle(self, angle):
        # Convert angle (0-180) to pulse width (500-2500us)
        pulse_width = int(500 + (angle / 180.0) * 2000)
        lgpio.gpio_write(self.handle, self.gpio_pin, pulse_width)

    def cleanup(self):
        lgpio.gpio_free(self.handle, self.gpio_pin)
        lgpio.gpiochip_close(self.handle)

class ServoTest:
    def __init__(self, servo: ServoController):
        self.servo = servo

    def run_test(self):
        try:
            for angle in range(0, 181, 90):
                print(f"Setting angle to {angle}")
                self.servo.set_angle(angle)
                time.sleep(0.5)
            for angle in range(180, -1, -90):
                print(f"Setting angle to {angle}")
                self.servo.set_angle(angle)
                time.sleep(0.5)
        except KeyboardInterrupt:
            pass
        finally:
            self.servo.cleanup()

if __name__ == "__main__":
    print(sys.argv)
    pin = sys.argv[1] if len(sys.argv) > 1 else "17"
    servo = ServoController(gpio_pin=int(pin))
    test = ServoTest(servo)
    test.run_test()