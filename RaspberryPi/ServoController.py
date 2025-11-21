import sys
import time
import lgpio

class ServoController:
    def __init__(self, gpio_pin=18):
        self.gpio_pin = 18  # GPIO18 = physical pin 12
        self.chip = lgpio.gpiochip_open(0)  # main GPIO chip
        self.position = 0
        self.handle = lgpio.gpio_claim_output(self.chip, self.gpio_pin)
        print(f"Initialized ServoController on GPIO pin {self.gpio_pin}.")

    def set_position(self, frequency, position):
        lgpio.tx_pwm(self.chip, self.gpio_pin, frequency, self._position_to_duty_cycle(position))
        self.position = position
        print(f"Set servo to position {position}° with frequency {frequency}Hz.")

    def _position_to_duty_cycle(self, position):
        return 10 + (position / 180) * 10
    
    def stop(self):
        lgpio.tx_pwm(self.chip, self.gpio_pin, self.frequency)
        print("Stopped PWM signal to servo.")
    
    def close(self):
        lgpio.gpiochip_close(self.chip)
        print("Closed GPIO chip.")
    
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, tb):
        # best-effort stop of PWM, then close resources
        try:
            if getattr(self, "chip", None) is not None and getattr(self, "gpio_pin", None) is not None:
                try:
                    lgpio.tx_pwm(self.chip, self.gpio_pin, 0, 0)
                except Exception:
                    pass
        finally:
            try:
                self.close()
            except Exception:
                pass
        return False

    def __del__(self):
        # ensure resources are released on deletion (best-effort, suppress errors during interpreter shutdown)
        try:
            if getattr(self, "chip", None) is not None and getattr(self, "gpio_pin", None) is not None:
                try:
                    lgpio.tx_pwm(self.chip, self.gpio_pin, 0, 0)
                except Exception:
                    pass
                try:
                    self.close()
                except Exception:
                    pass
        except Exception:
            pass

if __name__ == "__main__":
    controller = ServoController()
    controller.set_position(50, 0)  # middle position
    time.sleep(1)
    controller.set_position(50, 50)  # middle position
    time.sleep(1)
