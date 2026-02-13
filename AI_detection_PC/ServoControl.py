import lgpio
import asyncio

class ContinuousServo:
    """Servo control class using lgpio on RaspberryPi"""
    def __init__(self, chip, pin, min_us=1000, max_us=2000, freq=50):
        self.handle = lgpio.gpiochip_open(chip) # Open GPIO chip 0
        self.pin = pin
        self.min_us = min_us
        self.max_us = max_us
        self.freq = freq
        self.period_us = 1_000_000 / freq  # 20000 μs for 50 Hz
        
        self.stop_value = 90  # Value to stop (0-180)
        self.degrees_per_second = 360  # Rotation speed (determine by measurement!)
        
        lgpio.gpio_claim_output(self.handle, self.pin)

    def angle_to_pulse(self, angle: int):
        angle = max(0, min(180, angle))
        return self.min_us + (angle / 180) * (self.max_us - self.min_us)
        
    def pulse_to_duty_cycle(self, pulse_us: float):
        """Converts microseconds to duty cycle in percent"""
        return (pulse_us / self.period_us) * 100
    
    def set_speed(self, speed: int):
        """
        Sets speed and direction
        0 = max speed backward
        90 = STOP
        180 = max speed forward
        """
        pulse_us = self.angle_to_pulse(speed)
        duty_cycle = self.pulse_to_duty_cycle(pulse_us)
        lgpio.tx_pwm(self.handle, self.pin, self.freq, duty_cycle)
    
    def stop(self):
        """Stop servo"""
        lgpio.tx_pwm(self.handle, self.pin, 0, 0)
    
    async def rotate_degrees(self, degrees: float, speed_percent=50, direction=None):
        """
        Rotates servo by given degrees
        degrees: positive = forward, negative = backward
        speed_percent: speed 0-100%    
        WARNING: This is an ESTIMATE based on time!
        Actual precision depends on:
        - Calibration of degrees_per_second
        - Servo load
        - Battery voltage
        """
        if degrees == 0: return  
        # Calculate rotation time
        time_needed = abs(degrees) / self.degrees_per_second
        # Set direction and speed
        if degrees > 0:
            # Forward (90-180)
            speed_value = 90 + (speed_percent / 100) * 90
        else:
            # Backward (0-90)
            speed_value = 90 - (speed_percent / 100) * 90
        
        # Start rotation
        self.set_speed(int(speed_value))
        #sleep(time_needed)
        await asyncio.sleep(time_needed)
        self.stop()
        return speed_value  # Returns last set speed for info

    def cleanup(self):
        self.stop()
        lgpio.gpiochip_close(self.handle)
