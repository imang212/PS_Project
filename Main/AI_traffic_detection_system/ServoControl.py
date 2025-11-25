import lgpio
import asyncio

class ContinuousServo:
    """Servo control class using lgpio"""
    def __init__(self, chip, pin, min_us=1000, max_us=2000, freq=50):
        self.handle = lgpio.gpiochip_open(chip) # Open GPIO chip 0
        self.pin = pin
        self.min_us = min_us
        self.max_us = max_us
        self.freq = freq
        self.period_us = 1_000_000 / freq  # 20000 μs pro 50 Hz
        
        self.stop_value = 90  # Hodnota pro zastavení (0-180)
        self.degrees_per_second = 360  # Rychlost otáčení (zjistit měřením!)
        
        lgpio.gpio_claim_output(self.handle, self.pin)

    def angle_to_pulse(self, angle: int):
        angle = max(0, min(180, angle))
        return self.min_us + (angle / 180) * (self.max_us - self.min_us)
        
    def pulse_to_duty_cycle(self, pulse_us: float):
        """Převede mikrosekundy na duty cycle v procentech"""
        return (pulse_us / self.period_us) * 100
    
    def set_speed(self, speed: int):
        """
        Nastaví rychlost a směr
        0 = max rychlost zpět
        90 = STOP
        180 = max rychlost vpřed
        """
        pulse_us = self.angle_to_pulse(speed)
        duty_cycle = self.pulse_to_duty_cycle(pulse_us)
        lgpio.tx_pwm(self.handle, self.pin, self.freq, duty_cycle)
    
    def stop(self):
        """Zastav servo"""
        lgpio.tx_pwm(self.handle, self.pin, 0, 0)
    
    async def rotate_degrees(self, degrees: float, speed_percent=50, direction=None):
        """
        Otočí servo o daný počet stupňů
        degrees: kladné = vpřed, záporné = vzad
        speed_percent: rychlost 0-100%    
        POZOR: Toto je ODHAD založený na času!
        Skutečná přesnost závisí na:
        - Kalibraci degrees_per_second
        - Zatížení serva
        - Napětí baterie
        """
        if degrees == 0: return  
        # Vypočti čas otáčení
        time_needed = abs(degrees) / self.degrees_per_second
        # Nastav směr a rychlost
        if degrees > 0:
            # Dopředu (90-180)
            speed_value = 90 + (speed_percent / 100) * 90
        else:
            # Dozadu (0-90)
            speed_value = 90 - (speed_percent / 100) * 90
        
        # Start otáčení
        self.set_speed(int(speed_value))
        #sleep(time_needed)
        await asyncio.sleep(time_needed)
        self.stop()
        return speed_value  # Vrátí poslední nastavenou rychlost pro info

    def cleanup(self):
        self.stop()
        lgpio.gpiochip_close(self.handle)
