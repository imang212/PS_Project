import time
import lgpio

CHIP = 0           # hlavní GPIO čip
PIN = 18           # GPIO18 = fyzický pin 12
FREQ = 50          # 50 Hz pro servo

# Otevři GPIO čip
h = lgpio.gpiochip_open(CHIP)

# Nastav pin jako PWM
lgpio.gpio_claim_output(h, PIN)
lgpio.tx_pwm(h, PIN, FREQ, 7.5)   # střední pozice (90°)
time.sleep(1)

# 0° pozice
lgpio.tx_pwm(h, PIN, FREQ, 5)
time.sleep(1)

# 180° pozice
lgpio.tx_pwm(h, PIN, FREQ, 10)
time.sleep(1)

# Zastavení
lgpio.tx_pwm(h, PIN, FREQ, 0)
lgpio.gpiochip_close(h)
print("Hotovo.")
