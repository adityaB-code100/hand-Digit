import random

def generate_captcha_digit():
    """Generates a random digit (0-9) for the user to draw."""
    return random.randint(0, 9)
