import random

def generate_captcha_character(alphanumeric=False):
    """
    Generates a random character to draw:
    - If alphanumeric is False: returns a random digit '0'-'9'.
    - If alphanumeric is True: returns a random character among all 62 classes (0-9, A-Z, a-z).
    """
    if not alphanumeric:
        return str(random.randint(0, 9))
    
    # 62 classes: '0'-'9', 'A'-'Z', 'a'-'z'
    classes = [str(i) for i in range(10)]
    classes += [chr(ord('A') + i) for i in range(26)]
    classes += [chr(ord('a') + i) for i in range(26)]
    
    return random.choice(classes)

def generate_captcha_digit():
    """Generates a random digit (0-9) for backward compatibility."""
    return generate_captcha_character(alphanumeric=False)

