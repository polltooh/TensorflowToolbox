from termcolor import colored

def color_message(message, color='white'):
    message = colored(message, color)
    return message

def print_color(message, color='white'):
    message = color_message(message, color)
    print(message)
