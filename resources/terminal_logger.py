import sys
class TeeLogger:
    def __init__(self, file):
        self.file = file
        self.terminal = sys.stdout  # Save the original stdout

    def write(self, message):
        self.terminal.write(message)  # Write to the terminal
        self.file.write(message)     # Write to the file

    def flush(self):
        self.terminal.flush()
        self.file.flush()

class InputLogger:
    def __init__(self, file):
        self.file = file
        self.original_stdin = sys.stdin  # Save the original stdin

    def readline(self):
        user_input = self.original_stdin.readline()  # Read input from terminal
        self.file.write(user_input)  # Log the input to the file
        self.file.flush()
        return user_input

    def __getattr__(self, attr):
        # Delegate attribute access to the original stdin
        return getattr(self.original_stdin, attr)