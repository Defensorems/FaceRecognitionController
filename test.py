import subprocess
    
result = subprocess.run(['./blink'], capture_output=True, text=True)
