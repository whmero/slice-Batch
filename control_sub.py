import subprocess
import threading

def run_anaconda_prompt(commands):
    """
    Runs multiple commands in a new Anaconda Prompt window.
    :param commands: A list of commands to execute inside Anaconda Prompt.
    """
    try:
        command_string = " & ".join(commands)
        subprocess.Popen(f'start cmd.exe /K "C:\\Users\\whmer\\anaconda3\\Scripts\\activate.bat ti & {command_string}"', shell=True)
    except Exception as e:
        print(f"Error: {e}")

# Example usage
if __name__ == "__main__":
    threads = []
    
    for i in range(8):
        commands = ["cd C:\\Users\\whmer\\Desktop\\usual", f"python tclient.usual.py Cww{i+1} swin 3"]
        thread = threading.Thread(target=run_anaconda_prompt, args=(commands,))
        thread.start()
        threads.append(thread)
    
    # Optionally, wait for all threads to finish
    for thread in threads:
        thread.join()


# import subprocess

# def run_anaconda_prompt(commands):
#     """
#     Runs multiple commands in a new Anaconda Prompt window.
#     :param commands: A list of commands to execute inside Anaconda Prompt.
#     """
#     try:
#         command_string = " & ".join(commands)
#         subprocess.Popen(f'start cmd.exe /K "C:\\Users\\whmer\\anaconda3\\Scripts\\activate.bat py39 &{command_string}"', shell=True)
#     except Exception as e:
#         print(f"Error: {e}")

# # Example usage
# if __name__ == "__main__":


#     for i in range(16):
#         # print(i+1)
#         run_anaconda_prompt(["cd C:\\Users\\whmer\\OneDrive\\Desktop\\usual", f"python tclient.usual.py CC{i+1} vit 3"])
