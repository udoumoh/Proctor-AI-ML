import subprocess

command = "build/x64/Release/OpenPoseDemo.exe --image_dir C:/Users/JOHN/Desktop/examMalpracticeDetection/frames/cheating --display 0 --render_pose 0 --write_json C:/Users/JOHN/Desktop/examMalpracticeDetection/frames/cheatingJson" # replace this with the command you want to run on your shell
process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell = False)
output, error = process.communicate()

if process.returncode != 0:
    print(f"Error running command: {error}")
else:
    print(f"Output: {output.decode('utf-8')}")