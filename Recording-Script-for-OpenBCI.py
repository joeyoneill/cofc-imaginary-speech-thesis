’’’
Automation script for recording EEG Data with OpenBCI Software
REQUIREMENTS:
- OpenBCI Software is open, and dongle is connected
- Music application is open, and beep file is set to beginning
LAYOUT ORDER FOR TASKBAR:
- Google Chrome
- File Explorer
- Anaconda Prompt
- OpenBCI Software
- Music Tab
- (Anything Else)
Start from opened OpenBCI software + opened Anaconda prompt
’’’

# REQUIRED PACKAGES
import pyautogui
import time

# OPEN MUSIC TAB
pyautogui.click(710, 1060)
print(“opened music”)

# PLAY RYTHMIC BEEPS
pyautogui.click(955, 982)
print(“started sound”)

# OPEN OPENBCI SOFTWARE
pyautogui.click(661, 1059)
print(“opened OpenBCI”)

# START RECORDING
pyautogui.press(‘space’)
print(“started recording”)

# WAIT 30 SECONDS FOR RECORDING
time.sleep(30)

# END RECORDING
pyautogui.press(‘space’)
print(“recording ended”)

# OPENS ANACONDA PROMPT BACK UP AT END OF AUTOMATION
pyautogui.click(611, 1059)

# PRESSES UP IN ANACONDA TO GET BACK TO SCRIPT RUN
pyautogui.press(‘up’)
