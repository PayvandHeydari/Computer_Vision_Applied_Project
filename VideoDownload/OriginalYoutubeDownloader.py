import time
import subprocess
import pyautogui


# Automated youtube downloader


def runYTDownload():
    
    # Launch Command Line Terminal
    o = subprocess.Popen(["start", "cmd", "/k"],
                            shell=True)
    time.sleep(1)

    # List of Youtube Links 
    videoURLs=["https://www.youtube.com/watch?v=fuuBpBQElv4", 
    "https://www.youtube.com/watch?v=1EiC9bvVGnk", 
    "https://www.youtube.com/watch?v=5_XSYlAfJZM"]

    for url in videoURLs:

        time.sleep(5)
        pyautogui.typewrite(r"cd C:\Users\14087\Desktop\Python_Personal_Projects\AppliedProject")
        pyautogui.hotkey("enter")
        pyautogui.typewrite(f"yt-dlp.exe {url}")
        pyautogui.hotkey("enter")

        time.sleep(10)
        pyautogui.hotkey('ctrl', 'c')

    o.kill()

runYTDownload()

#test https://www.youtube.com/watch?v=5_XSYlAfJZM
