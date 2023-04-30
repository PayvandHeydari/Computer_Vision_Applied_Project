import subprocess
import time

def CLI_Launcher():
    cmds = [
        "wine yt-dlp.exe https://www.youtube.com/watch?v=fuuBpBQElv4",
        "wine yt-dlp.exe https://www.youtube.com/watch?v=ByED80IKdIU",
        "wine yt-dlp.exe https://www.youtube.com/watch?v=1EiC9bvVGnk",
        "wine yt-dlp.exe https://www.youtube.com/watch?v=5_XSYlAfJZM"
        #"wine yt-dlp.exe https://www.youtube.com/watch?v=1-iS7LArMPA"
    ]

    while True:
        xterms = []
        for cmd in cmds:
            xterm = subprocess.Popen(["xterm", "-e", cmd])
            xterms.append(xterm)

        time.sleep(23000) # 6 hours in seconds
        for xterm in xterms:
            xterm.terminate()

if __name__ == "__main__":
    CLI_Launcher()


# https://www.youtube.com/watch?v=1-iS7LArMPA - EarthCam Live: Times Square in 4K - 4K
# yt-dlp.exe https://www.youtube.com/watch?v=fuuBpBQElv4 - Sharx Security Demo Live Cam: rotary traffic circle Derry NH USA - 1440P
# yt-dlp.exe https://www.youtube.com/watch?v=ByED80IKdIU - downtown coldwater - 1080P
# https://www.youtube.com/watch?v=1EiC9bvVGnk - Jackson Hole Wyoming USA Town Square Live Cam - 1080 P
# https://www.youtube.com/watch?v=5_XSYlAfJZM - Village of Tilton - Traffic Camera - 720 P


# yt-dlp.exe https://www.youtube.com/watch?v=Uxe_A8IEpTw - not sure about this one