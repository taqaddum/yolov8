import functools
import os
import signal
from multiprocessing.shared_memory import ShareableList
from typing import List

from ultralytics import YOLO
from ultralytics.engine.results import Results

MODEL = YOLO("glass-detection/logs/weights/best.pt")
DIR = "/home/jhzou/yolov8/runs/detect"


def handler(sig, frame, share: ShareableList):
    try:
        if sig in {signal.SIGINT, signal.SIGTERM, signal.SIGABRT}:
            share.shm.close()
            share.shm.unlink()
            os._exit(0)

        print(share, "模型正在预测...")
        results: List[Results] = MODEL.predict(share[1], device=0)
        path = os.path.join(DIR, os.path.basename(share[1]))

        if path and path != "":
            results[0].save(path)
            share[3] = path
        else:
            share[3] = ""

        os.kill(share[0], signal.SIGUSR2)
    except Exception as e:
        # 处理异常，避免程序因为异常而退出
        print(e)


if __name__ == "__main__":
    share = ShareableList([-1, "#" * 100, os.getpid(), "#" * 100], name="yolo")

    print(share, "模型已启动，等待命令中...")

    handler_ext = functools.partial(handler, share=share)
    signal.signal(signal.SIGUSR1, handler_ext)
    signal.signal(signal.SIGINT, handler_ext)

    while True:
        signal.pause()
