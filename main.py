import functools
import os
from multiprocessing import Lock
from multiprocessing.shared_memory import ShareableList
from signal import SIGINT, SIGUSR1, SIGUSR2, signal, sigwait
from typing import List

from ultralytics import YOLO
from ultralytics.engine.results import Results

MODEL = YOLO("glass-detection/logs/weights/best.pt")
DIR = "/home/jhzou/yolov8/runs/detect"
LOCK = Lock()


def handler(sig, frame, share: ShareableList):
    try:
        print("模型正在预测...")
        results: List[Results] = MODEL.predict(share[1], device="cpu")
        path = os.path.join(DIR, os.path.basename(share[1]))

        if path and path != "":
            results[0].save(path)
            share[3] = path
        else:
            share[3] = ""

        os.kill(share[0], SIGUSR2)
    except Exception as e:
        # 处理异常，避免程序因为异常而退出
        print("Exception occurred:", e)


if __name__ == "__main__":
    share = ShareableList([0, "#" * 100, os.getpid(), "#" * 100], name="yolo")
    handler_ext = functools.partial(handler, share=share)
    print("模型已启动，等待命令中...")

    signal(SIGUSR1, handler_ext)

    try:
        while True:
            signum = sigwait({SIGUSR1, SIGINT})
            if signum in {SIGINT}:
                break
    except KeyboardInterrupt as e:
        print("\n", "正在释放内存...")
        share.shm.close()
        share.shm.unlink()
