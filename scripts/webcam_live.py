"""
webcam_live.py — LIVE on-screen webcam detection in a real window.

Uses Tkinter + Pillow for the display (both already available) so it needs NO GUI
OpenCV build — capture + inference run on the pinned opencv-python-headless, and
frames are shown in a Tk window. Stays within the project's secure/pinned deps.

  ./.venv/Scripts/python.exe scripts/webcam_live.py \
      --weights dist/detector.safetensors --num-classes 10 --imgsz 768 --score 0.3

Close the window (or press q / Esc) to stop. Optional --record out.mp4 saves it too.

NOTE: this model was trained on VisDrone AERIAL imagery, so a normal front-facing
webcam scene is out-of-domain — expect few/no detections. Lower --score to see more.
"""
from __future__ import annotations

import argparse
import time
import tkinter as tk
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageTk

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.model import CustomDetector
from src.model.postprocess import detections_from_outputs
from src.secure_io import load_model
from src.train import pick_device

VISDRONE = {0: "pedestrian", 1: "people", 2: "bicycle", 3: "car", 4: "van",
            5: "truck", 6: "tricycle", 7: "awn-tricycle", 8: "bus", 9: "motor"}


def draw(frame, boxes, scores, labels, imgsz):
    h, w = frame.shape[:2]
    sx, sy = w / imgsz, h / imgsz
    for (x1, y1, x2, y2), s, c in zip(boxes.tolist(), scores.tolist(), labels.tolist()):
        p1 = (int(x1 * sx), int(y1 * sy)); p2 = (int(x2 * sx), int(y2 * sy))
        cv2.rectangle(frame, p1, p2, (0, 255, 0), 2)
        name = VISDRONE.get(int(c), str(int(c)))
        cv2.putText(frame, f"{name}:{s:.2f}", (p1[0], max(p1[1] - 4, 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
    return frame


class LiveApp:
    def __init__(self, args):
        self.args = args
        self.device = pick_device()
        print(f"device: {self.device}")
        self.model = CustomDetector(args.num_classes).to(self.device).eval()
        self.model.load_state_dict(load_model(args.weights, args.verify_key, self.device),
                                   strict=False)
        print(f"loaded weights (signature verified): {args.weights}")

        self.cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            raise SystemExit(
                f"cannot open camera {args.camera}. Close other apps using it, and allow "
                "camera access for desktop apps in Windows Privacy settings.")

        self.writer = None
        if args.record:
            w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
            h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
            self.writer = cv2.VideoWriter(str(args.record),
                                          cv2.VideoWriter_fourcc(*"mp4v"), 15.0, (w, h))

        self.root = tk.Tk()
        self.root.title("Live detection — q / Esc to quit")
        self.label = tk.Label(self.root)
        self.label.pack()
        self.root.bind("<q>", lambda e: self.stop())
        self.root.bind("<Escape>", lambda e: self.stop())
        self.root.protocol("WM_DELETE_WINDOW", self.stop)

        self.n, self.t0, self.running, self._imgtk = 0, time.time(), True, None
        self.update()

    @torch.no_grad()
    def update(self):
        if not self.running:
            return
        ok, frame = self.cap.read()
        if not ok:
            print("frame grab failed; stopping."); self.stop(); return

        resized = cv2.resize(frame, (self.args.imgsz, self.args.imgsz))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        t = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(self.device)
        boxes, scores, labels = detections_from_outputs(
            self.model(t), self.args.num_classes, self.args.imgsz, self.args.score)

        frame = draw(frame, boxes.cpu(), scores.cpu(), labels.cpu(), self.args.imgsz)
        self.n += 1
        fps = self.n / max(time.time() - self.t0, 1e-6)
        cv2.putText(frame, f"{len(boxes)} det   {fps:.1f} FPS", (8, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
        if self.writer is not None:
            self.writer.write(frame)

        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        self._imgtk = ImageTk.PhotoImage(image=img)      # keep a ref (avoid GC)
        self.label.configure(image=self._imgtk)
        self.root.after(1, self.update)

    def stop(self):
        self.running = False
        try: self.cap.release()
        except Exception: pass
        if self.writer is not None:
            self.writer.release()
            print(f"saved recording -> {self.args.record}")
        try: self.root.destroy()
        except Exception: pass

    def run(self):
        self.root.mainloop()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=Path, default=Path("dist/detector.safetensors"))
    ap.add_argument("--num-classes", type=int, default=10)
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--imgsz", type=int, default=768)
    ap.add_argument("--score", type=float, default=0.3)
    ap.add_argument("--record", type=Path, default=None, help="optional: also save an .mp4")
    ap.add_argument("--verify-key", type=Path, default=Path("keys/model_ed25519.pub"))
    args = ap.parse_args()
    LiveApp(args).run()


if __name__ == "__main__":
    main()
