"""Poll whole-device VRAM via NVML; not just PyTorch allocator statistics."""
import threading
import time

class VramMonitor:
    def __init__(self, interval=.25):
        self.interval=interval; self.peak=None; self.start=None
        self._stop=threading.Event(); self.thread=None; self._nv=None; self._h=None
        try:
            import pynvml
            pynvml.nvmlInit()
            self._nv=pynvml
            self._h=pynvml.nvmlDeviceGetHandleByIndex(0)
        except Exception as e:
            print(f'[WARN] NVML unavailable ({e}); GPU VRAM will be null. Install nvidia-ml-py.')

    def used_mib(self):
        if not self._nv: return None
        try: return float(self._nv.nvmlDeviceGetMemoryInfo(self._h).used)/1024**2
        except Exception: return None

    def _poll(self):
        while not self._stop.is_set():
            val=self.used_mib()
            if val is not None: self.peak=max(val,self.peak or 0)
            self._stop.wait(self.interval)

    def __enter__(self):
        self.start=self.used_mib(); self.peak=self.start
        self.thread=threading.Thread(target=self._poll,daemon=True); self.thread.start()
        return self

    def __exit__(self,*args):
        self._stop.set(); self.thread.join(timeout=3)
        val=self.used_mib()
        if val is not None: self.peak=max(val,self.peak or 0)
        if self._nv:
            try: self._nv.nvmlShutdown()
            except Exception: pass
