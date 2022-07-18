import ctypes
import inspect
import threading
import time

class LoopThread(threading.Thread):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.keep_alive = False

    def run(self):
        self.keep_alive = True
        try:
            if self._target:
                while self.keep_alive:
                    self._target(*self._args, **self._kwargs)
        finally:
            del self._target, self._args, self._kwargs

    def stop(self):
        self.keep_alive = False


class ReturnThread(threading.Thread):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.result = None

    def run(self):
        try:
            if self._target:
                self.result = self._target(*self._args, **self._kwargs)
        finally:
            del self._target, self._args, self._kwargs

    def get_result(self, timeout=None):
        self.join(timeout=timeout)
        return self.result

class IntervalThread(threading.Thread):

    def __init__(self, interval, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.interval = interval

    def run(self):
        self.keep_alive = True
        try:
            if self._target:
                while self.keep_alive:
                    start = time.time()
                    self._target(*self._args, **self._kwargs)
                    end = time.time()
                    cost = end - start
                    if cost < self.interval:
                        time.sleep(self.interval - cost)
        finally:
            del self._target, self._args, self._kwargs
    
    def stop(self):
        self.keep_alive = False


def _async_raise(tid, exctype):
    """raises the exception, performs cleanup if needed"""
    tid = ctypes.c_long(tid)
    if not inspect.isclass(exctype):
        exctype = type(exctype)
    res = ctypes.pythonapi.PyThreadState_SetAsyncExc(
        tid, ctypes.py_object(exctype))
    if res in [0, 1]:
        return res
    else:
        # """if it returns a number greater than one, you're in trouble,
        # and you should call it again with exc=NULL to revert the effect"""
        ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, None)
        raise SystemError("PyThreadState_SetAsyncExc failed")


def stop_thread(thread):
    for i in range(10):
        res = _async_raise(thread.ident, SystemExit)
        print('number of thread killed:{}, ident={}, is_alive={}'.format(
            res, thread.ident, thread.is_alive()))
        if res == 1 or not thread.is_alive():
            return
