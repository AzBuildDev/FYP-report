"""等待用户按回车开始；窗口停止、Esc 和 Ctrl+C 可取消，不阻塞资源释放。"""


def wait_for_start(stop, read_key=None, poll_interval=0.05):
    if poll_interval <= 0:
        raise ValueError("轮询间隔必须为正数")
    if read_key is None:
        import msvcrt

        def read_key():
            return msvcrt.getwch() if msvcrt.kbhit() else None

    while not stop.is_set():
        key = read_key()
        if key == "\x03":
            raise KeyboardInterrupt
        if key == "\x1b":
            return False
        if key in ("\r", "\n"):
            return not stop.is_set()
        stop.wait(poll_interval)
    return False
