"""Bring a window that this (background) server opened to the front.

Windows does not let a background process take the focus, so a folder or a file
opened from the GUI used to appear *behind* the app window. After launching it we
wait for its window (matched by title) and activate it the way Windows allows:
attach to the input of the current foreground window, or as a fallback tap Alt.
"""
import ctypes
import threading
import time
from ctypes import wintypes

SW_RESTORE = 9
VK_MENU = 0x12
KEYEVENTF_KEYUP = 0x2


def _user32():
    return ctypes.windll.user32


def _windows():
    """Visible top-level windows as (hwnd, title), topmost first."""
    user32, out = _user32(), []

    @ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    def collect(hwnd, _):
        if user32.IsWindowVisible(hwnd):
            n = user32.GetWindowTextLengthW(hwnd)
            if n:
                buf = ctypes.create_unicode_buffer(n + 1)
                user32.GetWindowTextW(hwnd, buf, n + 1)
                out.append((hwnd, buf.value))
        return True

    user32.EnumWindows(collect, 0)
    return out


def activate(hwnd):
    user32, kernel32 = _user32(), ctypes.windll.kernel32
    if user32.IsIconic(hwnd):
        user32.ShowWindow(hwnd, SW_RESTORE)
    fg = user32.GetForegroundWindow()
    me = kernel32.GetCurrentThreadId()
    other = user32.GetWindowThreadProcessId(fg, None) if fg else 0
    attached = bool(other and other != me and user32.AttachThreadInput(me, other, True))
    try:
        user32.BringWindowToTop(hwnd)
        user32.SetForegroundWindow(hwnd)
    finally:
        if attached:
            user32.AttachThreadInput(me, other, False)
    if user32.GetForegroundWindow() != hwnd:
        user32.keybd_event(VK_MENU, 0, 0, 0)               # an Alt tap also unlocks SetForegroundWindow
        user32.keybd_event(VK_MENU, 0, KEYEVENTF_KEYUP, 0)
        user32.SetForegroundWindow(hwnd)


def open_in_front(launch, title_part, timeout=6.0):
    """Run launch(), then bring the window whose title contains title_part to the front.

    A new window is preferred; if none appears within a second (Explorer reuses a
    window that already shows that folder), the existing one is used.
    """
    try:
        before = {h for h, _ in _windows()}
    except Exception:
        before = set()
    launch()
    needle = title_part.lower()

    def wait():
        start = time.monotonic()
        while time.monotonic() - start < timeout:
            time.sleep(0.15)
            try:
                matches = [h for h, t in _windows() if needle in t.lower()]
                new = [h for h in matches if h not in before]
                pick = new[0] if new else (matches[0] if matches and time.monotonic() - start > 1.0 else None)
                if pick:
                    activate(pick)
                    return
            except Exception:
                return

    threading.Thread(target=wait, daemon=True).start()
