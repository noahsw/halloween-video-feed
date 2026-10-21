"""Put the OpenCV window fullscreen on a specific display (e.g. an AirPlay TV).

OpenCV's own fullscreen always uses the "main" screen, so on macOS we reach into
AppKit (same process, same NSApplication) and do it ourselves.
"""

import sys
from typing import List, Optional

try:
    from AppKit import NSApplication, NSScreen  # type: ignore[import-not-found]

    HAVE_APPKIT = sys.platform == "darwin"
except ImportError:  # not macOS
    HAVE_APPKIT = False


def screen_names() -> List[str]:
    if not HAVE_APPKIT:
        return []
    names = []
    for s in NSScreen.screens():
        size = s.frame().size
        name = s.localizedName() if hasattr(s, "localizedName") else "Display"
        names.append(f"{name} ({int(size.width)}x{int(size.height)})")
    return names


def find_screen(query: str) -> Optional[int]:
    """Matches a display by index ('1') or by name substring ('living room')."""
    names = screen_names()
    if query.isdigit():
        idx = int(query)
        return idx if 0 <= idx < len(names) else None
    for i, n in enumerate(names):
        if query.lower() in n.lower():
            return i
    return None


class FullscreenController:
    def __init__(self, window_title: str):
        self.window_title = window_title
        self.view = None
        self.screen_index: Optional[int] = None

    @property
    def is_fullscreen(self) -> bool:
        return self.view is not None and bool(self.view.isInFullScreenMode())

    def _find_view(self):
        if self.view is not None:
            return self.view
        for w in NSApplication.sharedApplication().windows():
            if w.title() == self.window_title:
                self.view = w.contentView()
                return self.view
        return None

    def enter(self, screen_index: int) -> bool:
        if not HAVE_APPKIT:
            return False
        screens = NSScreen.screens()
        view = self._find_view()
        if view is None or not screens:
            return False
        if view.isInFullScreenMode():
            view.exitFullScreenModeWithOptions_(None)
        screen_index %= len(screens)
        view.enterFullScreenMode_withOptions_(screens[screen_index], {"NSFullScreenModeAllScreens": False})
        self.screen_index = screen_index
        return True

    def exit(self) -> None:
        if self.is_fullscreen:
            self.view.exitFullScreenModeWithOptions_(None)

    def cycle(self) -> Optional[int]:
        """Moves fullscreen to the next display. Returns the new display index."""
        n = len(screen_names())
        if n == 0:
            return None
        nxt = 0 if self.screen_index is None else (self.screen_index + 1) % n
        return nxt if self.enter(nxt) else None
