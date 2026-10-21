"""Put the OpenCV window fullscreen on a specific display (e.g. an AirPlay TV).

OpenCV's own fullscreen always uses the "main" screen, so on macOS we reach into AppKit
(same process, same NSApplication) and do it ourselves: the window itself is moved to the
target screen, sized to fill it, its title bar hidden, and raised above the menu bar.

(AppKit's enterFullScreenMode moves the window's view into a separate fullscreen window;
OpenCV keeps drawing into the original, now-empty window, so the TV just shows gray.
A borderless window would avoid that too, but macOS won't give it keyboard focus, which
would break the hotkeys.)
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


# AppKit constants (NSWindow.h), spelled out to keep the import list short
_FULL_SIZE_CONTENT_VIEW = 1 << 15
_TITLE_HIDDEN = 1
_ABOVE_MENU_BAR = 25  # NSMainMenuWindowLevel (24) + 1


def _allow_covering_menu_bar(window_class) -> None:
    """macOS shrinks windows so they stay below the menu bar (a 30 px strip of desktop at the
    top of the TV). Overriding constrainFrameRect:toScreen: on OpenCV's window class lets the
    window keep the full-screen frame we give it."""
    if getattr(window_class, "_partycam_unconstrained", False):
        return
    try:
        import objc  # type: ignore[import-not-found]

        def constrainFrameRect_toScreen_(self, rect, screen):
            return rect

        rect_type = b"{CGRect={CGPoint=dd}{CGSize=dd}}"
        objc.classAddMethods(
            window_class,
            [objc.selector(constrainFrameRect_toScreen_, selector=b"constrainFrameRect:toScreen:",
                           signature=rect_type + b"@:" + rect_type + b"@")],  # fmt: skip
        )
        window_class._partycam_unconstrained = True
    except Exception as e:
        print(f"[!] Couldn't let the window cover the menu bar ({e}); a thin strip may show at the top")


class FullscreenController:
    def __init__(self, window_title: str):
        self.window_title = window_title
        self.window = None
        self.saved = None  # (style mask, frame, level) to restore on exit
        self.screen_index: Optional[int] = None

    @property
    def is_fullscreen(self) -> bool:
        return self.saved is not None

    def _find_window(self):
        if self.window is None:
            for w in NSApplication.sharedApplication().windows():
                if w.title() == self.window_title:
                    self.window = w
                    _allow_covering_menu_bar(type(w))
                    break
        return self.window

    def _set_chrome_hidden(self, hidden: bool) -> None:
        w = self.window
        w.setTitlebarAppearsTransparent_(hidden)
        w.setTitleVisibility_(_TITLE_HIDDEN if hidden else 0)
        for button in range(3):  # close, minimize, zoom
            b = w.standardWindowButton_(button)
            if b is not None:
                b.setHidden_(hidden)

    def enter(self, screen_index: int) -> bool:
        if not HAVE_APPKIT:
            return False
        screens = NSScreen.screens()
        w = self._find_window()
        if w is None or not screens:
            return False
        screen_index %= len(screens)
        if self.saved is None:
            self.saved = (w.styleMask(), w.frame(), w.level())
            w.setStyleMask_(w.styleMask() | _FULL_SIZE_CONTENT_VIEW)
            self._set_chrome_hidden(True)
            w.setLevel_(_ABOVE_MENU_BAR)
        w.setFrame_display_(screens[screen_index].frame(), True)
        w.makeKeyAndOrderFront_(None)
        self.screen_index = screen_index
        return True

    def exit(self) -> None:
        if self.saved is None or self.window is None:
            return
        style, frame, level = self.saved
        self.saved = None
        self._set_chrome_hidden(False)
        self.window.setStyleMask_(style)
        self.window.setLevel_(level)
        self.window.setFrame_display_(frame, True)

    def cycle(self) -> Optional[int]:
        """Moves fullscreen to the next display. Returns the new display index."""
        n = len(screen_names())
        if n == 0:
            return None
        nxt = 0 if self.screen_index is None else (self.screen_index + 1) % n
        return nxt if self.enter(nxt) else None
