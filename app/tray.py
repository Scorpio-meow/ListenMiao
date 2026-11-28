from __future__ import annotations

import logging
import os
import threading
import webbrowser
from collections.abc import Callable
from pathlib import Path

log = logging.getLogger(__name__)

ICON = Path(__file__).resolve().parent.parent / "scripts" / "listenmiao.ico"
_NAME = "ListenMiao"
_STOP_TIMEOUT_SECONDS = 5


class Tray:
    def __init__(self, url: str, logs_dir: Path, on_quit: Callable[[], None]) -> None:
        import pystray
        from PIL import Image

        self._icon = pystray.Icon(
            _NAME,
            Image.open(ICON),
            f"{_NAME}：{url}",
            pystray.Menu(
                pystray.MenuItem("開啟 ListenMiao", lambda: webbrowser.open(url), default=True),
                pystray.MenuItem("開啟日誌資料夾", lambda: os.startfile(logs_dir)),
                pystray.Menu.SEPARATOR,
                pystray.MenuItem("結束 ListenMiao", self._quit),
            ),
        )
        self._on_quit = on_quit
        self._thread = threading.Thread(target=self._icon.run, name="tray", daemon=True)

    def _quit(self) -> None:
        log.info("從系統匣選擇結束，開始正常關閉服務")
        self._on_quit()

    def start(self) -> None:
        self._thread.start()
        log.info("系統匣圖示已顯示；Windows 11 可能把它收在工作列的「顯示隱藏的圖示」裡")

    def stop(self) -> None:
        self._icon.stop()
        self._thread.join(_STOP_TIMEOUT_SECONDS)
        if self._thread.is_alive():
            log.warning("系統匣圖示 %d 秒內沒有收起來，工作列上可能留下殘影，滑鼠移過去就會消失", _STOP_TIMEOUT_SECONDS)
