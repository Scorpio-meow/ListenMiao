import sys
import types
from types import SimpleNamespace

import pytest

from app import __main__ as main_module
from app import tray as tray_module


class _Icon:
    def __init__(self, name, image, title, menu) -> None:
        self.title = title
        self.menu = menu
        self.calls = []

    def run(self) -> None:
        self.calls.append("run")

    def stop(self) -> None:
        self.calls.append("stop")


@pytest.fixture
def pystray(monkeypatch):
    module = types.ModuleType("pystray")
    module.Icon = _Icon
    module.Menu = lambda *items: list(items)
    module.Menu.SEPARATOR = None
    module.MenuItem = lambda text, action, default=False: SimpleNamespace(text=text, action=action, default=default)
    image = types.ModuleType("PIL.Image")
    image.open = lambda path: path
    pil = types.ModuleType("PIL")
    pil.Image = image
    monkeypatch.setitem(sys.modules, "pystray", module)
    monkeypatch.setitem(sys.modules, "PIL", pil)
    monkeypatch.setitem(sys.modules, "PIL.Image", image)
    return module


def test_menu_opens_the_page_by_default_and_quit_stops_the_service(pystray, monkeypatch, tmp_path):
    opened = []
    quits = []
    monkeypatch.setattr(tray_module.webbrowser, "open", opened.append)
    tray = tray_module.Tray("http://127.0.0.1:7860", tmp_path, lambda: quits.append("quit"))
    items = {item.text: item for item in tray._icon.menu if item is not None}
    assert [item.text for item in items.values() if item.default] == ["開啟 ListenMiao"]
    items["開啟 ListenMiao"].action()
    items["結束 ListenMiao"].action()
    assert opened == ["http://127.0.0.1:7860"]
    assert quits == ["quit"]
    assert "http://127.0.0.1:7860" in tray._icon.title
    tray.start()
    tray.stop()
    assert sorted(tray._icon.calls) == ["run", "stop"]
    assert not tray._thread.is_alive()


def test_icon_file_is_shipped():
    assert tray_module.ICON.is_file()


def test_quitting_from_the_tray_asks_the_server_to_exit(pystray, monkeypatch, tmp_path):
    monkeypatch.setenv("ASR_HOST", "127.0.0.1")
    monkeypatch.setenv("ASR_PORT", "7860")
    server = SimpleNamespace(should_exit=False)
    tray = main_module._start_tray(tmp_path, server)
    next(item for item in tray._icon.menu if item is not None and item.text == "結束 ListenMiao").action()
    assert server.should_exit is True


def test_tray_is_refused_outside_windows(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["app", "--tray"])
    monkeypatch.setattr(main_module.os, "name", "posix")
    with pytest.raises(SystemExit, match="只能在 Windows"):
        main_module.main()
