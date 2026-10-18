import hashlib
import tkinter as tk

import pytest

import launcher
from launcher import ApiConfigDialog, MadneApp
from madne.image.mock import render_placeholder_png
from madne.models import (
    Branch,
    DialogueLine,
    DialogueScript,
    GameOutput,
    Illustration,
    SceneOutline,
)
from madne.user_config import DEFAULT_IMG_BASE_URL, DEFAULT_IMG_MODEL


def make_output() -> GameOutput:
    return GameOutput(
        scene=SceneOutline(
            scene_id="scene_gui",
            summary="GUI 测试场景",
            setting="GUI 测试环境",
        ),
        script=DialogueScript(
            scene_id="scene_gui",
            lines=[DialogueLine(speaker="林默", text="这是测试台词。")],
        ),
        choices=[
            Branch(
                branch_id=f"G{index}",
                description=f"测试选项 {index}",
            )
            for index in range(1, 4)
        ],
        world_hash=hashlib.sha256(b"gui-test").hexdigest(),
        session_id="gui-test",
    )


def test_api_config_dialog_and_choice_buttons(tmp_path, monkeypatch):
    monkeypatch.setenv("APPDATA", str(tmp_path))
    monkeypatch.setattr(MadneApp, "_startup_config_check", lambda self: None)
    monkeypatch.setattr(ApiConfigDialog, "grab_set", lambda self: None)
    try:
        app = MadneApp()
    except tk.TclError:
        pytest.skip("当前环境无法创建 Tk 窗口")

    app.withdraw()
    try:
        assert app._ui_mode == "player"
        assert app._player_frame.winfo_manager() == "pack"
        assert app._admin_frame.winfo_manager() == ""
        assert app._player_api_btn.cget("text") == "API 设置"
        assert app._player_api_btn.winfo_manager() == "pack"

        dialog = ApiConfigDialog(app)
        dialog.withdraw()
        assert dialog._base_var.get() == "https://api.deepseek.com/v1"
        assert dialog._model_var.get() == "deepseek-chat"
        assert dialog._max_tokens_var.get() == 8192
        assert dialog._img_base_var.get() == DEFAULT_IMG_BASE_URL
        assert dialog._img_model_var.get() == DEFAULT_IMG_MODEL
        assert dialog._img_size_var.get() == ""
        assert dialog._img_width_var.get() == 1280
        assert dialog._img_height_var.get() == 720

        dialog._key_var.set("sk-test-admin")
        dialog._admin_code_var.set("admin")
        unlocked = dialog._read_config(check_admin=True)
        assert unlocked is not None
        assert unlocked.admin_unlocked is True
        assert unlocked.img_enabled is True
        assert unlocked.img_base_url == DEFAULT_IMG_BASE_URL
        assert unlocked.img_size == ""

        dialog._admin_code_var.set("wrong")
        assert dialog._read_config(check_admin=True) is None
        dialog.destroy()

        app._api_config = unlocked
        app._switch_ui("admin")
        assert app._ui_mode == "admin"
        assert app._admin_frame.winfo_manager() == "pack"
        app._switch_ui("player")
        assert app._ui_mode == "player"
        assert app._player_frame.winfo_manager() == "pack"

        output = make_output()
        app._render_output(output)
        buttons = [
            child
            for child in app._choices_frame.winfo_children()
            if isinstance(child, tk.Button)
        ]
        assert len(buttons) == len(output.choices)
        assert all(str(button.cget("state")) == "normal" for button in buttons)
        assert buttons[0].cget("text").startswith("▶ G1.")
        assert "风险" in buttons[0].cget("text")

        player_buttons = [
            child
            for child in app._player_choices_frame.winfo_children()
            if isinstance(child, tk.Button)
        ]
        assert len(player_buttons) == len(output.choices)
        assert all("风险" not in button.cget("text") for button in player_buttons)
        assert all("G1" not in button.cget("text") for button in player_buttons)
        player_text = app._player_story_text.get("1.0", "end")
        assert "scene_gui" not in player_text
        assert "风险" not in player_text
    finally:
        app.destroy()


def test_player_interface_renders_illustration(tmp_path, monkeypatch):
    monkeypatch.setenv("APPDATA", str(tmp_path))
    monkeypatch.setattr(MadneApp, "_startup_config_check", lambda self: None)
    monkeypatch.setattr(ApiConfigDialog, "grab_set", lambda self: None)
    monkeypatch.setattr(launcher, "DATA_DIR", tmp_path)
    try:
        app = MadneApp()
    except tk.TclError:
        pytest.skip("当前环境无法创建 Tk 窗口")

    app.withdraw()
    try:
        relative_path = "sessions/gui-test/assets/cycle_001.png"
        asset = tmp_path / relative_path
        asset.parent.mkdir(parents=True, exist_ok=True)
        asset.write_bytes(render_placeholder_png(64, 64, 3))

        output = make_output()
        output.illustration = Illustration(
            scene_id="scene_gui",
            backend="mock",
            prompt="GUI 测试插图",
            relative_path=relative_path,
            url="/assets/gui-test/cycle_001.png",
            width=64,
            height=64,
        )

        app._render_output(output)
        app.update_idletasks()

        assert app._player_image is not None
        assert app._player_image.width() == 64
        assert app._player_image.height() == 64
        assert app._player_image_label.winfo_manager() == "grid"
    finally:
        app.destroy()
