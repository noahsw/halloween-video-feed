from partycam.config import redact, rtsp_url_from_env


def test_builds_rtsp_url_from_env(monkeypatch):
    monkeypatch.setenv("CAMERA_HOST", "192.168.1.100")
    monkeypatch.setenv("CAMERA_USER", "partycam")
    monkeypatch.setenv("CAMERA_PASSWORD", "hunter2")
    monkeypatch.delenv("CAMERA_PORT", raising=False)
    monkeypatch.delenv("CAMERA_STREAM", raising=False)
    assert rtsp_url_from_env() == "rtsp://partycam:hunter2@192.168.1.100:554/stream1"


def test_special_characters_in_password_are_escaped(monkeypatch):
    monkeypatch.setenv("CAMERA_HOST", "cam.local")
    monkeypatch.setenv("CAMERA_USER", "me")
    monkeypatch.setenv("CAMERA_PASSWORD", "p@ss:w/rd")
    url = rtsp_url_from_env()
    assert "p%40ss%3Aw%2Frd@cam.local" in url
    assert url.count("@") == 1  # an unescaped @ would confuse the RTSP client about where the host starts


def test_no_host_means_no_camera(monkeypatch):
    monkeypatch.delenv("CAMERA_HOST", raising=False)
    assert rtsp_url_from_env() is None


def test_redact_hides_credentials():
    shown = redact("rtsp://partycam:hunter2@192.168.1.100:554/stream1")
    assert "hunter2" not in shown and "partycam" not in shown
    assert shown == "rtsp://***:***@192.168.1.100:554/stream1"


def test_redact_hides_escaped_passwords_too(monkeypatch):
    monkeypatch.setenv("CAMERA_HOST", "cam.local")
    monkeypatch.setenv("CAMERA_USER", "me")
    monkeypatch.setenv("CAMERA_PASSWORD", "p@ss:w/rd")
    shown = redact(rtsp_url_from_env())
    assert "p%40ss" not in shown and "p@ss" not in shown


def test_redact_leaves_other_sources_alone():
    assert redact("0") == "0"
    assert redact("/videos/party.mp4") == "/videos/party.mp4"
