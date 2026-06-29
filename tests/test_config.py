from market_forecaster.config import load_config
from market_forecaster.logging_setup import _redact, register_secret


def test_defaults():
    cfg = load_config(env={})
    assert cfg.symbols == ("NASDAQ", "NZX50")
    assert cfg.forecast_horizon == 5
    assert cfg.ollama_model == "qwen3"
    assert cfg.spec("nasdaq").finnhub == "^IXIC"
    assert cfg.spec("NZX50").yahoo == "^NZ50"


def test_global_market_symbols_present():
    cfg = load_config(env={})
    expected = {
        "ASX": "^AXJO",
        "EUROPE": "^STOXX50E",
        "SINGAPORE": "^STI",
        "KOREA": "^KS11",
        "INDIA": "^NSEI",
        "UK": "^FTSE",
    }
    for canonical, yahoo in expected.items():
        spec = cfg.spec(canonical)
        assert spec.yahoo == yahoo
        assert spec.market in {"AU", "EU", "SG", "KR", "IN", "UK"}


def test_symbol_map_override():
    cfg = load_config(env={"SYMBOL_MAP_JSON": '{"NZX50":{"yahoo":"ENZL","proxy":true}}'})
    assert cfg.spec("NZX50").yahoo == "ENZL"
    assert cfg.spec("NZX50").proxy is True


def test_finnhub_key_registered_as_secret():
    load_config(env={"FINNHUB_API_KEY": "key-to-redact-1234"})
    register_secret("key-to-redact-1234")
    assert "key-to-redact-1234" not in _redact("token using key-to-redact-1234")


def test_symbols_parsed_from_env():
    cfg = load_config(env={"SYMBOLS": "nasdaq"})
    assert cfg.symbols == ("NASDAQ",)
