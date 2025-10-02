from pycharge.config import Config, RootFindConfig


def test_config_defaults():
    c = Config()
    assert c.field_component in ("total", "velocity", "acceleration")
    assert isinstance(c.root_find, RootFindConfig)


def test_config_custom_values():
    root_cfg = RootFindConfig(rtol=1e-6, atol=1e-9, max_steps=512)
    c = Config(field_component="acceleration", root_find=root_cfg)

    assert c.field_component == "acceleration"
    assert c.root_find == root_cfg
