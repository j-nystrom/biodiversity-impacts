from box import Box

from core.model.model_data_task import get_ecological_effects_settings


def test_get_ecological_effects_settings_reads_nested_config() -> None:
    """Nested ecological_effects settings are used by current configs."""
    settings = Box(
        {
            "ecological_effects": {
                "rolled_up_predictions": True,
                "min_studies_per_group": 5,
                "hierarchy": {
                    "level_1": ["Biome", "Custom_taxonomic_group"],
                    "level_2": ["Realm"],
                },
            }
        }
    )

    ecological_effects = get_ecological_effects_settings(settings)

    assert ecological_effects["rolled_up_predictions"] is True
    assert ecological_effects["min_studies_per_group"] == 5
    assert ecological_effects["hierarchy"] == {
        "level_1": ["Biome", "Custom_taxonomic_group"],
        "level_2": ["Realm"],
        "level_3": [],
    }


def test_get_ecological_effects_settings_supports_legacy_config() -> None:
    """Legacy top-level hierarchy settings remain valid."""
    settings = Box(
        {
            "rolled_up_predictions": True,
            "min_studies_per_group": 3,
            "hierarchy": {
                "level_1": ["Biome"],
            },
        }
    )

    ecological_effects = get_ecological_effects_settings(settings)

    assert ecological_effects["rolled_up_predictions"] is True
    assert ecological_effects["min_studies_per_group"] == 3
    assert ecological_effects["hierarchy"] == {
        "level_1": ["Biome"],
        "level_2": [],
        "level_3": [],
    }
