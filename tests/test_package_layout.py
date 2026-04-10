"""Smoke tests for the initial package scaffold."""


def test_point_packages_import() -> None:
    """The repository should expose the planned top-level packages."""
    import benchmark.constructionsite10k
    import common.schemas
    import point1
    import point2

    assert benchmark.constructionsite10k is not None
    assert common.schemas is not None
    assert point1 is not None
    assert point2 is not None
