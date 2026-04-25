from physics.base_physics import DronePhysics


def get_physics(backend: str = 'simple', **kwargs) -> DronePhysics:
    """
    Factory function. Returns a DronePhysics instance.
    backend: 'simple' | 'pyflyt'
    """
    if backend == 'pyflyt':
        try:
            import PyFlyt  # noqa: F401 — import check only
            from physics.pyflyt_physics import PyFlytPhysics
            return PyFlytPhysics(**kwargs)
        except ImportError:
            print('[Physics] PyFlyt not installed — using simple physics')
            print('[Physics] Install with: pip install pyflyt')
    from physics.simple_physics import SimplePhysics
    return SimplePhysics()
