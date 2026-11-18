import sys, os
import types

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def _stand_in_for_pkg_resources():
    """
    `adbutils` imports `pkg_resources` for its version and one file path. On Python 3.12 and
    later that module is gone, or is there and fails on the old `packaging` ALAS pins, and
    then nothing under `module.device` can be imported. No test here talks to a device, so a
    stand-in with the two calls `adbutils` makes is enough. Left alone where the real one works.
    """
    try:
        import pkg_resources  # noqa: F401
        return
    except Exception:
        pass
    stand_in = types.ModuleType('pkg_resources')
    stand_in.get_distribution = lambda name: types.SimpleNamespace(version='0')
    stand_in.resource_filename = lambda package, name: name
    stand_in.DistributionNotFound = type('DistributionNotFound', (Exception,), {})
    sys.modules['pkg_resources'] = stand_in


_stand_in_for_pkg_resources()
