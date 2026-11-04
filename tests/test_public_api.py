"""Installed namespace parity prevents drift from canonical SDK exports."""

import importlib
import inspect
import unittest


class PublicAPITests(unittest.TestCase):
    """Both namespaces expose the same callable SDK surface."""

    def test_export_parity(self) -> None:
        """The compatibility namespace imports all public objects from odr."""
        self.assertEqual(exported("odr"), exported("odr.decisionrag"))


def exported(module_name: str) -> dict[str, object]:
    """Inspect public callable exports without importing implementation names twice.

    :param module_name: SDK namespace under inspection.
    :returns: Public callable objects indexed by export name.
    """
    module = importlib.import_module(module_name)
    return {
        name: record
        for name, record in inspect.getmembers(module, callable)
        if not name.startswith("_")
    }
