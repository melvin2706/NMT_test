"""KamerNMT: multilingual NMT for Cameroonian languages, French (pivot) and English.

    from kamernmt import Translator
    tr = Translator()                                    # finds the model by itself
    tr.translate("Je vais au marché.", src="fr", tgt="dua")
    tr.translate({"titre": "Bonjour", "prix": 2500}, src="fr", tgt="dua")   # same JSON back
"""

__version__ = "0.2.0"
__all__ = ["Translator", "translate"]

_default = None


def __getattr__(name):
    # imported lazily so `import kamernmt` stays fast and light
    if name == "Translator":
        from .engine import TranslationEngine

        return TranslationEngine
    raise AttributeError(name)


def translate(content, src: str = "fr", tgt: str = "dua", **options):
    """One-call translation with a shared, automatically found model."""
    global _default
    if _default is None:
        from .engine import TranslationEngine

        _default = TranslationEngine()
    return _default.translate(content, src, tgt, **options)
