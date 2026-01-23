import os
import sys
import tempfile
import sign_language_translator as slt

WRITABLE_BASE = os.path.join(tempfile.gettempdir(), "slt_persistent_storage")
slt.Assets.ROOT_DIR = WRITABLE_BASE

translator = slt.models.ConcatenativeSynthesis(
    text_language="urdu",
    sign_language="psl",
    sign_format="vid"
)

# Test tokens
test_words = ["سیب", "دنیا", "پاکستان", "اچھا", "لال", "ہے", "یہ", "وہ", "کھانا"]
for w in test_words:
    try:
        clip = translator.translate(w)
        print(f"✅ Token '{w}' -> OK ({len(clip)} frames)")
    except Exception as e:
        print(f"❌ Token '{w}' -> FAIL: {e}")

# Try English as fallback
test_en = ["apple", "food", "eat"]
for w in test_en:
    try:
        clip = translator.translate(w)
        print(f"✅ Token '{w}' -> OK ({len(clip)} frames)")
    except Exception as e:
        print(f"❌ Token '{w}' -> FAIL: {e}")
