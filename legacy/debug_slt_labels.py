import sign_language_translator as slt

def find_all_labels():
    # Stable 8 vocabulary
    words = {
        "apple": "سیب",
        "world": "دنیا",
        "good": "اچھا",
        "school": "اسکول",
        "mother": "ماں",
        "father": "باپ",
        "help": "مدد",
        "home": "گھر",
    }
    
    print("CHECKING DNA AVAILABILITY FOR STABLE 8 VOCABULARY")
    print("-" * 50)
    
    translator = slt.models.ConcatenativeSynthesis(
        text_language="urdu",
        sign_language="psl",
        sign_format="vid"
    )
    
    results = {"success": [], "failed": []}
    
    for eng, urdu in words.items():
        try:
            clip = translator.translate(urdu)
            if clip:
                print(f"[OK] {eng} -> {urdu}")
                results["success"].append(eng)
            else:
                print(f"[FAIL] {eng} -> {urdu} (No clip)")
                results["failed"].append(eng)
        except Exception as e:
            print(f"[FAIL] {eng} -> {urdu} ({str(e)[:50]})")
            results["failed"].append(eng)
    
    print("-" * 50)
    print(f"SUCCESS: {len(results['success'])}/8 - {results['success']}")
    print(f"FAILED: {len(results['failed'])}/8 - {results['failed']}")

if __name__ == "__main__":
    find_all_labels()
