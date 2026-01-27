import sign_language_translator as slt
import os

def find_labels():
    words = {
        "mother": "ماں",
        "father": "باپ",
        "help": "مدد",
        "thanks": "شکریہ",
        "home": "گھر",
        "yes": "ہاں",
        "no": "نہیں"
    }
    
    translator = slt.models.ConcatenativeSynthesis(
        text_language="urdu",
        sign_language="psl",
        sign_format="vid"
    )
    
    for eng, urdu in words.items():
        print(f"--- Querying: {eng} ({urdu}) ---")
        try:
            # ConcatenativeSynthesis.translate returns a SignClip
            clip = translator.translate(urdu)
            print(f"Label/Path: {clip}")
            # If clip is a SignVideo or similar, it might have a path or label
            if hasattr(clip, 'sign_filenames'):
                print(f"Sign Filenames: {clip.sign_filenames}")
        except Exception as e:
            print(f"Error querying {eng}: {e}")

if __name__ == "__main__":
    find_labels()
