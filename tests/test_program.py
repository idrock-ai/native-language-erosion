from src.program import is_compliant, rescue_letter, parse_letter

def test_is_compliant():
    assert is_compliant("Bitta gap.")
    assert is_compliant("Birinchi gap. Ikkinchi gap!")
    assert is_compliant("x" * 250 + ". a. b. c.")            # >2 sentences but <=300 chars
    long3 = ("Bu juda uzun birinchi gap bilan boshlanadi. " * 8 + "Yana bir gap. "
             "Va yakuniy xulosa gap.")                        # >300 chars and >2 sentences
    assert len(long3) > 300 and not is_compliant(long3)
    assert is_compliant("$x_{20} = 31.5$ bo'ladi")            # decimals are not sentence ends
    assert is_compliant("")

def test_rescue_letter_adapter_format():
    raw = "[[ ## reasoning ## ]]\nQisqa mulohaza.\n[[ ## answer_letter ## ]]\nC\n[[ ## completed ## ]]"
    assert rescue_letter(raw) == "C"

def test_rescue_letter_fallbacks():
    assert rescue_letter("Javob: B") == "B"
    assert rescue_letter("") == ""
    assert rescue_letter("uzilib qolgan matn ...") in ("", "A", "B", "C", "D")

def test_parse_letter_still_works():
    assert parse_letter("The answer is D") == "D"
