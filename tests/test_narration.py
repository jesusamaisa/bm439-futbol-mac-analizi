"""Gösterim anlatımı: sayıların Türkçe okunuşu (TTS'e rakam yerine yazıyla verilir)."""
from __future__ import annotations


def test_turkish_number_reading_for_narration():
    """Anlatım metnindeki sayıların okunuşu (TTS'e rakam yerine yazıyla verilir)."""
    from bgoru.experiments.make_narration import sayi, tamsayi_oku

    assert sayi(0.353, 3) == ("0,353", "sıfır virgül üç yüz elli üç")
    assert sayi(0.047, 3)[1] == "sıfır virgül sıfır kırk yedi"
    assert sayi(30, 0) == ("30", "otuz")
    assert tamsayi_oku(1280) == "bin iki yüz seksen"
    assert tamsayi_oku(2115) == "iki bin yüz on beş"
