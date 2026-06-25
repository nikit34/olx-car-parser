"""Tests for the condition-NLP minor-fault detector (2026-06-25 cheap-tail audit)."""

import pytest

from src.analytics.condition_signal import detect_minor_fault, minor_fault_cost


@pytest.mark.parametrize("text", [
    "Carro impecável, luz da injeção acesa no painel",
    "tudo ok mas tem o check engine ligado",
    "catalisador a precisar de substituição",
    "embraiagem a patinar em subida",
    "deita fumo branco no arranque",
    "reprovado na inspeção por emissões",
    "precisa de reparação na caixa",
    "necessita de mão de obra mecânica",
    "tem fuga de óleo no motor",
    "avaria eletrónica intermitente",
])
def test_detects_minor_faults(text):
    assert detect_minor_fault("", text) is not None


@pytest.mark.parametrize("text", [
    "Carro impecável, sempre na marca, full extras",
    "não precisa de qualquer reparação, está perfeito",
    "sem qualquer avaria, motor impecável",
    "luz de avaria apagada, tudo a funcionar",
    "revisão feita, distribuição nova, pronto a andar",
])
def test_clean_or_negated_text_no_fault(text):
    assert detect_minor_fault("Volkswagen Golf 1.6 TDI", text) is None


def test_cost_clamps_and_scales():
    # below floor -> floor
    cost, flag = minor_fault_cost("", "catalisador avariado", 1500)
    assert flag is not None and cost == 400        # 0.18*1500=270 -> floored to 400
    # mid -> %-of-price
    cost, _ = minor_fault_cost("", "check engine", 5000)
    assert cost == 900                              # 0.18*5000
    # cap
    cost, _ = minor_fault_cost("", "check engine", 50000)
    assert cost == 1500                             # capped
    # never exceeds half the car
    cost, _ = minor_fault_cost("", "catalisador a precisar", 600)
    assert cost == 300                              # min(max(400,108),1500)=400 -> min(400, 300)=300


def test_no_fault_zero_cost():
    cost, flag = minor_fault_cost("Audi A4", "estado impecável, full extras", 4000)
    assert cost == 0.0 and flag is None
