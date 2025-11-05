# Návrh zadání projektu

## Název tématu v češtině:
**Systém pro analýzu, klasifikaci a třídění objektů v reálném čase**

## Název tématu v angličtině:
**Real-time Object Analysis, Classification, and Sorting System**

---

### Anotace tématu

#### Motivace tématu:
V mnoha průmyslových a zemědělských odvětvích je klíčová automatizovaná kontrola kvality a třídění produktů. Tato práce se zaměřuje na návrh a implementaci systému, který dokáže v reálném čase analyzovat video signál, rozpoznávat, segmentovat a klasifikovat objekty a na základě této klasifikace je fyzicky třídit. Příkladem může být třídička brambor, jablek nebo jiných produktů, kde je potřeba oddělit vadné kusy. Cílem je vytvořit prototyp běžící na dostupném hardwaru, jako je Raspberry Pi s AI akcelerátorem, a demonstrovat tak praktickou aplikaci moderních metod počítačového vidění a strojového učení.

#### Cíl práce:
Cílem této bakalářské práce je **navrhnout, implementovat a otestovat systém pro analýzu videosignálu, klasifikaci objektů a následné hardwarové roztřídění**. Systém bude postaven na platformě Raspberry Pi s AI modulem a bude schopen v reálném čase zpracovávat obraz, sbírat data o klasifikovaných objektech, vizualizovat je a ovládat externí hardware pro fyzické třídění.

#### Cíle práce (Podrobný rozpis):
*   **Rešerše:** Přehled metod pro rozpoznávání, segmentaci a klasifikaci objektů v obraze. Analýza dostupných hardwarových řešení pro AI na vestavěných systémech (např. AI HAT pro Raspberry Pi).
*   **Návrh systému:** Architektura systému zahrnující softwarové i hardwarové komponenty. Definice komunikačních protokolů mezi jednotlivými částmi.
*   **Sběr a příprava dat:** Vytvoření nebo získání datové sady pro trénování modelu (např. obrázky různých kvalit jablek/brambor).
*   **Implementace modelu:** Výběr, natrénování a optimalizace modelu pro klasifikaci objektů pro nasazení na Raspberry Pi.
*   **Vývoj softwaru:** Implementace softwaru pro:
    *   1. Analýzu videosignálu z kamery.
    *   2. Zprostředkování sběru dat o detekovaných objektech (kategorie, čas detekce).
    *   3. Vizualizaci dat (např. v jednoduchém webovém rozhraní).
*   **Hardwarová integrace:** Napojení na hardwarový systém (např. servo, páka), který provede fyzické roztřídění klasifikovných objektů.
*   **Testování a evaluace:** Otestování celého systému v reálném scénáři, zhodnocení přesnosti klasifikace a spolehlivosti třídění.

#### Výstupy práce:
Výstupem práce bude **funkční prototyp zařízení**, které je schopné v reálném čase analyzovat obraz, klasifikovat objekty a na základě toho je třídit. Součástí bude také **softwarová aplikace** pro sběr a vizualizaci dat a **dokumentace** popisující návrh, implementaci a výsledky testování.

---
### Literatura:
*   Dokumentace k Raspberry Pi a použitému AI modulu (např. Raspberry Pi AI Kit).
*   Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press.
*   Szeliski, R. (2010). *Computer Vision: Algorithms and Applications*. Springer.
*   Knihovny pro počítačové vidění a strojové učení (OpenCV, TensorFlow, PyTorch).
*   Články a tutoriály týkající se implementace AI modelů na vestavěných zařízeních.
*   Odkaz na konkrétní hardware: https://rpishop.cz/516728/raspberry-pi-ai-kit/

