# sEMG-Detector

Code Struktur
========
Im Grunde besteht der Code aus zwei Teilen:
 1. Erzeugung der Trainings- und Validierungsdaten (selbst)
 2. Aufbau der NN Architektur und Durchführung des Trainings (Nutzung von KERAS)

1.:
--
Hier kann beispielhaft Creator001.py (noch data_creator.py) angesehen werden. (sollte mit entsprechend nummeriertem Datensatz korrellieren.)

2.:
-- 
Durch Ausführung von main.py gehandhabt. (Aufruf z.B. in script.py)
Hier kann vor allem mit KERAS gepielt werden: Ausprobieren anderer Modell-Architekturen, oder NN-Arten, wie zum Beispiel Recurrent NN.
