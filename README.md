# NNFAUContest
**TODO:**

RandomWeigths (evtl. Inteface) anpassen, sodass man nicht jedes mal in der Klasse Magic Numbers ändern muss
^^gleiches für constbias

Layers anpassen:

       Dropout implementieren
       early Prediction (Zwischenwerte ausgeben lassen)
       Recurring

Passende initial Werte für Test2 (Travel) / Normalisieren für die Activationfunc!?

Data Augmentation für image Test
       
Hochgeladen -> Nehme Präsentationen vom Marcel (wird bald im Forum Hochgeladen) und erstelle eine TODO liste ;)

...

**Log File Sascha**

**7.4.**

TestHelper und AccuracyFunction: Varianz und andere Kleinigkeiten werden ausgegeben.

Main: Epochen hinzugefügt um zB verschiedene Anzahl Layers/Gewichte leichter zu vergleichen

Normalisation in TestHelper: Habe ich versucht, hat die Varianzen erhöt :(

SigmoidFunction: Ich hab die Funktion auf 0 bis 1 gedeckelt Ergebnis:
     
- Test 1 Varianz von 28 auf 21 
- Test 2 Um einiges Langsamer geworden (da passiert was?) 
         Varianz eher schlechter geworden Viele richtige Antworten sind auf 0.4 gelangt
               
     

**Log File Chris**

**4.Jul**

Test 1 ~80 Acc mit Outputlayer Standard RandomW und Bias, Tanh und LearningRate ~0.0002f  (2 Hidden Layer verbessern auf 81%...noch was möglich?)

Test 3 hat nun in meiner main One-Hot-Encoding, dafür hab ich den Datareader angepasst

Alle Tests in meiner main sind lauffähig/getestet i.e prediction online

**Log File Benny**

