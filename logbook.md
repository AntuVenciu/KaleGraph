# Training Logbook

To be sure that work is done properly and that no effort is done in vain,
we shall write a logbook about training tests, in view of finding optimal
GNN architecture and training parameters.

## Rules for training

Each training should last at least 10 epochs.
Meaningful results should be saved in plot form and/or registered in this logbook.
The structure of the logbook should be:

date "command line with args to train" loss-after-10-epochs thrd TPR accuracy

If optimizer is changed, or major changes in the GNN architecture are made, they
should be properly documented before reporting the output of the training.
Unless overwise specified, new set-ups are assumed to be the new default.

## Logbook Entries

- 12062024 "python3 train.py --batch-size 32 --lr 0.002 --step-size 2 --hidden-size 40" 1.21 0.25 0.7 0.7

Use larger dataset with 200k events
(Takes too long, stopped after 2 epochs)
- 12062024 "python3 train.py --batch-size 128 --lr 0.002 --step-size 2 --hidden-size 40" 1.15 0.17 0.7 0.7

Make 2 iterations on node features and edges updates
-12062024 "python3 train.py --batch-size 128 --lr 0.001 --step-size 1 --hidden-size 80" 1.17 0.39 0.7 0.7

Add Dropout layers both in Relational and in Objective Model
-12062024 "python3 train.py --batch-size 128 --lr 0.001 --step-size 1 --hidden-size 80" 1.17 0.39 0.7 0.7

Change features of hits and nodes:
no more wireID, layers. Time in ns units.
deltaPhi, deltaZ for edges also.
In addition, we use the "max" aggregator
Train on smaller dataset for faster response.
Set dropout fraction to 0.1
Add ReLU at the beginning -> This is terrible
Set f_cdch and f_spx to 0.05

-12062024 "python3 train.py --batch-size 32 --lr 0.002 --step-size 1 --hidden-size 40" 1.15 0.5 0.68 0.68

### WARNING: Very important!

A new training goal is needed for the algorithm to converge:
1. Either we try to classify good nodes
2. Or we try to classify CONSECUTIVE edges between good nodes
We concentrate on 2. since it is more like what has been already developed.
We train with 5% cut on CDCH connection. NO SPX data
Add as a feature for the edges the average charge and amplitude of the two nodes

Try filtering only good hits
-14062024 "python3 train.py --batch-size 128 --lr 0.001 --step-size 1 --hidden-size 40" 1.29 0.16 0.68 0.68

Use a smaller dataset for quicker answers (10k events)
-14062024 "python3 train.py --batch-size 10 --lr 0.001 --step-size 5 --epochs 100 --hidden-size 60" 1.18 0.14 0.68 0.7
-14062024 "python3 train.py --batch-size 64 --epochs 100 --lr 0.005 --step-size 5 --gamma 0.9 --hidden-size 60" 0.98 0.15 0.78 0.78
-16062024 "python3 train.py --batch-size 64 --epochs 250 --lr 0.009 --step-size 10 --gamma 0.95 --hidden-size 60" 0.99 0.05 0.71 0.86

### WARNING: VERY IMPORTANT!

We log here results from a new approach, segmenting the CDCH
to have smaller graphs.

* 20032025 "python3 train.py --batch-size 128 --epochs 25 --lr 0.009 --step-size 10 --gamma 0.95 --hidden-size 140" 
	  * on 150k graphs with SPX and graph depth 4 (adjacent wires), 3 (adjacent layers), 3 (spx cdch connection)
	  * total number of trainable parameters ~60000


### 24 March 2025
New architecture: tried three different models:
* 24032025 Model 1: "python3 train.py --epochs 100 --lr 0.5e-4  --gamma 1 --hidden-size 250" 
* 24032025 Model 2: "python3 train.py --epochs 100 --lr 0.5e-4  --gamma 1 --hidden-size 500" 
* 24032025 Model 3: "python3 train.py --epochs 100 --lr 0.5e-4 --gamma 1 --hidden-size 1000" 

Model1 1st turn Sensitivity on test: 87%  
Model2 1st turn Sensitivity on test: 89%  
Model3 1st turn Sensitivity on test: 92%  
  
The most performant model on testing seems to be the third with hidden size equal to 1000.  

1)Aggregation function: 'sum'--> added before removing weighting of classes, it performed better compared to 'mean', 'max' aggregations. In order. 'sum'>'max'>'mean'  
2)Message passing steps: 2
3)Feature used: 'x0', 'y0', 'zTimeDiff', 'isSPX', 'amplitude', 'thetaStereo', 'phiStereo', 'Time'  
4)For SPXHits, phiStereo and thetaStereo are put to 0.  
5)Training was performed on full graph  
6)Removed weighting of classes  

### 31 March 2025
Listing all tried trainings in the last days:  

* python3 train.py --lr 1e-4 --gamma 1 --epochs 70 --hidden 1000  
  Training/Validation/Testing splitting was performed on dataset from 01002 to 01030, Message steps number is 2. Cdch depth conn = 5, Spx with cdch conn = 2  
  
  The reached Sensitivity for each turn is : 0.996, 0.935, 0.938, 0.929, 0.897, 0.895, 0.946  

* python3 train.py --lr 1e-4 --gamma 1 --epochs 50 --hidden 1000  
  Training/Validation/Testing splitting was performed on dataset from 01002 to 01070, Message steps number is 2.  Cdch depth conn = 5, Spx with cdch conn = 2  
  The reached Sensitivity for each turn is : 0.995, 0.938, 0.932, 0.935, 0.910, 0.881, 0.852  

It seems there is no significant improvement in enhancing the quantity of data.  

New scheduler used:  
epochs < 5: lr 3e-3  
5 < epochs < 10: lr 1e-3  
10 < epochs < 20:  lr: 1e-4  
20 < epochs < 40:  lr : 1e-5  

* python3 train.py --epochs 30 --hidden-size 250 --gamma 1  
  The reached Sensitivity for each turn is : 0.995, 0.932, 0.934, 0.933, 0.907, 0.906, 0.857  
  This training was performed on data from 01002 to 01030 (test split, val split etc..)Cdch depth conn = 4, Spx with cdch conn = 2  
  
* python3 train.py --epochs 30 --hidden-size 250 --gamma 1    
  The reached Sensitivity for each turn is : 0.995, 0.934, 0.938, 0.936, 0.906, 0.887, 0.774  
  This training was performed on data from 01002 to 01030 (test split, val split etc..)Cdch depth conn = 4, Spx with cdch conn = 2  
  
It seems that we are unable to go further, let us then try to play with the number of connections in the chamber  

1) Raise depth cdch connection from 4 to 5  


* python3 train.py --epochs 20 --hidden-size 250 --gamma 1  
  The reached Sensitivity for each turn is : 0.995, 0.929, 0.922, 0.922, 0.879, 0.873, 0.914  
  This training was performed on data from 01002 to 01030 (test split, val split etc..)Cdch depth conn = 5, Spx with cdch conn = 2. Message steps 2  

2) Raise depth cdch connection from 5 to 7  
* python3 train.py --epochs 20 --hidden-size 500 --gamma 1  
  The reached Sensitivity for each turn is : 0.995, 0.934, 0.930, 0.924, 0.878, 0.869, 0.900  
  This training was performed on data from 01002 to 01030 (test split, val split etc..)Cdch depth conn = 7, Spx with cdch conn = 2. Message steps 2  

Worse result or no improvement at all.  

3) Add layer information on the nodes: if spx the layer is 10. Lower cdch conn depth back to 4  
* python3 train.py --epochs 20 --hidden-size 500 --gamma 1  
  The reached Sensitivity for each turn is : 0.995, 0.928, 0.921, 0.923, 0.881, 0.891, 0.909  
  This training was performed on data from 01002 to 01030 (test split, val split etc..)Cdch depth conn = 4, Spx with cdch conn = 2. Message steps 2  
No improvement at all.  

4) Lower depth cdch connection from 7 to 4, the old one. Raise spx connection from 2 to 7.  
* python3 train.py --epochs 20 --hidden-size 250 --gamma 1  
The reached Sensitivity for each turn is : 0.996, 0.932, 0.935, 0.944, 0.924, 0.904, 0.678  
  This training was performed on data from 01002 to 01030 (test split, val split etc..)Cdch depth conn = 4, Spx with cdch conn = 7. Message steps 2  
No improvement at all.  

* python3 train.py --epochs 20 --hidden-size 250 --gamma 1  
The reached Sensitivity for each turn is : 0.996, 0.928, 0.936, 0.937, 0.921, 0.925, 0.924  
  This training was performed on data from 01002 to 01070 (test split, val split etc..)Cdch depth conn = 4, Spx with cdch conn = 7. Message steps 2  
No improvement at all. Except it is learning more about the last turns.  

* python3 train.py --epochs 20 --hidden-size 1000 --gamma 1  
The reached Sensitivity for each turn is : 0.996, 0.936, 0.938, 0.939, 0.925, 0.913, 0.759  
  This training was performed on data from 01002 to 01070 (test split, val split etc..)Cdch depth conn = 4, Spx with cdch conn = 7. Message steps 2  
No improvement at all.  



* python3 train.py --epochs 20 --hidden-size 250 --gamma 1  
From previous run the only difference is the number of message steps:3, efficiency is belov 90% for all turns.  

4) Remove z from features
* python3 train.py --epochs 20 --hidden-size 250 --gamma 1  
The reached Sensitivity for each turn is : 0.996, 0.936, 0.938, 0.939, 0.925, 0.913, 0.759  
 This training was performed on data from 01002 to 01030 (test split, val split etc..)Cdch depth conn = 4, Spx with cdch conn = 7. Message steps 2  
 efficiency is belov 90% for all turns.  



5) Replace z feature and time feature with MC values. Is it a model problem or noise problem?  
* python3 train.py --epochs 20 --hidden-size 250 --gamma 1  
The reached Sensitivity for each turn is : 0.995, 0.955, 0.944, 0.948, 0.943, 0.867, 0.926  
 This training was performed on data from 01002 to 01030 (test split, val split etc..)Cdch depth conn = 4, Spx with cdch conn = 7. Message steps 2  

### Conclusioni di oggi (31 Marzo 2025)

Sembrerebbe che effettuando un tuning degli iperparametri non si sia riusciti a salire oltre il 93.5 % di efficienza per ogni giro.  Aumentare i dati di allenamento non da' risultati significativi se non un aumento del 10% della loss (che si traduce in qualche 0.1 percentuale su ogni giro).  
L'ultimo training effettuato inserendo i valori MC di z e tempo sembrerebbe indicare che questa inefficienza non possa essere dovuta all'incertezza su z e sui tempi della camera. Perlomeno non in questa quantita ma in massimo 1-2 punti percentuali.  

Le ipotesi sono tre:  
* Modello da cambiare: magari grafi eterogenei?  
* Questa potrebbe essere un'inefficienza dovuta alla mal ricostruzione di alcuni hit che non passano la condizione di taglio del bartender. (le inefficienze nella hit reconstruction).  
* Forse l'inefficienza e' dovuta alle connessioni quando la particella esce dalla camera? (Connessioni tra due cluster diversi della cdch).  

Si riportano altre strategie provate, pero' solo con primi risultati e non approfondite.
* Inserimento di nuove feature: ampl -> ampl1, ampl2; sigmaZ : non sembrerebbero esserci miglioramenti.  
* Meccanismo di attenzione per tenere conto dei nodi rumorosi (anche in z): la performance e' peggiore, ma non e' stato effettuato alcun tipo di tuning. Potrebbe essere un'aggiunta utile quando introdurremo gli hit di rumore.  

### Come proseguire?
Il primo passo fondamentale e' prendere i dati senza alcuna inefficienza di ricostruzione e verificare se l'inefficienza possa essere potenzialmente dovuta a questa. Inoltre vedere se togliendo le connessioni tra due cluster diversi questa inefficienza diminuisce e di quanto (come fare e' da capire...)  
Poi:  

Opzione a)  Sviluppare nuovo modello (ad esempio grafi eterogenei, cio' permette di vedere hit della CYLDCH e del TC come oggetti diverse e toglie le forzature delle feature sugli hit del TC) (questo nel caso in cui l'inefficienza si riveli non essere dovuta alla ricostruzione)
Opzione b) Passare ai dati rumorosi e continuare.  

Altra cosa importante: e' vero che usare 1000 nodi dia risultati lievemente migliori rispetto a 250/ 500, tuttavia questo si traduce potenzialmente in un tempo di inferenza piu' grande: bisogna tenere conto di cio'.  


Side Quests:  
1) Togliere gli hit spx e vedere qual'e' l'efficienza per ogni giro.  
2) Fare una confusion matrix a parte sia per TC sia per CYLDCH.  
3) Pensare eventualmente a qualche forma di data augmentation 
4) Far scrivere ogni volta in automatico i trial effettuati ed i risultati sul logbook da un codice

