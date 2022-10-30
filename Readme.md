# dl4aed-p7: CODEC-DETEKTION IN AUDIODATEN MIT CNNS

Dieses Git-Repo enthält:
 * alle Python Skripte,
 * Notebooks und 
 * Dateien 

um das CNN aus dem [Paper](DL4AED___Project_7___CODEC_DETECTION_OF_AUDIO_DATA_USING_CNNS.pdf) zum Projekt **CODEC-DETEKTION IN AUDIODATEN MIT CNNS** zu rekonstruieren, weiter zu trainieren und anzuwenden.<br>
Im Folgenden soll auf die Verwendung der Notebooks sowie auf die einzelnen Schritte der Erstellung des Datensatzen eingegangen werden.

---

## **0. How to Use:**
Zunächst sollte das Python environment mit Hilfe der [yml-Datei](environment.yml) erstellt werden:

``` conda env create -f environment.yml ```

Falls ausschließlich mit den bereits erstellten Datensätzen gearbeitet werden soll, reichen die Python Notebooks aus. Deren Nutzung wird [hier](#1.0-Nutzung-der-Python-Notebooks:) erklärt.<br>
Auf die Generierung eines neuen Datensatzes wird [hier](#2.0-Generieung-eines-neue-Datensatzes:) genauer eingegangen. 
>**Voraussetzung hierfür:**<br>
>[`ffmpeg`](https://ffmpeg.org/) muss auf der verwendeten Maschine installiert sein. Um alle voreingestellten Codecs verwenden zu können, sollte die ffmpeg Version 4.2.4 installiert sein.

---

## **1. Nutzung der Python Notebooks:**
Mithilfe der Python Notebooks kann das CNN per Hypertuning optimiert werden oder
mit den bereitgestellten Datensätzen trainiert und evaluiert werden.<br>
Alle Hypertuning Searches und Trainings sind bereits ausgeführt und wurden in [hyperband_search](Code/Colab_Notebooks/_hyperband_search.zip) und in [_logs](Code/Colab_Notebooks/_logs.zip) festgehalten. Die daraus entstanden Modelle sind [hier](./Code/Colab_Notebooks) gespeichert.

>Um die Notebooks per Google Colab zu nutzen und Zugriff auf die Datensätze und Modelle zu bekommen, kann folgender Link mit dem Google Drive Konto verknüpft werden:<br>
>https://drive.google.com/drive/folders/10zlHC4l7S9PDZeE3nYVIF9rp0rAmHhUi?usp=sharing 

### **1.1 [Hypertuning](./Code/Colab_Notebooks/Hypertuning.ipynb):**
Notebook zum Anwenden des Hyperband Search Algorithmus zur Optimierung der CNN Parameter.<br>
Das beste Model wird als [best_model.h5](./Code/Colab_Notebooks/best_model.h5) gespeichert.<br>
### **1.2 [Training_Prediction_Hypertuned_Model](./Code/Colab_Notebooks/Training_Prediction_Hypertuned_Model.ipynb):**
Notebook zum Trainieren von [best_model.h5](./Code/Colab_Notebooks/best_model.h5) mit MedleyDB.<br>
Das trainierte Modell wird in [hyperband_model_trained.h5](./Code/Colab_Notebookshyperband_model_trained.h5) gespeichert.<br>
Prädiktionsergebnisse werden mit dem Vinyl-Datensatz erzeugt.

### **1.3 [Training_Prediction_Hennequin](./Code/Colab_Notebooks/Training_Prediction_Hennequin_Model.ipynb):**
Notebook zum Trainieren des Modells aus [*Hennequin et. al.*](./Literatur/hennequin_et_al_2017.pdf) mit MedleyDB.<br>
Das Ergebnis ist in [hennequin_model_trained](./Code/Colab_Notebooks/hennequin_model_trained.h5) gespeichert.<br>
Prädiktionsergebnisse werden mit dem Vinyl-Datensatz erzeugt.

---

## **2. Generieung eines neue Datensatzes:**
Die Generierung eines neuen Datensatzes besteht aus zwei Schritten. Als erstes werden mit FFMPEG Audiodateien in verschiedene Codecs konvertiert und in 1s-Ausschnitte unterteilt.<br>
Anschließend wird mit den Daten ein `tensorflow` Datensatz erstellt, welcher die Spektrogramme und dazugehörigen Labels jedes Ausschnitts enthält und in ein `train` und `test` Datensatz aufgeteilt ist.
### **2.1 Erstellung der Rohdaten mit FFMPEG**
Um einen neuen Datensatz zu generieren werden zunächst unkomprimierte Audiodateien im `wav` Format benötigt. Im Folgenden als *seed files* bezeichnet, diese werden in die vorgegebenen Codecs konvertiert und in 1s Ausschnitte abgespeichert.<br>

#### **Vorbereitung und Ordnerstruktur:**

Eine beispielhafte Struktur ist unter [Example_raw_dataset](./Code/_data/Example_raw_dataset) gespeichert. 
Dieser besteht immer aus drei Unterordner:
* [`seed_files`](./Code/_data/Example_raw_dataset/seed_files): Muss händisch erstellt werden, die Text Datei `seed_list.txt` muss ebenso händisch eingefügt werden. Sie enthält eine Liste aller Audiodateien, die bereits im Datensatz enthalten sind um Dopplungen zu vermeiden. Bei der Erstellung sollte sie deshalb leer sein. Zusätzlich können in diesem Ordner die *seed_files* abgelegt werden, welche konvertiert werden sollen.
* [`uncompr_wav`](./Code/_data/Example_raw_dataset/uncompr_wav): Enthält die Ausschnitte der *seed_files* im unkomprimierten `PCM` Format und wird automatisch erstellt. Die Ausschnitte werden für jedes *seed_file* in einen eigenen Unterordner abgelegt.
* [`compressed_wav`](./Code/_data/Example_raw_dataset/compressed_wav): Enthält Unterordner mit den vorgegeben Codecs. In jedem Unterordner liegen wiederum für jedes *seed_file* ein Unterordner mit den Ausschnitten.

Zusätzlich zum Datensatzordner [Example_raw_dataset](./Code/_data/Example_raw_dataset) gibt es eine [Konfigurationsdatei](./Code/_data/dataset_config.json):
```json
{
    "reference_audio_path": "seed_files",
    "uncompr_audio_path": "uncompr_wav",
    "compressed_audio_path": "compressed_wav",
    "segment_length": 1,
    "number of min chunks": 50,
    "mp3_32k": {
        "codec": "libmp3lame",
        "format": "mp3",
        "bitrate": 32000,
        "channels": 2,
        "sampling_rate": 44100,
        "ffmpeg_command": "ffmpeg -y -i {} -acodec libmp3lame -ac 2 -ab 32000 -ar 44100 {}"
    },

    ...

    "uncompr_wav": {
        "codec_ffmpeg": "pcm_s16le",
        "codec_sf": "PCM_16",
        "format_ffmpeg": "s16le",
        "format": "wav",
        "channels": 2,
        "sampling_rate": 44100,
        "ffmpeg_command": "ffmpeg -i {} -acodec pcm_s16le -ac 2 -ar 44100 {}"
    }
}
```
In der Konfigurationsdatei wird festgehalten 
 * in welchem Verzeichnis die *seed_files* und komprimierten bzw. unkomprimierten Dateien gespeichert werden,
 * die Länge des Ausschnitts in Sekunden: `segment_length`,
 * wie viele Ausschnitte pro Audiodatei mindestens entstehen müssen (`number of min chunks`), damit diese in den Datensatz mit aufgenommen wird, 
 * welche Codecs zur Konvertierung verwendet werden sollen. Die Einträge `format` und `ffmpeg_command` sind hierbei essentiell und müssen in jedem Eintrag enthalten sein. `ffmpeg_command` muss ein gültiger `ffmpeg` Befehl sein. Die ersten {} sind Platzhalter für die Input Datei, die zweiten {} Platzhalter für die Output Datei. 

Der Eintrag `uncompr_wav` legt fest in welchem Format die Ausschnitte im Datensatz abgelegt werden und muss sowohl mit `ffmpeg` als auch mit [`soundfile`](https://pysoundfile.readthedocs.io/en/latest/) kompatibel sein.<br>

#### **Anwendung des `dataset_generator`**
Sobald ein Datensatzordner mit Unterordner *seed_files*, `seed_list.txt`, und einer `dataset_config.json` erstellt sind, kann mit Hilfe des Python Skripts [Example_dataset_script.py](./Code/Database_Generation/Example_dataset_script.py) ein Rohdatensatz erstellt werden. Das Skript liest jede *seed_file* nacheinander ein und erstellt für jeden Codec eine Unterordner mit den Audioausschnitten, dafür wird ein `Dataset` Objekt aus [`dataset_generator.py`](./Code/Database_Generation/dataset_generator.py) erstellt und verwendet.

### **2.2 Erstellung des `tensorflow` Datensatzes:**

Die Rohdaten müssen im zweiten Schritt in Spektogramme und dazugehörige Labelinformation konvertiert werden.
Dies wird in verschiedenen Preprocessing Schritten in [`tf_datset_Example.py`](./Code/tf_dataset_Example.py) mit Funktionen der `PreprocessWrapper` Klasse aus [`wrapper_functions.py`](./Code/wrapper_functions.py) gemacht.
Entscheidend für das Preprocessing sind die vorher festgelegten Einstellung, welche anschließend zusammen mit dem `tensorflow` Datensatz gespeichert werden:
```json                               
{                                     
    time_stamp': time_stamp,          
    'sr': 44100,                      
    'audio_length': 1,                
    'mono': True,                     
    'n_mels': 64,                     
    'n_fft': 1024,                    
    'hop_length': 256,                
    'win_length': 512,                
    'window': 'hamm',                 
    'center': True,                   
    'pad_mode': 'reflect',            
    'power': 2.0,                     
    'calculate_mel': False,           
    'filter_signal': True,            
    'filter_config': ['high', 4000],  
    'random_seed': 10,                
    'binary': False                   
}                                     
```                                   
Hier werden Einstellung für die Berechnung der STFT festgelegt und die Verwendung von Filtern spezifiziert:<br>
mit `filter_signal` und `filter_config` kann festgelegt werden, ob und wie und bei welcher Frequenz das Spektrum abgeschnitten wird.
Mit `'binary' : False` wird festgelegt, dass der Datensatz mehrer Label aufweist (Anzahl der Codecs), falls `'binary' : True` wird nur unterschieden, ob ein Element komprimiert oder unkomprimiert ist.<br>
Der erstellte Datensatz kann anschließend zum Training oder Evaluation des CNNs innerhalb der Notebooks verwendet werden. 
