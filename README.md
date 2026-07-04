# Clasificador de edad con UTKFace — PyTorch vs. TensorFlow/Keras

**Autores:** Fabián Salinas, Alessandro Lavezzi
**Ramo:** Frameworks de IA - UDD 
**Profesor:** Mauricio Alex Vásquez 
**04 de Julio, 2026**

Proyecto de clasificación de rango etario a partir de fotos de rostros del dataset **UTKFace**.
En cada notebook se construye la **misma arquitectura CNN en PyTorch y en TensorFlow/Keras**
para comparar ambos frameworks (accuracy, tiempos de entrenamiento, matrices de confusión y
curvas ROC), probando distintas formas de agrupar y balancear las edades.

---

## 1. Requisitos previos

- **Python 3.11** (probado en 3.11.9). Sirve cualquier versión entre 3.10 y 3.11.
- **Git** para clonar el repositorio.
- Una cuenta gratuita en **[Kaggle](https://www.kaggle.com/)** para descargar el dataset.
- No se necesita GPU: los notebooks corren en CPU (con GPU van más rápido, pero es opcional).

> **Tiempo aproximado:** cada notebook entrena dos modelos (PyTorch + Keras) durante 10 épocas.
> En CPU esto puede tardar varios minutos por notebook.

---

## 2. Clonar el repositorio

```bash
git clone https://github.com/alessandro-lavezzi/frameworks-ia-2026_udd
cd frameworks-ia-2026_udd
```

---

## 3. Descargar el dataset UTKFace

El dataset **no está incluido en el repositorio** ya que pesa demasiado para git. Son 23.000 fotos aprox.
Hay que bajarlo desde Kaggle y dejarlo dentro de la carpeta del proyecto.

**Dataset:** [UTKFace — jangedoo/utkface-new](https://www.kaggle.com/datasets/jangedoo/utkface-new)

### Opción A — Descarga manual (la más simple y recomendada)

1. Entra a <https://www.kaggle.com/datasets/jangedoo/utkface-new> (inicia sesión).
2. Presiona **Download** para bajar el `.zip`.
3. Descomprímelo y ordena los archivos de modo que **todas las imágenes** queden en:

   ```
   frameworks-ia-2026_udd/UTKFace_data/UTKFace/
   ```

   Es decir, dentro de `UTKFace/` deben quedar directamente las ~23.700 imágenes
   `*.jpg.chip.jpg` (por ejemplo `100_0_0_20170112213500903.jpg.chip.jpg`), **sin** subcarpetas
   intermedias.

### Opción B — Kaggle API (por línea de comandos)

```bash
pip install kaggle
# 1. Ve a kaggle.com/settings -> API -> "Create New Token" (descarga kaggle.json)
# 2. Coloca kaggle.json en:
#      Windows: C:\Users\<usuario>\.kaggle\kaggle.json
#      Linux/Mac: ~/.kaggle/kaggle.json
kaggle datasets download -d jangedoo/utkface-new
unzip utkface-new.zip -d UTKFace_data
```

Luego verifica que las imágenes hayan quedado en `UTKFace_data/UTKFace/`.

### Estructura esperada de carpetas

```
frameworks-ia-2026_udd/
├── UTKFace_data/
│   └── UTKFace/
│       ├── 100_0_0_20170112213500903.jpg.chip.jpg
│       ├── ...
│       └── (≈23.700 imágenes)
├── 1er_analisis.ipynb
├── 2do_analisis.ipynb
├── 3er_analisis.ipynb
├── 4to_analisis.ipynb
├── heatmap_gradcam.ipynb
├── requirements.txt
└── README.md
```

> Todos los notebooks leen las imágenes desde la ruta relativa `./UTKFace_data/UTKFace`.
> La edad, el género y la etnia se obtienen del **nombre del archivo**
> (`[edad]_[género]_[etnia]_[fecha].jpg`), así que no hay que ordenar nada por subcarpetas.

---

## 4. Entorno virtual e instalación de dependencias

Recomendado para no ensuciar la instalación global de Python. Desde la carpeta del proyecto:

### Windows (PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

### Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Las dependencias incluidas en `requirements.txt` son:

| Librería | Uso |
|---|---|
| `tensorflow` | Modelos y entrenamiento en Keras |
| `torch`, `torchvision` | Modelos, entrenamiento y transforms en PyTorch |
| `numpy`, `pandas` | Manejo de datos y tablas |
| `matplotlib`, `seaborn` | Gráficos, curvas y matrices de confusión |
| `scikit-learn` | Split de datos, métricas, ROC/AUC |
| `scipy` | Cálculos numéricos de apoyo |
| `Pillow` | Carga de imágenes |

> **Nota:** cada notebook tiene al inicio una celda `!pip install ...` por comodidad. Si ya
> instalaste `requirements.txt` en el entorno virtual, puedes saltarte esa celda (o dejarla
> correr, no hace daño).

---

## 5. Abrir los notebooks

Con el entorno virtual activado:

```bash
pip install jupyter        
jupyter notebook
```

También puedes abrirlos directamente en **VS Code** con la extensión de Jupyter, seleccionando
el intérprete de `.venv` como kernel.

---

## 6. Orden de revisión de los notebooks

`Si bien los notebooks son de ejecución independiente, se aconseja su revisión secuencial para entender la metodología aplicada. El proceso parte con el estudio del dataset y continúa con la iteración de diferentes configuraciones con el fin de optimizar los resultados obtenidos.`

### 1️⃣ `1er_analisis.ipynb` — Rango etario, banco completo (desbalanceado)
Estudio inicial, con el dataset desbalanceado en cuanto a n° de imágenes por categoría.
Consta de 5 rangos etarios

### 2️⃣ `2do_analisis.ipynb` — Rango etario, banco balanceado
Estudio del dataset balanceado en n° de imágenes por categoría.
Consta de 5 rangos etarios

### 3️⃣ `3er_analisis.ipynb` — Clasificación por década
Estudio del dataset balanceado en n° de imágenes por categoría.
Consta de 9 rangos etarios (por décadas)
Incluye análisis de distribución por etnia

### 4️⃣ `4to_analisis.ipynb` — Rango etario, doble balanceo (edad + etnia)
Estudio del dataset balanceado en n° de imágenes por categoría.
Consta de 5 rangos etarios (20 años c/u, con niños, jóvenes, adulto joven, adulto, adulto mayor)
Se balancea tanto para la edad como para la etnia, para que el modelo no aprenda a estimar la edad a
partir de rasgos étnicos. 
Versión que se reutiliza en el notebook de Grad-CAM.

### 5️⃣ `heatmap_gradcam.ipynb` — Interpretabilidad (Grad-CAM)
Notebook **independiente y de cierre**. Aplica **Grad-CAM** sobre la CNN (en PyTorch y en Keras)
para visualizar **en qué zona del rostro se fija el modelo** al predecir la edad. Las zonas rojas
del mapa de calor son las más influyentes; lo ideal es que se concentren en rasgos faciales
(ojos, frente, mejillas) y no en el fondo. Reutiliza el pipeline balanceado del 4º análisis.

---

## 7. Carpetas y archivos adicionales

- **`fotos_prueba/`** — Imágenes sueltas (algunas de UTKFace y otras propias) para probar los
  modelos con casos fuera del set de entrenamiento.
- **`Lab 1/`** — Laboratorio introductorio del ramo (clasificador "is that Santa"), no forma
  parte del proyecto principal de edades.

---

## 8. Notas y solución de problemas

- **`FileNotFoundError` / no encuentra imágenes:** revisa que las imágenes estén exactamente en
  `UTKFace_data/UTKFace/` y que ejecutes los notebooks desde la raíz del repositorio.
- **La primera ejecución es lenta:** es normal, TensorFlow y PyTorch tardan en importar y el
  entrenamiento por CPU toma su tiempo.
- **Kernel de Jupyter:** asegúrate de seleccionar el intérprete del entorno virtual `.venv`,
  no el Python global, para que use las dependencias instaladas.
