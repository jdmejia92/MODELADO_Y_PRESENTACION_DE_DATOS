# Segmentación de clientes de tarjeta de crédito

Avance 1 del curso **Modelado y Presentación de Datos** (Zegel). Análisis exploratorio, limpieza y modelado no supervisado (**clustering con K-Means**) para segmentar a 30 000 clientes de tarjeta de crédito según su comportamiento de pago.

**Integrantes:**
1. Hilario Vargas, Jorge Miguel
2. Mejía Hoyos, Juan Daniel
3. Villalobos Sánchez, Miguel Angel
4. Jacinto Hueda, Luis Alonso
5. Soto Santos, Renzo Gastón
6. Vera Espinoza, Paul Enrique

## Problema de negocio

Un emisor de tarjetas aplica la misma política de límites y cobranza a toda su cartera, aunque el 22 % de sus clientes no paga al mes siguiente. El objetivo es encontrar segmentos de clientes con comportamiento parecido, sin usar la etiqueta de impago para formarlos, y medir después cuánto difiere su riesgo.

## Resultados principales

| Segmento | Nombre | % clientes | % impago | % de la deuda | % de los impagos |
|---|---|---|---|---|---|
| 1 | Financiadores al día | 38.2 | 11.4 | 66.5 | 19.5 |
| 2 | Pagadores totales | 33.2 | 13.9 | 6.7 | 20.7 |
| 3 | Inactivos con atrasos | 4.6 | 34.5 | 0.5 | 7.1 |
| 4 | Morosos recurrentes | 24.0 | 49.1 | 26.3 | 52.7 |

- **Modelo:** K-Means con k = 4, elegido con el método del codo y el coeficiente de silueta (silueta máxima de 0.347 en k = 4).
- **Veredicto:** el segmento 4 tiene **2.2 veces** la tasa de impago promedio y concentra el **52.7 %** de todos los impagos con el 24 % de los clientes.
- La etiqueta de impago **no** se usó para formar los segmentos, solo para evaluarlos.

## ¿Por qué clustering con K-Means?

La rúbrica pide aplicar **una** técnica no supervisada: PCA o clustering, y en clustering, K-Means o jerárquico.

**Clustering en vez de PCA:**
- El problema de negocio pide grupos de clientes a los que aplicar políticas distintas. Clustering entrega un segmento por cliente; PCA entrega ejes continuos que mezclan variables.
- El veredicto cuantitativo (impago, deuda e impagos por segmento) solo existe si hay segmentos.
- Las 18 columnas mensuales ya se resumieron en 6 variables interpretables, así que no hacía falta reducir dimensiones.

**K-Means en vez de jerárquico:**
- **Escala:** con 29 566 clientes, el jerárquico necesita 437 millones de distancias entre pares (unos 3.5 GB de memoria) y solo sería viable con una muestra. K-Means procesa a todos los clientes: probar k de 2 a 8 tomó unos 2 segundos.
- **Uso operativo:** K-Means asigna un segmento a clientes nuevos con `predict`; el jerárquico no.
- **Limitaciones y cómo se manejan:**
  - Hay que elegir k: método del codo y silueta.
  - Es sensible a la escala: `StandardScaler`.
  - Depende de la inicialización: `n_init=10` y `random_state=42`.

## Datos

| | |
|---|---|
| Dataset | *Default of Credit Card Clients*: clientes de un emisor de Taiwán, abril a septiembre de 2005 |
| Archivo | `data/raw/UCI_Credit_Card.csv`: 30 000 filas y 25 columnas, todas numéricas (2.9 MB) |
| Fuente original | UCI Machine Learning Repository, licencia [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |
| Copia descargada | [Hugging Face `scikit-learn/credit-card-clients`](https://huggingface.co/datasets/scikit-learn/credit-card-clients), licencia CC0 1.0, descargada el 2026-11-08 |
| SHA-256 | `a0f0ab49d6326671d6cd83be5c88dcf18007025fe9a53ecd699119c871176ca1` |

Cita: Yeh, I. C. (2009). *Default of Credit Card Clients* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C55S3H

## Estructura

```text
├── data/
│   ├── raw/UCI_Credit_Card.csv          # dataset original, sin modificar
│   └── processed/                       # se genera al ejecutar (clientes con su segmento)
├── notebooks/
│   └── avance1_segmentacion_clientes.ipynb
├── reports/
│   ├── figures/                         # gráficos; el prefijo indica la diapositiva (s04_..., s10_...)
│   ├── perfil_segmentos.csv
│   └── resultados.json                  # todas las cifras clave, generadas por el notebook
├── requirements.txt
└── README.md
```

## Cómo ejecutarlo

Requiere **Python 3.12 o superior**. Todas las rutas son relativas a la carpeta del proyecto.

```bash
git clone https://github.com/jdmejia92/MODELADO_Y_PRESENTACION_DE_DATOS.git
cd MODELADO_Y_PRESENTACION_DE_DATOS

python3 -m venv .venv
source .venv/bin/activate            # en Windows: .venv\Scripts\activate
pip install -r requirements.txt

jupyter notebook notebooks/avance1_segmentacion_clientes.ipynb
```

En Jupyter: *Run* → *Run All Cells*. La ejecución completa tarda menos de un minuto.

Para ejecutarlo sin abrir Jupyter:

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/avance1_segmentacion_clientes.ipynb
```

El notebook regenera las figuras, `reports/resultados.json`, `reports/perfil_segmentos.csv` y `data/processed/clientes_segmentados.csv`. Los resultados son reproducibles porque la semilla está fijada (`random_state=42`).

## Metodología

| Etapa | Qué se hizo | Funciones principales |
|---|---|---|
| Carga y estructura | Forma, tipos de dato y primer vistazo | `read_csv`, `shape`, `info`, `head`, `describe` |
| EDA univariado | Distribuciones, asimetría y variables categóricas | `hist`, `boxplot`, `skew`, `value_counts` |
| EDA multivariado | Correlaciones, dispersión e impago por grupo | `corr`, `sns.heatmap`, `sns.scatterplot`, `groupby`, `qcut` |
| Faltantes | Códigos fuera del diccionario convertidos en `NaN`; eliminación de faltantes y duplicados (1.45 % de las filas) | `isnull`, `where`, `duplicated`, `dropna` |
| Atípicos y transformación | IQR, `log1p`, recorte en el percentil 99, variación mensual de la deuda y escalado | `quantile`, `clip`, `np.log1p`, `pct_change`, `StandardScaler` |
| Clustering | K-Means sobre 6 variables de comportamiento escaladas; k elegido con codo y silueta | `KMeans`, `inertia_`, `silhouette_score` |
| Hallazgos | Perfil, tasa de impago y peso de cada segmento | `groupby`, `sns.heatmap` |

**Decisiones clave:**
- `EDUCATION` 0/5/6 y `MARRIAGE` 0 no figuran en el diccionario: se tratan como faltantes.
- Los estados de pago -2 y 0 se interpretan como "sin atraso".
- Las facturas negativas (saldo a favor) cuentan como deuda 0.
