# Premier League Predictor (1X2)

Proyecto de **modelos de ML para apuestas 1X2** en la Premier League (temporadas 2018–2026).
Flujo actual implementado: **recopilación de datos**, **normalización** y **validación** (Paso 2).
A futuro: **feature engineering**, **entrenamiento** y **sistema de apuestas**.

---

## ⚙️ Requisitos

- **Python** 3.12 (vía Anaconda/Miniconda recomendado)
- **Git** y **VS Code** (opcional)
- Conexión a internet para obtener partidos (FBref u OpenFootball via GitHub).
  *Las **odds** se cargan desde **CSVs locales** (opción A).*

---

## 🚀 Setup rápido

1) Crear entorno con `conda` usando el `environment.yml` del repo:
```bash
conda env create -f environment.yml
conda activate pl-predictor
