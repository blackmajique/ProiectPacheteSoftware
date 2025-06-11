# Analiza Performanței Studenților – Proiect Pachete Software

Aplicație educațională dezvoltată în Python + Streamlit, cu suport SAS, pentru analiza stilului de viață al studenților și a impactului acestuia asupra performanței academice. Proiectul face parte din disciplina **Pachete Software**, Facultatea CSIE – ASE București.

---

## Obiectiv

Scopul aplicației este de a explora factorii care influențează performanța academică a studenților, precum stresul, somnul, venitul familiei sau activitățile extracurriculare. Se analizează un set de date real de pe Kaggle, folosind statistici descriptive, regresii, codificare și prelucrare a datelor.

---

## Structura proiectului

```
ProiectPacheteSoftware
├── project.py                 # Script principal Streamlit
├── requirements.txt          # Dependențe Python
├── .gitignore, .gitattributes
├── Proiect.docx / .pdf       # Documentația completă
└── datasets/
    ├── Students_Grading_Dataset.csv
```

---

## Cum se rulează aplicația

1. Instalează Streamlit și celelalte biblioteci:
```bash
pip install -r requirements.txt
```

2. Rulează aplicația:
```bash
streamlit run project.py
```

---

## Funcționalități principale

- Încărcare fișiere CSV cu anteturi prelucrate automat
- Tratare valori lipsă (zero, medie, interpolare, MICE)
- Codificare variabile categorice (Label Encoding)
- Standardizare și scalare numerică (Standard, MinMax, Robust)
- Eliminare outlieri (Z-Score, percentile)
- Vizualizări: histograme, boxplot, violine plot, heatmap corelații
- Test ANOVA, regresie liniară și regresie logistică
- SelectKBest pentru reducerea predictorilor
- Modele predictive (scor total peste medie)

---

## Pachet SAS integrat

Pe lângă partea Python, proiectul include:
- Import și manipulare date în SAS
- Crearea de subseturi pe bază de condiții (performanță, venit, acces internet)
- Etichetare studenți în funcție de implicare și eligibilitate la bursă
- Funcții SAS (mean, catx, abs, etc.)
- Join SQL pentru bursieri

---

## Setul de date

[Students Grading Dataset – Kaggle](https://www.kaggle.com/datasets/mahmoudelhemaly/students-grading-dataset)  
Conține variabile legate de: vârstă, gen, departament, note, participare, stres, venit, acces internet, somn, activități extracurriculare etc.

