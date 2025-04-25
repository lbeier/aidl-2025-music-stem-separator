# Stem Separation - Vocal & Accompaniment


This project focuses on separating audio tracks into **vocal** and **accompaniment** stems using deep learning. The goal is to build an MVP model trained on the [MUSDB18](https://sigsep.github.io/datasets/musdb.html) dataset.

> 🧑‍🎓 This is the **final project** for the **Artificial Intelligence with Deep Learning** postgraduate course at **Universitat Politècnica de Catalunya (UPC)**.


## 🧠 Objective

Train a model capable of separating a mixed music track into:
- **Vocal track**
- **Accompaniment track** (everything else)

## 📦 Dataset

- **MUSDB18** – a professionally produced dataset with 150 full-length music tracks, each with isolated stems (vocals, drums, bass, and others).
- Format: WAV files, multitrack
- Download instructions: [https://sigsep.github.io/datasets/musdb.html](https://sigsep.github.io/datasets/musdb.html)

## 🛠️ Tech Stack

- Python
- PyTorch

Other TBD.

## 🚀 MVP Goal

Basic stem separation into:
`input.wav → vocal.wav + accompaniment.wav`

## 🔧 Setup
TBD

## ⏳ Status

MVP in development. Currently working on:
	•	Data pipeline
	•	Baseline model

## 📌 Notes
	•	MUSDB18 is not included in this repo. Download it manually and place it in the data/ directory.
	•	Model and evaluation scripts to follow.