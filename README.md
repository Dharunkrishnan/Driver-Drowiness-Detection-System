# 🚗 Driver Drowsiness Detection System (Enhanced v3)

## 📌 Overview

Drowsy driving is a major cause of road accidents. This project implements a **real-time driver monitoring system** that detects fatigue using facial landmarks and behavioral metrics.

The system uses computer vision techniques to monitor eye closure, yawning, head nodding, blink rate, and micro-sleep events.

---

## 🚀 Features

* 👁️ Eye Aspect Ratio (EAR) for eye closure detection
* 📊 PERCLOS (Percentage of Eye Closure)
* 😮 Yawn detection using Mouth Aspect Ratio (MAR)
* 🤕 Head nod detection (fatigue indicator)
* 👀 Blink rate tracking (blinks per minute)
* ⏱️ Micro-sleep detection (continuous eye closure)
* 🔊 Real-time audio alert system
* 📄 Event logging (CSV file)
* 🖥️ Live dashboard with visual metrics

---

## 🧠 Tech Stack

* Python
* OpenCV
* dlib (Facial Landmark Detection)
* NumPy
* SciPy

---

## 📂 Project Structure

```bash
driver-drowsiness-detection/
│── main.py                # Main detection script
│── drowsiness_log.csv    # Event logs
│── requirements.txt
│── shape_predictor_68_face_landmarks.dat
│── README.md
```

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/driver-drowsiness-detection-ml.git
cd driver-drowsiness-detection-ml
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Download facial landmark model

Download from:
http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2

Extract and place in project folder.

---

## ▶️ Usage

```bash
python main.py
```

Controls:

* `F` → Toggle fullscreen
* `Esc` → Exit

---

## 📊 Detection Metrics

| Metric      | Description                     |
| ----------- | ------------------------------- |
| EAR         | Detects eye closure             |
| PERCLOS     | % of time eyes are closed       |
| MAR         | Detects yawning                 |
| Blink Rate  | Blinks per minute               |
| Micro-sleep | Continuous eye closure duration |
| Head Nod    | Sudden head drop detection      |

---

## 📁 Output

* Logs stored in:

  ```
  drowsiness_log.csv
  ```
* Each event includes timestamp, EAR, PERCLOS, MAR, blink rate, etc.

---

## ⚠️ Alerts

* Audio alarm triggered during:

  * Drowsiness
  * Micro-sleep
  * Excessive nodding

---

## 📈 Future Improvements

* Deep learning (CNN / LSTM)
* Mobile app integration
* Night vision enhancement
* Driver identity tracking
* Integration with vehicle systems

---

## 🛡️ Real-World Applications

* Smart vehicles
* Fleet monitoring systems
* Driver safety analytics
* IoT-based transport systems

---

## 🤝 Contributing

Pull requests are welcome!

---

## 🙋‍♂️ Author

Dharun Krishna
