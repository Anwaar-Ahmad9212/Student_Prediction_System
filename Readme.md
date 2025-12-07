
---

# **Grade Prediction Dashboard**

A **Streamlit-based academic forecasting system** that predicts student grades using pre-trained regression models. The dashboard provides analytics, predictions, and insights into student performance.

---

## **Features**

* Predict grades for Midterm 1, Midterm 2, and Final Exam.
* Multiple model options:

  * Simple Linear Regression
  * Multiple Linear Regression (Best)
  * Dummy Baseline
* Interactive data analytics with correlation matrices, histograms, and pairplots.
* Workflow visualization from raw data to predictions.
* Dataset explorer with statistical summaries.

---

## **Requirements**

Install Python packages listed in `requirements.txt`:

```bash
pip install -r requirements.txt
```

**Main dependencies:**

* `streamlit`
* `pandas`
* `numpy`
* `matplotlib`
* `seaborn`
* `joblib`

---

## **Files Needed**

| File                            | Description                                                                                       |
| ------------------------------- | ------------------------------------------------------------------------------------------------- |
| `app.py`                        | Main Streamlit application script                                                                 |
| `universal_student_dataset.csv` | Dataset containing student scores (`Assign_Pct`, `Quiz_Pct`, `Mid1_Pct`, `Mid2_Pct`, `Final_Pct`) |
| `logo.jpeg`                     | Logo for the sidebar                                                                              |
| `RQ1_Simple_2.pkl`              | Pre-trained model for RQ1 (Simple Linear Regression)                                              |
| `RQ1_Linear_Multi.pkl`          | Pre-trained model for RQ1 (Multiple Linear Regression)                                            |
| `RQ1_Dummy_2.pkl`               | Dummy baseline for RQ1                                                                            |
| `RQ2_Simple_2.pkl`              | Pre-trained model for RQ2 (Simple Linear Regression)                                              |
| `RQ2_Linear_Multi.pkl`          | Pre-trained model for RQ2 (Multiple Linear Regression)                                            |
| `RQ2_Dummy_2.pkl`               | Dummy baseline for RQ2                                                                            |
| `RQ3_Simple_2.pkl`              | Pre-trained model for RQ3 (Simple Linear Regression)                                              |
| `RQ3_Linear_Multi.pkl`          | Pre-trained model for RQ3 (Multiple Linear Regression)                                            |
| `RQ3_Dummy_2.pkl`               | Dummy baseline for RQ3                                                                            |

---

## **Folder Structure**

```
grade_prediction_dashboard/
│
├─ app.py
├─ universal_student_dataset.csv
├─ logo.jpeg
├─ RQ1_Simple_2.pkl
├─ RQ1_Linear_Multi.pkl
├─ RQ1_Dummy_2.pkl
├─ RQ2_Simple_2.pkl
├─ RQ2_Linear_Multi.pkl
├─ RQ2_Dummy_2.pkl
├─ RQ3_Simple_2.pkl
├─ RQ3_Linear_Multi.pkl
├─ RQ3_Dummy_2.pkl
├─ requirements.txt
```

---

## **How to Run Locally**

1. Clone/download the repository.
2. Ensure all `.pkl` models, dataset, and logo are in the same folder.
3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Run the Streamlit app:

```bash
streamlit run app.py
```

5. Open the URL provided by Streamlit in your browser.

---

## **Notes**

* Ensure that the dataset contains **all required columns**:
  `Assign_Pct`, `Quiz_Pct`, `Mid1_Pct`, `Mid2_Pct`, `Final_Pct`.
* Pre-trained models must match the features defined in `MODEL_FEATURES`.
* Works best with **small-to-medium datasets**; larger datasets may need optimization.

---
