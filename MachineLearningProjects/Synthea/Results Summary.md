First Run:

Accuracy: 85% - The model correctly predicts readmission status 85% of the time

Class 0 (No Readmission):

Precision: 0.69 - When predicting "no readmission", it's correct 69% of the time
Recall: 0.56 - Catches 56% of actual "no readmission" cases

Class 1 (Readmission):

Precision: 0.88 - When predicting "readmission", it's correct 88% of the time
Recall: 0.93 - Catches 93% of actual readmission cases


Second Run:

Class 0 (No Readmission):

Precision: 0.69 → **0.88** (big improvement)


Class 1 (Readmission):

Recall: 0.93 → **0.98** (now catches 98% of readmissions!)

## Clinical Impact

- **Before (0.5 threshold)**: Missing 4 patients who would be readmitted
- **After (0.35 threshold)**: Missing only 1 patient who would be readmitted