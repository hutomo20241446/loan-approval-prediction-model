"""modules/transform.py"""

import tensorflow as tf
import tensorflow_transform as tft

LABEL_KEY = "Loan_Status"
FEATURE_KEYS = ["Gender", "Married", "Dependents", "Education", "Self_Employed", 
                "ApplicantIncome", "CoapplicantIncome", "LoanAmount", 
                "Loan_Amount_Term", "Credit_History", "Property_Area"]

CATEGORICAL_FEATURES = ["Gender", "Married", "Education", "Self_Employed", "Property_Area"]
NUMERICAL_FEATURES = ["Dependents", "ApplicantIncome", "CoapplicantIncome", "LoanAmount", 
                      "Loan_Amount_Term", "Credit_History"]

def preprocessing_fn(inputs):
    outputs = {}

    # Transform fitur kategorikal
    for key in CATEGORICAL_FEATURES:
        outputs[key] = tft.compute_and_apply_vocabulary(
            tf.strings.lower(tf.strings.strip(inputs[key])),
            vocab_filename=f'{key}_vocab'
        )

    # Transform fitur numerik
    for key in NUMERICAL_FEATURES:
        outputs[key] = tf.cast(inputs[key], tf.float32)

    # Label (sudah 0/1 dari tahap cleaning)
    outputs[LABEL_KEY] = tf.cast(inputs[LABEL_KEY], tf.float32)

    return outputs
