"""modules/tuner.py"""

import os
import keras_tuner
from keras_tuner import Objective
import tensorflow as tf
import tensorflow_transform as tft
from tensorflow.keras import layers, metrics
from tfx.components.trainer.fn_args_utils import FnArgs
from tfx.components.tuner.component import TunerFnResult

# Constants
LABEL_KEY = "Loan_Status"
BATCH_SIZE = 64
EPOCHS = 10
TRAIN_STEPS = 1000
EVAL_STEPS = 500

# Feature keys should match transform.py
FEATURE_KEYS = [
    "Gender", "Married", "Dependents", "Education", "Self_Employed",
    "ApplicantIncome", "CoapplicantIncome", "LoanAmount", 
    "Loan_Amount_Term", "Credit_History", "Property_Area"
]

CATEGORICAL_FEATURES = ["Gender", "Married", "Education", "Self_Employed", "Property_Area"]
NUMERICAL_FEATURES = ["ApplicantIncome", "CoapplicantIncome", "LoanAmount", 
                     "Loan_Amount_Term", "Credit_History", "Dependents"]

def input_fn(file_pattern, tf_transform_output, batch_size=BATCH_SIZE):
    """Create dataset from transformed data"""
    transformed_feature_spec = tf_transform_output.transformed_feature_spec().copy()
    
    dataset = tf.data.experimental.make_batched_features_dataset(
        file_pattern=file_pattern,
        batch_size=batch_size,
        features=transformed_feature_spec,
        reader=lambda x: tf.data.TFRecordDataset(x, compression_type="GZIP"),
        label_key=LABEL_KEY,
        num_epochs=1
    )
    return dataset

def get_hyperparameters() -> keras_tuner.HyperParameters:
    """Define the hyperparameter search space."""
    hp = keras_tuner.HyperParameters()
    
    # Architecture hyperparameters
    hp.Int('num_layers', 1, 3)
    for i in range(3):  # Max layers we might use
        hp.Int(f'units_{i}', 32, 256, step=32)
        hp.Float(f'dropout_{i}', 0.1, 0.5)
    
    # Learning rate
    hp.Float('learning_rate', 1e-4, 1e-2, sampling='log')
    
    # Categorical features embedding
    for feature in CATEGORICAL_FEATURES:
        hp.Int(f'{feature}_vocab_size', 2, 16)
        hp.Int(f'{feature}_embed_dim', 1, 4)
    
    return hp

def build_model(hp: keras_tuner.HyperParameters):
    """Model builder with hyperparameters"""
    # Create input layers for all features
    inputs = {
        feature_name: layers.Input(shape=(1,), name=feature_name, dtype=tf.float32)
        for feature_name in FEATURE_KEYS
    }
    
    # Process numerical features
    numerical_features = [
        layers.Reshape((1,))(inputs[feature]) 
        for feature in NUMERICAL_FEATURES
    ]
    
    # Process categorical features
    categorical_features = []
    for feature in CATEGORICAL_FEATURES:
        embed = layers.Embedding(
            input_dim=hp.get(f'{feature}_vocab_size'),
            output_dim=hp.get(f'{feature}_embed_dim')
        )(inputs[feature])
        embed = layers.Reshape((-1,))(embed)
        categorical_features.append(embed)
    
    # Concatenate all features
    concatenated = layers.concatenate(numerical_features + categorical_features)
    
    # Hidden layers with hyperparameter tuning
    for i in range(hp.get('num_layers')):
        concatenated = layers.Dense(
            units=hp.get(f'units_{i}'),
            activation='relu'
        )(concatenated)
        concatenated = layers.Dropout(
            hp.get(f'dropout_{i}')
        )(concatenated)
    
    # Output layer
    outputs = layers.Dense(1, activation='sigmoid')(concatenated)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            hp.get('learning_rate')
        ),
        loss='binary_crossentropy',
        metrics=[metrics.AUC(name='auc'), metrics.BinaryAccuracy()]
    )
    return model

def tuner_fn(fn_args: FnArgs):
    """Tuner function with proper TFX-compatible return format."""
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_graph_path)
    
    # Initialize the tuner
    tuner = keras_tuner.RandomSearch(
        hypermodel=build_model,
        hyperparameters=get_hyperparameters(),
        objective=Objective('val_auc', direction='max'),
        max_trials=fn_args.custom_config['keras_tuner']['max_trials'],
        directory=os.path.join(fn_args.custom_config['keras_tuner']['directory']),
        project_name=fn_args.custom_config['keras_tuner']['project_name'],
        overwrite=True
    )

    # Prepare datasets
    train_dataset = input_fn(fn_args.train_files, tf_transform_output)
    eval_dataset = input_fn(fn_args.eval_files, tf_transform_output)

    # Run hyperparameter search
    tuner.search(
        train_dataset,
        validation_data=eval_dataset,
        epochs=EPOCHS,
        steps_per_epoch=TRAIN_STEPS,
        validation_steps=EVAL_STEPS,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor='val_auc',
                patience=3,
                mode='max',
                restore_best_weights=True
            )
        ]
    )

    # Return the proper TFX TunerFnResult
    return TunerFnResult(
        tuner=tuner,
        fit_kwargs={
            'x': train_dataset,
            'validation_data': eval_dataset,
            'steps_per_epoch': TRAIN_STEPS,
            'validation_steps': EVAL_STEPS
        }
    )
