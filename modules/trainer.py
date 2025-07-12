"""modules/trainer.py"""

import tensorflow as tf
import tensorflow_transform as tft
from tensorflow.keras import layers, models
from tfx.components.trainer.fn_args_utils import FnArgs

LABEL_KEY = "Loan_Status"
CATEGORICAL_FEATURES = ["Gender", "Married", "Education", "Self_Employed", "Property_Area"]
NUMERICAL_FEATURES = ["ApplicantIncome", "CoapplicantIncome", "LoanAmount",
                     "Loan_Amount_Term", "Credit_History", "Dependents"]
FEATURE_KEYS = CATEGORICAL_FEATURES + NUMERICAL_FEATURES

def input_fn(file_pattern, tf_transform_output, num_epochs, batch_size=64):
    file_paths = tf.io.gfile.glob(file_pattern)
    transformed_feature_spec = tf_transform_output.transformed_feature_spec()
    parse_feature_spec = transformed_feature_spec.copy()
    label = parse_feature_spec.pop(LABEL_KEY)

    def decode_fn(record_bytes):
        parsed = tf.io.parse_single_example(record_bytes, parse_feature_spec)
        label_value = tf.io.parse_single_example(record_bytes, {LABEL_KEY: label})[LABEL_KEY]
        return parsed, label_value

    dataset = tf.data.TFRecordDataset(file_paths, compression_type='GZIP')
    dataset = dataset.map(decode_fn, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.shuffle(2048)
    dataset = dataset.repeat(num_epochs).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset

def model_builder(tf_transform_output):
    inputs = {
        key: layers.Input(shape=(1,), name=key, dtype=tf.int64 if key in CATEGORICAL_FEATURES else tf.float32)
        for key in FEATURE_KEYS
    }

    categorical = []
    for key in CATEGORICAL_FEATURES:
        vocab_size = tf_transform_output.vocabulary_size_by_name(f'{key}_vocab')
        emb_dim = min(8, vocab_size)
        x = layers.Embedding(input_dim=vocab_size + 1, output_dim=emb_dim, name=f'{key}_embedding')(inputs[key])
        x = layers.Reshape((emb_dim,))(x)
        categorical.append(x)
    categorical = layers.concatenate(categorical)

    numerical = layers.concatenate([inputs[key] for key in NUMERICAL_FEATURES])
    numerical = layers.BatchNormalization()(numerical)
    numerical = layers.Dense(16, activation='relu')(numerical)

    x = layers.concatenate([categorical, numerical])
    x = layers.Dense(32, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(16, activation='relu')(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)

    model = models.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def run_fn(fn_args: FnArgs):
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_graph_path)

    train_dataset = input_fn(fn_args.train_files, tf_transform_output, num_epochs=10)
    eval_dataset = input_fn(fn_args.eval_files, tf_transform_output, num_epochs=10)

    model = model_builder(tf_transform_output)

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy', patience=5, restore_best_weights=True)

    model.fit(
        train_dataset,
        validation_data=eval_dataset,
        epochs=30,
        callbacks=[early_stop]
    )

    # Create serving function
    tft_layer = tf_transform_output.transform_features_layer()
    serving_inputs = {
        key: tf.keras.Input(shape=(1,), name=key,
            dtype=tf.string if key in CATEGORICAL_FEATURES else tf.float32)
        for key in FEATURE_KEYS
    }

    transformed_features = tft_layer(serving_inputs)
    outputs = model(transformed_features)
    serving_model = tf.keras.Model(serving_inputs, outputs)

    serving_model.save(fn_args.serving_model_dir, save_format='tf')
