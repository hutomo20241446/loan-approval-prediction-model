"""main.py
pipeline orchestration for loan prediction model"""

import os
from absl import logging
from tfx.orchestration import metadata, pipeline
from tfx.orchestration.beam.beam_dag_runner import BeamDagRunner
from modules.components import init_components

PIPELINE_NAME = "muhammad-fahmi-hutomo-pipeline"
DATA_ROOT = "data"
TRANSFORM_MODULE = os.path.abspath("modules/transform.py")
TRAINER_MODULE = os.path.abspath("modules/trainer.py")
TUNER_MODULE = os.path.abspath("modules/tuner.py")
OUTPUT_BASE = "output"

os.makedirs(OUTPUT_BASE, exist_ok=True)
serving_model_dir = os.path.abspath(os.path.join(OUTPUT_BASE, 'serving_model'))
pipeline_root = os.path.abspath(os.path.join(OUTPUT_BASE, PIPELINE_NAME))
metadata_path = os.path.abspath(os.path.join(pipeline_root, "metadata.sqlite"))

def init_pipeline(components):
    beam_args = [
        "--direct_running_mode=multi_processing",
        "--direct_num_workers=0"
    ]

    return pipeline.Pipeline(
        pipeline_name=PIPELINE_NAME,
        pipeline_root=pipeline_root,
        components=components,
        enable_cache=False,
        metadata_connection_config=metadata.sqlite_metadata_connection_config(metadata_path),
        beam_pipeline_args=beam_args
    )

if __name__ == "__main__":
    logging.set_verbosity(logging.INFO)
    logging.info(f"Serving model will be saved to: {serving_model_dir}")
    os.makedirs(serving_model_dir, exist_ok=True)

    if not os.access(serving_model_dir, os.W_OK):
        raise PermissionError(f"Cannot write to serving directory: {serving_model_dir}")

    components = init_components(
        DATA_ROOT,
        TRANSFORM_MODULE,
        TRAINER_MODULE,
        serving_model_dir,
        tuner_module=TUNER_MODULE
    )

    p = init_pipeline(components)
    logging.info("Starting pipeline execution...")
    BeamDagRunner().run(p)
    logging.info(f"Pipeline completed. Check {serving_model_dir} for the saved model.")
