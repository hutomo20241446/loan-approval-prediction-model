FROM tensorflow/serving:latest

COPY ./output/serving_model/1752317294 /models/loan_model/1
COPY ./config/prometheus.config /model_config/

# Gunakan environment variable PORT dari Railway
ENV MODEL_NAME=loan_model
ENV MONITORING_CONFIG="/model_config/prometheus.config"
ENV ENABLE_MONITORING="1"

# Entrypoint script yang adaptif dengan PORT dinamis
RUN echo '#!/bin/bash \n\n\
PORT=${PORT:-8501} \n\
env \n\
tensorflow_model_server \
--port=8500 \
--rest_api_port=${PORT} \
--model_name=${MODEL_NAME} \
--model_base_path=/models/${MODEL_NAME} \
--monitoring_config_file=${MONITORING_CONFIG} \
"$@"' > /usr/bin/tf_serving_entrypoint.sh \
&& chmod +x /usr/bin/tf_serving_entrypoint.sh
