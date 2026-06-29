docker run --gpus all -it --rm -p 8002:8002 -v $PWD/ft_models:/srv/models -e MODEL_NAME=/srv/models/finetuned_nemotron_final.nemo nemotron-asr
