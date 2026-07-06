root@cx-apps:/home/nikita_verma2/nemotron_finetuned_compressed# docker run --gpus all -it --rm -v $PWD:/workspace -v $PWD/ft_models:/srv/models nemotron_finetuned bash
docker: Error response from daemon: could not select device driver "" with capabilities: [[gpu]]

Run 'docker run --help' for more information
root@cx-apps:/home/nikita_verma2/nemotron_finetuned_compressed# 
