root@cx-apps:/home/nikita_verma2/nemotron_finetuned_compressed# docker run --gpus all -it --rm -v $PWD:/workspace -v $PWD/ft_models:/srv/models nemotron_finetuned bash
docker: Error response from daemon: could not select device driver "" with capabilities: [[gpu]]

Run 'docker run --help' for more information
root@cx-apps:/home/nikita_verma2/nemotron_finetuned_compressed# 
nikita_verma2@cx-speech:~/nemotron_finetuned_compressed$ docker build -t nemotron_finetuned .
[+] Building 118.6s (15/17)                                                                     docker:default
 => [internal] load build definition from Dockerfile                                                      0.0s
 => => transferring dockerfile: 3.24kB                                                                    0.0s
 => [internal] load metadata for docker.io/nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04                   0.4s
 => [internal] load .dockerignore                                                                         0.0s
 => => transferring context: 2B                                                                           0.0s
 => [ 1/13] FROM docker.io/nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04@sha256:2fcc4280646484290cc50dce5  0.0s
 => => resolve docker.io/nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04@sha256:2fcc4280646484290cc50dce5e6  0.0s
 => [internal] load build context                                                                         0.0s
 => => transferring context: 1.13kB                                                                       0.0s
 => CACHED [ 2/13] WORKDIR /srv                                                                           0.0s
 => CACHED [ 3/13] RUN apt-get update &&     apt-get upgrade -y &&     apt-get dist-upgrade -y &&     ap  0.0s
 => CACHED [ 4/13] RUN wget --no-check-certificate https://www.python.org/ftp/python/3.11.9/Python-3.11.  0.0s
 => CACHED [ 5/13] RUN python3.11 -m ensurepip &&     python3.11 -m pip install --upgrade pip setuptools  0.0s
 => CACHED [ 6/13] COPY requirements.txt .                                                                0.0s
 => CACHED [ 7/13] RUN python3.11 -m pip install --no-cache-dir -r requirements.txt                       0.0s
 => CACHED [ 8/13] RUN python3.11 -m pip install --no-cache-dir     "nemo_toolkit[asr] @ git+https://git  0.0s
 => CACHED [ 9/13] RUN python3.11 -m pip install --no-cache-dir --force-reinstall     torch==2.6.0     t  0.0s
 => CACHED [10/13] RUN python3.11 - <<'EOF'                                                               0.0s
 => CACHED [11/13] RUN python3.11 - <<'EOF'                                                               0.0s
 => [12/13] RUN python3.11 - <<'EOF'                                                                    118.1s
 => => #       mt-MT: 102                                                                                     
 => => #       auto: 101                                                                                      
 => => #     num_prompts: 128                                                                                 
 => => #     subsampling_factor: 8                                                                            
 => => #     training_mode: false  
