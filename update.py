re_nikitav@EC03-E01-AICOE1:~/nemotron_finetuned$ cd /home/CORP/re_nikitav/nemotron_finetuned && rsync -avz --progress --exclude 'ft_models/' --exclude 'lightning_logs/' --exclude '__pycache__/' --exclude '*.pyc' ./ nikita_verma2@cx_speech:~/nemotron_finetuned/
ssh: Could not resolve hostname cx_speech: Temporary failure in name resolution
rsync: connection unexpectedly closed (0 bytes received so far) [sender]
rsync error: error in rsync protocol data stream (code 12) at io.c(232) [sender=3.2.7]
