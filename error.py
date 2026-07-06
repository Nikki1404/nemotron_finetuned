watch -n 5 'ps -ef | grep -E "docker build|buildkit|containerd" | grep -v grep; echo "---"; docker system df; echo "---"; docker images | grep nemotron_finetuned'
