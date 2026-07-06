(base) root@EC03-E01-AICOE1:/home/CORP/re_nikitav/nemotron_finetuned# git remote set-url origin https://ucgithub.exlserv
ice.com/Unified-Cloud-DevOps/bu-digital-cx-speech-asr-realtime-ml.git
(base) root@EC03-E01-AICOE1:/home/CORP/re_nikitav/nemotron_finetuned# git push origin main:feature/nemotron_finetuned_3.5
fatal: unable to access 'https://ucgithub.exlservice.com/Unified-Cloud-DevOps/bu-digital-cx-speech-asr-realtime-ml.git/': Proxy CONNECT aborted
(base) root@EC03-E01-AICOE1:/home/CORP/re_nikitav/nemotron_finetuned# unset http_proxy
(base) root@EC03-E01-AICOE1:/home/CORP/re_nikitav/nemotron_finetuned# unset https_proxy
(base) root@EC03-E01-AICOE1:/home/CORP/re_nikitav/nemotron_finetuned# git push origin main:feature/nemotron_finetuned_3.5
Username for 'https://ucgithub.exlservice.com': re-nikitav
Password for 'https://re-nikitav@ucgithub.exlservice.com':
To https://ucgithub.exlservice.com/Unified-Cloud-DevOps/bu-digital-cx-speech-asr-realtime-ml.git
 ! [rejected]        main -> feature/nemotron_finetuned_3.5 (fetch first)
error: failed to push some refs to 'https://ucgithub.exlservice.com/Unified-Cloud-DevOps/bu-digital-cx-speech-asr-realtime-ml.git'
hint: Updates were rejected because the remote contains work that you do
hint: not have locally. This is usually caused by another repository pushing
hint: to the same ref. You may want to first integrate the remote changes
hint: (e.g., 'git pull ...') before pushing again.
hint: See the 'Note about fast-forwards' in 'git push --help' for details.
