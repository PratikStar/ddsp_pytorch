# Development
```
# 1. Rsync repo
cd /Users/pratik/repos/ddsp_pytorch
watch -d -n5 "rsync -av --exclude-from=\".rsyncignore_upload\" \"/Users/pratik/repos/ddsp_pytorch\" w:/work/gk77/k77021/repos"

nohup watch -d -n5 rsync -av --exclude-from=".rsyncignore_upload" "/Users/pratik/repos/ddsp_pytorch" w:/work/gk77/k77021/repos 0<&- &> /dev/null &

# 2. Rsync data
cd /Users/pratik/data/timbre
rsync -avz "/Users/pratik/data/timbre" w:/work/gk77/k77021/data

# 3. Files from wisteria
rsync -av w:/work/gk77/k77021/repos/ddsp_pytorch/runs/ddsp-all "/Users/pratik/Downloads"

```
