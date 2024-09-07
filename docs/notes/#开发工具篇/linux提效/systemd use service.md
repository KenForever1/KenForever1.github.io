+ 查看服务的日志
```
journalctl -u xxx.service -e
```

+ 查看本机所有的running的service
```
systemctl list-units --type=service --state=running | grep xxx
```

+ 重启服务
```
systemctl start xxx.service
```
+ 部署服务
```
sudo cp xxx.service  /etc/systemd/system/
sudo cp xxx.sh /usr/local/bin/
```

### watch dog

```
#!/bin/bash

SERVICE_COMMAND_NAME="xxx"
SERVICE_NAME="xxx.service"

while true; do
  if ! pgrep -x "$SERVICE_COMMAND_NAME" > /dev/null; then
    echo "$SERVICE_NAME is not running, restarting..."
    sudo systemctl restart "$SERVICE_NAME"
  else
    echo "$SERVICE_NAME is running..."
  fi
  sleep 10
done
```