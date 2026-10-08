# 保留最佳旧实验权重并清理其余训练状态

唯一清理目录为 `/xiliang/LXY/baseline_v3_lab_20261004`。按用户最新授权，保留一个最佳版本的完整180组权重，其余旧权重和全部旧续训状态删除；不再完整下载已决定丢弃的权重。

选择标准是相对同一历史原版180任务的平均相对MSE，每任务等权。V3插件版约改善1.07%，V4插件版约退化1.92%；V3、V4无插件control分别约改善0.79%、0.84%。因此保留V3插件版180个checkpoint，约5.56GiB。MAE排序不同，完整排名写入retention_plan.json。这只是旧版本保留依据：训练协议有差异，不能当作严格因果比较。

`backup_cleanup.py` 后台SSH读取服务器清单，将全部旧版本超参数、完成记录、指标下载至 `D:/LXY_server_backup/20261008_best_V3_internal` 并校验SHA256，再推送GitHub。权重留在服务器；GitHub保存其路径、大小和SHA256，二进制权重不上传Git。密码通过隐藏输入及匿名管道传入内存，不写文件、命令行或日志。

`remote.py` 只使用 `/xiliang/LXY/envs/lxy/bin/python`。它保护180个选中checkpoint，在全部保留权重及待删文件核验通过后逐项unlink其余旧checkpoint、全部resume和已安装的offline_wheels副本。预训练模型、源码、数据、指标、预测及日志保留。当前r3代码、日志和lxy环境均在清理范围之外。不会递归删除目录或结束服务器进程。

GitHub发布失败、指标归档损坏、保留权重缺失或变化、待删文件变化、不安全链接或仍有旧实验进程都会拒绝清理。旧“完整备份所有文件”后台任务已停止，没有按旧方案删除生产文件。状态见本地STATUS.json，实际删除清单和完成标记为deleted_files.json、CLEANUP_COMPLETED.json。
