## Tensorboard

We provide basic tensorboard to track your training status. Run the tensorboard with the following command: 

```bash
tensorboard --logdir ./outputs --port 50001 --bind_all
```

where `PORT` for tensorboard is 50001.  
Note that the default directory of saving result will be `./outputs` directory.