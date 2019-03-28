This repo contains only one of a several custom video processing 
neural nets.  The one available here is a custom version of 
the Productive Corrective Network (PCN).  (It's model based on 
differential image processing and Kalman filters.)  Other video 
processing networks may be released at a later point.

Although this repo does not contain any working examples with 
this particular PCN model, you can find an example using my NBA 
play-by-play dataset and Horovod (for distributed training) in 
the 'distrib_training_scripts' repository under the 'sample_train_scripts'
directory.

Finally, it should be noted that every video processing network 
found in this repo is deliberately designed to consume a lot 
of GPU memory and therefore may only work on super end-high GPUs with 
their default parameters.  However, every model can also be customized 
(using their respective constructors) to consume a much smaller memory 
footprint if needed.  