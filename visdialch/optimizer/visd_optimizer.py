from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
import bisect

class WarmupLinearSegSchedule(LambdaLR): 
     def __init__(self, optimizer, warmup_steps, t_total, seg_points, seg_ratio, last_epoch=-1): 
         self.warmup_steps = warmup_steps 
         self.t_total = t_total 
         self.alpha = [seg_ratio[i]*(t_total - self.warmup_steps)/(t_total - seg_points[i]) for i in range(len(seg_points))]
         self.alpha = [1.] + self.alpha
         self.seg_points = seg_points + [self.t_total]
         super(WarmupLinearSegSchedule, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch) 
 
     def lr_lambda(self, step): 
         if step < self.warmup_steps: 
             return float(step) / float(max(1, self.warmup_steps))
         idx = bisect.bisect_left(self.seg_points, step) 
         return self.alpha[idx] * max(0.0, float(self.t_total - step) / float(max(1.0, self.t_total - self.warmup_steps))) 

