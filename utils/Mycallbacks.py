## 自定义callbacks, 每1000epoch，打印变量的值

from deepxde.callbacks import Callback
import os
import torch

class printVariable(Callback):
    def __init__(self, var_list, period=1000, filename=None, precision=2):
        super().__init__()
        self.var_list = var_list if isinstance(var_list, list) else [var_list]
        self.period = period
        self.precision = precision

        #self.file = sys.stdout if filename is None else open(filename, "w", buffering=1)
        self.value = None
        self.epochs_since_last = 0
        self.model = None
    
    def on_train_begin(self):

        self.value = [var.detach().item() for var in self.var_list]
        print(
            'Variable pred',
            self.value
        )
    
    def on_epoch_end(self):
        self.epochs_since_last += 1
        if self.epochs_since_last >= self.period:
            self.epochs_since_last = 0
            self.on_train_begin()

    def get_value(self):
        """Return the variable values."""
        return self.value
    
    def set_model(self, model):
        if model is not self.model:
            self.model = model
            self.init()
   
class saveModel(Callback):
    def __init__(self, saveDir, saveName, period=1000, filename=None, precision=2):
        super().__init__()
        #self.var_list = var_list if isinstance(var_list, list) else [var_list]
        self.period = period
        self.precision = precision

        #self.file = sys.stdout if filename is None else open(filename, "w", buffering=1)
        self.value = None
        self.epochs_since_last = 0
        self.model = None
        self.modelNum = 0
        self.saveDir = saveDir
        self.saveName = saveName
        
    def set_model(self, model):
        if model is not self.model:
            self.model = model
            self.init()    
            
    def on_epoch_end(self):
        self.epochs_since_last += 1
        if self.epochs_since_last >= self.period:
            self.modelNum += 1
            self.epochs_since_last = 0  
            if not os.path.exists(self.saveDir):
                #os.mkdir(self.saveDir)
                os.makedirs(self.saveDir)
            savePath = os.path.join(self.saveDir, self.saveName+str(self.modelNum)+'.pt')
            #print(self.model.net.state_dict())
            checkpoint = {
                "model_state_dict": self.model.net.state_dict(),
                "optimizer":self.model.opt.state_dict()
            }
            torch.save(checkpoint, savePath)            

class loadModel(Callback):
    def __init__(self, loadPath):
        super().__init__()
        #self.var_list = var_list if isinstance(var_list, list) else [var_list]
        self.loadPath = loadPath
        
    def set_model(self, model):
        if model is not self.model:
            self.model = model
            self.init()    
            
    def on_train_begin(self):
        checkpoint = torch.load(self.loadPath)
        self.model.net.load_state_dict(checkpoint["model_state_dict"])
        self.model.opt.load_state_dict(checkpoint["optimizer"])
        print('Load model successful')
        
class EarlyStoping(Callback):
    def __init__(self,sigma=5e-4):
        super().__init__()
        #self.var_list = var_list if isinstance(var_list, list) else [var_list]
        self.epochs_since_last = 0
        self.sigma = sigma
        
    def on_epoch_end(self):
        self.epochs_since_last += 1
        train_loss = sum(self.model.train_state.loss_train)
        if train_loss < self.sigma:
            self.model.stop_training = True
            print('Early stoping')
    