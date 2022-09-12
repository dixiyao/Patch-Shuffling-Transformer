import torch

import utilsenc

class t2t_pretrain(torch.nn.Module):
    def __init__(self,net,k1=1):
        super(t2t_pretrain, self).__init__()
        self.model = net
        self.k1=k1

    def forward_features(self, x):
        with torch.no_grad():
            x = self.model.tokens_to_token(x)
            x =utilsenc.BatchPatchPartialShuffle(x,self.k1)
            cls_token = self.model.cls_token.expand(x.shape[0], -1,-1)  # stole cls_tokens impl from Phil Wang, thanks
            x = torch.cat((cls_token, x), dim=1)
            x = self.model.blocks[:1](x)
            x =x[:,1:,:]
        cls_token = self.model.cls_token.expand(x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_token, x), dim=1)
        x  = self.model.blocks[1:](x)
        x = self.model.norm(x)
        return x[:, 0]

    def forward(self, x):
        x= self.forward_features(x)
        x = self.model.head(x)
        return x
