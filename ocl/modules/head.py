import torch
import torch.nn as nn

# multi-layer perceptron classifier
class MLP(nn.Module):
    def __init__(self, 
                 input_dims:int, 
                 num_classes:int):
        super(MLP, self).__init__()
        
        self.classifier = nn.Sequential(nn.BatchNorm1d(input_dims),
                                        nn.Dropout(p=0.25, inplace=False),
                                        nn.Linear(input_dims, 512, bias=False),
                                        nn.ReLU(inplace=True),
                                        nn.BatchNorm1d(512),
                                        nn.Dropout(p=0.5, inplace=False),
                                        nn.Linear(512, num_classes, bias=False))
        
        
    def forward(self, x):
        assert isinstance(x, torch.Tensor)
        
        out = self.classifier(x)
        return out



class KNNClassifier(nn.Module):
    def __init__(self, feat_dim=2048, num_classes=7, feat_type='cl2n', dist_type='l2'):
        super(KNNClassifier, self).__init__()
        assert feat_type in ['un', 'l2n', 'cl2n'], "feat_type is wrong!!!"
        assert dist_type in ['l2', 'cos'], "dist_type is wrong!!!"
        self.feat_dim = feat_dim
        self.num_classes = num_classes
        self.centroids = torch.randn(num_classes, feat_dim)
        self.feat_mean = torch.randn(feat_dim)
        self.feat_type = feat_type
        self.dist_type = dist_type
        self.initialized = False
    
    def get_centroids(self, features, labels):
        # get centroids
        centroids = list()
        for c in range(1, self.num_classes, 1):
            class_indices = (labels == c).nonzero()
            class_features = features[class_indices]
            centroid = torch.mean(class_features, dim=0)
            centroids.append(centroid)
        
        self.centroids.copy_(torch.tensor(centroids))
        assert self.centroids.shape == torch.size([self.num_classes, self.feat_dim])
        
    def update(self, features: torch.Tensor, labels: torch.Tensor):
        mean = torch.mean(features, dim=0)
        self.feat_mean.copy_(mean)
        self.get_centroids(features, labels)
        self.initialized = True

    def forward(self, inputs, *args):
        centroids = self.centroids
        feat_mean = self.feat_mean

        # Feature transforms
        if self.feat_type == 'cl2n':
            inputs = inputs - feat_mean
            #centroids = centroids - self.feat_mean

        if self.feat_type in ['l2n', 'cl2n']:
            norm_x = torch.norm(inputs, 2, 1, keepdim=True)
            inputs = inputs / norm_x

            #norm_c = torch.norm(centroids, 2, 1, keepdim=True)
            #centroids = centroids / norm_c
        
        # Logit calculation
        if self.dist_type == 'l2':
            logit = self.l2_similarity(inputs, centroids)
        elif self.dist_type == 'cos':
            logit = self.cos_similarity(inputs, centroids)
        
        return logit

    def l2_similarity(self, A, B):
        # input A: [bs, fd] (batch_size x feat_dim)
        # input B: [nC, fd] (num_classes x feat_dim)
        feat_dim = A.size(1)

        AB = torch.mm(A, B.t())
        AA = (A**2).sum(dim=1, keepdim=True)
        BB = (B**2).sum(dim=1, keepdim=True)
        dist = AA + BB.t() - 2*AB

        return -dist
    
    def cos_similarity(self, A, B):
        feat_dim = A.size(1)
        AB = torch.mm(A, B.t())
        AB = AB / feat_dim
        return AB