import torch 
import time, os
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import math
import torch.distributions.log_normal as ln 
from pdb import set_trace as st
from torch.autograd import Variable
import torch.nn.functional as F

eps = 0.000001

def weightNormalize(weights, drop_prob=0.0):
    out = []
    for row in weights:
        if drop_prob==0.0:
            out.append(row**2/torch.sum(row**2))
        else:
            p = torch.randint(0, 2, (row.size())).float().cuda() 
            out.append((row**2/torch.sum(row**2))*p)
    return torch.stack(out)

def weightNormalize1(weights):
    return ((weights**2)/torch.sum(weights**2))


def weightNormalize2(weights):
    return weights/torch.sum(weights**2)

class SURE_pure4D(nn.Module):
    def __init__(self, params, input_shape, out_channels):
        #input_shape must be torch.Size object
        super(SURE_pure4D, self).__init__()
        num_classes = params['num_classes'] 
        num_distr = params['num_distr']
        self.num_repeat = 1
        input_shapes = list(input_shape)
        input_shapes[1] = out_channels
        self.out = out_channels
        
        shapes = torch.Size([num_classes]+input_shapes)
        self.classes = num_classes
        self.shapes = shapes
        self.num_distr = num_distr
        self.sigmas = Variable(torch.ones(self.classes), requires_grad=False).cuda()
        self.X_LEs = Variable(torch.zeros(shapes)+eps, requires_grad=False).cuda()
        self.X_weights = Variable(torch.zeros((num_classes, 1))+eps, requires_grad=False).cuda() 
        self.X_LEs_xy = Variable(torch.zeros(shapes)+eps, requires_grad=False).cuda()
        self.w1 = torch.nn.Parameter(torch.rand(num_distr), requires_grad=True)
        self.w2 = torch.nn.Parameter(torch.rand(num_distr), requires_grad=True)
        
        
        self.miu = torch.nn.Parameter(torch.rand(torch.Size([num_distr]+input_shapes)), requires_grad=True) #backprop
        self.tao = torch.nn.Parameter(torch.rand(num_distr), requires_grad=True) #gets updated by backprop
        self.weight = torch.nn.Parameter(torch.rand([3]), requires_grad=True)
        self.ls = torch.nn.LogSigmoid()
        self.pool = torch.nn.MaxPool1d(kernel_size=num_classes, stride=num_classes)
    
    def Xmetric(self, X, Y):
        return torch.abs(X-Y)
    def SOmetric(self, X, Y):
        return torch.abs(X-Y)
    def Euclmetric(self, X, Y):
        return torch.sqrt(torch.sum(X-Y, dim=1))
    def P1metric(self, X, Y):
        return torch.abs(torch.log(X/(Y+eps)))
        
    def forward(self, x_LE, labels=None, sigmas=None): 
        #Size of x is [B, features, in, H, W]
        #Size of sigmas is [num_distr]
       
        sigmas = self.sigmas
            
        w1 = weightNormalize1(self.w1)
        w2 = weightNormalize1(self.w1)
        #Apply wFM to inputs
        #x_LE is of shape [B, features, in, H, W]
        B, features, in_channel, H, W = x_LE.shape
        
        x_LE_xy = x_LE[:, 2:, ...]
        x_LE = x_LE[:, :2, ...]
        if labels is not None:
            #During training stage only:
            #Select Tensors of Same class and group together
            inputs = x_LE.contiguous().view(B, -1)
            inputs_xy = x_LE_xy.contiguous().view(B, -1)
            
            label_used = labels.unsqueeze(-1).repeat(1, torch.cumprod(torch.tensor(self.shapes[1:]), 0)[-1])

            temp_bins = self.X_LEs.view(self.classes, -1)
            temp_bins_xy = self.X_LEs_xy.view(self.classes, -1)
            
            self.X_LEs = temp_bins.scatter_add(0, label_used, inputs).reshape(self.shapes).detach()
            self.X_LEs_xy = temp_bins_xy.scatter_add(0, label_used, inputs_xy).reshape(self.shapes).detach()
            #Since we are taking the average, better keep track of number of samples in each class
            labels_weights = labels.unsqueeze(-1)
            src = torch.ones((labels.shape[0], 1))
            
            self.X_weights = self.X_weights.scatter_add(0, labels_weights, src.cuda()).detach()
            
        #Size of [num_classes, features, in, H, W]
        x_LE_out = (self.X_LEs.view(self.classes, -1) / self.X_weights).view(self.shapes)


        #Size of [num_classes, num_distr, features, in , H, W]
        x_LE_expand = x_LE_out.unsqueeze(1).repeat(1, self.num_distr, 1, 1, 1, 1)

        #Size of [num_distr, num_classes] 
        tao_sqrd = (self.tao ** 2).unsqueeze(-1).repeat(1, self.classes)

        #Size of [num_classes]
        sigma_sqrd = sigmas ** 2
        
        if labels is not None:
          #####THIS BLOCK OF CODE GENERATES THE LOSS TERM TO BACKPROP###
            #Size of [num_distr, num_classes]
            term1 = sigma_sqrd / (tao_sqrd + sigma_sqrd) ** 2

            #Size of [num_distr, num_classes]
            LE_miu_dist = (self.ls(x_LE_expand).cuda() - self.ls(self.miu)) ** 2
            LE_miu_norm = torch.sum(LE_miu_dist.view(self.classes, self.num_distr, -1), dim=2).transpose(1, 0)

            #Size of [num_distr, num_classes]
            term2 = sigma_sqrd.unsqueeze(0).repeat(self.num_distr, 1) * LE_miu_norm

            #Size of [num_distr, num_classes]
            term3 = 2*self.out*H*W*(tao_sqrd ** 2 - sigma_sqrd ** 2).cuda() / self.X_weights.repeat(1, self.num_distr).transpose(1, 0).cuda()

            #Size of [num_distr]
            loss = torch.mean(term1 * (term2 + term3), dim=1)
        else:
            loss = None


      #####THIS BLOCK OF CODE GENERATES THE LOSS TERM TO BACKPROP###


        #Size of [num_classes, num_distr] 
        tao_sqrd = (self.tao ** 2).unsqueeze(0).repeat(self.classes, 1)

        #Size of [num_classes, num_distr]
        sigma_sqrd = (sigmas ** 2).unsqueeze(-1).repeat(1, self.num_distr)

      #####THIS BLOCK OF CODE GENERATES THE MEANS FOR EACH CLASS###

        #These are of shape [num_classes, num_distr, in, H, W]
        theta_x_LE = x_LE_expand[:, :, 0, ...]
        mag_x_LE = x_LE_expand[:, :, 1, ...]
        x_LE_mag = (tao_sqrd / (sigma_sqrd + tao_sqrd)).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).repeat(1, 1, in_channel, H, W) * self.ls(mag_x_LE+eps)

        miu_bins_mag = (sigma_sqrd / (sigma_sqrd + tao_sqrd)).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).repeat(1, 1, in_channel, H, W) *  self.ls(self.miu.unsqueeze(0).repeat(self.classes, 1, 1, 1, 1, 1)[:, :, 1, ...]+eps)


        exp_sum_mag = torch.exp((x_LE_mag + miu_bins_mag) * w1.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).repeat(1, in_channel, H, W))

        means_mag = torch.sum(exp_sum_mag, dim=1)

        x_LE_theta = mag_x_LE * (tao_sqrd / (sigma_sqrd + tao_sqrd)).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).repeat(1, 1, in_channel, H, W) 

        miu_bins_theta = self.miu.unsqueeze(0).repeat(self.classes, 1, 1, 1, 1, 1)[:, :, 0, ...] * (sigma_sqrd / (sigma_sqrd + tao_sqrd)).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).repeat(1, 1, in_channel, H, W) 

        means_theta = torch.sum((x_LE_theta + miu_bins_theta) * w2.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).repeat(1, in_channel, H, W), dim=1)

        #[num_classes, features, in, H, W]
        means = torch.cat((means_theta.unsqueeze(1), means_mag.unsqueeze(1)), 1)
        means_expand = means.unsqueeze(1).repeat(1, B, 1, 1, 1, 1)
        x_LE = x_LE.unsqueeze(0).repeat(self.classes, 1, 1, 1, 1, 1)
        dist_rot = self.SOmetric(x_LE[:, :, 0, ...].contiguous().view(-1), means_expand[:, :, 0, ...].contiguous().view(-1))
        dist_rot = dist_rot.view(self.classes, B, in_channel, H, W)   
        dist_abs = self.P1metric(x_LE[:, :, 1, ...].contiguous().view(-1), means_expand[:, :, 1, ...].contiguous().view(-1)).view(self.classes, B, in_channel, H, W)   
        (self.X_LEs.view(self.classes, -1) / self.X_weights).view(self.shapes)
        
        #[classes, 2, in, H, W]
        x_LEs_xy_out = (self.X_LEs_xy.view(self.classes, -1) / self.X_weights).view(self.shapes)
        x_LEs_xy_out = x_LEs_xy_out.unsqueeze(1).repeat(1, B, 1, 1, 1, 1)
        x_LE_xy = x_LE_xy.unsqueeze(0).repeat(self.classes, 1, 1, 1, 1, 1)
        
        #[class, B, 2, in, H, W]
        dist_xy = (x_LE_xy - x_LEs_xy_out) ** 2
        dist_xy = torch.sum(dist_xy, dim=2)
        
        #[num_classes, B, in, H, W]
        dist_l1 = (self.weight[0]**2)*dist_rot + (self.weight[1]**2)*dist_abs + (self.weight[2]**2)*dist_xy 
        classes, B, in_channel, H, W = dist_l1.shape
        x_LE = dist_l1.permute(1, 2, 3, 4, 0).view(B, in_channel*H*W, self.classes) * (-1)
        x_LE = self.pool(x_LE).view(B, in_channel, H, W) * (-1)

        return x_LE, loss
        
    
    def clear_LE(self):
        self.X_LEs = Variable(torch.zeros(self.shapes)+eps, requires_grad=False).cuda()
        self.X_weights = Variable(torch.zeros((self.classes, 1))+eps, requires_grad=False).cuda()

        

class ComplexConv2Deffgroup(nn.Module):
    
    def __init__(self, in_channels, out_channels, kern_size, stride):
        super(ComplexConv2Deffgroup, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kern_size = kern_size
        self.stride = stride
        self.wmr = torch.nn.Parameter(torch.rand(in_channels, kern_size[0]*kern_size[1]), requires_grad=True)
        self.wma = torch.nn.Parameter(torch.rand(in_channels, kern_size[0]*kern_size[1]), requires_grad=True) 
        self.complex_conv = ComplexConv2Deffangle(in_channels, out_channels, kern_size, stride)

    def forward(self, x):
        x_shape = x.shape
        out_spatial_x = int(math.floor((x_shape[3]-(self.kern_size[0]-1)-1)/self.stride[0] + 1))
        out_spatial_y = int(math.floor((x_shape[4]-(self.kern_size[1]-1)-1)/self.stride[1] + 1))
        x = x.view(-1,self.in_channels,x_shape[3],x_shape[4])
        temporal_buckets = nn.Unfold(kernel_size=self.kern_size, stride=self.stride)(x).view(x_shape[0], x_shape[1],  self.in_channels, self.kern_size[0]*self.kern_size[1], -1)
        temporal_buckets_rot = temporal_buckets[:,0,...]
        temporal_buckets_abs = temporal_buckets[:,1,...]
        
        tbr_shape0 = temporal_buckets_rot.shape
        temporal_buckets_rot = temporal_buckets_rot.permute(0,3,1,2).contiguous().view(-1, tbr_shape0[1], tbr_shape0[2])
        temporal_buckets_abs = temporal_buckets_abs.permute(0,3,1,2).contiguous().view(-1, tbr_shape0[1],tbr_shape0[2])
        tbr_shape = temporal_buckets_rot.shape 
        
        in_rot = temporal_buckets_rot * weightNormalize2(self.wmr)
        in_abs = temporal_buckets_abs + weightNormalize1(self.wma)
        in_rot = in_rot.view(tbr_shape0[0], out_spatial_x, out_spatial_y, -1).permute(0,3,1,2).contiguous().unsqueeze(1)
        in_abs = in_abs.view(tbr_shape0[0], out_spatial_x, out_spatial_y, -1).permute(0,3,1,2).contiguous().unsqueeze(1)
        in_ = torch.cat((in_rot, in_abs), 1).view(tbr_shape0[0], -1, out_spatial_x*out_spatial_y)
        in_fold = nn.Fold(output_size=(x_shape[3],x_shape[4]), kernel_size=self.kern_size, stride=self.stride)(in_)
        in_fold = in_fold.view(x_shape[0],x_shape[1],x_shape[2],x_shape[3],x_shape[4])
        out = self.complex_conv(in_fold)
        
        return out 
    
class ComplexConv2Deffangle(nn.Module):
    
    def __init__(self, in_channels, out_channels, kern_size, stride, drop_prob=0.0):
        super(ComplexConv2Deffangle, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kern_size = kern_size
        self.stride = stride
        self.drop_prob = drop_prob
        self.weight_matrix_rot1 = torch.nn.Parameter(torch.rand(in_channels, kern_size[0]*kern_size[1]), requires_grad=True)
        self.weight_matrix_rot2 = torch.nn.Parameter(torch.rand(out_channels, in_channels), requires_grad=True)
    
    def forward(self, x):
        x_shape = x.shape
        out_spatial_x = int(math.floor((x_shape[3]-(self.kern_size[0]-1)-1)/self.stride[0] + 1))
        out_spatial_y = int(math.floor((x_shape[4]-(self.kern_size[1]-1)-1)/self.stride[1] + 1))
        x = x.view(-1,self.in_channels,x_shape[3],x_shape[4])
        temporal_buckets = nn.Unfold(kernel_size=self.kern_size, stride=self.stride)(x).view(x_shape[0], x_shape[1],  self.in_channels, self.kern_size[0]*self.kern_size[1], -1)
        temporal_buckets_rot = temporal_buckets[:,0,...]
        temporal_buckets_abs = temporal_buckets[:,1,...]
        tbr_shape = temporal_buckets_rot.shape 
        out_rot = ((torch.sum(temporal_buckets_rot.permute(0,3,1,2).contiguous().view(-1, tbr_shape[1], tbr_shape[2])*weightNormalize1(self.weight_matrix_rot1),2))).view(tbr_shape[0],tbr_shape[3],tbr_shape[1]).permute(0,2,1).contiguous()
        out_rot_shape = out_rot.shape
        out_rot = out_rot.permute(0,2,1).contiguous().view(-1,1,self.in_channels).repeat(1,self.out_channels,1)
        out_rot = (torch.sum(out_rot*weightNormalize1(self.weight_matrix_rot2),2)).view(out_rot_shape[0], 1, out_spatial_x, out_spatial_y, self.out_channels).permute(0,1,4,2,3).contiguous()
        tba_shape = temporal_buckets_abs.shape
        temporal_buckets_abs = torch.log((temporal_buckets_abs+eps))
        tba_shape = temporal_buckets_abs.shape   
        out_abs = ((torch.sum(temporal_buckets_abs.permute(0,3,1,2).contiguous().view(-1, tba_shape[1], tba_shape[2])*weightNormalize(self.weight_matrix_rot1,self.drop_prob),2))).view(tba_shape[0],tba_shape[3],tba_shape[1]).permute(0,2,1).contiguous()
        out_abs_shape = out_abs.shape
        out_abs = out_abs.permute(0,2,1).contiguous().view(-1,1,self.in_channels).repeat(1,self.out_channels,1)
        out_abs = torch.exp(torch.sum(out_abs*weightNormalize(self.weight_matrix_rot2,self.drop_prob),2)).view(out_abs_shape[0], 1, out_spatial_x, out_spatial_y, self.out_channels).permute(0,1,4,2,3).contiguous()
        return torch.cat((out_rot,out_abs),1)
    
def weightNormalizexy(weights1, weights2):
    weights = (weights1+weights2)**2
    weights = weights / torch.sum(weights)
    return weights
    
class ComplexConv2Deffangle4Dxy(nn.Module):
    def __init__(self, in_channels, out_channels, kern_size, stride, dilation = (1, 1), padding = (0, 0), drop_prob=0.0):
        super(ComplexConv2Deffangle4Dxy, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kern_size = kern_size
        self.stride = stride
        self.dilation = dilation
        self.padding = padding
        self.drop_prob = drop_prob
        self.weight_matrix_rot1x = torch.nn.Parameter(torch.rand(in_channels, kern_size[0]*kern_size[1]), requires_grad=True)
        self.weight_matrix_rot1y = torch.nn.Parameter(torch.rand(in_channels, kern_size[0]*kern_size[1]), requires_grad=True)
        self.weight_matrix_rot2x = torch.nn.Parameter(torch.rand(out_channels, in_channels), requires_grad=True)
        self.weight_matrix_rot2y = torch.nn.Parameter(torch.rand(out_channels, in_channels), requires_grad=True)
        
    def forward(self, x):
        x_shape = x.shape
        out_spatial_x = int(math.floor((x_shape[3]+2*self.padding[0]-self.dilation[0]*(self.kern_size[0]-1)-1)/self.stride[0] + 1))
        out_spatial_y = int(math.floor((x_shape[4]+2*self.padding[1]-self.dilation[1]*(self.kern_size[1]-1)-1)/self.stride[1] + 1))
        
        #Shape: [batches, features, in_channels, spatial_x, spatial_y] -> [batches*features, in_channels, spatial_x, spatial_y]
        x = x.view(-1,self.in_channels,x_shape[3],x_shape[4])
        #Shape: [batches, features, in_channels, kern_size[0]*kern_size[1], L]
        temporal_buckets = nn.Unfold(kernel_size=self.kern_size, stride=self.stride, dilation = self.dilation, padding = self.padding)(x).view(x_shape[0], x_shape[1],  self.in_channels, self.kern_size[0]*self.kern_size[1], -1)
        
        #Shape: [batches, in_channels, kern_size[0]*kern_size[1], L]
        temporal_buckets_rot = temporal_buckets[:,0,...]
        
        #Shape: [batches, in_channels, kern_size[0]*kern_size[1], L]
        temporal_buckets_abs = temporal_buckets[:,1,...]
        temporal_buckets_x = temporal_buckets[:,2,...]
        temporal_buckets_y = temporal_buckets[:,3,...]
        tbr_shape = temporal_buckets_rot.shape 
       
         
        out_x = ((torch.sum(temporal_buckets_x.permute(0,3,1,2).contiguous().view(-1, tbr_shape[1], tbr_shape[2])*(self.weight_matrix_rot1x),2))).view(tbr_shape[0],tbr_shape[3],tbr_shape[1]).permute(0,2,1).contiguous()
        
        out_y = ((torch.sum(temporal_buckets_y.permute(0,3,1,2).contiguous().view(-1, tbr_shape[1], tbr_shape[2])*(self.weight_matrix_rot1y),2))).view(tbr_shape[0],tbr_shape[3],tbr_shape[1]).permute(0,2,1).contiguous()
        
        #Shape: [Batch, in_channels, L], 
        out_rot = ((torch.sum(temporal_buckets_rot.permute(0,3,1,2).contiguous().view(-1, tbr_shape[1], tbr_shape[2])*weightNormalizexy(self.weight_matrix_rot1x, self.weight_matrix_rot1y),2))).view(tbr_shape[0],tbr_shape[3],tbr_shape[1]).permute(0,2,1).contiguous()
        
        
        out_rot_shape = out_rot.shape
        #[Batch, L, in_channels] -> [Batch*L, 1, in_channels] -> [Batch*L, out_channels, in_channels]
        out_rot = out_rot.permute(0,2,1).contiguous().view(-1,1,self.in_channels).repeat(1,self.out_channels,1)
        # [Batch*L, out_channels]
        out_rot = (torch.sum(out_rot*weightNormalizexy(self.weight_matrix_rot2x, self.weight_matrix_rot2y),2)).view(out_rot_shape[0], 1, out_spatial_x, out_spatial_y, self.out_channels).permute(0,1,4,2,3).contiguous()
       
        out_x = out_x.permute(0,2,1).contiguous().view(-1,1,self.in_channels).repeat(1,self.out_channels,1)
        out_x = (torch.sum(out_x*(self.weight_matrix_rot2x),2)).view(out_rot_shape[0], 1, out_spatial_x, out_spatial_y, self.out_channels).permute(0,1,4,2,3).contiguous()
        
        
        out_y = out_y.permute(0,2,1).contiguous().view(-1,1,self.in_channels).repeat(1,self.out_channels,1)
        out_y = (torch.sum(out_y*(self.weight_matrix_rot2y),2)).view(out_rot_shape[0], 1, out_spatial_x, out_spatial_y, self.out_channels).permute(0,1,4,2,3).contiguous()
        tba_shape = temporal_buckets_abs.shape
        
        #Shape: [batches,  in_channels, kern_size[0]*kern_size[1], L]
        temporal_buckets_abs = torch.log((temporal_buckets_abs+eps))
        
        tba_shape = temporal_buckets_abs.shape   
        
        #Shape: [batches, in_channels, L]
        out_abs = ((torch.sum(temporal_buckets_abs.permute(0,3,1,2).contiguous().view(-1, tba_shape[1], tba_shape[2])*weightNormalizexy(self.weight_matrix_rot1x, self.weight_matrix_rot1y),2))).view(tba_shape[0],tba_shape[3],tba_shape[1]).permute(0,2,1).contiguous()
        out_abs_shape = out_abs.shape
        
        out_abs = out_abs.permute(0,2,1).contiguous().view(-1,1,self.in_channels).repeat(1,self.out_channels,1)
        out_abs = torch.exp(torch.sum(out_abs*weightNormalizexy(self.weight_matrix_rot2x,self.weight_matrix_rot2y),2)).view(out_abs_shape[0], 1, out_spatial_x, out_spatial_y, self.out_channels).permute(0,1,4,2,3).contiguous()
        #Shape: [batches, 1, out_channels, out_spatial_x, out_spatial_y]
        return torch.cat((out_rot,out_abs,out_x,out_y),1)
    
class ReLU4Dsp(nn.Module):
    
    def __init__(self,channels):
        super(ReLU4Dsp, self).__init__()
        self.weight_rot = torch.nn.Parameter(torch.rand(1,channels), requires_grad=True)
        self.channels = channels
        self.relu = nn.ReLU() 
        
    def forward(self, x):
        #Shape: [batches, features, in_channels, spatial_x, spatial_y]
        x_shape = x.shape  
        temp_rot = x[:,0,...]
        temp_abs = x[:,1,...]  
        temp_x = self.relu(x[:,2,...]).unsqueeze(1)
        temp_y = self.relu(x[:,3,...]).unsqueeze(1)
        temp_rot_prod = (temp_rot.unsqueeze(1)*(weightNormalize2(self.weight_rot+eps)).unsqueeze(0).unsqueeze(3).unsqueeze(4).repeat(x_shape[0],1,1,x_shape[3],x_shape[4]))
        temp_abs = (temp_abs.unsqueeze(1)+(weightNormalize1(self.weight_rot+eps)).unsqueeze(0).unsqueeze(3).unsqueeze(4).repeat(x_shape[0],1,1,x_shape[3],x_shape[4]))
        return torch.cat((temp_rot_prod, temp_abs, temp_x, temp_y),1)

class ComplexLinearangle4Dmw_outfield(nn.Module):
    #input_dim should equal channels*frames of previous layer.
    def __init__(self, input_dim):
        super(ComplexLinearangle4Dmw_outfield, self).__init__()
        self.input_dim = input_dim
        self.weight = torch.nn.Parameter(torch.rand([4]), requires_grad=True)
        self.bias = torch.nn.Parameter(torch.rand([4]), requires_grad=True)
        self.weightsx = torch.nn.Parameter(torch.rand(input_dim), requires_grad=True)
        self.weightsy = torch.nn.Parameter(torch.rand(input_dim), requires_grad=True)
        self.weights = torch.nn.Parameter(torch.rand(input_dim), requires_grad=True)

    def ComplexunweightedMeanLinear(self, x_rot, x_abs, x_x, x_y):
    #x_rot.shape: [batches, out_channels* out_spatial_x* out_spatial_y]
    #x_abs.shape: [batches, out_channels* out_spatial_x* out_spatial_y]
        
        out_x = torch.sum(x_x*(self.weightsx),1) + self.bias[2]
        out_y = torch.sum(x_y*(self.weightsy),1) + self.bias[3]
        out_rot = torch.sum(x_rot*weightNormalize1(self.weights),1) * torch.tanh(-self.bias[0])
        x_abs_log = torch.log(x_abs+eps)
        out_abs = torch.exp(torch.sum(x_abs_log*weightNormalize1(self.weights),1))+torch.exp(-self.bias[1]**2)    
    
        return (out_rot,out_abs,out_x,out_y)

    def unweightedFMComplex(self, point_list_rot, point_list_abs,point_list_x,point_list_y):
        return self.ComplexunweightedMeanLinear(point_list_rot, point_list_abs, point_list_x, point_list_y)
    
    def Xmetric(self, X, Y):
        return torch.abs(X-Y)
    def SOmetric(self, X, Y):
        return torch.abs(X-Y)
       
    def P1metric(self, X, Y):
        return torch.abs(torch.log(X/(Y+eps)))

    def forward(self, x):
        
        
        #shape: [batches, out_channels* out_spatial_x* out_spatial_y, 2]
        
        all_data = x.permute(0,2,3,4,1).contiguous()
        all_data_shape = all_data.shape
        all_data = all_data.view(all_data_shape[0], all_data_shape[1]*all_data_shape[2]*all_data_shape[3], all_data_shape[4])
           
        all_data_rot = all_data[:,:,0]
        all_data_abs = all_data[:,:,1]
        all_data_x = all_data[:,:,2]
        all_data_y = all_data[:,:,3]
           
        all_shape = all_data_rot.shape
           
        M_rot, M_abs, M_x, M_y = self.unweightedFMComplex(all_data_rot, all_data_abs,all_data_x,all_data_y)
        
        dist_x = self.Xmetric(all_data_x.view(-1), M_x.unsqueeze(1).repeat(1, all_shape[1]).view(-1)).view(all_shape[0],all_shape[1])       
        dist_y = self.Xmetric(all_data_y.view(-1), M_y.unsqueeze(1).repeat(1,all_shape[1]).view(-1)).view(all_shape[0],all_shape[1])       
        dist_rot = self.SOmetric(all_data_rot.view(-1), M_rot.unsqueeze(1).repeat(1,all_shape[1]).view(-1)).view(all_shape[0],all_shape[1])       
        dist_abs = self.P1metric(all_data_abs.view(-1), M_abs.unsqueeze(1).repeat(1,all_shape[1]).view(-1)).view(all_shape[0],all_shape[1])
        dist_l1 = (self.weight[0]**2)*dist_rot + (self.weight[1]**2)*dist_abs + (self.weight[2]**2)*dist_x + (self.weight[3]**2)*dist_y
        dist_l1 = dist_l1.view(all_data_shape[0], all_data_shape[1], all_data_shape[2], all_data_shape[3]) 
        return dist_l1
    
