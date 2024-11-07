import torch.nn as nn
from itertools import combinations
from torch.autograd import Variable
import torchvision.models as models
from utils import *


class TriplePrior(nn.Module):
    def __init__(self):
        super(TriplePrior, self).__init__()
        # self.args = args
        self.conv_st = nn.Conv3d(in_channels=1, out_channels=1, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False)
        self.conv_ch1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.conv_ch2 = nn.Conv2d(in_channels=16, out_channels=3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.sigmoid = nn.Sigmoid()

    def calculate_channel_diff(self, input):
        n, t, c = input.shape
        channel_diff = nn.Conv1d(in_channels=t, out_channels=t, kernel_size=3, stride=1, padding=1).cuda()
        channel_input = channel_diff(input)
        channel_output = channel_input.reshape(n*t, c, 1, 1)

        return channel_output

    def split_frame_with_interval(self, input, interval):
        num_frames = input.shape[1]
        num_splits = num_frames - interval + 1
        split_input = {}
        for i in range(num_splits):
            split_input[f"split_{i}"] = input[:, i:i+interval, ...]

        return split_input

    def recover_frame_diff_shape(self, input):
        n, num_diff, c, h, w = input.shape
        input = input.reshape(n*num_diff, c, h, w)
        recover_channel = nn.Conv2d(in_channels=c, out_channels=3, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)).cuda()
        recover_input = recover_channel(input).reshape(n, num_diff, -1, h, w)

        return recover_input

    def calculate_frame_diff(self, input):
        n, s, c, h, w = input[0].shape
        frame_diff_conv = nn.Conv2d(in_channels=s*c, out_channels=s*c, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)).cuda()
        concatenated_diff = None
        for i in range(len(input) - 1):
            diff_shape = [n, s*c, h, w]
            frame_diff = (input[i].reshape(diff_shape) - frame_diff_conv(input[i + 1].reshape(diff_shape))).unsqueeze(1)

            concatenated_diff = frame_diff if concatenated_diff is None else torch.cat([concatenated_diff, frame_diff], dim=1)

        concatenated_diff = self.recover_frame_diff_shape(concatenated_diff)

        return concatenated_diff

    def forward(self, x, seq_len):

        n, c, h, w = x.shape
        x_st = x.reshape(n//seq_len, c, -1, h, w)
        x_st = x_st.mean(dim=1, keepdim=True)
        x_st = self.conv_st(x_st)
        x_st = x_st.transpose(1, 2).contiguous()
        x_st = x_st.reshape(n, -1, h, w)
        x_st = x * self.sigmoid(x_st) + x

        x_ch = self.conv_ch1(x)
        x_ch = self.avg_pool(x_ch)
        num_channels = x_ch.shape[1]
        x_ch = x_ch.reshape(n//seq_len, -1, num_channels)
        x_ch = self.calculate_channel_diff(x_ch)
        x_ch = self.conv_ch2(x_ch)
        x_ch = x * self.sigmoid(x_ch) + x
        
        x_m= x.reshape(n//seq_len, c, -1, h, w)
        x_m = x_m.transpose(1, 2).contiguous()
        internal_set = [1, 2, 3]
        x_mo = None
        for value in internal_set:
            x_split = self.split_frame_with_interval(x_m, value)
            x_diff = self.calculate_frame_diff(list(x_split.values()))
            x_mo = x_diff if x_mo is None else torch.cat([x_mo, x_diff], dim=1)
        x_mo = x_mo.mean(dim=1, keepdim=True)
        x_mo = x_mo.expand(n//seq_len, seq_len, c, h, w)
        x_mo = x_mo.reshape(n, -1, h, w)
        x_mo = self.avg_pool(x_mo)

        x_m = x* self.sigmoid(x_mo) + x

        x_fin = x_st + x_ch + x_m + x

        return x_fin


class CNN_FSHead(nn.Module):

    def __init__(self, args):
        super(CNN_FSHead, self).__init__()
        self.train()
        self.args = args

        last_layer_idx = self.args.last_layer_idx
        
        if self.args.backbone == "resnet18":
            backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        elif self.args.backbone == "resnet34":
            backbone = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
        elif self.args.backbone == "resnet50":
            backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

        if self.args.pretrained_backbone is not None:
            checkpoint = torch.load(self.args.pretrained_backbone)
            backbone.load_state_dict(checkpoint)

        self.backbone = nn.Sequential(*list(backbone.children())[:last_layer_idx])
        self.tripel_prior = TriplePrior()

    def get_feats(self, support_images, target_images):

        support_images = self.tripel_prior(support_images, self.args.seq_len)
        target_images = self.tripel_prior(target_images, self.args.seq_len)
        support_features = self.backbone(support_images).squeeze()
        target_features = self.backbone(target_images).squeeze()

        dim = int(support_features.shape[1])

        support_features = support_features.reshape(-1, self.args.seq_len, dim)
        target_features = target_features.reshape(-1, self.args.seq_len, dim)

        return support_features, target_features

    def forward(self, support_images, support_labels, target_images):

        raise NotImplementedError

    def distribute_model(self):

        if self.args.num_gpus > 1:
            self.backbone.cuda(0)
            self.backbone = torch.nn.DataParallel(self.backbone, device_ids=[i for i in range(0, self.args.num_gpus)])

    def loss(self, test_logits_sample, test_labels, device):

        size = test_logits_sample.size()
        sample_count = size[0]  # scalar for the loop counter
        num_samples = torch.tensor([sample_count], dtype=torch.float, device=device, requires_grad=False)

        log_py = torch.empty(size=(size[0], size[1]), dtype=torch.float, device=device)
        for sample in range(sample_count):
            log_py[sample] = -F.cross_entropy(test_logits_sample[sample], test_labels, reduction='none')
        score = torch.logsumexp(log_py, dim=0) - torch.log(num_samples)
        return -torch.sum(score, dim=0)


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout, max_len=5000, pe_scale_factor=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.pe_scale_factor = pe_scale_factor

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term) * self.pe_scale_factor
        pe[:, 1::2] = torch.cos(position * div_term) * self.pe_scale_factor
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
                          
    def forward(self, x):
       x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
       return self.dropout(x)



class TemporalCrossTransformer(nn.Module):

    def __init__(self, args, temporal_set_size=3):
        super(TemporalCrossTransformer, self).__init__()
       
        self.args = args
        self.temporal_set_size = temporal_set_size

        max_len = int(self.args.seq_len * 1.5)
        self.pe = PositionalEncoding(self.args.trans_linear_in_dim, self.args.trans_dropout, max_len=max_len)

        self.k_linear = nn.Linear(self.args.trans_linear_in_dim * temporal_set_size, self.args.trans_linear_out_dim)#.cuda()
        self.v_linear = nn.Linear(self.args.trans_linear_in_dim * temporal_set_size, self.args.trans_linear_out_dim)#.cuda()

        self.norm_k = nn.LayerNorm(self.args.trans_linear_out_dim)
        self.norm_v = nn.LayerNorm(self.args.trans_linear_out_dim)
        
        self.class_softmax = torch.nn.Softmax(dim=1)
        
        frame_idxs = [i for i in range(self.args.seq_len)]
        frame_combinations = combinations(frame_idxs, temporal_set_size)
        self.tuples = nn.ParameterList([nn.Parameter(torch.tensor(comb), requires_grad=False) for comb in frame_combinations])
        self.tuples_len = len(self.tuples) 
    
    
    def forward(self, support_set, support_labels, queries):
        global align
        n_queries = queries.shape[0]
        n_support = support_set.shape[0]
        
        support_set = self.pe(support_set)
        queries = self.pe(queries)

        s = [torch.index_select(support_set, -2, p).reshape(n_support, -1) for p in self.tuples]
        q = [torch.index_select(queries, -2, p).reshape(n_queries, -1) for p in self.tuples]
        support_set = torch.stack(s, dim=-2)
        queries = torch.stack(q, dim=-2)

        support_set_ks = self.k_linear(support_set)
        queries_ks = self.k_linear(queries)
        support_set_vs = self.v_linear(support_set)
        queries_vs = self.v_linear(queries)
        
        mh_support_set_ks = self.norm_k(support_set_ks)
        mh_queries_ks = self.norm_k(queries_ks)
        mh_support_set_vs = support_set_vs
        mh_queries_vs = queries_vs
        
        unique_labels = torch.unique(support_labels)

        all_distances_tensor = torch.zeros(n_queries, self.args.way, device=queries.device)

        for label_idx, c in enumerate(unique_labels):
        
            class_k = torch.index_select(mh_support_set_ks, 0, extract_class_indices(support_labels, c))
            class_v = torch.index_select(mh_support_set_vs, 0, extract_class_indices(support_labels, c))
            k_bs = class_k.shape[0]

            class_scores = torch.matmul(mh_queries_ks.unsqueeze(1), class_k.transpose(-2,-1)) / math.sqrt(self.args.trans_linear_out_dim)

            class_scores = class_scores.permute(0,2,1,3)
            class_scores = class_scores.reshape(n_queries, self.tuples_len, -1)

            class_scores = [self.class_softmax(class_scores[i]) for i in range(n_queries)]
            class_scores = torch.cat(class_scores)
            class_scores = class_scores.reshape(n_queries, self.tuples_len, -1, self.tuples_len)
            class_scores = class_scores.permute(0,2,1,3)

            align = class_scores.sum(dim=1)
     
            query_prototype = torch.matmul(class_scores, class_v)
            query_prototype = torch.sum(query_prototype, dim=1)
            
            diff = mh_queries_vs - query_prototype
            norm_sq = torch.norm(diff, dim=[-2,-1])**2
            distance = torch.div(norm_sq, self.tuples_len)
            
            distance = distance * -1
            c_idx = c.long()
            all_distances_tensor[:,c_idx] = distance
        
        return_dict = {'logits': all_distances_tensor, 'align': align}
        
        return return_dict


class SOAP(CNN_FSHead):

    def __init__(self, args):
        super(SOAP, self).__init__(args)

        #fill default args
        self.args.trans_linear_out_dim = 1152
        self.args.temp_set = [1]
        self.args.trans_dropout = 0.1

        self.transformers = nn.ModuleList([TemporalCrossTransformer(args, s) for s in args.temp_set])

    def forward(self, support_images, support_labels, target_images):
        support_features, target_features = self.get_feats(support_images, target_images)
        all_logits = [t(support_features, support_labels, target_features)['logits'] for t in self.transformers]
        align = [t(support_features, support_labels, target_features)['align'] for t in self.transformers]
        all_logits = torch.stack(all_logits, dim=-1)
        sample_logits = all_logits 
        sample_logits = torch.mean(sample_logits, dim=[-1])

        return_dict = {'logits': split_first_dim_linear(sample_logits, [1, target_features.shape[0]])}

        return return_dict, align[0]

    def distribute_model(self):
        if self.args.num_gpus > 1:
            self.backbone.cuda(0)
            self.backbone = torch.nn.DataParallel(self.backbone, device_ids=[i for i in range(0, self.args.num_gpus)])

            self.transformers.cuda(0)