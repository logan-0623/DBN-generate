import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torch.autograd import Variable

# Set a random seed for reproducibility
torch.manual_seed(42)

# RBM (Restricted Boltzmann Machine) with Attention Mechanism
class RBM_att(nn.Module):
    def __init__(self, num_visible, num_hidden, attention_size=64, residual=True):
        super(RBM_att, self).__init__()
        self.W = nn.Parameter(torch.randn(num_hidden, num_visible))  # Initialize weight matrix from visible to hidden
        self.v_bias = nn.Parameter(torch.randn(num_visible))  # Initialize visible bias
        self.h_bias = nn.Parameter(torch.randn(num_hidden))  # Initialize hidden bias

        # Attention mechanism
        self.attention_weights = nn.Parameter(torch.randn(attention_size, num_visible))  # Initialize attention weights
        self.conv = nn.Conv1d(260, 512, 1)
        self.res = residual

    def content_attention(self, input_data):
        # Compute attention scores
        attention_scores = torch.matmul(input_data, self.attention_weights.t())
        attention_weights = torch.softmax(attention_scores, dim=1)  # Use softmax to compute attention weights
        # Weighted input data
        attended_input = torch.matmul(attention_weights, input_data)

        return attended_input

    def sample_hidden(self, visible_probabilities):
        attended_input = self.content_attention(visible_probabilities)
        hidden_activations = torch.matmul(attended_input, self.W.t()) + self.h_bias
        hidden_probabilities = torch.sigmoid(hidden_activations)

        return hidden_probabilities, torch.bernoulli(hidden_probabilities)

    def sample_visible(self, hidden_probabilities):
        visible_activations = torch.matmul(hidden_probabilities, self.W) + self.v_bias
        visible_probabilities = torch.sigmoid(visible_activations)
        return visible_probabilities, torch.bernoulli(visible_probabilities)

    def initialize(self):  # Initialize model parameters
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.unsqueeze(0))
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.xavier_uniform_(m.weight.unsqueeze(0))

    def forward(self, input_data):
        _, hidden_sample = self.sample_hidden(input_data)

        # Add residual connection
        if self.res:
            input_data_0 = self.conv(input_data.unsqueeze(2))

            output = input_data_0.squeeze(2) + hidden_sample
            return output

        return hidden_sample

# Define a DBN (Deep Belief Network) for dual-modal data
class DBN(nn.Module):
    def __init__(self, num_visible_mod1, num_hidden_mod1, num_visible_mod2, num_hidden_mod2):
        super(DBN, self).__init__()

        # Define the RBM (Restricted Boltzmann Machine) for the first modality
        self.rbm_mod1 = RBM_att(num_visible_mod1, num_hidden_mod1, attention_size=32)

        # Define the RBM for the second modality
        self.rbm_mod2 = RBM_att(num_visible_mod2, num_hidden_mod2, attention_size=32, residual=False)

    def generate_mod2_from_mod1(self, input_mod1):
        hidden_mod1 = self.rbm_mod1(input_mod1)
        visible_mod2_probs, visible_mod2_gen = self.rbm_mod2.sample_visible(hidden_mod1)
        return visible_mod2_probs, visible_mod2_gen

    def generate_mod1_from_mod2(self, input_mod2):
        hidden_mod2 = self.rbm_mod2(input_mod2)
        visible_mod1_probs, visible_mod1_gen = self.rbm_mod1.sample_visible(hidden_mod2)
        return visible_mod1_probs, visible_mod1_gen

    def initialize(self):  # Initialize model parameters
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.unsqueeze(0))
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.xavier_uniform_(m.weight.unsqueeze(0))

    def forward(self, input_mod1, input_mod2):
        hidden_mod1 = self.rbm_mod1(input_mod1)
        hidden_mod2 = self.rbm_mod2(input_mod2)

        return hidden_mod1, hidden_mod2

# Dual-Modal DBN (Deep Belief Network) with Fusion
class DualModalDBN(nn.Module):
    def __init__(self, Encoder_image, Encoder_seq, dual_fusion_modal, dual_shape, out_class):
        super(DualModalDBN, self).__init__()

        # Define encoders for each modality
        self.encoder_modal1 = Encoder_seq
        self.encoder_modal2 = Encoder_image

        # Define the DBN (Deep Belief Network) layer for modality fusion
        self.dbn = dual_fusion_modal

        # Define fully connected layers for classification
        self.fc = nn.Sequential(nn.Flatten(), nn.Dropout(0.5),
                                nn.Linear(64 * 512, 512), nn.ReLU(), nn.Dropout(0.5),
                                nn.Linear(512, out_class))

        self.fusion_1 = FusionBlock(dim=512, dim_out=512, heads=8, dim_head=64)

    def initialize(self):  # Initialize model parameters
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.unsqueeze(0))
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.xavier_uniform_(m.weight.unsqueeze(0))

    def forward(self, x, img):
        # Missing modality generation
        if img is None:
            _, img = self.dbn.generate_mod2_from_mod1(x)
        elif x is None:
            img = rearrange(img, 'b n h d -> b (h n d)')
            _, x = self.dbn.generate_mod1_from_mod2(img)
        else:  # Both modalities are present
            img = rearrange(img, 'b n h d -> b (h n d)')

        x_out, img_out = self.dbn(x, img)

        # Modality generation
        _, gen_img = self.dbn.generate_mod2_from_mod1(x)
        _, gen_x = self.dbn.generate_mod1_from_mod2(img)

        encoding_modal_x = self.encoder_modal1(x_out)
        encoding_modal_img = self.encoder_modal2(img_out)

        x_embed = repeat(encoding_modal_x.squeeze(1), 'h w -> h w c', c=512)
        img_embed = repeat(encoding_modal_img, 'h w -> h w c', c=512)

        out = self.fusion_1(img_embed, x_embed)
        output = self.fc(out)

        return output, gen_x, gen_img

# Layer normalization module
class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        gamma = self.gamma[:x.shape[-1]]
        beta = self.beta[:x.shape[-1]]
        return F.layer_norm(x, x.shape[1:], gamma.repeat(x.shape[1], 1), beta.repeat(x.shape[1], 1))

# Fusion Block
class FusionBlock(nn.Module):
    def __init__(self, dim, dim_out, heads, dim_head):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.norm = LayerNorm(dim)

        inner_dim = heads * dim_head

        # Identity for image
        self.identity_pic = nn.Linear(dim, inner_dim)
        self.identity_context = nn.Linear(dim, dim_head * 2)

        # ResNet and CNN layers
        self.resnet = nn.Sequential(nn.Conv1d(512, 512, 1, 1),
                                    nn.Conv1d(512, 8, 1, 1))
        self.cnn = nn.Sequential(nn.Conv1d(512, 512, 1, 1))
        self.relu = nn.ReLU()

    def initialize(self):  # Initialize model parameters
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.unsqueeze(0))
            elif isinstance(m, nn.Conv1d):
                nn.init.xavier_uniform_(m.weight.unsqueeze(0))
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.xavier_uniform_(m.weight.unsqueeze(0))

    def forward(self, pic, context):
        # Normalize extracted features
        pic = self.norm(pic)
        context = self.norm(context)

        # Get the identity for the image
        identity_pic = self.identity_pic(pic)
        identity_pic = rearrange(identity_pic, 'b n (h d) -> b h n d', h=self.heads)
        identity_pic = identity_pic * self.scale

        # Chunk the identity for context
        identity_context, Value = self.identity_context(context).chunk(2, dim=-1)

        # Handle dimensions for context and Value if needed
        if identity_context.ndim == 2 or Value.ndim == 2:
            identity_context = repeat(identity_context, 'h w -> h w c', c=64)
            Value = repeat(Value, 'h w -> h w c', c=64)

        # Compute similarity
        sim = einsum('b h i d, b j d -> b h i j', identity_pic, identity_context)
        sim = sim - sim.amax(dim=-1, keepdim=True)

        # Compute attention scores
        attn = sim.softmax(dim=-1)

        # Apply attention to Value
        out = einsum('b h i j, b j d -> b h i d', attn, Value)
        out = rearrange(out, 'b h n d -> b n (h d)')

        # Sum the identity_pic along dimension 3
        identity_pic = torch.sum(identity_pic, dim=3)

        # First residual connection
        out_1 = self.resnet(out)
        out_1 = out_1 + identity_pic

        # Transpose and apply CNN
        out_1 = torch.transpose(out_1, 1, 2)
        out_1 = self.cnn(out_1)

        # Interpolate the result
        out_1 = F.interpolate(out_1, size=(64,), mode='linear', align_corners=False)

        # Second residual connection
        out_2 = self.relu(out_1) + identity_context
        out_2 = self.relu(out_2)

        return out_2

# Example Decoder
# Define a 1D Residual Block
class Residual_1D(nn.Module):
    def __init__(self, in_channels, out_channels, use_1x1conv=False, stride=1):
        super(Residual_1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        return F.relu(Y + X)

# Define a sequence of Residual Blocks
def resnet_block_1D(in_channels, out_channels, num_residuals, first_block=False):
    if first_block:
        assert in_channels == out_channels  # The number of channels in the first block is the same as input channels
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(Residual_1D(in_channels, out_channels, use_1x1conv=True, stride=2))
        else:
            blk.append(Residual_1D(out_channels, out_channels))
    return nn.Sequential(*blk)

# Define a 1D ResNet model
class ResNet_1D(nn.Module):
    def __init__(self, input_channels=3, output_channels=512):
        super().__init__()
        self.net2_1 = nn.Sequential(nn.Conv1d(input_channels, 64, kernel_size=7, stride=2, padding=3),
                                    nn.BatchNorm1d(64), nn.ReLU(),
                                    nn.MaxPool1d(kernel_size=3, stride=2, padding=1))

        self.net2_2 = nn.Sequential(*resnet_block_1D(64, 64, 2, first_block=True))

        self.net2_3 = nn.Sequential(*resnet_block_1D(64, 256, 2))  # Each module uses 2 residual blocks

        self.net2_6 = nn.Sequential(*resnet_block_1D(256, 512, 2))

        self.net2_7 = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten())

        self.linear = nn.Sequential(nn.Linear(512, 4096), nn.Dropout(0.5),
                                    nn.Linear(4096, output_channels))

    def initialize(self):  # Initialize model parameters
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.unsqueeze(0))
            elif isinstance(m, nn.Conv1d):
                nn.init.xavier_uniform_(m.weight.unsqueeze(0))
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.xavier_uniform_(m.weight.unsqueeze(0))

    def forward(self, img):
        # 2D image network
        img = rearrange(img, 'c (r p)  -> c r p ', r=1)
        feature2_1 = self.net2_1(img)
        feature2_2 = self.net2_2(feature2_1)
        feature2_3 = self.net2_3(feature2_2)
        feature2_6 = self.net2_6(feature2_3)
        feature2_7 = self.net2_7(feature2_6)
        output = self.linear(feature2_7)
        return output

# Define an LSTM-based RNN model
class LstmRNN(nn.Module):
    """
        Parameters:
        - input_size: feature size
        - hidden_size: number of hidden units
        - output_size: number of outputs
        - num_layers: layers of LSTM to stack
    """

    def __init__(self, input_size, hidden_size=64, output_size=128, num_layers=1):
        super().__init__()

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers)  # utilize the LSTM model in torch.nn
        self.linear1 = nn.Linear(hidden_size, output_size)  # Fully connected layer

    def initialize(self):  # Initialize model parameters
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.unsqueeze(0))

    def forward(self, _x):
        _x = rearrange(_x, 'c (r p)  -> c r p ', r=1)
        x, _ = self.lstm(_x)  # _x is input, size (seq_len, batch, input_size)

        s, b, h = x.shape  # x is output, size (seq_len, batch, hidden_size)
        x = x.view(s * b, h)
        x = self.linear1(x)
        x = x.view(s, -1)

        return x

# Define a Transformer Encoder model
class TransformerEncoder(nn.Module):
    def __init__(self, input_size, output_size):
        super(TransformerEncoder, self).__init__()
        self.transformer_layer = nn.TransformerEncoderLayer(d_model=output_size, nhead=4)
        self.transformer_encoder = nn.TransformerEncoder(self.transformer_layer, num_layers=6)

    def initialize(self):  # Initialize model parameters
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.unsqueeze(0))
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.xavier_uniform_(m.weight.unsqueeze(0))
            elif isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight.unsqueeze(0))

    def forward(self, x):
        x = x.unsqueeze(dim=1)
        out = self.transformer_encoder(x)
        return out


# Instantiate the network components
DBn_outsize = 512
DBn = DBN(num_visible_mod1=260, num_hidden_mod1=DBn_outsize, num_visible_mod2=196608, num_hidden_mod2=DBn_outsize)
encoder_outsize = 512

encoder_1 = LstmRNN(input_size=DBn_outsize, output_size=encoder_outsize)
encoder_2 = ResNet_1D(input_channels=1, output_channels=encoder_outsize)
# encoder_2 = TransformerEncoder(input_size = 512, output_size = 512)

fusion_net = DualModalDBN(encoder_1, encoder_2, dual_fusion_modal=DBn, dual_shape=encoder_outsize*2, out_class=5)

# Test the network
x = torch.randn([32, 260])
img = torch.randn([32, 3, 256, 256])
out, gen_x, gen_img = fusion_net(x, img)
