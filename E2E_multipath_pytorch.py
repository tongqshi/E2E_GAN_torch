import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
class generator_conditional(nn.Module):
    def __init__(self):
        super(generator_conditional, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=24, out_channels=256, kernel_size=5, padding='same')
        self.conv2 = nn.Conv1d(in_channels=256, out_channels=128, kernel_size=3, padding='same')
        self.conv3 = nn.Conv1d(in_channels=128, out_channels=64, kernel_size=3, padding='same')
        self.conv4 = nn.Conv1d(in_channels=64, out_channels=2, kernel_size=3, padding='same')
        self.relu = nn.LeakyReLU()
    def forward(self, z, conditioning):
        z_combine = torch.cat([z, conditioning], dim=-1).permute(0,2,1).to(torch.float32)
        conv1 = self.conv1(z_combine)
        conv1 = self.relu(conv1)
        conv2 = self.conv2(conv1)
        conv2 = self.relu(conv2)
        conv3 = self.conv3(conv2)
        conv3 = self.relu(conv3)
        conv4 = self.conv4(conv3)
        return conv4.permute(0,2,1)

class discriminator_condintional(nn.Module):
    def __init__(self):
        super(discriminator_condintional, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=10, out_channels=256, kernel_size=5, padding='same')
        self.conv2 = nn.Conv1d(in_channels=256, out_channels=128, kernel_size=3, padding='same')
        self.conv3 = nn.Conv1d(in_channels=128, out_channels=64, kernel_size=3, padding='same')
        self.conv4 = nn.Conv1d(in_channels=64, out_channels=16, kernel_size=3, padding='same')
        self.linear1 = nn.Linear(16, 100)
        self.linear2 = nn.Linear(100, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    def forward(self, x, conditioning):
        x_combine = torch.cat([x, conditioning], dim=-1).permute(0,2,1).to(torch.float32)
        conv1 = self.conv1(x_combine)
        conv1 = self.relu(conv1)
        conv1 = torch.mean(conv1, dim=0, keepdim=True)
        conv2 = self.conv2(conv1)
        conv2 = self.relu(conv2)
        conv3 = self.conv3(conv2)
        conv3 = self.relu(conv3)
        conv4 = self.conv4(conv3)
        conv4 = self.relu(conv4).permute(0,2,1)
        FC = self.linear1(conv4)
        D_logit = self.linear2(FC)
        D_prob = self.sigmoid(D_logit)
        return D_prob, D_logit

class Encoding(nn.Module):
    def __init__(self):
        super(Encoding, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=256, kernel_size=5, padding='same')
        self.conv2 = nn.Conv1d(in_channels=256, out_channels=128, kernel_size=3, padding='same')
        self.conv3 = nn.Conv1d(in_channels=128, out_channels=64, kernel_size=3, padding='same')
        self.conv4 = nn.Conv1d(in_channels=64, out_channels=2, kernel_size=3, padding='same')
        self.relu = nn.ReLU()
    def forward(self, x):
        conv1 = self.conv1(x.permute(0,2,1).to(torch.float32))
        conv1 = self.relu(conv1)
        conv2 = self.conv2(conv1)
        conv2 = self.relu(conv2)
        conv3 = self.conv3(conv2)
        conv3 = self.relu(conv3)
        conv4 = self.conv4(conv3)
        layer_4_normalized = torch.sqrt(torch.tensor(block_length / 2.0)) * torch.nn.functional.normalize(
            conv4, dim=1)
        return layer_4_normalized.permute(0,2,1)

class Decoding(nn.Module):
    def __init__(self):
        super(Decoding, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=8, out_channels=256, kernel_size=5, padding='same')
        self.conv2 = nn.Conv1d(in_channels=256, out_channels=128, kernel_size=5, padding='same')
        self.conv3 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=5, padding='same')
        self.conv4 = nn.Conv1d(in_channels=128, out_channels=64, kernel_size=5, padding='same')
        self.conv5 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=5, padding='same')
        self.conv6 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, padding='same')
        self.conv7 = nn.Conv1d(in_channels=64, out_channels=32, kernel_size=3, padding='same')
        self.conv8 = nn.Conv1d(in_channels=32, out_channels=1, kernel_size=3, padding='same')
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    def forward(self, x, channel_info):
        x_combine = torch.cat([x, channel_info], dim=-1).permute(0,2,1).to(torch.float32)
        conv1 = self.conv1(x_combine)
        conv1 = self.relu(conv1)
        conv2_ori = self.conv2(conv1)
        conv2 = self.relu(conv2_ori)
        conv2 = self.conv3(self.relu(self.conv3(conv2)))
        conv2 += conv2_ori
        conv2 = self.relu(conv2)
        conv3_ori = self.conv4(conv2)
        conv3 = self.relu(conv3_ori)
        conv3 = self.conv6(self.relu(self.conv5(conv3)))
        conv3 += conv3_ori
        conv3 += self.relu(conv3)
        conv4 = self.relu(self.conv7(conv3))
        conv4 = self.relu(conv4)
        D_logit = self.conv8(conv4).permute(0,2,1)
        D_prob = self.sigmoid(D_logit)

        return D_logit[:, 0:block_length], D_prob[:, 0:block_length]

def sample_Z(sample_size):
    ''' Sampling the generation noise Z from normal distribution '''
    return np.random.normal(size=sample_size)

def sample_uniformly(sample_size):
    return np.random.randint(size=sample_size, low=-15, high=15) / 10

def Multipath_layer(x, h_r, h_i, std): ###tensor
    # Pad x along the second dimension
    x_pad = torch.nn.functional.pad(x, (0, 0, 0, L, 0, 0))
    # Reshape h_r and h_i
    h_r = torch.Tensor(h_r).view(-1, L, 1)
    h_i = torch.Tensor(h_i).view(-1, L, 1)
    # Reshape x_pad to separate real and imaginary parts
    x_r = x_pad[:, :, 0].view(-1, block_length + L, 1)
    x_i = x_pad[:, :, 1].view(-1, block_length + L, 1)
    # Define the convolution function
    def convolution(x, h):
        y = x * h[:, 0, 0].view(-1, 1, 1)
        for i in range(1, L):
            cur = x * h[:, i, 0].view(-1, 1, 1)
            cur = torch.cat([cur[:, -i:, :], cur[:, :-i, :]], dim=1)
            y += cur
        return y
    # Perform convolution for real and imaginary parts
    o_r = convolution(x_r, h_r) - convolution(x_i, h_i)
    o_i = convolution(x_r, h_i) + convolution(x_i, h_r)
    # Concatenate real and imaginary parts
    output = torch.cat([o_r, o_i], dim=-1)
    # Add Gaussian noise
    noise = torch.normal(mean=0.0, std=std, size=output.size())
    output += noise
    return output

def generate_channel(PDP):
    """ Generate channel based on the PDP """
    h = 1 / np.sqrt(2) * (np.random.normal(size=len(PDP)) + 1j * np.random.normal(size=len(PDP))) * np.sqrt(PDP)
    return h

def generate_channel_parts(PDP, sample_size):
    """ Generate real and imagary part of channel """
    h = 1 / np.sqrt(2) * np.sqrt(PDP) * np.random.normal(size=sample_size)
    return h

def generate_PDP(L):
    """ Generate the PDP for channel generation """
    PDP = np.ones(L)
    PDP = PDP / sum(PDP)
    return PDP

def sample_h(sample_size):
    """ sampling the h """
    return np.random.normal(size=sample_size) / np.sqrt(2.)

def encoding_padding(encoding):
    """ padding the encodings s.t. the output number will be the same as the conv """
    padding = [0, 0, 0, L, 0, 0]  # 指定每个维度的填充数，形式为 (padding_left, padding_right, ...)
    encoding_padding = torch.nn.functional.pad(encoding, padding, mode='constant', value=0)  # 使用 F.pad 进行填充
    return encoding_padding

def generate_batch_data(batch_size):
    global start_idx, data
    if start_idx + batch_size >= data_size:
        start_idx = 0
        data = np.random.binomial(1, 0.5, [data_size, block_length, 1])
    batch_x = data[start_idx:start_idx + batch_size]
    start_idx += batch_size
    return batch_x

""" Start of the Main function """


''' Building the Graph'''
batch_size = 512
block_length = 64
condition_depth = 4
Z_dim_c = 16
learning_rate = 1e-4
L = 3
condition_length = block_length + L
channel_PDP = generate_PDP(L)
number_steps_receiver = 0
number_steps_channel = 0
number_steps_transmitter = 0
display_step = 100
batch_size = 1000
number_iterations = 1000  # in each iteration, the receiver, the transmitter and the channel will be updated

EbNo_train = 20.
EbNo_train = 10. ** (EbNo_train / 10.)

EbNo_train_GAN = 35
EbNo_train_GAN = 10. ** (EbNo_train_GAN / 10.)

EbNo_test = 15.
EbNo_test = 10. ** (EbNo_test / 10.)

R = 0.5

Disc_vars = [param_D for name_D, param_D in discriminator_condintional().named_parameters()]
Gen_vars = [param_G for name_G, param_G in generator_conditional().named_parameters()]
Tx_vars = [param_T for name_T, param_T in Encoding().named_parameters()]
Rx_vars = [param_R for name_R, param_R in Decoding().named_parameters()]

N_training = int(5e6)
data = np.random.binomial(1, 0.5, [N_training, block_length, 1])
data_size = len(data)
N_val = int(1e4)
val_data = np.random.binomial(1, 0.5, [N_val, block_length, 1])
N_test = int(1e3)
test_data = np.random.binomial(1, 0.5, [N_test, block_length, 1])
start_idx = 0

Encoder = Encoding()
Decoder = Decoding()
Gen = generator_conditional()
Dis = discriminator_condintional()

for iteration in range(number_iterations): #for iteration in range(number_iterations):
    print("iteration is ", iteration)
    number_steps_transmitter += 5000
    number_steps_receiver += 5000
    number_steps_channel += 2000
    ''' =========== Training the Channel Simulator ======== '''
    for step in range(number_steps_channel): #for step in range(number_steps_channel):
        if step % 100 == 0:
            print("Training ChannelGAN, step is ", step)
        # batch_x = data[start_idx:start_idx + int(batch_size / 2), :]
        batch_x = generate_batch_data(int(batch_size / 2))
        batch_x = torch.from_numpy(np.asarray(batch_x)).float()
        encoded_data = Encoder(batch_x)
        random_data = sample_uniformly([int(batch_size / 2), block_length, 2])
        input_data = torch.tensor(np.concatenate((encoded_data.detach().numpy().reshape([int(batch_size / 2), block_length, 2])
                                     + np.random.normal(0, 0.1, size=([int(batch_size / 2), block_length, 2])),
                                     random_data), axis=0))
        ########### D simulation ########
        h_i = generate_channel_parts(channel_PDP, [batch_size, L])
        h_r = generate_channel_parts(channel_PDP, [batch_size, L])
        E = encoding_padding(encoded_data)
        encodings_uniform_generated_padding = encoding_padding(input_data)
        Channel_info = torch.cat([torch.tensor(h_r).reshape(-1, 1, L),torch.tensor(h_i).reshape(-1, 1, L)], dim = -1).repeat(1, condition_length, 1)
        Conditions_uniform = torch.cat((encodings_uniform_generated_padding, torch.tensor(Channel_info)), -1) # conditional info.
        Z = torch.tensor(sample_Z([batch_size, condition_length, Z_dim_c])) #noise
        Noise_std = (np.sqrt(1 / (2 * R * EbNo_train_GAN)))
        D_solver = torch.optim.Adam(Disc_vars, lr=1e-4)
        G_solver = torch.optim.Adam(Gen_vars, lr=1e-4)
        G_sample_uniform = Gen(Z, Conditions_uniform) #fake data
        R_sample_uniform = Multipath_layer(input_data, h_r, h_i, Noise_std) # real data
        D_prob_real, D_logit_real = Dis(R_sample_uniform, Conditions_uniform) # dis. result for real data
        D_prob_fake, D_logit_fake = Dis(G_sample_uniform, Conditions_uniform) # dis. result for fake data
        D_loss_real = torch.mean(F.binary_cross_entropy_with_logits(torch.tensor(D_logit_real), torch.ones_like(D_logit_real))) # loss for real data
        D_loss_fake = torch.mean(F.binary_cross_entropy_with_logits(torch.tensor(D_logit_fake), torch.zeros_like(D_logit_fake))) # loss for fake data
        D_loss = D_loss_real + D_loss_fake
        D_loss.requires_grad_(True)
        G_loss = torch.mean(F.binary_cross_entropy_with_logits(torch.tensor(D_logit_fake), torch.ones_like(D_logit_fake)))
        G_loss.requires_grad_(True)

        D_solver.zero_grad()
        G_solver.zero_grad()
        D_loss.backward()
        G_loss.backward()
        D_solver.step()
        G_solver.step()

    ''' =========== Training the Transmitter ======== '''
    for step in range(number_steps_transmitter):
        if step % 100 == 0:
            print("Training transmitter, step is ", step)
        batch_x = generate_batch_data(batch_size)
        batch_x = torch.from_numpy(np.asarray(batch_x)).float()
        encoded_data = Encoder(batch_x)
        Z = torch.tensor(sample_Z([batch_size, condition_length, Z_dim_c]))  # noise
        h_i = generate_channel_parts(channel_PDP, [batch_size, L])
        h_r = generate_channel_parts(channel_PDP, [batch_size, L])
        Noise_std = (np.sqrt(1 / (2 * R * EbNo_train)))
        E = encoding_padding(encoded_data)
        Channel_info = torch.cat([torch.tensor(h_r).reshape(-1, 1, L), torch.tensor(h_i).reshape(-1, 1, L)],
                                 dim=-1).repeat(1, condition_length, 1)
        Conditions = torch.cat([E, Channel_info], axis=-1) # conditional info.
        G_sample = Gen(Z, Conditions) # generator for fake data
        G_decodings_logit, G_decodings_prob = Decoder(G_sample, Channel_info) # pass fake channel (generator)
        loss_receiver_G = torch.mean(F.binary_cross_entropy_with_logits(torch.tensor(G_decodings_logit), batch_x))
        loss_receiver_G.requires_grad_(True)
        Tx_Optimizer = torch.optim.Adam(Tx_vars, lr=1e-4)

        Tx_Optimizer.zero_grad()
        loss_receiver_G.backward()
        Tx_Optimizer.step()

    ''' ========== Training the Receiver ============== '''

    for step in range(number_steps_receiver):
        if step % 100 == 0:
            print("Training receiver, step is ", step)
        batch_x = generate_batch_data(batch_size)
        h_i = generate_channel_parts(channel_PDP, [batch_size, L])
        h_r = generate_channel_parts(channel_PDP, [batch_size, L])
        Noise_std = (np.sqrt(1 / (2 * R * EbNo_train)))
        E_r = Encoder(torch.tensor(batch_x)) #encode
        R_sample = Multipath_layer(E_r, h_r, h_i, Noise_std) #pass layer
        Channel_info = torch.cat([torch.tensor(h_r).reshape(-1, 1, L),torch.tensor(h_i).reshape(-1, 1, L)], dim = -1).repeat(1, condition_length, 1)
        R_decodings_logit, R_decodings_prob = Decoder(R_sample, Channel_info)
        loss_receiver_R = torch.mean(F.binary_cross_entropy_with_logits(R_decodings_logit, torch.tensor(batch_x, dtype=torch.float32)))
        Rx_Optimizer = torch.optim.Adam(Rx_vars, lr=1e-4)

        Rx_Optimizer.zero_grad()
        loss_receiver_R.backward()
        Rx_Optimizer.step()

    '''  ----- Testing ----  '''
    '''--------use batch sample -----------'''
    accuracy_R = np.mean(np.abs(R_decodings_prob.detach().numpy() - batch_x) > 0.5)
    print("Real Channel Evaluation:", "Step " + str(step) + ", Minibatch Loss= " + \
          "{:.4f}".format(loss_receiver_R) + ", Training Accuracy= " + \
          "{:.3f}".format(accuracy_R))
    accuracy_G = np.mean(np.abs(G_decodings_prob.detach().numpy() - batch_x) > 0.5)
    print("Generated Channel Evaluation:", "Step " + str(step) + ", Minibatch Loss= " + \
          "{:.4f}".format(loss_receiver_G) + ", Training Accuracy= " + \
          "{:.3f}".format(accuracy_G))
    '''---------- use test data ----------------------'''
    test_data = torch.from_numpy(np.random.binomial(1, 0.5, [N_test, block_length, 1])).float()
    E_r = Encoder(test_data)
    h_i = generate_channel_parts(channel_PDP, [batch_size, L])
    h_r = generate_channel_parts(channel_PDP, [batch_size, L])
    Noise_std = (np.sqrt(1 / (2 * R * EbNo_train)))
    R_sample = Multipath_layer(E_r, h_r, h_i, Noise_std)  # pass layer
    Channel_info = torch.cat([torch.tensor(h_r).reshape(-1, 1, L),torch.tensor(h_i).reshape(-1, 1, L)], dim = -1).repeat(1, condition_length, 1)
    R_decodings_logit, R_decodings_prob = Decoder(R_sample, Channel_info)
    loss_receiver_R = torch.mean(F.binary_cross_entropy_with_logits(torch.tensor(R_decodings_logit), test_data))
    accuracy_R = np.mean(np.abs(R_decodings_prob.detach().numpy() - test_data.detach().numpy()) > 0.5)
    print("Real Channel Evaluation:", "Step " + str(step) + ", Minibatch Loss= " + \
          "{:.4f}".format(loss_receiver_R) + ", Training Accuracy= " + \
          "{:.3f}".format(accuracy_R))

    Z = torch.tensor(sample_Z([batch_size, condition_length, Z_dim_c]))  # noise
    E = encoding_padding(E_r)
    Channel_info = torch.cat((torch.tensor(h_r).unsqueeze(1).repeat(1, condition_length, 1),
                              torch.tensor(h_i).unsqueeze(1).repeat(1, condition_length, 1)), dim=-1)  # (1000, 67, 6)
    Conditions = torch.cat([E, Channel_info], axis=-1)  # conditional info.
    G_sample = Gen(Z, Conditions)  # generator for fake data
    G_decodings_logit, G_decodings_prob = Decoder(G_sample, Channel_info)  # pass fake channel (generator)
    loss_receiver_G = torch.mean(F.binary_cross_entropy_with_logits(torch.tensor(G_decodings_logit), test_data))
    accuracy_G = np.mean(np.abs(G_decodings_prob.detach().numpy() - test_data.detach().numpy()) > 0.5)
    print("Generated Channel Evaluation:", "Step " + str(step) + ", Minibatch Loss= " + \
          "{:.4f}".format(loss_receiver_G) + ", Training Accuracy= " + \
          "{:.3f}".format(accuracy_G))
















"""
encoded_padding: (1, 500, 67, 2)
(1, 500, 64, 2)
(500, 64, 2)
input data (1000, 64, 2)
h_i (1000, 3)
encoded (1000, 67, 2)
G_sample_uniform (1000, 67, 2)
Z (1000, 67, 16)
Conditions_uniform (1000, 67, 8)
D_prob_real (1, 67, 1)
"""
