import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
class generator_conditional(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=20, out_channels=256, kernel_size=5, padding='same')
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
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=6, out_channels=256, kernel_size=5, padding='same')
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
        super().__init__()
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
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=4, out_channels=256, kernel_size=5, padding='same')
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

def gaussian_noise_layer(input_layer, std):
    noise = torch.normal(mean=0.0, std=std, size=input_layer.size())
    return input_layer + noise


def Rayleigh_noise_layer(input_layer, h_r, h_i, std):
    h_complex = torch.complex(torch.tensor(h_r), torch.tensor(h_i))
    input_layer_real = input_layer[:, :, 0]
    input_layer_imag = input_layer[:, :, 1]
    input_layer_complex = torch.complex(input_layer_real, input_layer_imag)
    noise_real = torch.normal(mean=0.0, std=std, size=input_layer_complex.size())
    noise_imag = torch.normal(mean=0.0, std=std, size=input_layer_complex.size())
    noise = torch.complex(noise_real, noise_imag)
    output_complex = h_complex * input_layer_complex + noise
    output_complex_reshape = output_complex.view(-1, block_length, 1)
    # print("Shape of the output complex", output_complex, output_complex_reshape)
    return torch.cat([output_complex_reshape.real, output_complex_reshape.imag], dim=-1)

def sample_h(sample_size):
    """ sampling the h """
    return np.random.normal(size=sample_size) / np.sqrt(2.)

def generate_batch_data(batch_size):
    global start_idx, data
    if start_idx + batch_size >= N_training:
        start_idx = 0
        data = np.random.binomial(1, 0.5, [N_training, block_length, 1])
    batch_x = data[start_idx:start_idx + batch_size]
    start_idx += batch_size
    return batch_x

""" Start of the Main function """


''' Building the Graph'''
batch_size = 512
block_length = 128
Z_dim_c = 16
learning_rate = 1e-4

number_steps_receiver = 5000
number_steps_channel = 5000
number_steps_transmitter = 5000
display_step = 100
batch_size = 320
number_iterations = 1000  # in each iteration, the receiver, the transmitter and the channel will be updated

EbNo_train = 20.
EbNo_train = 10. ** (EbNo_train / 10.)

EbNo_train_GAN = 35.
EbNo_train_GAN = 10. ** (EbNo_train_GAN / 10.)

EbNo_test = 15.
EbNo_test = 10. ** (EbNo_test / 10.)

R = 0.5

Disc_vars = [param_D for param_D in discriminator_condintional().parameters()]
Gen_vars = [param_G for param_G in generator_conditional().parameters()]
Tx_vars = [param_T for param_T in Encoding().parameters()]
Rx_vars = [param_R for param_R in Decoding().parameters()]

N_training = int(1e6)
data = np.random.binomial(1, 0.5, [N_training, block_length, 1])
N_val = int(1e4)
val_data = np.random.binomial(1, 0.5, [N_val, block_length, 1])
N_test = int(1e4)
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
        batch_x = generate_batch_data(int(batch_size / 2))
        batch_x = torch.from_numpy(np.asarray(batch_x)).float()
        E = Encoder(batch_x)
        random_data = sample_uniformly([int(batch_size / 2), block_length, 2])
        input_data = torch.tensor(np.concatenate((E.detach().numpy().reshape([int(batch_size / 2), block_length, 2])
                                     + np.random.normal(0, 0.1, size=([int(batch_size / 2), block_length, 2])),
                                     random_data), axis=0))
        ########### D simulation ########

        h_i = sample_h([batch_size, 1])
        h_r = sample_h([batch_size, 1])
        Z = torch.tensor(sample_Z([batch_size, block_length, Z_dim_c]))  # noise
        Noise_std = (np.sqrt(1 / (2 * R * EbNo_train_GAN)))
        Channel_info = torch.cat([torch.tensor(h_r).view(-1, 1, 1),torch.tensor(h_i).view(-1, 1, 1)], dim = -1).repeat(1, block_length, 1)
        Conditions_uniform = torch.cat((torch.tensor(input_data), torch.tensor(Channel_info)), -1) # conditional info.

        D_solver = torch.optim.Adam(Disc_vars, lr=1e-4)
        G_solver = torch.optim.Adam(Gen_vars, lr=1e-4)

        G_sample_uniform = Gen(Z, Conditions_uniform) #fake data
        R_sample_uniform = Rayleigh_noise_layer(input_data, h_r, h_i, Noise_std) # real data
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
        E = Encoder(batch_x)
        Z = torch.tensor(sample_Z([batch_size, block_length, Z_dim_c]))  # noise
        h_i = sample_h([batch_size, 1])
        h_r = sample_h([batch_size, 1])
        Noise_std = (np.sqrt(1 / (2 * R * EbNo_train)))
        Channel_info = torch.cat([torch.tensor(h_r).reshape(-1, 1, 1),torch.tensor(h_i).reshape(-1, 1, 1)], dim = -1).repeat(1, block_length, 1)
        Conditions = torch.cat((torch.tensor(E), torch.tensor(Channel_info)), -1) # conditional info.
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
        h_i = sample_h([batch_size, 1])
        h_r = sample_h([batch_size, 1])
        Noise_std = (np.sqrt(1 / (2 * R * EbNo_train)))
        E_r = Encoder(torch.tensor(batch_x)) #encode
        R_sample = Rayleigh_noise_layer(E_r, h_r, h_i, Noise_std) #pass layer
        Channel_info = torch.cat([torch.tensor(h_r).reshape(-1, 1, 1),torch.tensor(h_i).reshape(-1, 1, 1)], dim = -1).repeat(1, block_length, 1)
        R_decodings_logit, R_decodings_prob = Decoder(R_sample, Channel_info)
        loss_receiver_R = torch.mean(F.binary_cross_entropy_with_logits(R_decodings_logit, torch.tensor(batch_x, dtype=torch.float32)))
        Rx_Optimizer = torch.optim.Adam(Rx_vars, lr=1e-4)

        Rx_Optimizer.zero_grad()
        loss_receiver_R.backward()
        Rx_Optimizer.step()

    '''  ----- Testing ----  '''
    '''--------batch sample -----------'''

    accuracy_R = torch.mean(torch.cast(torch.abs(R_decodings_prob - batch_x) > 0.5, torch.float32))
    print("Real Channel Evaluation:", "Step " + str(step) + ", Minibatch Loss= " + \
          "{:.4f}".format(loss_receiver_R) + ", Training Accuracy= " + \
          "{:.3f}".format(accuracy_R))
    accuracy_G = torch.mean(torch.cast(torch.abs(G_decodings_prob - batch_x) > 0.5, torch.float32))
    print("Generated Channel Evaluation:", "Step " + str(step) + ", Minibatch Loss= " + \
          "{:.4f}".format(loss_receiver_G) + ", Training Accuracy= " + \
          "{:.3f}".format(accuracy_G))


    '''---------- test data ----------------------'''
    # test_data = torch.from_numpy(np.random.binomial(1, 0.5, [N_test, block_length, 1])).float()
    # E = Encoder(test_data)
    # h_i = sample_h([batch_size, 1])
    # h_r = sample_h([batch_size, 1])
    # Noise_std = (np.sqrt(1 / (2 * R * EbNo_train)))
    # R_sample = Rayleigh_noise_layer(E, h_r, h_i, Noise_std)  # pass layer
    # Channel_info = torch.cat([torch.tensor(h_r).reshape(-1, 1, 1), torch.tensor(h_i).reshape(-1, 1, 1)], dim=-1).repeat(
    #     1, block_length, 1)
    # R_decodings_logit, R_decodings_prob = Decoder(R_sample, Channel_info)
    # loss_receiver_R = torch.mean(F.binary_cross_entropy_with_logits(torch.tensor(R_decodings_logit), test_data))
    # accuracy_R = np.mean(np.abs(R_decodings_prob.detach().numpy() - test_data.detach().numpy()) > 0.5)
    # print("Real Channel Evaluation:", "Step " + str(step) + ", test_data Loss= " + \
    #       "{:.4f}".format(loss_receiver_R) + ", Test Accuracy= " + \
    #       "{:.3f}".format(accuracy_R))
    #
    # Z = torch.tensor(sample_Z([batch_size, block_length, Z_dim_c]))  # noise
    # Channel_info = torch.cat((torch.tensor(h_r).unsqueeze(1).repeat(1, block_length, 1),
    #                           torch.tensor(h_i).unsqueeze(1).repeat(1, block_length, 1)), dim=-1)  # (1000, 67, 6)
    # Conditions = torch.cat([E, Channel_info], axis=-1)  # conditional info.
    # G_sample = Gen(Z, Conditions)  # generator for fake data
    # G_decodings_logit, G_decodings_prob = Decoder(G_sample, Channel_info)  # pass fake channel (generator)
    # loss_receiver_G = torch.mean(F.binary_cross_entropy_with_logits(torch.tensor(G_decodings_logit), test_data))
    # accuracy_G = np.mean(np.abs(G_decodings_prob.detach().numpy() - test_data.detach().numpy()) > 0.5)
    # print("Generated Channel Evaluation:", "Step " + str(step) + ", test_data Loss= " + \
    #       "{:.4f}".format(loss_receiver_G) + ", Test Accuracy= " + \
    #       "{:.3f}".format(accuracy_G))



    EbNodB_range = np.arange(0, 30)
    ber = np.ones(len(EbNodB_range))
    wer = np.ones(len(EbNodB_range))
    for n in range(0, len(EbNodB_range)):
        EbNo = 10.0 ** (EbNodB_range[n] / 10.0)
        test_data = torch.from_numpy(np.random.binomial(1, 0.5, [N_test, block_length, 1])).float()
        E = Encoder(test_data)
        h_i = sample_h([batch_size, 1])
        h_r = sample_h([batch_size, 1])
        Noise_std = (np.sqrt(1 / (2 * R * EbNo_train)))
        R_sample = Rayleigh_noise_layer(E, h_r, h_i, Noise_std)
        Channel_info = torch.cat([torch.tensor(h_r).reshape(-1, 1, 1), torch.tensor(h_i).reshape(-1, 1, 1)],
                                 dim=-1).repeat(1, block_length, 1)
        R_decodings_logit, R_decodings_prob = Decoder(R_sample, Channel_info)
        ber[n] = np.mean(np.abs(R_decodings_prob.detach().numpy() - test_data.detach().numpy()) > 0.5)
        wer[n] = 1 - torch.mean(torch.cast(torch.all(torch.abs(R_decodings_prob - test_data) < 0.5, dim=1), torch.float32))
        E = Encoder(test_data)
        Noise_std = (np.sqrt(1 / (2 * R * EbNo)))
        h_i = sample_h([batch_size, 1])
        h_r = sample_h([batch_size, 1])

        print('SNR:', EbNodB_range[n], 'BER:', ber[n], 'WER:', wer[n])

    print(ber)
    print(wer)
