import torch
import torch.nn as nn

# Datos de entrada (1 muestra, 5 filas, 12 columnas)
input_data = torch.randn(1, 5, 12)

# Definir la capa de convolución 1D
# in_channels = número de canales de entrada (5 en este caso), out_channels = número de filtros de salida (1 en este caso)
# kernel_size = tamaño del kernel (3 en este caso), padding = 0 para 'valid' padding (sin padding)
conv1d_layer = nn.Conv1d(in_channels=5, out_channels=1, kernel_size=3, padding=1)

# Aplicar la convolución 1D
output_data = conv1d_layer(input_data)

# Imprimir la forma de la salida
print("Forma de la salida:", output_data.shape)
print("Forma de la salida:", output_data[0][0])

linear = nn.Linear(12, 6,False)
linear2 = nn.Linear(6, 4,False)

out_linear = linear2(linear(output_data))
print("out linear ", out_linear[0][0])


batch_size = 10  # Define el tamaño del lote

x = torch.randn(batch_size, 3, 64, 64)  # Batch size 10, 3 canales, 64x64 imágenes

conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=4, stride=2)
conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2)
conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2)

out_conv1 = nn.functional.relu(conv1(x))
out_conv2 = nn.functional.relu(conv2(out_conv1))
out_conv3 = nn.functional.relu(conv3(out_conv2))
out_conv4 = nn.functional.relu(conv4(out_conv3))

print(out_conv4.shape)  # torch.Size([10, 32, 32, 32])
print(out_conv4.size(0))

h = out_conv4.view(out_conv4.size(0), -1)
print(h.shape)

mu = nn.Linear(2*2*256, 128)
logvar = nn.Linear(2*2*256, 128)
out_mu = mu(h)
out_logvar = logvar(h)
print(out_mu.shape)