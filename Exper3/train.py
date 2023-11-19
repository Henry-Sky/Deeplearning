import torch
import torch.nn as nn
import torch.nn.functional as func
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.optim as optim
import torch

# 网络结构
class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()

        # 3 * 224 * 224
        self.conv1_1 = nn.Conv2d(3, 64, 3)  # 64 * 222 * 222
        self.conv1_2 = nn.Conv2d(64, 64, 3, padding=(1, 1))  # 64 * 222* 222
        self.maxpool1 = nn.MaxPool2d((2, 2), padding=(1, 1))  # pooling 64 * 112 * 112

        self.conv2_1 = nn.Conv2d(64, 128, 3)  # 128 * 110 * 110
        self.conv2_2 = nn.Conv2d(128, 128, 3, padding=(1, 1))  # 128 * 110 * 110
        self.maxpool2 = nn.MaxPool2d((2, 2), padding=(1, 1))  # pooling 128 * 56 * 56

        self.conv3_1 = nn.Conv2d(128, 256, 3)  # 256 * 54 * 54
        self.conv3_2 = nn.Conv2d(256, 256, 3, padding=(1, 1))  # 256 * 54 * 54
        self.conv3_3 = nn.Conv2d(256, 256, 3, padding=(1, 1))  # 256 * 54 * 54
        self.maxpool3 = nn.MaxPool2d((2, 2), padding=(1, 1))  # pooling 256 * 28 * 28

        self.conv4_1 = nn.Conv2d(256, 512, 3)  # 512 * 26 * 26
        self.conv4_2 = nn.Conv2d(512, 512, 3, padding=(1, 1))  # 512 * 26 * 26
        self.conv4_3 = nn.Conv2d(512, 512, 3, padding=(1, 1))  # 512 * 26 * 26
        self.maxpool4 = nn.MaxPool2d((2, 2), padding=(1, 1))  # pooling 512 * 14 * 14

        self.conv5_1 = nn.Conv2d(512, 512, 3)  # 512 * 12 * 12
        self.conv5_2 = nn.Conv2d(512, 512, 3, padding=(1, 1))  # 512 * 12 * 12
        self.conv5_3 = nn.Conv2d(512, 512, 3, padding=(1, 1))  # 512 * 12 * 12
        self.maxpool5 = nn.MaxPool2d((2, 2), padding=(1, 1))  # pooling 512 * 7 * 7

        self.fc1 = nn.Linear(512 * 7 * 7, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, 2)

    def forward(self, x):

        out = self.conv1_1(x)  # 222
        out = func.relu(out)
        out = self.conv1_2(out)  # 222
        out = func.relu(out)
        out = self.maxpool1(out)  # 112

        out = self.conv2_1(out)  # 110
        out = func.relu(out)
        out = self.conv2_2(out)  # 110
        out = func.relu(out)
        out = self.maxpool2(out)  # 56

        out = self.conv3_1(out)  # 54
        out = func.relu(out)
        out = self.conv3_2(out)  # 54
        out = func.relu(out)
        out = self.conv3_3(out)  # 54
        out = func.relu(out)
        out = self.maxpool3(out)  # 28

        out = self.conv4_1(out)  # 26
        out = func.relu(out)
        out = self.conv4_2(out)  # 26
        out = func.relu(out)
        out = self.conv4_3(out)  # 26
        out = func.relu(out)
        out = self.maxpool4(out)  # 14

        out = self.conv5_1(out)  # 12
        out = func.relu(out)
        out = self.conv5_2(out)  # 12
        out = func.relu(out)
        out = self.conv5_3(out)  # 12
        out = func.relu(out)
        out = self.maxpool5(out)  # 7

        # 展平
        out = torch.flatten(out, 1, -1)

        out = self.fc1(out)
        out = func.relu(out)
        out = self.fc2(out)
        out = func.relu(out)
        out = self.fc3(out)

        out = func.log_softmax(out, dim=1)
        return out
    
# 数据加载
data_transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 调整图片大小
    transforms.RandomHorizontalFlip(),  # 随机翻转图片
    transforms.ToTensor(),# 图片转tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # 数据预处理
])

train_data = datasets.ImageFolder(root="./data/catVSdog/train_data/", transform=data_transform)
test_data = datasets.ImageFolder(root="./data/catVSdog/test_data/", transform=data_transform)

train_loader = DataLoader(train_data, batch_size=24, shuffle=True)
test_loader = DataLoader(test_data, batch_size=24, shuffle=True)

# 模型初始化
net = VGG16().cuda()

for m in net.modules():
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.xavier_uniform_(m.weight)

loss_func = nn.NLLLoss()
optimizer = optim.SGD(net.parameters(), lr=0.005, momentum=0.9)

# 模型训练
epoch_num = 50

max_acc = 0.0
for epoch in range(epoch_num):
    loss_sum = 0.0
    total_batch = 0.0
    torch.cuda.empty_cache()
    for i, data in enumerate(train_loader):
        inputs, labels = data[0].cuda(), data[1].cuda()
        optimizer.zero_grad()
        outputs = net(inputs)
        # 默认为单批次损失的均值
        loss = loss_func(outputs, labels)
        loss_sum += loss.item()
        total_batch += 1
        loss.backward()
        optimizer.step()

    loss_mean = loss_sum / total_batch
    
    correct_sum = 0.0
    total = 0.0
    torch.cuda.empty_cache()
    for data in test_loader:
        inputs, labels = data[0].cuda(), data[1].cuda()
        with torch.no_grad():
            outputs = torch.exp(net(inputs))
            _, predicts = torch.max(outputs, 1)
            correct_sum += (predicts == labels).sum().item()
        total += labels.size(0)
    
    acc_mean = correct_sum / total * 100

    # 保存准确率更高的模型
    print(acc_mean)
    if acc_mean > 90.0 and acc_mean > max_acc:
        max_acc = acc_mean
        print("保存模型")
        torch.save(net.state_dict(),'./data/VGG16_{}.rui'.format(acc_mean))

    print("第{}次迭代完成,损失为:{:.8f},准确率为:{:.8f}%".format(epoch + 1, loss_mean, acc_mean))