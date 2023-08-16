import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import random
from torchvision import transforms
import torchvision
import logging
import time
import torch.nn as nn



device = "cuda" if torch.cuda.is_available() else "cpu"
dataset = "mnist"

minmax_num = 1000
minmax_sample_ratio = 0.05
target_class = 3
num_client = 10
victim_client = 9
attack_client = 7

# seed = np.random.randint(0, 999999)
seed = 2023
print(f"seed:{seed}")
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

logging.basicConfig(format='%(message)s', level=logging.DEBUG, filename=dataset+".log", filemode='a')
logging.getLogger('matplotlib.font_manager').disabled = True
handler = logging.StreamHandler()
handler.terminator = ""

def mms(grads, sample_ratio):
    randv = torch.rand_like(grads).to(device)
    weight = (1 / randv - 1) * grads ** 2
    weight = torch.where(weight.isnan(), torch.zeros_like(weight), weight).to(device)

    sort, idx = weight.sort(descending=True)

    kth = int(sample_ratio * grads.size(0))
    topk_idx = idx[:kth]

    L = torch.zeros_like(grads).to(device)
    L[topk_idx] = 1

    return L


def draw_imags(imgs, imgs_indices):
    plt.figure(figsize=(12, 8))
    for index, data in enumerate(imgs):
        plt.subplot(2, 5, index + 1)
        plt.imshow(data.cpu().numpy().squeeze(), cmap='gray')
        plt.title("clients{}".format(imgs_indices[index]))
    plt.savefig(dataset+"_imgs.png")
    plt.show()

def label_to_onehot(target, num_classes):
    target = torch.unsqueeze(target, 1)
    onehot_target = torch.zeros(target.size(0), num_classes, device=target.device)
    onehot_target.scatter_(1, target, 1)
    return onehot_target

def load_data():
    dst = torchvision.datasets.MNIST("./dataset/"+dataset, transform=transforms.Compose(
        [transforms.ToTensor()]), download=True)

    class_dst = [[] for i in range(10)]
    class_label = [[] for i in range(10)]
    for data in dst:
        class_dst[data[1]].append(data[0])
        class_label[data[1]].append(data[1])

    imgs = []
    label = []

    imgs_indices = random.sample([i for i in range(len(class_label[target_class]))], num_client)

    for i in imgs_indices:
        pic = class_dst[target_class][i]
        pic = pic.view(1, *pic.size())
        imgs.append(pic.to(device))

        target = torch.Tensor([class_label[target_class][i]]).long().view(1, ).to(device)
        gt_onehot_label = label_to_onehot(target, 10)
        label.append(gt_onehot_label)

    draw_imags(imgs, imgs_indices)

    return imgs, label

def deep_leakage(imgs, label, net, criterion, original_dy_dx, true_grads):

    history = []
    history.append(imgs[victim_client][0].cpu())
    history.append(imgs[attack_client][0].cpu())

    gt_data = imgs[victim_client]
    gt_onehot_label = label[victim_client]

    dummy_data = torch.randn(gt_data.size()).to(device).requires_grad_(True)
    dummy_label = torch.randn(gt_onehot_label.size()).to(device).requires_grad_(True)
    optimizer = torch.optim.LBFGS([dummy_data, dummy_label])

    loss = []
    loss_true = []
    for iters in range(24):
        def closure():
            optimizer.zero_grad()
            dummy_pred = net(dummy_data).to(device)
            dummy_onehot_label = F.softmax(dummy_label, dim=-1)
            dummy_loss = criterion(dummy_pred, dummy_onehot_label)
            dummy_dy_dx = torch.autograd.grad(dummy_loss, net.parameters(), create_graph=True)

            grad_diff = 0
            grad_true = 0
            for gx, gy, gz in zip(dummy_dy_dx, original_dy_dx, true_grads):
                grad_diff += ((gx - gy) ** 2).sum()
                grad_true += ((gx - gz) ** 2).sum()
            grad_diff.backward()

            loss_true.append(np.round(grad_true.item(), 4))

            return grad_diff

        optimizer.step(closure)
        if iters % 2 == 0:
            current_loss = closure()
            print("%.4f" % current_loss.item(), end=",  ")
            history.append(dummy_data[0].cpu().detach().numpy().squeeze())
            loss.append(np.round(current_loss.item(), 4))
            if iters == 50:
                if current_loss.item() > 50.:
                    break

    logging.info(f"loss:{loss}")
    loss_true = loss_true[::210]
    logging.info(f"loss:{loss_true}")
    print(f"loss:{loss_true}")
    a = np.array(loss).min()
    b = np.array(loss_true).min()

    plt.figure(figsize=(17, 8))
    plt.subplot(2, 7, 1)
    plt.imshow(history[0].squeeze(), cmap='gray')
    plt.title(f"victim{victim_client}")
    plt.subplot(2, 7, 2)
    plt.imshow(history[1].squeeze(), cmap='gray')
    plt.title(f"attack{attack_client}")
    for i in range(2, len(history)):
        plt.subplot(2, 7, i + 1)
        plt.imshow(history[i], cmap='gray')

        plt.title("iter=%d" % ((i - 2) * 2))
        plt.axis('off')

    print()
    return plt, a, b


class mnistCNN(nn.Module):
    def __init__(self):
        super(mnistCNN, self).__init__()
        act = nn.Sigmoid
        self.body = nn.Sequential(
            nn.Conv2d(1, 10, kernel_size=5),
            act(),
            nn.Conv2d(10, 20, kernel_size=5),
            act(),
            nn.Conv2d(20, 10, kernel_size=5),
            act(),
        )
        self.fc = nn.Sequential(
            nn.Linear(2560, 10)
        )

    def forward(self, x):
        out = self.body(x)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

def weights_init(m):
    if hasattr(m, "weight"):
        m.weight.data.uniform_(-0.5, 0.5)
    if hasattr(m, "bias"):
        m.bias.data.uniform_(-0.5, 0.5)

def cross_entropy_for_onehot(pred, target):
    return torch.mean(torch.sum(- target * F.log_softmax(pred, dim=-1), 1))
def flatten_network_function(grad):
    res = torch.tensor([]).to(device)
    for v in grad:
        v = v.view(1, -1).squeeze()
        res = torch.cat([res.squeeze(), v], 0).view(-1)
    return res

def get_grads(net):

    grads = []
    for idx in range(num_client):
        pred = net(imgs[idx])
        loss = cross_entropy_for_onehot(pred, gt_onehot_label[idx])
        grad = torch.autograd.grad(loss, net.parameters())
        grad = list((_.detach().clone() for _ in grad))
        grad = flatten_network_function(grad).clone().detach()     # ->tensor
        grads.append(grad)

    return grads


def shape_network_architecture(network, grads):     #grads: tensor
    end_grads = []
    begin = 0
    for _, para in network.named_parameters():
        temp_shape, num = para.shape, para.numel()
        end = begin + num
        temp = grads[begin:end]
        temp = temp.reshape(temp_shape)
        end_grads.append(temp.to(device))
        begin = end
    return end_grads

if __name__ == '__main__':

    imgs, gt_onehot_label = load_data()

    TIME = str(time.time())[:-8]

    logging.info(f"\nnum{minmax_num}_ratio{minmax_sample_ratio}_seed{seed}_{TIME}")

    net = mnistCNN().to(device)
    net.apply(weights_init)
    model_paras_num = sum(param.numel() for param in net.parameters())

    grads = get_grads(net)
    vic_grad = grads[victim_client]
    att_grad = grads[attack_client]
    aggregation_grad = torch.mean(torch.stack(grads), dim=0)

    sum_L = torch.zeros_like(vic_grad).to(device)
    for i in range(minmax_num):
        L = mms(vic_grad, minmax_sample_ratio)
        sum_L += L
    sort_vic, idx_vic = sum_L.sort(descending=True)

    # determine magnitude
    sort_abs, idx_abs = torch.abs(att_grad).sort(descending=True)
    end_grads = torch.tensor([0.] * model_paras_num).to(device)
    end_grads[idx_vic] = sort_abs

    # update sign
    ture_sign = torch.sign(sum_L)
    aggregation_sign = torch.sub(1, ture_sign)
    ture_sign = ture_sign * torch.sign(vic_grad)
    aggregation_sign = aggregation_sign * torch.sign(aggregation_grad)
    all_sign = torch.add(ture_sign, aggregation_sign)
    end_grads = end_grads * all_sign

    original_dy_dx = shape_network_architecture(net, end_grads)
    true_grads = shape_network_architecture(net, grads[victim_client])
    plt, loss, loss_true = deep_leakage(imgs, gt_onehot_label, net, cross_entropy_for_onehot, original_dy_dx, true_grads)

    plt.show()
    plt.savefig(
        f"{str(num_client)}_num{minmax_num}_ratio{minmax_sample_ratio}_seed{seed}_{TIME}_{str(loss)}_{str(loss_true)}.png")
    plt.suptitle(
        f"{str(num_client)}_num{minmax_num}_ratio{minmax_sample_ratio}_seed{seed}_{TIME}_{str(loss)}_{str(loss_true)}")