import os
import argparse
import numpy as np
from PIL import Image, ImageDraw
import torch
import torch.nn as nn
from torch.utils import data
from torchvision import transforms
from loss import calc_gan_loss, words_loss, sent_loss
from model import InpaintNet, PatchDiscriminator, RNN_ENCODER, CNN_ENCODER
from datasets import TextDataset, prepare_data
from torch.autograd import Variable
import warnings

warnings.filterwarnings("ignore")

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='birds')  # birds flowers celeba
parser.add_argument('--root', type=str, default='../data/birds')
parser.add_argument('--save_dir', type=str, default='./model/birds')
parser.add_argument('--training_image', type=str, default='./training/birds')
parser.add_argument('--CAPTIONS_PER_IMAGE', type=int, default=10)
parser.add_argument('--WORDS_NUM', type=int, default=16)
parser.add_argument('--lr', type=float, default=2e-4, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--threshold", type=float, default=0.8, help="RN_L threshold")
parser.add_argument('--max_epoch', type=int, default=300)
parser.add_argument('--start_epoch', type=int, default=0)
parser.add_argument('--batch_size', type=int, default=6)
parser.add_argument('--n_threads', type=int, default=4)
parser.add_argument('--save_model_interval', type=int, default=10000)
parser.add_argument('--vis_interval', type=int, default=1000)
parser.add_argument('--log_interval', type=int, default=100)
parser.add_argument('--image_size', type=int, default=256)
parser.add_argument('--resume', type=int, default=0)
args = parser.parse_args()

use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')
if use_cuda:
    torch.backends.cudnn.benchmark = True
if not os.path.exists(args.save_dir):
    os.makedirs('{:s}'.format(args.save_dir))
if not os.path.exists(args.training_image):
    os.makedirs('{:s}'.format(args.training_image))

size = (args.image_size, args.image_size)
train_tf = transforms.Compose([
    transforms.Resize(size),
    transforms.RandomHorizontalFlip(),
    transforms.RandomGrayscale(),
])

dataset_train = TextDataset(args.root, base_size=args.image_size, CAPTIONS_PER_IMAGE=args.CAPTIONS_PER_IMAGE,
                            WORDS_NUM=args.WORDS_NUM, transform=train_tf)
assert dataset_train
train_set = torch.utils.data.DataLoader(
    dataset_train, batch_size=args.batch_size,
    drop_last=True, shuffle=True, num_workers=args.n_threads)

print(len(train_set))

ixtoword_train = dataset_train.ixtoword

g_model = InpaintNet().to(device)
pd_model = PatchDiscriminator().to(device)
l1 = nn.L1Loss().to(device)

g_optimizer_t = torch.optim.Adam(
    g_model.parameters(),
    args.lr, (args.b1, args.b2))
pd_optimizer_t = torch.optim.Adam(
    pd_model.parameters(),
    args.lr, (args.b1, args.b2))

if args.resume:
    g_checkpoint = torch.load(f'{args.save_dir}/G_{args.resume}.pth', map_location=device)
    g_model.load_state_dict(g_checkpoint)
    pd_checkpoint = torch.load(f'{args.save_dir}/PD_{args.resume}.pth', map_location=device)
    pd_model.load_state_dict(pd_checkpoint)
    print('Models restored')

image_encoder = CNN_ENCODER(args.image_size)
img_encoder_path = '../DAMSMencoders/' + args.dataset + '/image_encoder500.pth'
state_dict = torch.load(img_encoder_path, map_location=lambda storage, loc: storage)
image_encoder.load_state_dict(state_dict)
for p in image_encoder.parameters():
    p.requires_grad = False
print('Load image encoder from:', img_encoder_path)
image_encoder.eval()

print('dataset_train.n_words', dataset_train.n_words)

text_encoder = RNN_ENCODER(dataset_train.n_words, nhidden=args.image_size)
text_encoder_path = '../DAMSMencoders/' + args.dataset + '/text_encoder500.pth'
state_dict = torch.load(text_encoder_path, map_location=lambda storage, loc: storage)
text_encoder.load_state_dict(state_dict)
for p in text_encoder.parameters():
    p.requires_grad = False
print('Load text encoder from:', text_encoder_path)
text_encoder.eval()

if use_cuda:
    text_encoder = text_encoder.cuda()
    image_encoder = image_encoder.cuda()


def prepare_labels():
    batch_size = args.batch_size
    real_labels = Variable(torch.FloatTensor(batch_size).fill_(1))
    fake_labels = Variable(torch.FloatTensor(batch_size).fill_(0))
    match_labels = Variable(torch.LongTensor(range(batch_size)))
    if use_cuda:
        real_labels = real_labels.cuda()
        fake_labels = fake_labels.cuda()
        match_labels = match_labels.cuda()

    return real_labels, fake_labels, match_labels


real_labels, fake_labels, match_labels = prepare_labels()


def get_mask():
    mask = []
    IMAGE_SIZE = args.image_size

    for i in range(args.batch_size):
        q1 = p1 = IMAGE_SIZE // 4
        q2 = p2 = IMAGE_SIZE - IMAGE_SIZE // 4

        m = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
        m[q1:q2 + 1, p1:p2 + 1] = 1
        m = np.expand_dims(m, 0)
        mask.append(m)

    mask = np.array(mask)
    mask = torch.from_numpy(mask)

    if use_cuda:
        mask = mask.float().cuda()
    return mask


nz = 100
noise = Variable(torch.FloatTensor(args.batch_size, nz))
fixed_noise = Variable(torch.FloatTensor(args.batch_size, nz).normal_(0, 1))
if use_cuda:
    noise, fixed_noise = noise.cuda(), fixed_noise.cuda()

for i in range(args.start_epoch, args.max_epoch + 1):

    iterator_train = iter(train_set)
    train_step = 1
    num_batches = len(train_set) + 1
    while train_step < num_batches:

        data_train = iterator_train.next()
        real_masks, imgs, captions, cap_lens, class_ids, keys = prepare_data(data_train)

        hidden = text_encoder.init_hidden(args.batch_size)
        words_embs, sent_emb = text_encoder(captions, cap_lens, hidden)
        words_embs, sent_emb = words_embs.detach(), sent_emb.detach()
        text_mask = (captions == 0)
        num_words = words_embs.size(2)
        if text_mask.size(1) > num_words:
            text_mask = text_mask[:, :num_words]

        img = imgs[-1]
        real_mask = real_masks[-1]
        mask = get_mask()
        masked = img * (1. - mask)

        noise.data.normal_(0, 1)
        coarse_result_t, refine_result_t, attn_loss, attn_refine_loss = g_model(masked, mask, noise, sent_emb,
                                                                                words_embs, text_mask)

        ################### Text-Image Matching Loss ####################
        region_features, cnn_code = image_encoder(refine_result_t)
        w_loss0, w_loss1, _ = words_loss(region_features, words_embs, match_labels, cap_lens, class_ids,
                                         args.batch_size, use_cuda)
        w_loss = (w_loss0 + w_loss1) * 1.0

        s_loss0, s_loss1 = sent_loss(cnn_code, sent_emb, match_labels, class_ids, args.batch_size, use_cuda)
        s_loss = (s_loss0 + s_loss1) * 1.0

        matching_loss = w_loss + s_loss

        ################### Text Inpainting Loss ####################
        pg_loss_t, pd_loss_t = calc_gan_loss(pd_model, refine_result_t, img)

        recon_loss_t = l1(coarse_result_t, img) + l1(refine_result_t, img) + \
                       l1(coarse_result_t * attn_loss, img * attn_loss) + \
                       l1(refine_result_t * attn_refine_loss, img * attn_refine_loss) + \
                       l1(coarse_result_t * real_mask, img * real_mask) + \
                       l1(refine_result_t * real_mask, img * real_mask)
        gan_loss_t = pg_loss_t
        total_loss_t = 0.5 * recon_loss_t + 0.002 * gan_loss_t + 0.001 * matching_loss
        g_optimizer_t.zero_grad()
        total_loss_t.backward(retain_graph=True)
        g_optimizer_t.step()

        pd_optimizer_t.zero_grad()
        pd_loss_t.backward()
        pd_optimizer_t.step()

        num_save_interval = num_batches * i + train_step

        if num_save_interval % args.save_model_interval == 0 or i == args.max_epoch:
            torch.save(g_model.state_dict(), f'{args.save_dir}/G_{num_save_interval}.pth')
            torch.save(pd_model.state_dict(), f'{args.save_dir}/PD_{num_save_interval}.pth')
            print("model saved.")
            if i == args.max_epoch:
                break

        if num_save_interval % args.log_interval == 0:
            print('\n', num_save_interval)
            print('g_loss_t/total_loss_t', total_loss_t.item())
            print('g_loss_t/recon_loss_t', recon_loss_t.item())
            print('g_loss_t/matching_loss', matching_loss.item())
            print('d_loss_t/pd_loss_t', pd_loss_t.item())


        def denorm(x):
            out = (x + 1) / 2  # [-1,1] -> [0,1]
            return out.clamp_(0, 1)


        if num_save_interval % args.vis_interval == 0:
            ims = torch.cat([masked, refine_result_t, img], dim=3)
            ims_train = ims.add(1).div(2).mul(255).clamp(0, 255).byte()
            ims_train = ims_train[0].permute(1, 2, 0).data.cpu().numpy()

            cap_back = Image.new('RGB', (ims_train.shape[1], 30), (255, 255, 255))
            cap = captions[0].data.cpu().numpy()
            sentence = []
            for j in range(len(cap)):
                if cap[j] == 0:
                    break
                word = ixtoword_train[cap[j]].encode('ascii', 'ignore').decode('ascii')
                sentence.append(word)
            sentence = ' '.join(sentence)
            draw = ImageDraw.Draw(cap_back)
            draw.text((0, 10), sentence, (0, 0, 0))
            cap_back = np.array(cap_back)

            ims_text = np.concatenate([ims_train, cap_back], 0)
            ims_out = Image.fromarray(ims_text)
            fullpath = '%s/epoch%d_iteration%d.png' % (args.training_image, i, num_save_interval)
            ims_out.save(fullpath)

            print("train image saved.")

        train_step = train_step + 1
