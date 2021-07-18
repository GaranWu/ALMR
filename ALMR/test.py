import argparse
import os

import numpy as np
from PIL import Image, ImageDraw
import torch
from torchvision import transforms

from model import InpaintNet, RNN_ENCODER

from datasets import TextDataset
from datasets import prepare_data
from torch.autograd import Variable

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='birds')  # birds flowers celeba15000
parser.add_argument('--root', type=str, default='../data/birds')
parser.add_argument('--CAPTIONS_PER_IMAGE', type=int, default=10)
parser.add_argument('--checkpoint', type=str, default='./model/birds-16/',
                    help='The filename of pickle checkpoint.')
parser.add_argument('--WORDS_NUM', type=int, default=16)
parser.add_argument('--save_dir', type=str, default='./images/birds/three')
parser.add_argument('--save_only', type=str, default='./images/birds/only')
parser.add_argument('--save_real', type=str, default='./images/birds/real')
parser.add_argument('--batch_size', type=int, default=6)
parser.add_argument('--image_size', type=int, default=256)
args = parser.parse_args()

use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')
if use_cuda:
    torch.backends.cudnn.benchmark = True

size = (args.image_size, args.image_size)
train_tf = transforms.Compose([
    transforms.Resize(size)
])

dataset_test = TextDataset(args.root, 'test',
                           base_size=args.image_size,
                           CAPTIONS_PER_IMAGE=args.CAPTIONS_PER_IMAGE,
                           WORDS_NUM=args.WORDS_NUM,
                           transform=train_tf)
assert dataset_test
test_set = torch.utils.data.DataLoader(
    dataset_test, batch_size=args.batch_size, drop_last=True)

print(len(test_set))

ixtoword_test = dataset_test.ixtoword

text_encoder = RNN_ENCODER(dataset_test.n_words, nhidden=args.image_size)
text_encoder_path = '../DAMSMencoders/' + args.dataset + '/text_encoder500.pth'
state_dict = torch.load(text_encoder_path, map_location=lambda storage, loc: storage)
text_encoder.load_state_dict(state_dict)
for p in text_encoder.parameters():
    p.requires_grad = False
print('Load text encoder from:', text_encoder_path)
text_encoder.eval()

if use_cuda:
    text_encoder = text_encoder.cuda()


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
if use_cuda:
    noise = noise.cuda()

for i in range(15, 43):

    if not os.path.exists(args.save_dir + str(i * 10000)):
        os.makedirs('{:s}'.format(args.save_dir + str(i * 10000)))

    if not os.path.exists(args.save_only + str(i * 10000)):
        os.makedirs('{:s}'.format(args.save_only + str(i * 10000)))

    if not os.path.exists(args.save_real + str(i * 10000)):
        os.makedirs('{:s}'.format(args.save_real + str(i * 10000)))

    g_model = InpaintNet().to(device)
    g_checkpoint = torch.load(args.checkpoint + 'G_' + str(i * 10000) + '.pth', map_location=device)
    g_model.load_state_dict(g_checkpoint)
    g_model.eval()

    print('```````````````````````````````')
    print(args.checkpoint + 'G_' + str(i * 10000) + '.pth')
    print('```````````````````````````````')

    for step, data_test in enumerate(test_set, 0):

        if step % 100 == 0:
            print('step: ', step)

        real_mask, imgs, captions, cap_lens, class_ids, keys = prepare_data(data_test)

        hidden = text_encoder.init_hidden(args.batch_size)
        words_embs, sent_emb = text_encoder(captions, cap_lens, hidden)
        words_embs, sent_emb = words_embs.detach(), sent_emb.detach()
        text_mask = (captions == 0)
        num_words = words_embs.size(2)
        if text_mask.size(1) > num_words:
            text_mask = text_mask[:, :num_words]

        img = imgs[-1]
        mask = get_mask()
        masked = img * (1. - mask)

        noise.data.normal_(0, 1)
        coarse_result, refine_result, attnloss, attn = g_model(masked, mask, noise, sent_emb, words_embs, text_mask)


        def denorm(x):
            out = (x + 1) / 2
            return out.clamp_(0, 1)


        for bb in range(args.batch_size):
            # three
            ims = torch.cat([masked, refine_result, img], dim=3)
            ims_test = ims.add(1).div(2).mul(255).clamp(0, 255).byte()
            ims_test = ims_test[bb].permute(1, 2, 0).data.cpu().numpy()
            cap_back = Image.new('RGB', (ims_test.shape[1], 30), (255, 255, 255))

            cap = captions[bb].data.cpu().numpy()
            sentence = []
            for j in range(len(cap)):
                if cap[j] == 0:
                    break
                word = ixtoword_test[cap[j]].encode('ascii', 'ignore').decode('ascii')
                sentence.append(word)
            sentence = ' '.join(sentence)

            draw = ImageDraw.Draw(cap_back)
            draw.text((0, 10), sentence, (0, 0, 0))
            cap_back = np.array(cap_back)

            ims_text = np.concatenate([ims_test, cap_back], 0)

            ims_out = Image.fromarray(ims_text)
            fullpath = '%s/%s.png' % (args.save_dir + str(i * 10000), keys[bb].split('/')[-1])
            ims_out.save(fullpath)

            # only
            ims_test = refine_result.add(1).div(2).mul(255).clamp(0, 255).byte()
            ims_test = ims_test[bb].permute(1, 2, 0).data.cpu().numpy()
            ims_out = Image.fromarray(ims_test)
            fullpath = '%s/%s.png' % (args.save_only + str(i * 10000), keys[bb].split('/')[-1])
            ims_out.save(fullpath)

            # real
            ims_test = img.add(1).div(2).mul(255).clamp(0, 255).byte()
            ims_test = ims_test[bb].permute(1, 2, 0).data.cpu().numpy()
            ims_out = Image.fromarray(ims_test)
            fullpath = '%s/%s.png' % (args.save_real + str(i * 10000), keys[bb].split('/')[-1])
            ims_out.save(fullpath)
