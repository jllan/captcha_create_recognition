from data_utils_torch import *
from model import *
import os
import time
import matplotlib.pyplot as plt
from PIL import Image

BATCH_SIZE = 16
TEST_SIZE = 800
HEIGHT = 70
WIDTH = 160
HIDDEN_SIZE = 160
NUM_RNN_LAYERS = 2
DROPOUT = 0
LR = 0.0003
CLIP = 10.
NUM_EPOCHS = 50
PRINT_EVERY_N_ITER = 100
ATTN_TYPE = 'dot'
ATTN_CLASS = 'type1' #type1 (Luong) | type2
ENC_TYPE = 'CNNRNN' #CNN|CNNRNN
SAVE_DIR = 'CNNRNNdot128_lr0.0003cp10type1'
if not os.path.exists("results"):
    os.mkdir("results")
SAVE_DIR = os.path.join("results", SAVE_DIR)
if not os.path.exists(SAVE_DIR):
    os.mkdir(SAVE_DIR)

USE_CUDA = torch.cuda.is_available()

#.....................模型初始化........................#

def model_init(vocab_size):
    """
    模型初始化
    """
    encoder = Encoder(ENC_TYPE,
                          num_rnn_layers=NUM_RNN_LAYERS,
                          rnn_hidden_size=HIDDEN_SIZE,
                          dropout=DROPOUT)

    if ATTN_CLASS == 'type1':
        decoder = RNNAttnDecoder(ATTN_TYPE,
                                 input_vocab_size=vocab_size,
                                 hidden_size=HIDDEN_SIZE,
                                 output_size=vocab_size,
                                 num_rnn_layers=NUM_RNN_LAYERS,
                                 dropout=DROPOUT)
    elif ATTN_CLASS == 'type2':
        decoder = RNNAttnDecoder2(ATTN_TYPE,
                                  input_vocab_size=vocab_size,
                                  hidden_size=HIDDEN_SIZE,
                                  output_size=vocab_size,
                                  num_rnn_layers=NUM_RNN_LAYERS,
                                  dropout=DROPOUT)
    else:
        raise NotImplementedError

    if USE_CUDA:
        encoder.cuda()
        decoder.cuda()

    return encoder, decoder


def trainn(encoder, decoder):
    encoder_params = list(filter(lambda p: p.requires_grad, encoder.parameters()))
    decoder_params = list(filter(lambda p: p.requires_grad, decoder.parameters()))
    encoder_optimizer = optim.Adam(encoder_params, lr=LR)
    decoder_optimizer = optim.Adam(decoder_params, lr=LR)
    criterion = nn.CrossEntropyLoss()

    epoch_train_loss = []
    epoch_train_accclevel = []
    epoch_train_accuracy = []
    batch_train_loss = []
    batch_train_accclevel = []
    batch_train_accuracy = []
    test_loss = []
    test_accclevel = []
    test_accuracy = []
    best_acc = 0
    for epoch in range(1, NUM_EPOCHS+1):
        start = time.time()
        loss = accuracy = accclevel = 0
        batches_loss = batches_acc = batches_acccl = 0
        for num_iter, (x, y) in enumerate(dl_train):
            # print('x: ', x)
            vx = Variable(x)
            vy = Variable(y)
            if USE_CUDA:
                vx = vx.cuda()
                vy = vy.cuda()
            a_loss, a_accclevel, a_accuracy = train(vx, vy,
                                                    encoder, decoder,
                                                    encoder_optimizer, decoder_optimizer,
                                                    criterion, CLIP, use_cuda=USE_CUDA)
            loss += a_loss
            accuracy += a_accuracy
            accclevel += a_accclevel
            batches_loss += a_loss
            batches_acc += a_accuracy
            batches_acccl += a_accclevel

            if (num_iter+1) % PRINT_EVERY_N_ITER == 0:
                batches_loss /= PRINT_EVERY_N_ITER
                batches_acc /= PRINT_EVERY_N_ITER
                batches_acccl /= PRINT_EVERY_N_ITER
                print("Iteration: {}/{}; Epoch: {}/{}\n".format(num_iter+1, len(dl_train), epoch, NUM_EPOCHS))
                print("recent batches:\n"
                      "loss: {}; accuracy: {}; accclevel: {}\n".format(batches_loss, batches_acc, batches_acccl))
                batch_train_loss.append(batches_loss)
                batch_train_accuracy.append(batches_acc)
                batch_train_accclevel.append(batches_acccl)
                batches_loss = batches_acc = batches_acccl = 0

        epoch_train_loss.append(loss / len(dl_train))
        epoch_train_accuracy.append(accuracy / len(dl_train))
        epoch_train_accclevel.append(accclevel / len(dl_train))
        print("epoch train loss: {}; epoch train accuracy: {}; epoch train accclevel: {}".format(epoch_train_loss[-1],
                                                                                     epoch_train_accuracy[-1],
                                                                                     epoch_train_accclevel[-1]))

        # test
        loss = accuracy = accclevel = 0
        for num_iter, (x, y) in enumerate(dl_test):
            vx = Variable(x)
            vy = Variable(y)
            if USE_CUDA:
                vx = vx.cuda()
                vy = vy.cuda()
            a_loss, a_accclevel, a_accuracy, outputs = evaluate(vx, vy, encoder, decoder, criterion, use_cuda=USE_CUDA)
            loss += a_loss
            accuracy += a_accuracy
            accclevel += a_accclevel
        test_loss.append(loss / len(dl_test))
        test_accclevel.append(accclevel / len(dl_test))
        test_accuracy.append(accuracy / len(dl_test))
        print("test loss: {}; test accuracy: {}; accclevel: {}\n".format(test_loss[-1], test_accuracy[-1],
                                                                         test_accclevel[-1]))

        c = np.random.choice(y.size()[0])
        print('pred: ' + ''.join(vocab['id2token'][i] for i in outputs[c]) + ' ; ' +
              'real: ' + ''.join(vocab['id2token'][i] for i in y[c][1:]))

        if test_accuracy[-1]>best_acc:
            best_acc = test_accuracy[-1]
            torch.save({'state_dict': encoder.state_dict(),
                        'accuracy': test_accuracy[-1]},
                        os.path.join('results', 'encoder_model_best_16.pkl'))
            torch.save({'state_dict': decoder.state_dict(),
                        'accuracy': test_accuracy[-1]},
                        os.path.join('results', 'decoder_model_best_16.pkl'))
        print('epoch time:', time.time()-start)
    print("Training over")

    # save figures
    fig = plt.figure(figsize=(20, 10))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    ax1.plot(batch_train_loss, 'r', label='loss')
    ax1.legend()
    ax2.plot(batch_train_accuracy, label='acc')
    ax2.plot(batch_train_accclevel, label='acccl')
    ax2.legend()
    fig.savefig(os.path.join(SAVE_DIR, "sampled_batch_error.png"))
    print("A figure is saved.")

    fig = plt.figure(figsize=(20, 10))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    ax1.plot(epoch_train_loss, 'r', label='train_loss')
    ax1.plot(test_loss, 'b', label='test_loss')
    ax1.legend()
    ax2.plot(epoch_train_accuracy, 'r', label='train_acc')
    ax2.plot(test_accuracy, 'b', label='test_acc')
    ax2.plot(test_accclevel, 'g', label='test_acccl')
    ax2.legend()
    fig.savefig(os.path.join(SAVE_DIR, "epoch_error.png"))
    print("Another fig is saved.")
    plt.show()
    np.savetxt(os.path.join(SAVE_DIR, "acc.txt"),
               np.vstack([epoch_train_accuracy, test_accuracy, epoch_train_accclevel, test_accclevel]).T)


def load_model(model_path):
    if not os.path.exists(model_path):
        raise RuntimeError('cannot find model path: {}'.format(model_path))
    checkpoint = torch.load(model_path)
    print('load model done.')
    print('accuracy: {:.2f}'.format(checkpoint['accuracy']))
    return checkpoint['state_dict']


def evulatee(encoder, decoder):
    loss = 0
    for num_iter, (x, y) in enumerate(dl_test):
        test_x = Variable(x)
        test_y = Variable(y)
        outputs = predict(test_x, test_y, encoder, decoder, use_cuda=USE_CUDA)
        for img in range(y.size()[0]):
            pred = ''.join(vocab['id2token'][i] for i in outputs[img])
            real = ''.join(vocab['id2token'][i] for i in y[img][1:])
            if pred != real:
                print('pred: ' + pred + ' ; ' + 'real: ' + real)
                loss += 1
    print(loss, loss/TEST_SIZE)


def predictt(encoder, decoder, img_file):
    images = np.zeros((2, 70, 160, 3))
    labels = np.zeros((2, 8))
    img = Image.open(img_file).convert('RGB')
    img = np.array(img).astype(np.float32)/127.5-1
    images[0] = img
    images = torch.Tensor(images).permute(0, 3, 1, 2)
    test_x = Variable(images)
    test_y = Variable(torch.LongTensor(labels))
    outputs = predict(test_x, test_y, encoder, decoder, use_cuda=USE_CUDA)
    pred = ''.join(vocab['id2token'][i] for i in outputs[0]).strip().strip('$')
    return pred


if __name__ == '__main__':
    ## 数据加载
    dl_train, dl_test, vocab = load_dataset(batch_size=BATCH_SIZE, test_size=TEST_SIZE)
    vocab_size = len(vocab['token2id'])

    ## 模型初始化
    encoder, decoder = model_init(vocab_size)

    ## 模型训练
    # trainn(encoder, decoder)

    # ## 模型加载
    encoder.load_state_dict(load_model('./results/encoder_model_best_8.pkl'))
    decoder.load_state_dict(load_model('./results/decoder_model_best_8.pkl'))
    # #
    # # ## 模型评估
    # evulatee(encoder, decoder)
    #
    # ## 预测
    test_dir = '/home/jlan/WorkSpace/cv/dataset/test'
    pics = os.listdir(test_dir)
    errors = 0
    for pic in pics:
        pred = predictt(encoder, decoder, os.path.join(test_dir, pic))
        real = pic.split('.')[0]
        if pred != real:
            errors += 1
            print('real: {}; pred: {}'.format(real, pred))
    print('错误率： ', errors/len(pics))







