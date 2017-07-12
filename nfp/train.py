import argparse

import chainer
import chainer.functions as F
import chainer.iterators as I
import chainer.links as L
import chainer.optimizers as O
from chainer import training
import chainer.training.extensions as E

import data
import data2
import data_ggnn
import ggnn
import model
import model2
import model3

parser = argparse.ArgumentParser(description='Chainer example: MNIST')
parser.add_argument('--mode', '-m', type=int, default=2,
                    help='Preprocessing mode type 1 is compress adj, 2 uses '
                         'fixed size array')
parser.add_argument('--batchsize', '-b', type=int, default=128,
                    help='Number of images in each mini-batch')
parser.add_argument('--epoch', '-e', type=int, default=20,
                    help='Number of sweeps over the dataset to train')
parser.add_argument('--gpu', '-g', type=int, default=-1,
                    help='GPU ID (negative value indicates CPU)')
parser.add_argument('--radius', '-r', type=int, default=3,
                    help='Radius parameter of NFP')
parser.add_argument('--hidden-dim', '-H', type=int, default=128,
                    help='Number of hidden units of NFP')
parser.add_argument('--out-dim', '-O', type=int, default=128,
                    help='Number of output units of NFP')
args = parser.parse_args()


if args.mode == 1:
    train, val, max_degree, atom2id, C = data.load()
    nfp = model.NFP(args.hidden_dim, args.out_dim, max_degree, len(atom2id),
                    args.radius, concat_hidden=True)
    converter = data.concat_example
elif args.mode == 2:
    train, val, max_degree, atom2id, C = data2.load()
    nfp = model2.NFP(args.hidden_dim, args.out_dim, max_degree, len(atom2id),
                    args.radius, concat_hidden=True)
    converter = chainer.dataset.concat_examples
elif args.mode == 3:
    train, val, max_degree, atom2id, C = data2.load()
    nfp = model3.NFP(args.hidden_dim, args.out_dim, max_degree, len(atom2id),
                     args.radius, concat_hidden=True)
    converter = chainer.dataset.concat_examples
else:
    train, val, max_degree, atom2id, C = data_ggnn.load()
    nfp = ggnn.GGNN(args.hidden_dim, args.out_dim, len(atom2id),
                    args.radius, concat_hidden=True)
    converter = chainer.dataset.concat_examples

print('data', max_degree, len(atom2id), C)
predictor = model.Predictor(nfp, C)
model = L.Classifier(predictor,
                     lossfun=F.sigmoid_cross_entropy,
                     accfun=F.binary_accuracy)
if args.gpu >= 0:
    chainer.cuda.get_device_from_id(args.gpu).use()
    model.to_gpu()

optimizer = O.Adam()
optimizer.setup(model)

train_iter = I.SerialIterator(train, args.batchsize)
val_iter = I.SerialIterator(val, args.batchsize, repeat=False, shuffle=False)

updater = training.StandardUpdater(train_iter, optimizer,
                                   converter, args.gpu)
trainer = training.Trainer(updater, (args.epoch, 'epoch'))

trainer.extend(E.Evaluator(val_iter, model, converter, args.gpu),
               trigger=(5, 'iteration'))
trainer.extend(E.LogReport(trigger=(5, 'iteration')))

if E.PlotReport.available():
    trainer.extend(
        E.PlotReport(['main/loss', 'validation/main/loss'], 'epoch',
                     file_name='loss.png'))
    trainer.extend(
        E.PlotReport(['main/accuracy', 'validation/main/accuracy'], 'epoch',
                     file_name='accuracy.png'))

trainer.extend(
    E.PrintReport(['epoch', 'iteration', 'main/loss', 'validation/main/loss',
                   'main/accuracy', 'validation/main/accuracy',
                   'elapsed_time']), trigger=(5, 'iteration'))
trainer.extend(E.ProgressBar(update_interval=1))

trainer.run()
