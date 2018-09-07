import tensorflow.contrib.slim as slim
from .baseop import BaseOp
import tensorflow as tf
import numpy as np
import models.research.inception.inception.slim.variables as vars


class reorg(BaseOp):
    def _forward(self):
        inp = self.inp.out
        shape = inp.get_shape().as_list()
        _, h, w, c = shape
        s = self.lay.stride
        out = list()
        for i in range(int(h / s)):
            row_i = list()
            for j in range(int(w / s)):
                si, sj = s * i, s * j
                boxij = inp[:, si: si + s, sj: sj + s, :]
                flatij = tf.reshape(boxij, [-1, 1, 1, c * s * s])
                row_i += [flatij]
            out += [tf.concat(row_i, 2)]

        self.out = tf.concat(out, 1)

    def forward(self):
        inp = self.inp.out
        s = self.lay.stride
        self.out = tf.extract_image_patches(
            inp, [1, s, s, 1], [1, s, s, 1], [1, 1, 1, 1], 'VALID')

    def speak(self):
        args = [self.lay.stride] * 2
        msg = 'local flatten {}x{}'
        return msg.format(*args)


class local(BaseOp):
    def forward(self):
        pad = [[self.lay.pad, self.lay.pad]] * 2;
        temp = tf.pad(self.inp.out, [[0, 0]] + pad + [[0, 0]])

        k = self.lay.w['kernels']
        ksz = self.lay.ksize
        half = int(ksz / 2)
        out = list()
        for i in range(self.lay.h_out):
            row_i = list()
            for j in range(self.lay.w_out):
                kij = k[i * self.lay.w_out + j]
                i_, j_ = i + 1 - half, j + 1 - half
                tij = temp[:, i_: i_ + ksz, j_: j_ + ksz, :]
                row_i.append(
                    tf.nn.conv2d(tij, kij,
                                 padding='VALID',
                                 strides=[1] * 4))
            out += [tf.concat(row_i, 2)]

        self.out = tf.concat(out, 1)

    def speak(self):
        l = self.lay
        args = [l.ksize] * 2 + [l.pad] + [l.stride]
        args += [l.activation]
        msg = 'loca {}x{}p{}_{}  {}'.format(*args)
        return msg


class convolutional(BaseOp):

    def forward(self):
        pad = [[self.lay.pad, self.lay.pad]] * 2;
        temp1 = tf.pad(self.inp.out, [[0, 0]] + pad + [[0, 0]])
        # temp = tf.nn.conv2d(temp1, self.lay.w['kernel'], padding = 'VALID',
        #    name = self.scope, strides = [1] + [self.lay.stride] * 2 + [1])

        kernel_h = int(self.lay.wshape['kernel'][0])
        # print("kernel size: ", self.lay.w['kernel'].get_shape())
        # print("kernel_h: ", kernel_h)

        kernel_w = int(self.lay.wshape['kernel'][1])
        # print("kernel_w: ", kernel_w)
        # print("kernel size2: ", self.lay.wshape['kernel'])
        stride_h = self.lay.stride
        stride_w = self.lay.stride
        num_filters_in = int(temp1.get_shape()[-1])
        num_filters_out = int(self.lay.wshape['kernel'][-1])

        # print("num_filters_out: ", num_filters_out)

        def block_indx(k, rc, cc):
            rc = int((rc + k - 1) // k) * k
            cc = int((cc + k - 1) // k) * k
            i = np.arange(0, k, 1).reshape([1, k])
            j = np.arange(0, -k, -1).reshape([k, 1])
            indx = (i + j).T
            indx = (indx + k) % k
            m = np.tile(indx, [int(rc / k), int(cc / k)])
            offset = np.arange(0, rc * cc)
            i = (offset / cc) // k
            j = (offset % cc) // k
            offset = (i * cc + j * k).reshape([rc, cc])
            return m + offset

        #if np.min([num_filters_out, num_filters_in]) == 16:
            #partition_size = 16
        #else:
        partition_size = 16

        if partition_size and partition_size <= np.min([num_filters_out, num_filters_in]):
            k = partition_size
            indx = block_indx(k, num_filters_out, num_filters_in)
            # print(indx)
            target_c = num_filters_in * num_filters_out // k
            print("Leo: congratulations!!!!!!!!!!!!!!!!!! you are using BlockCircConv2D", partition_size)
        else:
            print("Leo: sorry, not enough size for partitoning", num_filters_out, num_filters_in, kernel_h, kernel_w)
            target_c = np.max([num_filters_in, num_filters_out])
            if target_c < 32:
                target_c = 32
            a, b = np.ogrid[0:target_c, 0:-target_c:-1]
            indx = a + b

        print('target_c:{}'.format(target_c))
        indx = (indx + target_c) % target_c
        # np.set_printoptions(threshold=np.inf)
        print(indx[:num_filters_out, :num_filters_in].astype(np.int32))
        indx = tf.constant(indx[:num_filters_out, :num_filters_in].astype(np.int32))

        with tf.variable_scope(self.scope, 'conv', [temp1], reuse=tf.AUTO_REUSE):
            weights_shape = [target_c, kernel_h * kernel_w]
            n = kernel_h * kernel_w * num_filters_out
            weights_initializer = tf.truncated_normal_initializer(stddev=np.sqrt(2.0 / int(n)))  # stddev)

            # l2_regularizer = losses.l2_regularizer(0.0005)

            weights = vars.variable('weights',
                                    shape=weights_shape,
                                    initializer=weights_initializer,
                                    regularizer=None,
                                    trainable=True,
                                    restore=True)

            self.lay.w["weights"] = weights

            weights = tf.reshape(tf.transpose(tf.gather(weights, indx), [2, 1, 0]),
                                 [int(kernel_h), int(kernel_w), int(num_filters_in), int(num_filters_out)])

            conv = tf.nn.conv2d(temp1, weights, [1, stride_h, stride_w, 1],
                                name=self.scope, padding='VALID')
            self.temp_out=conv
            if self.lay.batch_norm:
                conv = self.batchnorm(self.lay, conv)
            self.out = tf.nn.bias_add(conv, self.lay.w['biases'])
            self.flag = 0

    def batchnorm(self, layer, inp):
        if not self.var:
            temp = (inp - layer.w['moving_mean'])
            temp /= (np.sqrt(layer.w['moving_variance']) + 1e-5)
            temp *= layer.w['gamma']
            return temp
        else:
            args = dict({
                'center': False, 'scale': True,
                'epsilon': 1e-5, 'scope': self.scope,
                'updates_collections': None,
                'is_training': layer.h['is_training'],
                'param_initializers': layer.w
            })
            return slim.batch_norm(inp, **args)

    def speak(self):
        l = self.lay
        args = [l.ksize] * 2 + [l.pad] + [l.stride]
        args += [l.batch_norm * '+bnorm']
        args += [l.activation]
        msg = 'conv {}x{}p{}_{}  {}  {}'.format(*args)
        return msg


class conv_select(convolutional):
    def speak(self):
        l = self.lay
        args = [l.ksize] * 2 + [l.pad] + [l.stride]
        args += [l.batch_norm * '+bnorm']
        args += [l.activation]
        msg = 'sele {}x{}p{}_{}  {}  {}'.format(*args)
        return msg


class conv_extract(convolutional):
    def speak(self):
        l = self.lay
        args = [l.ksize] * 2 + [l.pad] + [l.stride]
        args += [l.batch_norm * '+bnorm']
        args += [l.activation]
        msg = 'extr {}x{}p{}_{}  {}  {}'.format(*args)
        return msg
