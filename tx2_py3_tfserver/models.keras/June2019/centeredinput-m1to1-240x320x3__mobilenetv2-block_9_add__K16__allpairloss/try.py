import tensorflow as tf
from tensorflow.keras import backend as K




# This was copied from the training code's CustomNets.py
class NetVLADLayer( tf.keras.layers.Layer ):

    def __init__( self, num_clusters, **kwargs ):
        self.num_clusters = num_clusters
        super(NetVLADLayer, self).__init__(**kwargs)

    def build( self, input_shape ):
        self.K = self.num_clusters
        self.D = input_shape[-1]

        self.kernel = self.add_weight( name='kernel',
                                    shape=(1,1,self.D,self.K),
                                    initializer='uniform',
                                    trainable=True )

        self.bias = self.add_weight( name='bias',
                                    shape=(1,1,self.K),
                                    initializer='uniform',
                                    trainable=True )

        self.C = self.add_weight( name='cluster_centers',
                                shape=[1,1,1,self.D,self.K],
                                initializer='uniform',
                                trainable=True)

    def call( self, x ):
        # soft-assignment.
        s = K.conv2d( x, self.kernel, padding='same' ) + self.bias
        print( 's.shape=', s.shape)
        a = K.softmax( s )
        self.amap = K.argmax( a, -1 )
        # print 'amap.shape', self.amap.shape

        # Dims used hereafter: batch, H, W, desc_coeff, cluster
        a = K.expand_dims( a, -2 )
        # print 'a.shape=',a.shape

        # Core
        v = K.expand_dims(x, -1) + self.C
        # print 'v.shape', v.shape
        v = a * v
        # print 'v.shape', v.shape
        v = K.sum(v, axis=[1, 2])
        # print 'v.shape', v.shape
        v = K.permute_dimensions(v, pattern=[0, 2, 1])
        # print 'v.shape', v.shape
        #v.shape = None x K x D

        # Normalize v (Intra Normalization)
        v = K.l2_normalize( v, axis=-1 )
        v = K.batch_flatten( v )
        v = K.l2_normalize( v, axis=-1 )

        # return [v, self.amap]
        return v

    def compute_output_shape( self, input_shape ):

        # return [(input_shape[0], self.K*self.D ), (input_shape[0], input_shape[1], input_shape[2]) ]
        return (input_shape[0], self.K*self.D )

    def get_config( self ):
        pass
        # base_config = super(NetVLADLayer, self).get_config()
        # return dict(list(base_config.items()))

        # As suggested by: https://github.com/keras-team/keras/issues/4871#issuecomment-269731817
        config = {'num_clusters': self.num_clusters}
        base_config = super(NetVLADLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


# model = tf.keras.models.load_model( 'modelarch_and_weights.800.h5', custom_objects={'NetVLADLayer': NetVLADLayer} )
input_img = tf.keras.layers.Input( batch_shape=(1,60,80,256) )
out = NetVLADLayer( num_clusters=16 )(input_img)
# out = tf.keras.layers.Conv2D(10, (2,2))( input_img )
model = tf.keras.models.Model( inputs=input_img, outputs=out )
# model.summary()
