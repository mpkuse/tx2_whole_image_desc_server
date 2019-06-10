#!/usr/bin/python3
import rospy

from TerminalColors import bcolors
tcol = bcolors()

import tensorflow as tf
from tensorflow.keras import backend as K
import numpy as np

import os

from tx2_whole_image_desc_server.srv import WholeImageDescriptorCompute, WholeImageDescriptorComputeResponse

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



class HDF5ModelImageDescriptor:
    """
    This class loads the net structure from the .h5 file. This file contains
    the model weights as well as architecture details.
    In the argument `kerasmodel_file`
    you need to specify the full path (keras model file).
    """
    def __init__(self, kerasmodel_file, im_rows=600, im_cols=960, im_chnls=3):
        ## Build net
        # from keras.backend.tensorflow_backend import set_session
        # tf.set_random_seed(42)
        # config = tf.ConfigProto()
        # config.gpu_options.per_process_gpu_memory_fraction = 0.2
        # config.gpu_options.visible_device_list = "0"
        # tf.keras.backend.tensorflow_backend.set_session(tf.Session(config=config))
        #TODO set learning phase(0)
        tf.keras.backend.set_learning_phase(0)

        # Blackbox 4
        # self.im_rows = 512
        # self.im_cols = 640
        # self.im_chnls = 3

        # point grey
        # self.im_rows = 600
        # self.im_cols = 960
        # self.im_chnls = 3

        # EuroC
        # self.im_rows = 480
        # self.im_cols = 752
        # self.im_chnls = 3

        self.im_rows = int(im_rows)
        self.im_cols = int(im_cols)
        self.im_chnls = int(im_chnls)

        LOG_DIR = '/'.join( kerasmodel_file.split('/')[0:-1] )
        print( '+++++++++++++++++++++++++++++++++++++++++++++++++++++++' )
        print(  '++++++++++ (HDF5ModelImageDescriptor) LOG_DIR=', LOG_DIR )
        print( '++++++++++ im_rows=', im_rows, ' im_cols=', im_cols, ' im_chnls=', im_chnls )
        print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++' )
        model_type = LOG_DIR.split('/')[-1]


        assert os.path.isdir( LOG_DIR ), "The LOG_DIR doesnot exist, or there is a permission issue. LOG_DIR="+LOG_DIR


        # Load from HDF5
        assert os.path.isfile( kerasmodel_file ), 'The model weights file doesnot exists or there is a permission issue.'+"kerasmodel_file="+kerasmodel_file
        model_fname = kerasmodel_file
        print( tcol.OKGREEN, 'Load model: ', model_fname, tcol.ENDC )
        model = tf.keras.models.load_model(  model_fname, custom_objects={'NetVLADLayer': NetVLADLayer} )
        old_input_shape = model._layers[0].input_shape
        print( 'OLD MODEL: ', 'input_shape=', str(old_input_shape) )
        model.summary()
        model_visual_fname = None
        if model_visual_fname is not None:
            print( 'Writing Model Visual to: ', model_visual_fname )
            keras.utils.plot_model( model, to_file=model_visual_fname, show_shapes=True )


        self.model = model
        self.model_type = model_type


        # Doing this is a hack to force keras to allocate GPU memory. Don't comment this,
        tmp_zer = np.zeros( (1,self.im_rows,self.im_cols,self.im_chnls), dtype='float32' )
        tmp_zer_out = self.model.predict( tmp_zer )
        print( 'model input.shape=', tmp_zer.shape, '\toutput.shape=', tmp_zer_out.shape )
        print( 'model_type=', self.model_type )

        print( '-----' )
        print( '\tinput_image.shape=', tmp_zer.shape )
        print( '\toutput.shape=', tmp_zer_out.shape )
        print( '\tminmax(tmp_zer_out)=', np.min( tmp_zer_out ), np.max( tmp_zer_out ) )
        print( '\tnorm=', np.linalg.norm( tmp_zer_out ) )
        print( '\tdtype=', tmp_zer_out.dtype )
        print( '-----' )


    def handle_req( self, req ):
        print ('handle_req')
        result = WholeImageDescriptorComputeResponse()
        result.desc = [ 2122., 46.2 ]
        result.model_type = self.model_type
        return result

    #
    # def handle_req( self, req ):
    #     """ The received image from CV bridge has to be [0,255]. In function makes it to
    #     intensity range [-0.5 to 0.5]
    #     """
    #     ## Get Image out of req
    #     cv_image = CvBridge().imgmsg_to_cv2( req.ima )
    #     print '[HDF5ModelImageDescriptor Handle Request] cv_image.shape', cv_image.shape, '\ta=', req.a, '\tt=', req.ima.header.stamp
    #     if len(cv_image.shape)==2:
    #         # print 'Input dimensions are NxM but I am expecting it to be NxMxC, so np.expand_dims'
    #         cv_image = np.expand_dims( cv_image, -1 )
    #     elif len( cv_image.shape )==3:
    #         pass
    #     else:
    #         assert( False )
    #
    #
    #
    #
    #     assert (cv_image.shape[0] == self.im_rows and
    #             cv_image.shape[1] == self.im_cols and
    #             cv_image.shape[2] == self.im_chnls),\
    #             "\n[whole_image_descriptor_compute_server] Input shape of the image \
    #             does not match with the allocated GPU memory. Expecting an input image of \
    #             size %dx%dx%d, but received : %s" %(self.im_rows, self.im_cols, self.im_chnls, str(cv_image.shape) )
    #
    #     # cv2.imshow( 'whole_image_descriptor_compute_server:imshow', cv_image.astype('uint8') )
    #     # cv2.waitKey(10)
    #     # cv2.imwrite( '/app/tmp/%s.jpg' %( str(req.ima.header.stamp) ), cv_image )
    #
    #     ## Compute Descriptor
    #     start_time = time.time()
    #     # i__image = np.expand_dims( cv_image.astype('float32'), 0 )
    #     # i__image = (np.expand_dims( cv_image.astype('float32'), 0 ) - 128.)/255. [-0.5,0.5]
    #     i__image = (np.expand_dims( cv_image.astype('float32'), 0 ) - 128.)*2.0/255. #[-1,1]
    #
    #     u = self.model.predict( i__image )
    #
    #     print tcol.HEADER, 'Descriptor Computed in %4.4fms' %( 1000. *(time.time() - start_time) ), tcol.ENDC
    #     print '\tinput_image.shape=', cv_image.shape,
    #     print '\tinput_image dtype=', cv_image.dtype
    #     print tcol.OKBLUE, '\tinput image (to neuralnet) minmax=', np.min( i__image ), np.max( i__image ), tcol.ENDC
    #     print '\tdesc.shape=', u.shape,
    #     print '\tdesc minmax=', np.min( u ), np.max( u ),
    #     print '\tnorm=', np.linalg.norm(u[0])
    #     print '\tmodel_type=', self.model_type
    #
    #
    #
    #     ## Populate output message
    #     result = WholeImageDescriptorComputeResponse()
    #     # result.desc = [ cv_image.shape[0], cv_image.shape[1] ]
    #     result.desc = u[0,:]
    #     result.model_type = self.model_type
    #     return result

if __name__ == '__main__':
    rospy.init_node( 'whole_image_descriptor_compute_server' )

    ##
    ## Load the config file and read image row, col
    ##
    fs_image_width = -1
    fs_image_height = -1
    fs_image_chnls = 1

    if True: # read from param `config_file`
        if not rospy.has_param( '~config_file'):
            print( 'FATAL...cannot find param ~config_file. This is needed to determine size of the input image to allocate GPU memory. If you do not specify the config_file, you need to atleast specify the nrows, ncols, nchnls' )
            rospy.logerr( '[whole_image_descriptor_compute_server]FATAL...cannot find param ~config_file. This is needed to determine size of the input image to allocate GPU memory. If you do not specify the config_file, you need to atleast specify the nrows, ncols, nchnls' )

            if ( rospy.has_param( '~nrows') and rospy.has_param( '~ncols') ):
                print( tcol.OKGREEN, 'However, you seem to have set the parameters nrows and ncols, so will read those.', tcol.ENDC )
            else:
                quit()
            # quit only if you cannot see nrows and ncols

        else:
            config_file = rospy.get_param('~config_file')
            print( '++++++++\n++++++++ config_file: ', config_file )
            if not os.path.isfile(config_file):
                print( 'FATAL...cannot find config_file: ', config_file )
                rospy.logerr( '[whole_image_descriptor_compute_server]FATAL...cannot find config_file: '+ config_file )
                quit()


            print( '++++++++ READ opencv-yaml file: ', config_file )
            fs = cv2.FileStorage(config_file, cv2.FILE_STORAGE_READ)
            fs_image_width = int(  fs.getNode( "image_width" ).real() )
            fs_image_height = int( fs.getNode( "image_height" ).real() )
            print( '++++++++ opencv-yaml:: image_width=', fs_image_width, '   image_height=', fs_image_height )
            print( '++++++++' )


    ##
    ## Load nrows and ncols directly as config
    ##
    if True:  # read from param `nrows` and `ncols`
        if fs_image_width < 0 :
            if ( not rospy.has_param( '~nrows') or not rospy.has_param( '~ncols') or not rospy.has_param( '~nchnls') ):
                print( 'FATAL...cannot find param either of ~nrows, ~ncols, ~nchnls. This is needed to determine size of the input image to allocate GPU memory' )
                rospy.logerr( '[whole_image_descriptor_compute_server] FATAL...cannot find param either of ~nrows, ~ncols, nchnls. This is needed to determine size of the input image to allocate GPU memory' )
                quit()
            else:
                fs_image_height = rospy.get_param('~nrows')
                fs_image_width = rospy.get_param('~ncols')
                fs_image_chnls = rospy.get_param('~nchnls')

                print ( '~~~~~~~~~~~~~~~~' )
                print ( '~nrows = ', fs_image_height, '\t~ncols = ', fs_image_width, '\t~nchnls = ', fs_image_chnls )
                print ( '~~~~~~~~~~~~~~~~' )


    ##
    ## Load Channels
    ##
    if True:
        if not rospy.has_param( '~nchnls' ):
            rospy.logerr( "[whole_image_descriptor_compute_server] FATAL....cannot file cmd param nchnls.")
            quit()
        else:
            fs_image_chnls = rospy.get_param('~nchnls')


    print( '~~~@@@@ OK...' )
    ##
    ## Start Server
    ##
    # kerasmodel_file = '/models.keras/Apr2019/gray_conv6_K16Ghost1__centeredinput/core_model.%d.keras' %(500)
    if rospy.has_param( '~kerasmodel_file'):
        kerasmodel_file = rospy.get_param('~kerasmodel_file')
    else:
        print( tcol.ERROR, 'FATAL...missing specification of model file. You need to specify ~kerasmodel_file', tcol.ENDC )
        quit()

    gpu_netvlad = HDF5ModelImageDescriptor( kerasmodel_file=kerasmodel_file, im_rows=fs_image_height, im_cols=fs_image_width, im_chnls=fs_image_chnls )

    s = rospy.Service( 'whole_image_descriptor_compute', WholeImageDescriptorCompute, gpu_netvlad.handle_req  )
    print (tcol.OKGREEN )
    print( '++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    print( '+++  whole_image_descriptor_compute_server is running +++' )
    print( '++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    print( tcol.ENDC )
    rospy.spin()
