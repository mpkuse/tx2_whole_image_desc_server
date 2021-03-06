#!/usr/bin/env python

# Sample Client to call the `whole_image_descriptor_compute_server` on TX2

from tx2_whole_image_desc_server.srv import *
import rospy
import numpy as np
import cv2

from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image

rospy.wait_for_service( 'whole_image_descriptor_compute' )
try:
    res = rospy.ServiceProxy( 'whole_image_descriptor_compute', WholeImageDescriptorCompute )

    X = np.zeros( (100, 100), dtype=np.uint8 )
    # X = cv2.resize( cv2.imread( '/app/lena.jpg' ), (752,480) )
    i = CvBridge().cv2_to_imgmsg( X )
    u = res( i, 23  )
    print 'received: ', u
except rospy.ServiceException, e:
    print 'failed', e
