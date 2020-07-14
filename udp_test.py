import os
import six.moves.urllib as urllib
import sys

import socket
from struct import *


udp_socket = socket.socket(
        socket.AF_INET, #internet
        socket.SOCK_DGRAM # udp
        )
_req = pack('<BBBB',0x7f,10,0,0)

udp_socket.sendto(_req,('localhost',20105))
udp_socket.settimeout(3.0)
try :
    _data, _rinfo = udp_socket.recvfrom(1024)
    _packet = unpack("<BBBB",_data)
    print( list(_data) )
except Exception as ex:
    print(ex)

