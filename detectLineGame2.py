'''
약어 사전
mbc: marked boundary camera 의 약자
wfnliiocn: write_file_name_list_index_instead_of_correct_name 의 약자
'''


import tensorflow as tf
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.base_env import ActionTuple
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
import numpy as np
import datetime
import time
import math
from collections import deque
import os

import random
import CustomFuncionFor_mlAgent as CF
from PIL import Image
from tqdm import tqdm