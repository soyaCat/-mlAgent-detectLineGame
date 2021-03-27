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

game = "DetectLineGame.exe"
env_path = "./build/" + game
save_picture_path = "./made_data/"

channel = EngineConfigurationChannel()
channel.set_configuration_parameters(time_scale=1.0, target_frame_rate=60, capture_frame_rate=60)
env = UnityEnvironment(file_name=env_path, side_channels=[channel])
env.reset()
behavior_names = list(env.behavior_specs)
ConversionDataType = CF.ConversionDataType()
AgentsHelper = CF.AgentsHelper(env, string_log=None, ConversionDataType=ConversionDataType)

connection_test_count = 50
train_count = 1
test_count =0

write_file_name_list_index_instead_of_correct_name = False
list_index_for_main = 0
list_index_for_mbc0 = 1
list_index_for_mbc1 = 2
list_index_for_mbc2 = 3
list_index_for_mbc3 = 4
list_index_for_bc0 = 5
list_index_for_bc1 = 6
list_index_for_bc2 = 7
list_index_for_bc3 = 8
generate_main = True
generate_mbc0 = True
generate_mbc1 = True
generate_mbc2 = True
generate_mbc3 = True
generate_boundary_cam = True

state_size = [128,128,3]
action_size = 6

epsilon_init = 1.0
epsilon_min = 0.1

class DQN_Network():
    def __init__(self):
        pass

class DQN():
    def __init__(self):
        self.epsilon = epsilon_init
        self.action_size = action_size

    def get_action(self, state):
        if self.epsilon > np.random.rand():
            return np.random.randint(0, self.action_size)

class Get_Reward():
    def __init__(self):
        self.reward = 0
        self.target_bc_area = 0
        self.pre_target_hide_area = 0
        self.target_bc_index = -1

    def init_reward(self, mbc_dic, bc_dic):
        bc_sum_list = [np.sum(bc_dic[0]), np.sum(bc_dic[1]), np.sum(bc_dic[2]), np.sum(bc_dic[3])]
        self.target_bc_index = bc_sum_list.index(max(bc_sum_list))
        self.target_bc_area = bc_sum_list[self.target_bc_index]
        self.pre_target_hide_area = np.sum(bc_dic[self.target_bc_index])-np.sum(mbc_dic[self.target_bc_index])
        self.reward = 0


    def update_reward(self, mbc_dic, bc_dic):
        # target_hide_area가 target_bc이면 라인을 맞춘 것, 0 넓이라면 라인을 전혀 못 맞춘것
        # target_hide_area가 커지는 방향이면 올바르게 찾는것, 작아지는 방향이면 올바르지 못하게 찾는 것
        target_hide_area = self.target_bc_area-np.sum(mbc_dic[self.target_bc_index])
        game_fin = False
        print("\n")
        print("시작")
        print("index :", self.target_bc_index)
        print("tba   :", self.target_bc_area)
        print("tmba  :", np.sum(mbc_dic[self.target_bc_index]))
        print("tha   :", target_hide_area)
        print("ptha  :", self.pre_target_hide_area)

        if target_hide_area == 0:
            self.reward = -1
        elif target_hide_area == self.target_bc_area:
            self.reward = 1
            game_fin = True

        else:
            area_diff = int(target_hide_area) - int(self.pre_target_hide_area)
            print("arad   :", area_diff)
            self.reward = area_diff/self.target_bc_area
            self.pre_target_hide_area = target_hide_area

        return self.reward, game_fin





def save_numpy_file(append_name, list_index, wfnliiocn, episodeCount):
    im = Image.fromarray(vis_observation_list[list_index].astype('uint8'), 'RGB')
    if wfnliiocn == False:
        im.save(save_picture_path + str(episodeCount) + append_name + '.jpg')
    else:
        im.save(save_picture_path + str(list_index) + '.jpg')

def save_gray_numpy_file(append_name, list_index, wfnliiocn, episodeCount):
    target = ConversionDataType.Reduction_Dimention_for_grayIMG(vis_observation_list[list_index])
    im = Image.fromarray(target.astype('uint8'), 'L')
    if wfnliiocn == False:
        im.save(save_picture_path + str(episodeCount) + append_name + '.jpg')
    else:
        im.save(save_picture_path + str(list_index) + '.jpg')



if __name__ == '__main__':
    totalEpisodeCount = train_count + test_count
    Get_Reward = Get_Reward()
    DQN = DQN()


    for episodeCount in tqdm(range(connection_test_count)):
        behavior_name = behavior_names[0]
        decision_steps, terminal_steps = env.get_steps(behavior_name)
        vec_observation, vis_observation_list, done = AgentsHelper.getObservation(behavior_name)
        wfnliiocn = write_file_name_list_index_instead_of_correct_name
        if generate_main is True:
            save_numpy_file('_main', list_index_for_main, wfnliiocn, episodeCount)
        if generate_mbc0 is True:
            save_gray_numpy_file('_mbc0', list_index_for_mbc0, wfnliiocn, episodeCount)
        if generate_mbc1 is True:
            save_gray_numpy_file('_mbc1', list_index_for_mbc1, wfnliiocn, episodeCount)
        if generate_mbc2 is True:
            save_gray_numpy_file('_mbc2', list_index_for_mbc2, wfnliiocn, episodeCount)
        if generate_mbc3 is True:
            save_gray_numpy_file('_mbc3', list_index_for_mbc3, wfnliiocn, episodeCount)
        if generate_boundary_cam is True:
            save_gray_numpy_file('_bc0', list_index_for_bc0, wfnliiocn, episodeCount)
            save_gray_numpy_file('_bc1', list_index_for_bc1, wfnliiocn, episodeCount)
            save_gray_numpy_file('_bc2', list_index_for_bc2, wfnliiocn, episodeCount)
            save_gray_numpy_file('_bc3', list_index_for_bc3, wfnliiocn, episodeCount)
        action = [1, 0, 0, 0, 0, 0, 0]
        actionTuple = ConversionDataType.ConvertList2DiscreteAction(action, behavior_name)
        env.set_actions(behavior_name, actionTuple)
        env.step()


    for episodeCount in tqdm(range(train_count)):
        behavior_name = behavior_names[0]
        decision_steps, terminal_steps = env.get_steps(behavior_name)
        vec_observation, vis_observation_list, done = AgentsHelper.getObservation(behavior_name)
        bc_dic = {
            0: vis_observation_list[list_index_for_bc0],
            1: vis_observation_list[list_index_for_bc1],
            2: vis_observation_list[list_index_for_bc2],
            3: vis_observation_list[list_index_for_bc3]}
        mbc_dic = {
            0: vis_observation_list[list_index_for_mbc0],
            1: vis_observation_list[list_index_for_mbc1],
            2: vis_observation_list[list_index_for_mbc2],
            3: vis_observation_list[list_index_for_mbc3]}
        Get_Reward.init_reward(mbc_dic, bc_dic)

        while 1:
            behavior_name = behavior_names[0]
            decision_steps, terminal_steps = env.get_steps(behavior_name)
            vec_observation, vis_observation_list, done = AgentsHelper.getObservation(behavior_name)

            state = vis_observation_list[list_index_for_main]
            calculated_action_index = DQN.get_action(state)
            action = [2, 0, 0, 0, 0, 0, 0]
            action[calculated_action_index+1] = 1
            actionTuple = ConversionDataType.ConvertList2DiscreteAction(action, behavior_name)
            env.set_actions(behavior_name, actionTuple)
            env.step()

            bc_dic = {
                0: vis_observation_list[list_index_for_bc0],
                1: vis_observation_list[list_index_for_bc1],
                2: vis_observation_list[list_index_for_bc2],
                3: vis_observation_list[list_index_for_bc3]}
            mbc_dic = {
                0: vis_observation_list[list_index_for_mbc0],
                1: vis_observation_list[list_index_for_mbc1],
                2: vis_observation_list[list_index_for_mbc2],
                3: vis_observation_list[list_index_for_mbc3]}

            reward, game_fin = Get_Reward.update_reward(mbc_dic, bc_dic)
            if game_fin is True:
                break

            print(reward)
            time.sleep(0.5)




        action = [1, 0, 0, 0, 0, 0, 0]
        actionTuple = ConversionDataType.ConvertList2DiscreteAction(action, behavior_name)
        env.set_actions(behavior_name, actionTuple)
        env.step()






    env.close()

'''
game = "DetectLineGame.exe"
env_path = "./build/" + game
save_picture_path = "./made_data/"
channel = EngineConfigurationChannel()
channel.set_configuration_parameters(time_scale=1.0, target_frame_rate=60, capture_frame_rate=60)
env = UnityEnvironment(file_name=env_path, side_channels=[channel])
env.reset()
behavior_names = list(env.behavior_specs)

ConversionDataType = CF.ConversionDataType()
normal_get_dataCount = 2
super_Random_EpisodeCount = 0
super_close_EpisodeCount = 0

AgentsHelper = CF.AgentsHelper(env, string_log=None, ConversionDataType=ConversionDataType)
write_file_name_list_index_instead_of_correct_name = False
list_index_for_ALL = 0
list_index_for_ball = 1
list_index_for_flag1 = 2
list_index_for_flag2 = 4
list_index_for_flag1_range = 3
list_index_for_flag2_range = 5
list_index_for_stage = 10
list_index_for_goal1_detection = 6
list_index_for_goal2_detection = 8
list_index_for_goal1_range = 7
list_index_for_goal2_range = 9

generate_ball_map = False
generate_stage = True
generate_flag = True
generate_ball = True
generate_goal_dectecion = True
generate_goal_range = True
generate_yolo_txt_file = True

write_txt_file_ball_pos = True
write_txt_file_goal_pos = True
write_txt_file_flag_pos = True


def get_result_about_Is_same_1D_npArr(myArr1, myArr2):
    result = True
    for index in range(np.shape(myArr1)[0]):
        if myArr1[index] != myArr2[index]:
            result = False
    return result


def get_rectangle_point_for_yolo(ball_npArr):
    im_width_size = np.shape(ball_npArr)[0]
    im_hight_size = np.shape(ball_npArr)[1]
    ball_map_npArr = np.zeros((im_width_size, im_hight_size))
    candidate_width_list = []
    candidate_hight_list = []

    for width in range(im_width_size):
        Is_width_list_append = False
        for hight in range(im_hight_size):
            result = get_result_about_Is_same_1D_npArr(ball_npArr[width][hight], np.array([0, 0, 0]))
            if result == False:
                candidate_hight_list.append(hight)
                if Is_width_list_append == False:
                    candidate_width_list.append(width)
                    Is_width_list_append = True
                ball_map_npArr[width][hight] = 255
    left = 0  # 큰수여야함
    bottom = 0  # 큰수여야함
    right = im_width_size  # 작은수여야함
    top = im_hight_size  # 작은수여야 함

    for candidate_width in candidate_width_list:
        if right > candidate_width:
            right = candidate_width
        if left < candidate_width:
            left = candidate_width

    for candidate_hight in candidate_hight_list:
        if top > candidate_hight:
            top = candidate_hight
        if bottom < candidate_hight:
            bottom = candidate_hight
    print(left, bottom, right, top)

    return left, bottom, right, top, ball_map_npArr


def better_get_rectangle_point_for_yolo_no_map(ball_npArr):
    candidates_arr = np.where(ball_npArr != 0)
    right = np.min(candidates_arr[0])
    left = np.max(candidates_arr[0])
    top = np.min(candidates_arr[1])
    bottom = np.max(candidates_arr[1])

    return left, bottom, right, top


def mapping_point_to_float_shape(npArr, left, bottom, right, top):
    im_width_size = np.shape(npArr)[0]
    im_hight_size = np.shape(npArr)[1]
    left = left / im_width_size
    bottom = bottom / im_hight_size
    right = right / im_width_size
    top = top / im_hight_size

    return left, bottom, right, top  # round(n,2)


def get_txt_line_for_yolo_txt(episodeCount, left, bottom, right, top, ID):
    height_distance = bottom - top
    width_distance = left - right
    height_center = bottom - (height_distance / 2)
    width_center = left - (width_distance / 2)
    data = str(ID) + " " + str(round(height_center, 6)) + " " + str(round(width_center, 6)) + " " + str(
        round(height_distance, 6)) + " " + str(round(width_distance, 6)) + "\n"

    return data


# (episodeCount, ball_pos, goal1_pos, goal2_pos, flag1_pos, flag2_pos, ball_result, goal1_result, goal2_result, flag1_result, flag2_result)
def write_txt_file_like_yolo_mark(episodeCount, ball_list, goal1_list, goal2_list, flag1_list, flag2_list, no_ball,
                                  no_goal1, no_goal2, no_flag1, no_flag2):
    f = open("./made_data/" + str(episodeCount) + "_ALL.txt", 'w')
    if no_ball == False:
        left = ball_list[0]
        bottom = ball_list[1]
        right = ball_list[2]
        top = ball_list[3]
        data = get_txt_line_for_yolo_txt(episodeCount, left, bottom, right, top, 0)
        f.write(data)
    if no_goal1 == False:
        left = goal1_list[0]
        bottom = goal1_list[1]
        right = goal1_list[2]
        top = goal1_list[3]
        data = get_txt_line_for_yolo_txt(episodeCount, left, bottom, right, top, 1)
        f.write(data)
    if no_goal2 == False:
        left = goal2_list[0]
        bottom = goal2_list[1]
        right = goal2_list[2]
        top = goal2_list[3]
        data = get_txt_line_for_yolo_txt(episodeCount, left, bottom, right, top, 1)
        f.write(data)
    if no_flag1 == False:
        left = flag1_list[0]
        bottom = flag1_list[1]
        right = flag1_list[2]
        top = flag1_list[3]
        data = get_txt_line_for_yolo_txt(episodeCount, left, bottom, right, top, 2)
        f.write(data)
    if no_flag2 == False:
        left = flag2_list[0]
        bottom = flag2_list[1]
        right = flag2_list[2]
        top = flag2_list[3]
        data = get_txt_line_for_yolo_txt(episodeCount, left, bottom, right, top, 2)
        f.write(data)

    f.close()


def write_train_txt_file_for_yolo(totalEpisodeCount):
    f = open("./made_data/" + "train.txt", 'w')
    for i in range(totalEpisodeCount):
        data = "data/kudos_obj/" + str(i) + "_ALL.jpg" + "\n"
        f.write(data)

    f.close()


def save_numpy_file(append_name, list_index, wfnliiocn):
    im = Image.fromarray(vis_observation_list[list_index].astype('uint8'), 'RGB')
    if wfnliiocn == False:
        im.save(save_picture_path + str(episodeCount) + append_name + '.jpg')
    else:
        im.save(save_picture_path + str(list_index) + '.jpg')


if __name__ == "__main__":
    totalEpisodeCount = super_close_EpisodeCount + super_Random_EpisodeCount + normal_get_dataCount
    start_super_Random_EpisodeCount = normal_get_dataCount
    start_super_close_EpisodeCount = normal_get_dataCount + super_Random_EpisodeCount
    print(totalEpisodeCount)
    print(start_super_Random_EpisodeCount)
    print(start_super_close_EpisodeCount)
    write_train_txt_file_for_yolo(totalEpisodeCount)
    for episodeCount in tqdm(range(totalEpisodeCount)):
        behavior_name = behavior_names[0]
        decision_steps, terminal_steps = env.get_steps(behavior_name)
        vec_observation, vis_observation_list, done = AgentsHelper.getObservation(behavior_name)
        vis_observation = vis_observation_list[0]

        im = Image.fromarray(vis_observation_list[list_index_for_ALL].astype('uint8'), 'RGB')
        im.save(save_picture_path + str(episodeCount) + '_ALL.jpg')
        wfnliiocn = write_file_name_list_index_instead_of_correct_name
        if generate_stage == True:
            save_numpy_file('_stage', list_index_for_stage, wfnliiocn)
        if generate_ball == True:
            save_numpy_file('_ball', list_index_for_ball, wfnliiocn)
        if generate_flag == True:
            save_numpy_file('_flag1', list_index_for_flag1, wfnliiocn)
            save_numpy_file('_flag2', list_index_for_flag2, wfnliiocn)
            save_numpy_file('_flag1_range', list_index_for_flag1_range, wfnliiocn)
            save_numpy_file('_flag2_range', list_index_for_flag2_range, wfnliiocn)
        if generate_goal_dectecion == True:
            save_numpy_file('_goal1_detection', list_index_for_goal1_detection, wfnliiocn)
            save_numpy_file('_goal2_detection', list_index_for_goal2_detection, wfnliiocn)
        if generate_goal_range == True:
            save_numpy_file('_goal1_range', list_index_for_goal1_range, wfnliiocn)
            save_numpy_file('_goal2_range', list_index_for_goal2_range, wfnliiocn)

        ball_npArr = vis_observation_list[list_index_for_ball]
        goal1_detection_npArr = vis_observation_list[list_index_for_goal1_detection]
        goal2_detection_npArr = vis_observation_list[list_index_for_goal2_detection]
        goal1_npArr = vis_observation_list[list_index_for_goal1_range]
        goal2_npArr = vis_observation_list[list_index_for_goal2_range]
        flag1_detection_npArr = vis_observation_list[list_index_for_flag1]
        flag1_npArr = vis_observation_list[list_index_for_flag1_range]
        flag2_detection_npArr = vis_observation_list[list_index_for_flag2]
        flag2_npArr = vis_observation_list[list_index_for_flag2_range]
        left = 0.0
        bottom = 0.0
        right = 1.0
        top = 1.0

        g1_left = 0.0
        g1_bottom = 0.0
        g1_right = 1.0
        g1_top = 1.0

        g2_left = 0.0
        g2_bottom = 0.0
        g2_right = 1.0
        g2_top = 1.0

        f1_left = 0.0
        f1_bottom = 0.0
        f1_right = 1.0
        f1_top = 1.0

        f2_left = 0.0
        f2_bottom = 0.0
        f2_right = 1.0
        f2_top = 1.0

        if write_txt_file_ball_pos == True:
            if np.sum(ball_npArr) != 0:
                ball_map_npArr = np.ones_like(ball_npArr)
                left, bottom, right, top = better_get_rectangle_point_for_yolo_no_map(ball_npArr)
                left, bottom, right, top = mapping_point_to_float_shape(ball_npArr, left, bottom, right, top)
            else:
                ball_map_npArr = np.zeros_like(ball_npArr)
            if generate_ball_map == True:
                try:
                    im_ball_map = Image.fromarray(ball_map_npArr.astype('uint8'), 'L')
                    im_ball_map.save(save_picture_path + str(episodeCount) + '_ball_map.jpg')
                except:
                    im_ball_map = Image.fromarray(ball_map_npArr.astype('uint8'), 'RGB')
                    im_ball_map.save(save_picture_path + str(episodeCount) + '_ball_map.jpg')

        if write_txt_file_goal_pos == True:
            if np.sum(goal1_detection_npArr) != 0:
                g1_left, g1_bottom, g1_right, g1_top = better_get_rectangle_point_for_yolo_no_map(goal1_npArr)
                g1_left, g1_bottom, g1_right, g1_top = mapping_point_to_float_shape(goal1_npArr, g1_left, g1_bottom,
                                                                                    g1_right, g1_top)
            if np.sum(goal2_detection_npArr) != 0:
                g2_left, g2_bottom, g2_right, g2_top = better_get_rectangle_point_for_yolo_no_map(goal2_npArr)
                g2_left, g2_bottom, g2_right, g2_top = mapping_point_to_float_shape(goal2_npArr, g2_left, g2_bottom,
                                                                                    g2_right, g2_top)

        if write_txt_file_flag_pos == True:
            if np.sum(flag1_detection_npArr) != 0:
                f1_left, f1_bottom, f1_right, f1_top = better_get_rectangle_point_for_yolo_no_map(flag1_npArr)
                f1_left, f1_bottom, f1_right, f1_top = mapping_point_to_float_shape(flag1_npArr, f1_left, f1_bottom,
                                                                                    f1_right, f1_top)
            if np.sum(flag2_detection_npArr) != 0:
                f2_left, f2_bottom, f2_right, f2_top = better_get_rectangle_point_for_yolo_no_map(flag2_npArr)
                f2_left, f2_bottom, f2_right, f2_top = mapping_point_to_float_shape(flag2_npArr, f2_left, f2_bottom,
                                                                                    f2_right, f2_top)

        ball_result = get_result_about_Is_same_1D_npArr(np.array([left, bottom, right, top]),
                                                        np.array([0.0, 0.0, 1.0, 1.0]))
        goal1_result = get_result_about_Is_same_1D_npArr(np.array([g1_left, g1_bottom, g1_right, g1_top]),
                                                         np.array([0.0, 0.0, 1.0, 1.0]))
        goal2_result = get_result_about_Is_same_1D_npArr(np.array([g2_left, g2_bottom, g2_right, g2_top]),
                                                         np.array([0.0, 0.0, 1.0, 1.0]))
        flag1_result = get_result_about_Is_same_1D_npArr(np.array([f1_left, f1_bottom, f1_right, f1_top]),
                                                         np.array([0.0, 0.0, 1.0, 1.0]))
        flag2_result = get_result_about_Is_same_1D_npArr(np.array([f2_left, f2_bottom, f2_right, f2_top]),
                                                         np.array([0.0, 0.0, 1.0, 1.0]))

        if generate_yolo_txt_file == True:
            ball_pos = [left, bottom, right, top]
            goal1_pos = [g1_left, g1_bottom, g1_right, g1_top]
            goal2_pos = [g2_left, g2_bottom, g2_right, g2_top]
            flag1_pos = [f1_left, f1_bottom, f1_right, f1_top]
            flag2_pos = [f2_left, f2_bottom, f2_right, f2_top]
            write_txt_file_like_yolo_mark(episodeCount, ball_pos, goal1_pos, goal2_pos, flag1_pos, flag2_pos,
                                          ball_result, goal1_result, goal2_result, flag1_result, flag2_result)

        if episodeCount > start_super_Random_EpisodeCount and episodeCount <= start_super_close_EpisodeCount:
            action = [2, 1, 0, 0]
        elif episodeCount > start_super_close_EpisodeCount:
            action = [2, 2, 0, 0]
        else:
            action = [2, 0, 0, 0]
        actionTuple = ConversionDataType.ConvertList2DiscreteAction(action, behavior_name)
        env.set_actions(behavior_name, actionTuple)

        env.step()
    env.close()
    '''