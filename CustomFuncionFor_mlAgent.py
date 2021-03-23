from PIL import Image
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.side_channel import(
    SideChannel,
    IncomingMessage,
    OutgoingMessage,
)
from mlagents_envs.base_env import ActionTuple
import uuid
import random
import numpy as np

class StringLogChannel(SideChannel):
    def __init__(self) -> None:
        super().__init__(uuid.UUID("621f0a70-4f87-11ea-a6bf-784f4387d1f7"))
    def send_string(self, data: str) -> None:
        msg = OutgoingMessage()
        msg.write_string(data)
        super().queue_message_to_send(msg)
    def on_message_received(self, msg : IncomingMessage):
        print(msg.read_string())

class ConversionDataType:
    def ConvertBehaviorname2Num(self, behavior_name):
        EnvNum = 0
        behavior_name = str(behavior_name).split('?')[0]
        behavior_name = behavior_name[8:]
        EnvNum = int(behavior_name)
        return EnvNum

    def ConvertList2DiscreteAction(self, arr, behavior_name):
        '''
        input data type = list or 1D array -> ex)[3]
                !!! Don't Input 2D Array or list like [(0, 2)]
        output data type = Actiontuple
        '''
        actionList = []
        actionList.append(arr)
        _discrete = np.array(actionList, dtype=np.int32)
        action = ActionTuple(discrete=_discrete)

        return action

    def ChangeArrayDimentionOrder_forPytorch(self, arr):
        '''
        input data shape(Count, width, height, channel)
        output data shape(channel, width, height)
        #The default of Count is 1..

        input_data_type : array
        output_data_type : array

        because pyTorch expect data shape:(batch_size, channel, width, height)
        use this after receive visual_observation from env
        '''
        arr = np.array(self.sliceVisualObservation_ChannelLevel(arr[0], 1))
        arr =  np.squeeze(arr, axis = 3)
        return arr

    def Reduction_Dimention_for_grayIMG(self, arr):
        target_shape = np.shape(arr)
        arr = arr.reshape(target_shape[0], target_shape[1])
        return arr

    def sliceVisualObservation_ChannelLevel(self, vis_obs, slice_channel_size):
        '''
        input shape: (width, height, channel)
        output shape: (channel/slice_channel_size, width, height, slice_channel_size)

        input datatype is numpy array
        output datatype is list with numpy array
        '''
        vis_obs_shape = np.shape(vis_obs)
        vis_obs_list = []
        if(int(vis_obs_shape[2]/slice_channel_size)==3):
            for i in range(int(vis_obs_shape[2]/slice_channel_size)):
                vis_obs_list.append(vis_obs[:,:,i*3:(i+1)*3])

        if(int(slice_channel_size)==1):
            for i in range(int(vis_obs_shape[2]/slice_channel_size)):
                vis_obs_list.append(vis_obs[:,:,i:(i+1)])
        
        return vis_obs_list

    def delete_last_char(self, message):
        message = message[:-1]
        return message

class AgentsHelper:
    def __init__(self, Env, string_log, ConversionDataType):
        self.env = Env
        self.string_log = string_log
        self.ConversionDataType = ConversionDataType
    def print_specs_of_Agents(self, behavior_names):
        for behavior_name in behavior_names:
            spec = self.env.behavior_specs[behavior_name]
            print(f"Name of the behavior : {behavior_name}")
            print("Number of observations : ", len(spec.observation_shapes))
            print("Observation shape : ", spec.observation_shapes)
            vis_obs_bool = any(len(shape) == 3 for shape in spec.observation_shapes)
            print("Is there a visual observation ?", vis_obs_bool)
            print("Is action is discrete ?", spec.action_spec.is_discrete())
            print("Is action is continus ?", spec.action_spec.is_continuous())
            print("\n")
        print("Examine finish....")
        print("======================================")

    def getObservation(self, behavior_name):
        '''
        output data shape(visual_observation):
        -> (num_of_vis_obs_per_behavior_name, vis_obs_chennel, vis_obs_width, vis_obs_height)
        output data shape(vector_observation):
        -> (1, num_of_vec_obs_per_behavior_name*stacked_data_num)

        output datatype(visual_observation)
            -> list array for visual_observation(so use index before use it in main_code)
        output datatype(vector_observation)
            -> array for vector_observation

        if terminal_steps.observations are exist, They overWrite decision_steps.observations
        '''
        decision_steps, terminal_steps = self.env.get_steps(behavior_name)
        spec = self.env.behavior_specs[behavior_name]
        done = False
        vis_obs_list = []
        vec_obs = 0
        
        for index, observation_spec in enumerate(spec.observation_specs):
            shape = observation_spec.shape
            if len(shape) == 3:
                if(terminal_steps.obs[index].size != 0):
                    vis_obs_list.append(terminal_steps.obs[index])
                    done = True
                else:
                    vis_obs_list.append(decision_steps.obs[index])
            elif len(shape) ==1:
                if(terminal_steps.obs[index].size != 0):
                    vec_obs = terminal_steps.obs[index]
                    done = True
                else:
                    vec_obs = decision_steps.obs[index]

        for index, vis_obs in enumerate(vis_obs_list):
            #print(np.shape(vis_obs[0]))
            #vis_obs = self.ConversionDataType.ChangeArrayDimentionOrder_forPytorch(vis_obs[0])
            vis_obs_list[index] = np.uint8(255*vis_obs[0])
        
        return vec_obs, vis_obs_list, done


    def get_reward(self, behavior_name):
        decision_steps, terminal_steps = self.env.get_steps(behavior_name)
        reward = decision_steps.reward
        tr_reward = terminal_steps.reward

        if(np.size(tr_reward)!=0):
            reward = tr_reward
        return reward[0]

    def UpdateEnvLevel(self, env_modes):
        sendMessage = ""
        print(env_modes)
        for index, env_mode in enumerate(env_modes):
            sendMessage +=str(index) + "?" + str(env_mode)+"/"
        sendMessage = self.ConversionDataType.delete_last_char(sendMessage)
        self.string_log.send_string("@" + sendMessage)
        self.env.step()

    def SendMessageToEnv(self, message):
        self.string_log.send_string(message)
        self.env.step()

    def saveArrayAsImagefile(self, array):
        '''
        input shape: (width, height, 3)
        '''
        im = Image.fromarray(array)
        im.save("your_file"+str(random.random())+".jpeg")
