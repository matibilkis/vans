import os


channel = "depolarizing"
channel_params=[0]
q_batch_size = 10**2

def create_dict(channel, channel_params, q_batch_size):

    d= '{\
        "channel\":\"' +channel + '\",\"channel_params\":'+str(channel_params)+',\"q_batch_size\":' + str(q_batch_size) + '}'
    return "\'"+d+ "\'"


d = create_dict(channel, channel_params, q_batch_size)
os.system("python3 tr.py --noise_model "+d)
