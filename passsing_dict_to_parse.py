import os

channel = "depolarizing"
channel_params=[0]
q_batch_size = 10**2

def channel_dict(channel, channel_params, q_batch_size):
    d= '{\
        "channel\":\"' +channel + '\",\"channel_params\":'+str(channel_params)+',\"q_batch_size\":' + str(q_batch_size) + '}'
    return "\'"+d+ "\'"

m = channel_dict(channel, channel_params, q_batch_size)
for d in ['{}', m]:
    os.system("python3 parse_dict.py --noise_model "+d)
    print("\n")
