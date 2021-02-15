import os



def dict_to_json1(problem_dict):
    d= '{\"problem\":\"' +problem_dict["problem"]
    if problem_dict["problem"].upper() in ["XXZ","TFIM"]:
        d+='\",\"g\":'+str(problem_dict["g"])+',\"J\":' + str(problem_dict["J"]) + '}'
    return "\'"+d+ "\'"

ori={"problem":"XXZ","g":1, "J":30}
dd = dict_to_json(ori)
ko = dict_to_json1(ori)
print(dd)
print(ko)


os.system("python3 dicts.py --dict {}".format(ko))
os.system("python3 dicts.py --dict {}".format(dd))
