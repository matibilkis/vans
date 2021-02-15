
def dict_to_json(dictionary):
    d="{"
    for k,v in dictionary.items():
        if isinstance(k,str) and k!= "geometry":
            d+='\"{}\":\"{}\",'.format(k,v)
        elif k == "geometry":
            skip_next=False
            d+='\"geometry:\"'
            d+='[('
            for tup in v:
                for indi, it in enumerate(tup):
                    if skip_next is True:
                        skip_next=False
                        continue
                    try:
                        k = it.isalpha()
                        if k:
                            d.pop(-1)
                            d+='\"{}\"'.format(it)
                            if not v[indi+1].isalpha():
                                skip_next=True
                    except Exception:
                        if indi == len(tup)-1:
                            d+=","+str(it)
                        elif indi==0:
                            d+=str(it)+"("
                        else:
                            d+=str(it)
                d+='),'
            d+=']'

        else:
            d+='\"{}\":{},'.format(k,v)
    d=d[:-1]
    d+="}" #kill the comma
    return "\'"+d+ "\'"
