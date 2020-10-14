from datetime import datetime

st = datetime.now()

print(type((datetime.now()-st).total_seconds()))
